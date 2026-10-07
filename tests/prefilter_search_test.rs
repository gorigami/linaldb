// tests/prefilter_search_test.rs
//
// SORTED index + SEARCH ... PREFILTER (CASMI_WORKLOADS_PLAN.md, P3b/c): a
// per-query mass window applied *before* ranking, then an exact cosine
// top-k over the rows in the window. Checked against an independent brute
// force, with and without the SORTED index.

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn run_err(db: &mut TensorDb, line: &str) -> String {
    match execute_line(db, line, 1) {
        Ok(out) => panic!("`{}` should have failed, got {:?}", line, out),
        Err(e) => e.to_string(),
    }
}

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn lit(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

struct Lib {
    mass: Vec<f64>,
    vecs: Vec<Vec<f32>>,
}

/// 300 library spectra with masses in [100, 400), and 6 queries.
fn setup(db: &mut TensorDb, sorted: bool) -> (Lib, Vec<(i64, f64, Vec<f32>)>) {
    let d = 6;
    let mut seed = 11u64;
    run(
        db,
        &format!(
            "DATASET lib COLUMNS (id: Int, mass: Float64, e: Vector({}))",
            d
        ),
    );
    let mut lib = Lib {
        mass: Vec::new(),
        vecs: Vec::new(),
    };
    for i in 0..300 {
        let mass = 100.0 + ((i * 7919) % 300) as f64 + 0.25;
        let v: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = v
            .iter()
            .map(|x| format!("{:.6}", x).parse().unwrap())
            .collect();
        run(
            db,
            &format!("INSERT INTO lib VALUES ({}, {}, {})", i, mass, lit(&v)),
        );
        lib.mass.push(mass);
        lib.vecs.push(v);
    }
    if sorted {
        run(db, "CREATE SORTED INDEX ON lib(mass)");
    }
    run(
        db,
        &format!(
            "DATASET q COLUMNS (spec: Int, pmass: Float64, e: Vector({}))",
            d
        ),
    );
    let mut queries = Vec::new();
    for j in 0..6 {
        let pmass = 120.0 + 45.0 * j as f64;
        let v: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = v
            .iter()
            .map(|x| format!("{:.6}", x).parse().unwrap())
            .collect();
        run(
            db,
            &format!("INSERT INTO q VALUES ({}, {}, {})", 100 + j, pmass, lit(&v)),
        );
        queries.push((100 + j as i64, pmass, v));
    }
    (lib, queries)
}

fn brute_force(lib: &Lib, q: &[f32], lo: f64, hi: f64, k: usize) -> Vec<i64> {
    let mut scored: Vec<(usize, f32)> = (0..lib.mass.len())
        .filter(|&i| lib.mass[i] >= lo && lib.mass[i] <= hi)
        .map(|i| (i, cosine(q, &lib.vecs[i])))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(i, _)| i as i64).collect()
}

fn hits_by_query(r: &[Vec<Value>]) -> std::collections::BTreeMap<i64, Vec<i64>> {
    let mut m = std::collections::BTreeMap::new();
    for row in r {
        let (Value::Int(q), Value::Int(id)) = (&row[0], &row[4]) else {
            panic!("{:?}", row)
        };
        m.entry(*q).or_insert_with(Vec::new).push(*id);
    }
    m
}

#[test]
fn per_query_mass_window_matches_brute_force_with_and_without_sorted_index() {
    for sorted in [false, true] {
        let (_dir, mut db) = db();
        let (lib, queries) = setup(&mut db, sorted);
        let r = rows(run(
            &mut db,
            "SEARCH lib ON e QUERIES q.e KEY spec PREFILTER mass BETWEEN q.pmass - 20.0 AND q.pmass + 20.0 LIMIT 5",
        ));
        let got = hits_by_query(&r);
        for (spec, pmass, v) in &queries {
            let expected = brute_force(&lib, v, pmass - 20.0, pmass + 20.0, 5);
            assert_eq!(
                got.get(spec).cloned().unwrap_or_default(),
                expected,
                "sorted={} spec={}",
                sorted,
                spec
            );
            assert_eq!(expected.len(), 5, "fixture: every window holds >= 5 rows");
        }
    }
}

#[test]
fn ppm_window_and_sorted_index_shows_in_explain() {
    let (_dir, mut db) = db();
    let (lib, queries) = setup(&mut db, true);
    let ppm = "mass >= q.pmass - q.pmass * 0.05 AND mass <= q.pmass + q.pmass * 0.05";
    let r = rows(run(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERIES q.e KEY spec PREFILTER {} LIMIT 3",
            ppm
        ),
    ));
    let got = hits_by_query(&r);
    for (spec, pmass, v) in &queries {
        let expected = brute_force(&lib, v, pmass - pmass * 0.05, pmass + pmass * 0.05, 3);
        assert_eq!(got[spec], expected);
    }
    let plan = format!(
        "{:?}",
        run(
            &mut db,
            &format!(
                "EXPLAIN SEARCH lib ON e QUERIES q.e PREFILTER {} LIMIT 3",
                ppm
            )
        )
    );
    assert!(
        plan.contains("narrowed by SORTED index on mass"),
        "{}",
        plan
    );
}

#[test]
fn prefilter_returns_k_rows_where_post_filter_cannot() {
    let (_dir, mut db) = db();
    let (lib, queries) = setup(&mut db, true);
    run(&mut db, "CREATE VECTOR INDEX ON lib(e)");
    let (_, _, v) = &queries[0];
    let post = rows(run(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERY {} LIMIT 5 FILTER mass < 110.0",
            lit(v)
        ),
    ));
    let pre = rows(run(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERY {} PREFILTER mass < 110.0 LIMIT 5",
            lit(v)
        ),
    ));
    assert!(
        post.len() < 5,
        "the post-filter fixture should starve: {}",
        post.len()
    );
    assert_eq!(pre.len(), 5);
    let ids: Vec<i64> = pre
        .iter()
        .map(|r| match r[0] {
            Value::Int(i) => i,
            _ => panic!(),
        })
        .collect();
    assert_eq!(ids, brute_force(&lib, v, f64::MIN, 110.0 - 1e-9, 5));
    // Same output shape as plain SEARCH: the dataset's own columns.
    assert_eq!(pre[0].len(), 3);
}

#[test]
fn prefilter_needs_no_vector_index_and_window_can_be_empty() {
    let (_dir, mut db) = db();
    setup(&mut db, false);
    let r = rows(run(
        &mut db,
        "SEARCH lib ON e QUERIES q.e PREFILTER mass > 100000.0 LIMIT 5",
    ));
    assert!(r.is_empty());
}

#[test]
fn where_ranges_through_a_sorted_index_match_a_full_scan() {
    let (_dir, mut a) = db();
    let (_dir2, mut b) = db();
    setup(&mut a, true);
    setup(&mut b, false);
    for pred in [
        "mass BETWEEN 150.0 AND 180.5",
        "mass > 390.0",
        "mass = 200.25",
        "mass >= 300.0 AND id < 100",
        "id < 50 AND mass <= 160.0",
    ] {
        let q = format!("SELECT id FROM lib WHERE {} ORDER BY id", pred);
        assert_eq!(rows(run(&mut a, &q)), rows(run(&mut b, &q)), "{}", pred);
        let plan = format!(
            "{:?}",
            run(
                &mut a,
                &format!("EXPLAIN SELECT id FROM lib WHERE {}", pred)
            )
        );
        assert!(plan.contains("SortedRangeScanExec"), "{}: {}", pred, plan);
    }
    // Rows inserted after the index was built, and after UPDATE/DELETE.
    run(&mut a, "INSERT INTO lib VALUES (999, 151.0, [1,0,0,0,0,0])");
    run(&mut b, "INSERT INTO lib VALUES (999, 151.0, [1,0,0,0,0,0])");
    run(&mut a, "UPDATE lib SET mass = 155.5 WHERE id = 3");
    run(&mut b, "UPDATE lib SET mass = 155.5 WHERE id = 3");
    run(&mut a, "DELETE FROM lib WHERE id = 10");
    run(&mut b, "DELETE FROM lib WHERE id = 10");
    let q = "SELECT id FROM lib WHERE mass BETWEEN 150.0 AND 160.0 ORDER BY id";
    assert_eq!(rows(run(&mut a, q)), rows(run(&mut b, q)));
}

#[test]
fn sorted_index_survives_save_and_load() {
    let (dir, mut db) = db();
    setup(&mut db, true);
    run(&mut db, "SAVE DATASET lib");
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db = TensorDb::with_config(config);
    let msg = format!("{:?}", run(&mut db, "LOAD DATASET lib"));
    assert!(msg.contains("indices restored on: mass"), "{}", msg);
    let plan = format!(
        "{:?}",
        run(&mut db, "EXPLAIN SELECT id FROM lib WHERE mass > 300.0")
    );
    assert!(plan.contains("SortedRangeScanExec"), "{}", plan);
}

#[test]
fn loud_errors() {
    let (_dir, mut db) = db();
    setup(&mut db, true);
    let e = run_err(
        &mut db,
        "SEARCH lib ON e QUERIES q.e PREFILTER mass < q.nope LIMIT 2",
    );
    assert!(e.contains("unknown query column 'q.nope'"), "{}", e);
    let e = run_err(
        &mut db,
        "SEARCH lib ON e QUERIES q.e PREFILTER nope < 3 LIMIT 2",
    );
    assert!(e.contains("unknown column 'nope'"), "{}", e);
    run(&mut db, "DATASET v COLUMNS (e: Vector(2))");
    run(&mut db, "INSERT INTO v VALUES ([1.0, 2.0])");
    let e = run_err(&mut db, "CREATE SORTED INDEX ON v(e)");
    assert!(e.contains("SORTED index supports"), "{}", e);
}
