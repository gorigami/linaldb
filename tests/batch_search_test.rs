// tests/batch_search_test.rs
//
// SEARCH ... QUERIES: top-k for many query vectors in one statement
// (CASMI_WORKLOADS_PLAN.md, P3a). Results must equal an independent brute
// force cosine ranking (below the IVF clustering threshold the index is an
// exact scan) and equal the single-query SEARCH for every query.

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

/// Deterministic pseudo-random vectors (LCG), no ties in practice.
fn vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        })
        .collect()
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Library of `n` vectors + a query dataset; returns the parsed-back
/// (rounded) library and query vectors exactly as the engine stores them.
fn setup(db: &mut TensorDb, n: usize, nq: usize, hnsw: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let d = 8;
    run(
        db,
        &format!(
            "DATASET lib COLUMNS (id: Int, mass: Float, e: Vector({}))",
            d
        ),
    );
    let round = |v: &Vec<f32>| -> Vec<f32> {
        v.iter()
            .map(|x| format!("{:.6}", x).parse().unwrap())
            .collect()
    };
    let lib: Vec<Vec<f32>> = vectors(n, d, 7).iter().map(round).collect();
    for (i, v) in lib.iter().enumerate() {
        run(
            db,
            &format!("INSERT INTO lib VALUES ({}, {}.0, {})", i, 100 + i, lit(v)),
        );
    }
    run(
        db,
        &format!(
            "CREATE VECTOR INDEX ON lib(e){}",
            if hnsw { " USING HNSW" } else { "" }
        ),
    );
    run(
        db,
        &format!("DATASET q COLUMNS (name: String, e: Vector({}))", d),
    );
    let qs: Vec<Vec<f32>> = vectors(nq, d, 99).iter().map(round).collect();
    for (i, v) in qs.iter().enumerate() {
        run(
            db,
            &format!("INSERT INTO q VALUES (\"q{}\", {})", i, lit(v)),
        );
    }
    (lib, qs)
}

#[test]
fn batch_top_k_matches_brute_force() {
    let (_dir, mut db) = db();
    let (lib, qs) = setup(&mut db, 40, 5, false); // < 64 rows: exact scan
    let k = 4;
    let r = rows(run(
        &mut db,
        &format!("SEARCH lib ON e QUERIES q.e KEY name LIMIT {}", k),
    ));
    assert_eq!(r.len(), qs.len() * k);
    for (qi, q) in qs.iter().enumerate() {
        let mut expected: Vec<(usize, f32)> = lib
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(q, v)))
            .collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (rank, (row_id, score)) in expected.iter().take(k).enumerate() {
            let row = &r[qi * k + rank];
            assert_eq!(row[0], Value::String(format!("q{}", qi)));
            assert_eq!(row[1], Value::Int(rank as i64 + 1));
            match row[2] {
                Value::Float(s) => assert!((s - score).abs() < 1e-5, "{} vs {}", s, score),
                ref other => panic!("score should be Float, got {:?}", other),
            }
            assert_eq!(row[3], Value::Int(*row_id as i64));
            assert_eq!(row[4], Value::Int(*row_id as i64)); // lib.id == row position
        }
    }
}

#[test]
fn batch_equals_single_query_search_with_ivf_and_hnsw() {
    for hnsw in [false, true] {
        let (_dir, mut db) = db();
        let (_lib, qs) = setup(&mut db, 300, 4, hnsw);
        let batch = rows(run(&mut db, "SEARCH lib ON e QUERIES q.e LIMIT 5"));
        for (qi, q) in qs.iter().enumerate() {
            let single = rows(run(
                &mut db,
                &format!("SEARCH lib ON e QUERY {} LIMIT 5", lit(q)),
            ));
            let batch_ids: Vec<Value> = batch
                .iter()
                .filter(|r| r[0] == Value::Int(qi as i64))
                .map(|r| r[4].clone())
                .collect();
            let single_ids: Vec<Value> = single.iter().map(|r| r[0].clone()).collect();
            assert_eq!(batch_ids, single_ids, "hnsw={} query {}", hnsw, qi);
        }
    }
}

#[test]
fn matrix_tensor_queries_and_filter_on_rank_and_score() {
    let (_dir, mut db) = db();
    let (_lib, qs) = setup(&mut db, 30, 2, false);
    run(
        &mut db,
        &format!("MATRIX qm = [{}, {}]", lit(&qs[0]), lit(&qs[1])),
    );
    let all = rows(run(&mut db, "SEARCH lib ON e QUERIES qm LIMIT 3"));
    assert_eq!(all.len(), 6);
    assert_eq!(all[0][0], Value::Int(0));
    assert_eq!(all[3][0], Value::Int(1));

    let top1 = rows(run(
        &mut db,
        "SEARCH lib ON e QUERIES qm LIMIT 3 FILTER rank = 1",
    ));
    assert_eq!(top1.len(), 2);

    run(
        &mut db,
        "SEARCH lib ON e QUERIES qm LIMIT 3 FILTER mass >= 110.0 INTO hits",
    );
    let ds = db.get_dataset("hits").unwrap();
    assert!(ds
        .rows
        .iter()
        .all(|r| matches!(r.values[5], Value::Float(m) if m >= 110.0)));
    let names: Vec<&str> = ds.schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["query_id", "rank", "score", "row_id", "id", "mass", "e"]
    );
}

#[test]
fn explain_reports_the_batch_operator_and_index() {
    let (_dir, mut db) = db();
    setup(&mut db, 20, 3, true);
    let out = run(&mut db, "EXPLAIN SEARCH lib ON e QUERIES q.e LIMIT 2");
    let text = format!("{:?}", out);
    assert!(text.contains("BatchVectorSearchExec"), "{}", text);
    assert!(text.contains("queries: 3"), "{}", text);
    assert!(text.contains("Hnsw"), "{}", text);
}

#[test]
fn loud_errors() {
    let (_dir, mut db) = db();
    setup(&mut db, 10, 2, false);
    run(&mut db, "MATRIX bad = [[1.0, 2.0], [3.0, 4.0]]");
    let e = run_err(&mut db, "SEARCH lib ON e QUERIES bad LIMIT 2");
    assert!(e.contains("dimension 2"), "{}", e);

    run(&mut db, "VECTOR one = [1.0, 2.0]");
    let e = run_err(&mut db, "SEARCH lib ON e QUERIES one LIMIT 2");
    assert!(e.contains("2-D matrix"), "{}", e);

    let e = run_err(&mut db, "SEARCH lib ON e QUERIES q.name LIMIT 2");
    assert!(e.contains("not a Vector"), "{}", e);

    let e = run_err(&mut db, "SEARCH lib ON e QUERIES q.e KEY nope LIMIT 2");
    assert!(e.contains("KEY column 'nope'"), "{}", e);

    let e = run_err(&mut db, "SEARCH lib ON e QUERIES bad KEY x LIMIT 2");
    assert!(e.contains("KEY needs a dataset"), "{}", e);

    run(
        &mut db,
        "DATASET clash COLUMNS (score: Float, e: Vector(8))",
    );
    run(&mut db, "INSERT INTO clash VALUES (1.0, [1,0,0,0,0,0,0,0])");
    run(&mut db, "CREATE VECTOR INDEX ON clash(e)");
    let e = run_err(&mut db, "SEARCH clash ON e QUERIES q.e LIMIT 1");
    assert!(e.contains("collides"), "{}", e);
}

// ── SEARCH ... RETURN (CASMI_WORKLOADS_PLAN_2.md, P10) ──────────────────────

fn table(out: DslOutput) -> (Vec<String>, Vec<Vec<Value>>) {
    match out {
        DslOutput::Table(ds) => (
            ds.schema.fields.iter().map(|f| f.name.clone()).collect(),
            ds.rows.iter().map(|r| r.values.clone()).collect(),
        ),
        other => panic!("expected a table, got {:?}", other),
    }
}

#[test]
fn return_projects_hit_columns_without_changing_hits() {
    let (_dir, mut db) = db();
    setup(&mut db, 40, 3, false);
    let base = "SEARCH lib ON e QUERIES q.e KEY name PREFILTER mass >= 110.0";
    let (_, full) = table(run(&mut db, &format!("{} LIMIT 5", base)));
    let (cols, some) = table(run(&mut db, &format!("{} RETURN mass, id LIMIT 5", base)));
    assert_eq!(cols, ["query_id", "rank", "score", "row_id", "mass", "id"]);
    let (cols, none) = table(run(&mut db, &format!("{} RETURN NONE LIMIT 5", base)));
    assert_eq!(cols, ["query_id", "rank", "score", "row_id"]);
    assert_eq!(full.len(), 15);
    for ((f, s), n) in full.iter().zip(&some).zip(&none) {
        assert_eq!(f[..4], s[..4]);
        assert_eq!(f[..4], n[..]);
        assert_eq!(s[4], f[5]); // mass
        assert_eq!(s[5], f[4]); // id
    }
    // Index path (no PREFILTER), and FILTER over a returned column.
    let (cols, r) = table(run(
        &mut db,
        "SEARCH lib ON e QUERIES q.e RETURN id LIMIT 3 FILTER rank = 1",
    ));
    assert_eq!(cols, ["query_id", "rank", "score", "row_id", "id"]);
    assert_eq!(r.len(), 3);
    // Single query: the dataset's own columns, projected.
    let (cols, r) = table(run(
        &mut db,
        "SEARCH lib ON e QUERY [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] RETURN id LIMIT 2",
    ));
    assert_eq!(cols, ["id"]);
    assert_eq!(r.len(), 2);
    // INTO stores only the projected columns.
    run(&mut db, &format!("{} RETURN id LIMIT 2 INTO hits", base));
    let (cols, _) = table(run(&mut db, "SELECT * FROM hits"));
    assert_eq!(cols, ["query_id", "rank", "score", "row_id", "id"]);
    let lineage = match run(&mut db, "EXPLAIN LINEAGE hits AS JSON") {
        DslOutput::Message(m) => m,
        other => panic!("{:?}", other),
    };
    assert!(
        lineage.contains("\"return\": [\n      \"id\"\n    ]"),
        "{}",
        lineage
    );

    let e = run_err(&mut db, &format!("{} RETURN nope LIMIT 2", base));
    assert!(e.contains("unknown column 'nope'"), "{}", e);
    let e = run_err(&mut db, &format!("{} RETURN id, id LIMIT 2", base));
    assert!(e.contains("listed twice"), "{}", e);
    let e = run_err(
        &mut db,
        "SEARCH lib ON e QUERY [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] RETURN NONE LIMIT 2",
    );
    assert!(e.contains("RETURN NONE needs a batch"), "{}", e);
}
