// tests/search_using_test.rs
//
// SEARCH ... USING <expression> (top-k by any score) and CANDIDATES n RERANK
// USING <expression> (two-stage ranking) -- CASMI_WORKLOADS_PLAN_2.md, P9 and
// P13. Each result is checked against a brute-force SELECT ... ORDER BY per
// query, and USING COSINE_SIM against the plain PREFILTER search.

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

fn table(out: DslOutput) -> (Vec<String>, Vec<Vec<Value>>) {
    match out {
        DslOutput::Table(ds) => (
            ds.schema.fields.iter().map(|f| f.name.clone()).collect(),
            ds.rows.iter().map(|r| r.values.clone()).collect(),
        ),
        other => panic!("expected a table, got {:?}", other),
    }
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x as f64,
        Value::Float64(x) => *x,
        Value::Int(x) => *x as f64,
        other => panic!("expected a number, got {:?}", other),
    }
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Int(x) => *x,
        other => panic!("expected an Int, got {:?}", other),
    }
}

/// Deterministic pseudo-random numbers in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn peaks_literal(rng: &mut Lcg, base: &[f64]) -> String {
    let mut mz: Vec<f64> = Vec::new();
    for m in base {
        if rng.next() < 0.8 {
            mz.push(((m + (rng.next() - 0.5) * 0.004) * 1000.0).round() / 1000.0);
        }
    }
    mz.push(((600.0 + rng.next() * 50.0) * 1000.0).round() / 1000.0);
    mz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mz.dedup();
    let int: Vec<String> = mz
        .iter()
        .map(|_| format!("{:.2}", 1.0 + rng.next() * 99.0))
        .collect();
    let mz: Vec<String> = mz.iter().map(|m| format!("{:.3}", m)).collect();
    format!("[[{}], [{}]]", mz.join(", "), int.join(", "))
}

/// `lib` (40 rows) and `q` (6 queries): masses, peak lists drawn from a few
/// shared fragment sets (so scores are informative), 4-D vectors, and 8-bit
/// fingerprints with per-bit query weights.
fn setup(db: &mut TensorDb) {
    let mut rng = Lcg(42);
    let families: [&[f64]; 3] = [
        &[69.04, 124.05, 142.06, 181.07],
        &[110.07, 138.07, 163.06, 195.09],
        &[118.07, 132.08, 146.06, 188.07, 205.10],
    ];
    run(
        db,
        "DATASET lib COLUMNS (id: Int, mass: DOUBLE, peaks: Matrix(2, *), e: Vector(4), fp: BitVector(8))",
    );
    for i in 0..40 {
        let fam = families[i % 3];
        let mass = 150.0 + (i as f64) * 7.0;
        let e: Vec<String> = (0..4).map(|_| format!("{:.4}", rng.next())).collect();
        let fp: String = (0..8)
            .map(|_| if rng.next() < 0.5 { '1' } else { '0' })
            .collect();
        run(
            db,
            &format!(
                "INSERT INTO lib VALUES ({}, {:.2}, {}, [{}], \"{}\")",
                i,
                mass,
                peaks_literal(&mut rng, fam),
                e.join(", "),
                fp
            ),
        );
    }
    run(
        db,
        "DATASET q COLUMNS (qid: Int, mass: DOUBLE, peaks: Matrix(2, *), e: Vector(4), z: Vector(8))",
    );
    for i in 0..6 {
        let fam = families[i % 3];
        let e: Vec<String> = (0..4).map(|_| format!("{:.4}", rng.next())).collect();
        let z: Vec<String> = (0..8).map(|_| format!("{:.3}", rng.next())).collect();
        run(
            db,
            &format!(
                "INSERT INTO q VALUES ({}, {:.2}, {}, [{}], [{}])",
                100 + i,
                200.0 + (i as f64) * 30.0,
                peaks_literal(&mut rng, fam),
                e.join(", "),
                z.join(", ")
            ),
        );
    }
    run(db, "CREATE SORTED INDEX ON lib(mass)");
}

/// The query rows, as `(qid, mass, peaks literal, e literal, z literal)`,
/// read back so brute-force SELECTs can inline them.
fn queries(db: &mut TensorDb) -> Vec<(i64, f64, String, String, String)> {
    let lit = |v: &Value| match v {
        Value::Matrix(m) => format!(
            "[{}]",
            m.iter()
                .map(|r| format!(
                    "[{}]",
                    r.iter()
                        .map(|x| format!("{}", x))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Vector(v) => format!(
            "[{}]",
            v.iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => panic!("{:?}", other),
    };
    let (_, rows) = table(run(db, "SELECT qid, mass, peaks, e, z FROM q ORDER BY qid"));
    rows.iter()
        .map(|r| (int(&r[0]), num(&r[1]), lit(&r[2]), lit(&r[3]), lit(&r[4])))
        .collect()
}

/// `(query_id, row_id, score)` of a batch SEARCH, in output order.
fn hits(db: &mut TensorDb, sql: &str) -> Vec<(i64, i64, f64)> {
    let (cols, rows) = table(run(db, sql));
    let at = |n: &str| cols.iter().position(|c| c == n).unwrap();
    let (q, r, s) = (at("query_id"), at("row_id"), at("score"));
    rows.iter()
        .map(|row| (int(&row[q]), int(&row[r]), num(&row[s])))
        .collect()
}

/// Per query, `SELECT id, <score> ... WHERE <window> ORDER BY s <dir>, id
/// LIMIT k` -- the brute-force answer (`lib.id` is the row position).
fn brute_force(
    db: &mut TensorDb,
    qs: &[(i64, f64, String, String, String)],
    score: impl Fn(&(i64, f64, String, String, String)) -> String,
    window: impl Fn(&(i64, f64, String, String, String)) -> String,
    ascending: bool,
    k: usize,
) -> Vec<(i64, i64, f64)> {
    let mut out = Vec::new();
    for q in qs {
        let (_, rows) = table(run(
            db,
            &format!(
                "SELECT id, {} AS s FROM lib WHERE {} ORDER BY s {}, id ASC LIMIT {}",
                score(q),
                window(q),
                if ascending { "ASC" } else { "DESC" },
                k
            ),
        ));
        out.extend(rows.iter().map(|r| (q.0, int(&r[0]), num(&r[1]))));
    }
    out
}

fn assert_same(got: &[(i64, i64, f64)], want: &[(i64, i64, f64)], tol: f64) {
    assert_eq!(got.len(), want.len(), "{:?}\nvs\n{:?}", got, want);
    for (g, w) in got.iter().zip(want) {
        assert_eq!((g.0, g.1), (w.0, w.1), "{:?}\nvs\n{:?}", got, want);
        assert!((g.2 - w.2).abs() <= tol, "{:?} vs {:?}", g, w);
    }
}

#[test]
fn using_cosine_equals_the_plain_prefilter_search() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let window = "PREFILTER mass BETWEEN q.mass - 60 AND q.mass + 60";
    let plain = hits(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERIES q.e KEY qid {} RETURN NONE LIMIT 5",
            window
        ),
    );
    let using = hits(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERIES q.e KEY qid USING COSINE_SIM(e, q.e) {} RETURN NONE LIMIT 5",
            window
        ),
    );
    assert!(!plain.is_empty());
    assert_same(&using, &plain, 1e-6);
}

#[test]
fn using_spectral_and_dot_scores_equal_brute_force() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let qs = queries(&mut db);
    let narrow = |q: &(i64, f64, String, String, String)| {
        format!("mass BETWEEN {} - 60 AND {} + 60", q.1, q.1)
    };
    let all = |_: &(i64, f64, String, String, String)| "true".to_string();

    let got = hits(&mut db, "SEARCH lib ON peaks QUERIES q.peaks KEY qid USING SPEC_ENTROPY(peaks, q.peaks, 0.01) PREFILTER mass BETWEEN q.mass - 60 AND q.mass + 60 RETURN NONE LIMIT 4");
    let want = brute_force(
        &mut db,
        &qs,
        |q| format!("SPEC_ENTROPY(peaks, {}, 0.01)", q.2),
        narrow,
        false,
        4,
    );
    assert_same(&got, &want, 1e-12);

    // Analog search: a wide window and a per-pair shift.
    let got = hits(&mut db, "SEARCH lib ON peaks QUERIES q.peaks KEY qid USING SPEC_COSINE_MOD(peaks, q.peaks, 0.01, mass - q.mass) PREFILTER mass BETWEEN q.mass - 200 AND q.mass + 200 RETURN NONE LIMIT 5");
    let want = brute_force(
        &mut db,
        &qs,
        |q| format!("SPEC_COSINE_MOD(peaks, {}, 0.01, mass - {})", q.2, q.1),
        |q| format!("mass BETWEEN {} - 200 AND {} + 200", q.1, q.1),
        false,
        5,
    );
    assert_same(&got, &want, 1e-12);

    // Fingerprint scoring: unnormalised f . z, no window (a full scan).
    let got = hits(
        &mut db,
        "SEARCH lib ON fp QUERIES q.z KEY qid USING DOT(fp, q.z) RETURN NONE LIMIT 6",
    );
    let want = brute_force(&mut db, &qs, |q| format!("DOT(fp, {})", q.4), all, false, 6);
    assert_same(&got, &want, 1e-6);

    // A distance, lowest first.
    let got = hits(
        &mut db,
        "SEARCH lib ON e QUERIES q.e KEY qid USING DISTANCE(e, q.e) ASC RETURN NONE LIMIT 3",
    );
    let want = brute_force(
        &mut db,
        &qs,
        |q| format!("DISTANCE(e, {})", q.3),
        all,
        true,
        3,
    );
    assert_same(&got, &want, 1e-6);
}

#[test]
fn ties_keep_row_order_and_null_scores_are_skipped() {
    let (_dir, mut db) = db();
    run(
        &mut db,
        "DATASET t COLUMNS (id: Int, x: Float?, v: Vector(1))",
    );
    for (id, x) in [(0, "2.0"), (1, "NULL"), (2, "2.0"), (3, "1.0"), (4, "2.0")] {
        run(
            &mut db,
            &format!("INSERT INTO t VALUES ({}, {}, [1.0])", id, x),
        );
    }
    run(&mut db, "DATASET q COLUMNS (k: Int, v: Vector(1))");
    run(&mut db, "INSERT INTO q VALUES (1, [1.0])");
    let r = hits(
        &mut db,
        "SEARCH t ON v QUERIES q.v KEY k USING x RETURN NONE LIMIT 10",
    );
    let ids: Vec<i64> = r.iter().map(|h| h.1).collect();
    assert_eq!(ids, [0, 2, 4, 3]);
    let r = hits(
        &mut db,
        "SEARCH t ON v QUERIES q.v KEY k USING x ASC RETURN NONE LIMIT 10",
    );
    let ids: Vec<i64> = r.iter().map(|h| h.1).collect();
    assert_eq!(ids, [3, 0, 2, 4]);
}

#[test]
fn rerank_equals_the_two_stages_run_separately() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let qs = queries(&mut db);
    let (cols, rows) = table(run(&mut db, "SEARCH lib ON e QUERIES q.e KEY qid USING DOT(fp, q.z) CANDIDATES 8 RERANK USING SPEC_ENTROPY(peaks, q.peaks, 0.01) RETURN id LIMIT 3"));
    assert_eq!(
        cols,
        ["query_id", "rank", "score", "score_stage1", "row_id", "id"]
    );
    for q in &qs {
        // Stage 1 by hand: the top 8 by DOT; stage 2: those, by entropy.
        let stage1 = brute_force(
            &mut db,
            std::slice::from_ref(q),
            |q| format!("DOT(fp, {})", q.4),
            |_| "true".to_string(),
            false,
            8,
        );
        let ids: Vec<String> = stage1.iter().map(|h| h.1.to_string()).collect();
        let want = brute_force(
            &mut db,
            std::slice::from_ref(q),
            |q| format!("SPEC_ENTROPY(peaks, {}, 0.01)", q.2),
            |_| format!("id IN ({})", ids.join(", ")),
            false,
            3,
        );
        let got: Vec<&Vec<Value>> = rows.iter().filter(|r| int(&r[0]) == q.0).collect();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(int(&g[4]), w.1);
            assert!((num(&g[2]) - w.2).abs() < 1e-12);
            let s1 = stage1.iter().find(|h| h.1 == w.1).unwrap().2;
            assert!((num(&g[3]) - s1).abs() < 1e-6);
        }
    }
    // RERANK after the default cosine stage; score_stage1 is then the cosine (Float).
    let (_, rows) = table(run(&mut db, "SEARCH lib ON e QUERIES q.e KEY qid CANDIDATES 10 RERANK USING SPEC_ENTROPY(peaks, q.peaks, 0.01) RETURN NONE LIMIT 2"));
    assert_eq!(rows.len(), 12);
    assert!(matches!(rows[0][3], Value::Float(_)));
    assert!(matches!(rows[0][2], Value::Float64(_)));
}

#[test]
fn explain_and_lineage_show_the_scores() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let plan = match run(&mut db, "EXPLAIN SEARCH lib ON peaks QUERIES q.peaks KEY qid USING DOT(fp, q.z) PREFILTER mass BETWEEN q.mass - 10 AND q.mass + 10 CANDIDATES 20 RERANK USING SPEC_ENTROPY(peaks, q.peaks, 0.01) LIMIT 5") {
        DslOutput::Message(m) => m,
        other => panic!("{:?}", other),
    };
    assert!(
        plan.contains(
            "DOT(fp, q.z) (top 20), then RERANK USING SPEC_ENTROPY(peaks, q.peaks, 0.01)"
        ),
        "{}",
        plan
    );
    assert!(
        plan.contains("narrowed by SORTED index on mass"),
        "{}",
        plan
    );

    run(&mut db, "SEARCH lib ON e QUERIES q.e KEY qid USING DISTANCE(e, q.e) ASC CANDIDATES 10 RERANK USING SPEC_ENTROPY(peaks, q.peaks, 0.01) RETURN NONE LIMIT 3 INTO hits");
    let lineage = match run(&mut db, "EXPLAIN LINEAGE hits AS JSON") {
        DslOutput::Message(m) => m,
        other => panic!("{:?}", other),
    };
    for want in [
        "\"using\": \"DISTANCE(e, q.e) ASC\"",
        "\"candidates\": 10",
        "\"rerank\": \"SPEC_ENTROPY(peaks, q.peaks, 0.01)\"",
    ] {
        assert!(lineage.contains(want), "{} not in {}", want, lineage);
    }
}

#[test]
fn loud_errors_before_any_row_runs() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let base = "SEARCH lib ON peaks QUERIES q.peaks KEY qid";
    let cases = [
        (
            format!("{} USING peaks LIMIT 3", base),
            "the score must be a number",
        ),
        (
            format!("{} USING SPEC_ENTROPY(peaks, q.nope, 0.01) LIMIT 3", base),
            "unknown query column 'q.nope'",
        ),
        (
            format!("{} USING SPEC_ENTROPY(nope, q.peaks, 0.01) LIMIT 3", base),
            "unknown column 'nope'",
        ),
        (
            format!("{} USING SPEC_ENTROPY(peaks, q.peaks) LIMIT 3", base),
            "takes 3 to 4 arguments",
        ),
        (
            format!(
                "{} USING SPEC_ENTROPY(peaks, q.peaks, 0.01) PREFILTER true APPROX LIMIT 3",
                base
            ),
            "drop APPROX",
        ),
        (
            format!(
                "{} USING DOT(fp, q.z) CANDIDATES 2 RERANK USING DOT(fp, q.z) LIMIT 3",
                base
            ),
            "CANDIDATES 2 is fewer than LIMIT 3",
        ),
        (
            format!("{} CANDIDATES 5 USING DOT(fp, q.z) LIMIT 3", base),
            "expected RERANK",
        ),
        (
            "SEARCH lib ON e QUERY [1.0, 0.0, 0.0, 0.0] USING DISTANCE(e, e) LIMIT 3".to_string(),
            "needs queries from a dataset",
        ),
        (
            "SEARCH lib ON fp QUERIES q.e KEY qid USING DOT(fp, q.e) LIMIT 3".to_string(),
            "BitVector has 8 bits, the Vector 4 elements",
        ),
        (
            "SEARCH lib ON fp QUERIES q.z KEY qid USING COSINE_SIM(fp, q.z) LIMIT 3".to_string(),
            "for a BitVector use TANIMOTO",
        ),
    ];
    for (sql, want) in cases {
        let e = run_err(&mut db, &sql);
        assert!(e.contains(want), "`{}`: expected '{}' in: {}", sql, want, e);
    }
    // A score error inside a row (unsorted peaks) fails the statement.
    run(&mut db, "INSERT INTO lib VALUES (99, 500.0, [[2.0, 1.0], [1.0, 1.0]], [0.1, 0.1, 0.1, 0.1], \"00000000\")");
    let e = run_err(
        &mut db,
        &format!("{} USING SPEC_ENTROPY(peaks, q.peaks, 0.01) LIMIT 3", base),
    );
    assert!(e.contains("sorted ascending"), "{}", e);
}
