// tests/sparse_vector_test.rs
//
// SparseVector(dim) columns (CASMI_WORKLOADS_PLAN.md, P2): results must
// equal the dense equivalent -- bit for bit, since the sparse loops visit
// the same nonzero terms in the same order -- and bad input must fail.

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

fn dense(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..40)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x % 4 == 0 {
                ((x >> 40) % 1000) as f32 / 37.0 - 10.0
            } else {
                0.0
            }
        })
        .collect()
}

fn lit(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The same 10 vectors in a SparseVector(40) and a Vector(40) column.
fn setup(db: &mut TensorDb) -> Vec<Vec<f32>> {
    run(
        db,
        "DATASET t COLUMNS (id: Int, s: SparseVector(40), d: Vector(40), mass: Float)",
    );
    let all: Vec<Vec<f32>> = (1..=10).map(dense).collect();
    for (i, v) in all.iter().enumerate() {
        run(
            db,
            &format!(
                "INSERT INTO t VALUES ({}, {}, {}, {}.0)",
                i,
                lit(v),
                lit(v),
                100 + i
            ),
        );
    }
    all
}

#[test]
fn equals_dense_bit_for_bit() {
    let (_dir, mut db) = db();
    let all = setup(&mut db);
    let q = lit(&all[2]);
    let r = rows(run(
        &mut db,
        &format!(
            "SELECT COSINE_SIM(s, {q}) AS cs, COSINE_SIM(d, {q}) AS cd, DOT(s, d) AS ds, DOT(d, d) AS dd, L2_NORM(s) AS ns, L2_NORM(d) AS nd, COSINE_SIM(s, s) AS css, CAST(NORMALIZE(s) AS VECTOR(40)) AS us, NORMALIZE(d) AS ud, CAST(VEC_SCALE(s, 3) AS VECTOR(40)) AS ks, VEC_SCALE(d, 3) AS kd FROM t ORDER BY id"
        ),
    ));
    for row in &r {
        for (a, b) in [(0, 1), (2, 3), (4, 5), (7, 8), (9, 10)] {
            assert_eq!(row[a], row[b], "columns {} and {}: {:?}", a, b, row);
        }
    }
    // A sparse vector stores only its nonzeros.
    let ds = run(&mut db, "SELECT s FROM t WHERE id = 0");
    let nnz = all[0].iter().filter(|x| **x != 0.0).count();
    match &rows(ds)[0][0] {
        Value::SparseVector(s) => assert_eq!(s.nnz(), nnz),
        other => panic!("{:?}", other),
    }
}

#[test]
fn constructor_cast_and_exact_search() {
    let (_dir, mut db) = db();
    let all = setup(&mut db);
    let r = rows(run(
        &mut db,
        "SELECT CAST(SPARSE(5, [1, 3], [2.0, -1.0]) AS VECTOR(5)) AS v, CAST(SPARSE(5, [1, 3], [2.0, -1.0]) AS TEXT) AS t",
    ));
    assert_eq!(r[0][0], Value::Vector(vec![0.0, 2.0, 0.0, -1.0, 0.0]));
    run(
        &mut db,
        "UPDATE t SET s = CAST(d AS SPARSEVECTOR(40)) WHERE id = 1",
    );

    // No vector index on a sparse column -- searched exactly via PREFILTER.
    let e = run_err(&mut db, "CREATE VECTOR INDEX ON t(s)");
    assert!(e.contains("need a dense Vector column"), "{}", e);
    let e = run_err(
        &mut db,
        &format!("SEARCH t ON s QUERY {} LIMIT 3", lit(&all[4])),
    );
    assert!(e.contains("add PREFILTER"), "{}", e);
    let sparse_hits = rows(run(
        &mut db,
        &format!(
            "SEARCH t ON s QUERY {} PREFILTER mass >= 102.0 LIMIT 3",
            lit(&all[4])
        ),
    ));
    let dense_hits = rows(run(
        &mut db,
        &format!(
            "SEARCH t ON d QUERY {} PREFILTER mass >= 102.0 LIMIT 3",
            lit(&all[4])
        ),
    ));
    let ids = |r: &Vec<Vec<Value>>| r.iter().map(|x| x[0].clone()).collect::<Vec<_>>();
    assert_eq!(ids(&sparse_hits), ids(&dense_hits));
    assert_eq!(sparse_hits[0][0], Value::Int(4));
}

#[test]
fn save_and_load_round_trip_with_nulls() {
    let (dir, mut db) = db();
    setup(&mut db);
    run(&mut db, "DATASET n COLUMNS (id: Int, s: SparseVector(4)?)");
    run(&mut db, "INSERT INTO n VALUES (1, [0, 1.5, 0, 0])");
    run(&mut db, "INSERT INTO n VALUES (2, null)");
    run(&mut db, "INSERT INTO n VALUES (3, [0, 0, 0, 0])");
    run(&mut db, "SAVE DATASET t");
    run(&mut db, "SAVE DATASET n");
    let before_t = rows(run(&mut db, "SELECT * FROM t ORDER BY id"));
    let before_n = rows(run(&mut db, "SELECT * FROM n ORDER BY id"));
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db2 = TensorDb::with_config(config);
    run(&mut db2, "LOAD DATASET t");
    run(&mut db2, "LOAD DATASET n");
    assert_eq!(rows(run(&mut db2, "SELECT * FROM t ORDER BY id")), before_t);
    assert_eq!(rows(run(&mut db2, "SELECT * FROM n ORDER BY id")), before_n);
    let raw = std::fs::read_to_string(dir.path().join("default/datasets/n/schema.json")).unwrap();
    assert!(raw.contains(r#""SparseVector": 4"#), "{}", raw);
}

#[test]
fn loud_errors() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(&mut db, "DATASET o COLUMNS (id: Int, s: SparseVector(30))");
    run(
        &mut db,
        &format!("INSERT INTO o VALUES (1, {})", lit(&[1.0; 30])),
    );

    let e = run_err(&mut db, "INSERT INTO t VALUES (99, [1, 2], [1, 2], 1.0)");
    assert!(
        e.contains("SparseVector(40), got a vector of length 2"),
        "{}",
        e
    );
    let e = run_err(
        &mut db,
        "SELECT COSINE_SIM(t.s, o.s) AS c FROM t JOIN o ON t.id = o.id",
    );
    assert!(
        e.contains("COSINE_SIM: dimensions differ (40 vs 30)"),
        "{}",
        e
    );
    let e = run_err(&mut db, "SELECT DOT(s, mass) AS x FROM t");
    assert!(
        e.contains("DOT expects Vector or SparseVector arguments"),
        "{}",
        e
    );
    for (bad, msg) in [
        ("SPARSE(5, [3, 1], [1.0, 1.0])", "must be increasing"),
        ("SPARSE(5, [1, 1], [1.0, 1.0])", "duplicate index 1"),
        ("SPARSE(5, [7], [1.0])", "out of range for dimension 5"),
        ("SPARSE(5, [1.5], [1.0])", "not a non-negative integer"),
        ("SPARSE(5, [1, 2], [1.0])", "2 indices but 1 values"),
    ] {
        let e = run_err(&mut db, &format!("SELECT {} AS x FROM t WHERE id = 0", bad));
        assert!(e.contains(msg), "{}: {}", bad, e);
    }
    let e = run_err(&mut db, "SELECT SPARSE(5, [1]) AS x FROM t");
    assert!(e.contains("SPARSE takes 3 arguments"), "{}", e);
}
