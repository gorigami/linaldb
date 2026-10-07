// tests/arg_rrf_aggregate_test.rs
//
// ARG_MAX / ARG_MIN / RRF group aggregates (CASMI_WORKLOADS_PLAN.md, P6).
// ARG_MAX(col, by) must agree with the existing window formulation
// (ROW_NUMBER() OVER (PARTITION BY g ORDER BY by DESC) = 1), break ties by
// first appearance, and fail loudly on inputs that have no ordering.

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

/// Spectra of three molecules: (molecule, spectrum id, candidate, score).
fn setup(db: &mut TensorDb) {
    run(
        db,
        "DATASET hits COLUMNS (mol: Int, spec: Int, cand: String, score: Float, rnk: Int)",
    );
    let data = [
        (1, 10, "a", 0.50, 2),
        (1, 11, "b", 0.90, 1),
        (1, 12, "c", 0.90, 1), // ties with "b": first appearance wins
        (2, 20, "d", 0.10, 3),
        (2, 21, "e", 0.70, 1),
        (3, 30, "f", 0.30, 1),
    ];
    for (m, s, c, sc, r) in data {
        run(
            db,
            &format!(
                "INSERT INTO hits VALUES ({}, {}, \"{}\", {}, {})",
                m, s, c, sc, r
            ),
        );
    }
}

#[test]
fn arg_max_matches_the_window_formulation() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let native = rows(run(
        &mut db,
        "SELECT mol, ARG_MAX(cand, score) AS best FROM hits GROUP BY mol",
    ));
    let window = rows(run(
        &mut db,
        "SELECT mol, cand FROM (SELECT mol, cand, ROW_NUMBER() OVER (PARTITION BY mol ORDER BY score DESC) AS rn FROM hits) AS w WHERE rn = 1",
    ));
    let as_pairs = |r: Vec<Vec<Value>>| {
        let mut v: Vec<(i64, String)> = r
            .into_iter()
            .map(|row| match (&row[0], &row[1]) {
                (Value::Int(m), Value::String(c)) => (*m, c.clone()),
                other => panic!("unexpected row {:?}", other),
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(as_pairs(native.clone()), as_pairs(window));
    assert_eq!(
        as_pairs(native),
        vec![(1, "b".into()), (2, "e".into()), (3, "f".into())]
    );
}

#[test]
fn ties_keep_the_first_row_for_both_directions() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let r = rows(run(
        &mut db,
        "SELECT mol, ARG_MAX(spec, score) AS hi, ARG_MIN(spec, rnk) AS lo FROM hits GROUP BY mol",
    ));
    // mol 1: score 0.90 ties between spec 11 and 12 -> 11; rnk 1 ties -> 11.
    assert_eq!(r[0], vec![Value::Int(1), Value::Int(11), Value::Int(11)]);
    assert_eq!(r[1], vec![Value::Int(2), Value::Int(21), Value::Int(21)]);
    assert_eq!(r[2], vec![Value::Int(3), Value::Int(30), Value::Int(30)]);
}

#[test]
fn arg_max_without_group_by_reduces_the_whole_input() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let r = rows(run(
        &mut db,
        "SELECT ARG_MAX(cand, score) AS best FROM hits",
    ));
    assert_eq!(r, vec![vec![Value::String("b".into())]]);
}

#[test]
fn arg_max_selects_a_vector_value() {
    let (_dir, mut db) = db();
    run(
        &mut db,
        "DATASET v COLUMNS (g: Int, s: Float, e: Vector(2))",
    );
    run(&mut db, "INSERT INTO v VALUES (1, 0.1, [1.0, 0.0])");
    run(&mut db, "INSERT INTO v VALUES (1, 0.8, [0.0, 1.0])");
    let r = rows(run(
        &mut db,
        "SELECT g, ARG_MAX(e, s) AS best FROM v GROUP BY g",
    ));
    assert_eq!(r, vec![vec![Value::Int(1), Value::Vector(vec![0.0, 1.0])]]);
}

#[test]
fn rrf_fuses_ranks_with_default_and_explicit_k() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let r = rows(run(
        &mut db,
        "SELECT mol, RRF(rnk) AS d, RRF(rnk, 1) AS one FROM hits GROUP BY mol",
    ));
    let expect = |ranks: &[f64], k: f64| ranks.iter().map(|r| 1.0 / (k + r)).sum::<f64>();
    let f = |v: &Value| match v {
        Value::Float64(x) => *x,
        other => panic!("RRF should be Float64, got {:?}", other),
    };
    assert!((f(&r[0][1]) - expect(&[2.0, 1.0, 1.0], 60.0)).abs() < 1e-12);
    assert!((f(&r[0][2]) - expect(&[2.0, 1.0, 1.0], 1.0)).abs() < 1e-12);
    assert!((f(&r[1][1]) - expect(&[3.0, 1.0], 60.0)).abs() < 1e-12);
    assert!((f(&r[2][2]) - expect(&[1.0], 1.0)).abs() < 1e-12);
}

#[test]
fn group_by_dataset_form_and_schema_types() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(
        &mut db,
        "DATASET best FROM hits GROUP BY mol SELECT mol, ARG_MAX(cand, score) AS cand, RRF(rnk) AS fused",
    );
    let ds = db.get_dataset("best").unwrap();
    let types: Vec<String> = ds
        .schema
        .fields
        .iter()
        .map(|f| format!("{:?}", f.value_type))
        .collect();
    assert_eq!(types, vec!["Int", "String", "Float64"]);
    assert_eq!(ds.rows.len(), 3);
}

#[test]
fn loud_errors_for_bad_arguments() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(&mut db, "DATASET v COLUMNS (g: Int, e: Vector(2))");
    run(&mut db, "INSERT INTO v VALUES (1, [1.0, 0.0])");

    let e = run_err(&mut db, "SELECT mol, ARG_MAX(cand) FROM hits GROUP BY mol");
    assert!(e.contains("two arguments"), "{}", e);

    let e = run_err(
        &mut db,
        "SELECT mol, ARG_MAX(cand, score) OVER (PARTITION BY mol) AS x FROM hits",
    );
    assert!(e.contains("window function"), "{}", e);

    let e = run_err(&mut db, "SELECT g, ARG_MAX(g, e) FROM v GROUP BY g");
    assert!(e.contains("must be a scalar"), "{}", e);

    let e = run_err(&mut db, "SELECT mol, RRF(cand) FROM hits GROUP BY mol");
    assert!(e.contains("numeric rank"), "{}", e);

    let e = run_err(&mut db, "SELECT mol, RRF(rnk, abc) FROM hits GROUP BY mol");
    assert!(e.contains("numeric literal"), "{}", e);
}
