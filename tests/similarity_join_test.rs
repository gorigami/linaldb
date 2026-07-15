// tests/similarity_join_test.rs
// Regression tests for CONSISTENCY_PLAN.md Track D / D2: index-accelerated
// similarity JOIN — `JOIN <dataset> ON COSINE_SIM(a.col, b.col) > threshold`.
//
// Uses a Vector index on the right dataset's join column when one exists
// (index-accelerated path via Index::search), falling back to a brute-force
// O(n*m) comparison otherwise — the same index-or-fallback pattern as
// CosineFilterExec/IndexScanExec. Both paths must produce identical results.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn setup(db: &mut TensorDb) {
    exec(db, "DATASET a COLUMNS (aid: Int, v: Vector(3))", 1);
    exec(db, "INSERT INTO a VALUES (1, [1.0, 0.0, 0.0])", 2);
    exec(db, "INSERT INTO a VALUES (2, [0.0, 1.0, 0.0])", 3);

    exec(db, "DATASET b COLUMNS (bid: Int, v: Vector(3))", 4);
    exec(db, "INSERT INTO b VALUES (10, [0.9, 0.1, 0.0])", 5);
    exec(db, "INSERT INTO b VALUES (20, [0.0, 0.9, 0.1])", 6);
    exec(db, "INSERT INTO b VALUES (30, [0.0, 0.0, 1.0])", 7);
}

fn pairs(ds: &linal::core::dataset_legacy::Dataset) -> Vec<(Value, Value)> {
    ds.rows
        .iter()
        .map(|r| (r.values[0].clone(), r.values[1].clone()))
        .collect()
}

#[test]
fn test_inner_similarity_join_brute_force() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT aid, bid FROM a JOIN b ON COSINE_SIM(a.v, b.v) > 0.8",
        8,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = pairs(&ds);
    result.sort_by_key(|(a, _)| format!("{:?}", a));
    assert_eq!(
        result,
        vec![
            (Value::Int(1), Value::Int(10)),
            (Value::Int(2), Value::Int(20)),
        ]
    );
}

#[test]
fn test_inner_similarity_join_index_accelerated_matches_brute_force() {
    let mut db = TensorDb::new();
    setup(&mut db);
    exec(&mut db, "CREATE VECTOR INDEX ON b(v)", 8);

    let out = exec(
        &mut db,
        "SELECT aid, bid FROM a JOIN b ON COSINE_SIM(a.v, b.v) > 0.8",
        9,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = pairs(&ds);
    result.sort_by_key(|(a, _)| format!("{:?}", a));
    assert_eq!(
        result,
        vec![
            (Value::Int(1), Value::Int(10)),
            (Value::Int(2), Value::Int(20)),
        ],
        "index-accelerated path must produce the same result as brute force"
    );
}

#[test]
fn test_left_similarity_join_pads_unmatched_left_rows_with_null() {
    let mut db = TensorDb::new();
    setup(&mut db);
    // c(2)'s vector is opposite (negative cosine sim) to everything in b —
    // should not match anything.
    exec(&mut db, "DATASET c COLUMNS (cid: Int, v: Vector(3))", 8);
    exec(&mut db, "INSERT INTO c VALUES (1, [1.0, 0.0, 0.0])", 9);
    exec(&mut db, "INSERT INTO c VALUES (2, [-1.0, 0.0, 0.0])", 10);

    let out = exec(
        &mut db,
        "SELECT cid, bid FROM c LEFT JOIN b ON COSINE_SIM(c.v, b.v) > 0.8",
        11,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = pairs(&ds);
    result.sort_by_key(|(a, _)| format!("{:?}", a));
    assert_eq!(
        result,
        vec![
            (Value::Int(1), Value::Int(10)),
            (Value::Int(2), Value::Null),
        ]
    );
}

#[test]
fn test_right_similarity_join_pads_unmatched_right_rows_with_null() {
    let mut db = TensorDb::new();
    setup(&mut db);
    // b(30)'s vector [0,0,1] doesn't match either a(1) [1,0,0] or a(2) [0,1,0].

    let out = exec(
        &mut db,
        "SELECT aid, bid FROM a RIGHT JOIN b ON COSINE_SIM(a.v, b.v) > 0.8",
        8,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = pairs(&ds);
    result.sort_by_key(|(_, b)| format!("{:?}", b));
    assert_eq!(
        result,
        vec![
            (Value::Int(1), Value::Int(10)),
            (Value::Int(2), Value::Int(20)),
            (Value::Null, Value::Int(30)),
        ]
    );
}

#[test]
fn test_similarity_join_no_matches_returns_empty() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT aid, bid FROM a JOIN b ON COSINE_SIM(a.v, b.v) > 0.999",
        8,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 0);
}
