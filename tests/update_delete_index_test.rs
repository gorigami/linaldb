// tests/update_delete_index_test.rs
//
// UPDATE and DELETE must keep a dataset's derived structures in sync with
// its rows: indexes (hash, IVF, HNSW, sorted), the per-partition zone maps
// used for range pruning, and the metadata stats. They used to edit
// `Dataset.rows` in place and leave all of these stale, so a later query
// could silently return wrong rows or miss matching ones.

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

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

#[test]
fn search_after_delete_returns_the_right_row() {
    for using in ["", " USING HNSW"] {
        let (_dir, mut db) = db();
        run(&mut db, "DATASET t COLUMNS (id: Int, e: Vector(2))");
        run(&mut db, "INSERT INTO t VALUES (1, [1.0, 0.0])");
        run(&mut db, "INSERT INTO t VALUES (2, [0.0, 1.0])");
        run(&mut db, "INSERT INTO t VALUES (3, [-1.0, 0.0])");
        run(&mut db, &format!("CREATE VECTOR INDEX ON t(e){}", using));
        run(&mut db, "DELETE FROM t WHERE id = 1");
        let r = rows(run(&mut db, "SEARCH t ON e QUERY [-1.0, 0.0] LIMIT 1"));
        assert_eq!(r[0][0], Value::Int(3), "index{}", using);
    }
}

#[test]
fn search_after_update_sees_the_new_vector() {
    let (_dir, mut db) = db();
    run(&mut db, "DATASET t COLUMNS (id: Int, e: Vector(2))");
    run(&mut db, "INSERT INTO t VALUES (1, [1.0, 0.0])");
    run(&mut db, "INSERT INTO t VALUES (2, [0.0, 1.0])");
    run(&mut db, "CREATE VECTOR INDEX ON t(e)");
    run(&mut db, "UPDATE t SET e = [0.0, -1.0] WHERE id = 1");
    let r = rows(run(&mut db, "SEARCH t ON e QUERY [0.0, -1.0] LIMIT 1"));
    assert_eq!(r[0][0], Value::Int(1));
}

#[test]
fn hash_index_after_update() {
    let (_dir, mut db) = db();
    run(&mut db, "DATASET t COLUMNS (id: Int, cat: String)");
    run(&mut db, "INSERT INTO t VALUES (1, \"a\")");
    run(&mut db, "INSERT INTO t VALUES (2, \"b\")");
    run(&mut db, "CREATE INDEX ON t(cat)");
    run(&mut db, "UPDATE t SET cat = \"z\" WHERE id = 1");
    assert_eq!(
        rows(run(&mut db, "SELECT id FROM t WHERE cat = \"z\"")).len(),
        1
    );
    assert_eq!(
        rows(run(&mut db, "SELECT id FROM t WHERE cat = \"a\"")).len(),
        0
    );
}

#[test]
fn range_pruning_after_update_and_delete() {
    let (_dir, mut db) = db();
    run(&mut db, "DATASET t COLUMNS (id: Int, x: Int)");
    for i in 0..2100 {
        run(&mut db, &format!("INSERT INTO t VALUES ({}, {})", i, i));
    }
    run(&mut db, "UPDATE t SET x = 9000 WHERE id = 10");
    assert_eq!(
        rows(run(&mut db, "SELECT id FROM t WHERE x > 8000")).len(),
        1
    );
    run(&mut db, "DELETE FROM t WHERE id < 1500");
    assert_eq!(
        rows(run(&mut db, "SELECT id FROM t WHERE x >= 2000")).len(),
        100
    );
    assert_eq!(
        rows(run(&mut db, "SELECT id FROM t WHERE x < 1600")).len(),
        100
    );
}

#[test]
fn update_type_checks_before_changing_anything() {
    let (_dir, mut db) = db();
    run(
        &mut db,
        "DATASET t COLUMNS (id: Int, score: Float, e: Vector(2))",
    );
    run(&mut db, "INSERT INTO t VALUES (1, 0.5, [1.0, 0.0])");
    run(&mut db, "INSERT INTO t VALUES (2, 0.7, [0.0, 1.0])");

    // Numeric widening like INSERT: an Int literal into a Float column.
    run(&mut db, "UPDATE t SET score = 1 WHERE id = 1");
    let r = rows(run(&mut db, "SELECT score FROM t WHERE id = 1"));
    assert_eq!(r[0][0], Value::Float(1.0));

    // A non-integral value into an Int column is an error, not a truncation.
    let e = execute_line(&mut db, "UPDATE t SET id = 2.5", 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("column 'id' is Int"), "{}", e);

    // Wrong vector dimension: error, and no row changed (all-or-nothing).
    let e = execute_line(&mut db, "UPDATE t SET e = [1.0, 2.0, 3.0]", 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("column 'e'"), "{}", e);
    let r = rows(run(&mut db, "SELECT e FROM t ORDER BY id"));
    assert_eq!(r[0][0], Value::Vector(vec![1.0, 0.0]));
    assert_eq!(r[1][0], Value::Vector(vec![0.0, 1.0]));

    // Computed assignments go through the full evaluator.
    run(&mut db, "UPDATE t SET score = score * 2 WHERE id = 2");
    let r = rows(run(&mut db, "SELECT score FROM t WHERE id = 2"));
    assert!(matches!(r[0][0], Value::Float(x) if (x - 1.4).abs() < 1e-6));
}
