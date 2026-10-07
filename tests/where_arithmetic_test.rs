// tests/where_arithmetic_test.rs
//
// Arithmetic inside a WHERE/FILTER comparison (`WHERE price * qty > 100`).
// The predicate evaluator only resolved columns, literals and function
// calls as comparison operands; an arithmetic operand evaluated to
// "unknown", so the comparison was false for every row and the query
// silently returned nothing. Found while building SEARCH ... PREFILTER
// (`mass BETWEEN q.mass - 0.01 AND q.mass + 0.01`).

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

fn ids(db: &mut TensorDb, q: &str) -> Vec<i64> {
    match execute_line(db, q, 1).unwrap_or_else(|e| panic!("{}: {}", q, e)) {
        DslOutput::Table(ds) => ds
            .rows
            .iter()
            .map(|r| match r.values[0] {
                Value::Int(i) => i,
                ref v => panic!("{:?}", v),
            })
            .collect(),
        other => panic!("{:?}", other),
    }
}

#[test]
fn arithmetic_operands_in_where() {
    let (_dir, mut db) = db();
    for l in [
        "DATASET t COLUMNS (id: Int, price: Float, qty: Int, mass: Float64)",
        "INSERT INTO t VALUES (1, 10.0, 5, 100.0)",
        "INSERT INTO t VALUES (2, 3.0, 20, 180.5)",
        "INSERT INTO t VALUES (3, 50.0, 1, 250.25)",
    ] {
        execute_line(&mut db, l, 1).unwrap();
    }
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE price * qty > 55 ORDER BY id"
        ),
        vec![2]
    );
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE price * qty >= 50 ORDER BY id"
        ),
        vec![1, 2, 3]
    );
    assert_eq!(ids(&mut db, "SELECT id FROM t WHERE qty - 1 = 0"), vec![3]);
    assert_eq!(
        ids(&mut db, "SELECT id FROM t WHERE mass / 2 < 60"),
        vec![1]
    );
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE 200 < mass + 20 ORDER BY id"
        ),
        vec![2, 3]
    );
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE mass BETWEEN 180.0 - 1.0 AND 180.0 + 1.0"
        ),
        vec![2]
    );
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE id IN (1, 3) AND price + 0 > 20"
        ),
        vec![3]
    );
    assert_eq!(
        ids(
            &mut db,
            "SELECT id FROM t WHERE NOT (price * qty > 55) ORDER BY id"
        ),
        vec![1, 3]
    );
    // DELETE/UPDATE share the evaluator.
    execute_line(&mut db, "UPDATE t SET qty = 0 WHERE price * qty > 55", 1).unwrap();
    assert_eq!(ids(&mut db, "SELECT id FROM t WHERE qty = 0"), vec![2]);
    execute_line(&mut db, "DELETE FROM t WHERE mass - 100.0 < 1", 1).unwrap();
    assert_eq!(ids(&mut db, "SELECT id FROM t ORDER BY id"), vec![2, 3]);
}

/// A literal boolean predicate. `WHERE true` (and `PREFILTER true`) used to
/// match no rows: the predicate evaluator had no case for a literal.
#[test]
fn literal_boolean_predicates() {
    let (_dir, mut db) = db();
    execute_line(&mut db, "DATASET t COLUMNS (id: Int)", 1).unwrap();
    execute_line(&mut db, "INSERT INTO t VALUES (1)", 1).unwrap();
    execute_line(&mut db, "INSERT INTO t VALUES (2)", 1).unwrap();
    assert_eq!(
        ids(&mut db, "SELECT id FROM t WHERE true ORDER BY id"),
        vec![1, 2]
    );
    assert_eq!(
        ids(&mut db, "SELECT id FROM t WHERE false"),
        Vec::<i64>::new()
    );
    assert_eq!(
        ids(&mut db, "SELECT id FROM t WHERE true AND id > 1"),
        vec![2]
    );
    execute_line(&mut db, "DATASET v COLUMNS (id: Int, e: Vector(2))", 1).unwrap();
    execute_line(&mut db, "INSERT INTO v VALUES (7, [1.0, 0.0])", 1).unwrap();
    assert_eq!(
        ids(
            &mut db,
            "SEARCH v ON e QUERY [1.0, 0.0] PREFILTER true LIMIT 1"
        ),
        vec![7]
    );
}
