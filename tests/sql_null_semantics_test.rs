// tests/sql_null_semantics_test.rs
//
// Comparisons with NULL are unknown, not true or false, and a WHERE keeps a
// row only when its predicate is true -- SQL's three-valued logic. Before,
// `Value::compare` ordered NULL below everything, so `x < 2`, `x <> 2` and
// `NULL = NULL` were true for NULL rows; a sorted index (which doesn't hold
// NULLs) then answered differently from a scan of the same data, and a hash
// join matched NULL keys. `NOT` also bound tighter than a comparison, so
// `NOT x = 2` was `(NOT x) = 2` and matched nothing.
//
// Found by linal-hub notebook 18's generated-predicate check against a SQL
// reference evaluator.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

/// Every row of a query's first column, as i64.
fn ints(db: &mut TensorDb, query: &str) -> Vec<i64> {
    let rows = match run(db, query) {
        DslOutput::Table(ds) => ds.rows,
        other => panic!("`{}`: expected a table, got {:?}", query, other),
    };
    rows.iter()
        .map(|r| match &r.values[0] {
            Value::Int(i) => *i,
            other => panic!("`{}`: expected Int, got {:?}", query, other),
        })
        .collect()
}

fn strings(db: &mut TensorDb, query: &str) -> Vec<String> {
    let rows = match run(db, query) {
        DslOutput::Table(ds) => ds.rows,
        other => panic!("`{}`: expected a table, got {:?}", query, other),
    };
    rows.iter()
        .map(|r| match &r.values[0] {
            Value::String(s) => s.clone(),
            other => format!("{:?}", other),
        })
        .collect()
}

/// id 1: x=1, y=NULL; id 2: x=NULL, y=NULL; id 3: x=3, y=3.
fn small() -> TensorDb {
    let mut db = TensorDb::new();
    run(
        &mut db,
        "DATASET t COLUMNS (id: Int, x: Int?, y: Int?, s: String?)",
    );
    run(&mut db, "INSERT INTO t VALUES (1, 1, NULL, \"a\")");
    run(&mut db, "INSERT INTO t VALUES (2, NULL, NULL, NULL)");
    run(&mut db, "INSERT INTO t VALUES (3, 3, 3, \"b\")");
    db
}

fn ids(db: &mut TensorDb, predicate: &str) -> Vec<i64> {
    let mut v = ints(db, &format!("SELECT id FROM t WHERE {}", predicate));
    v.sort();
    v
}

#[test]
fn comparisons_with_null_select_nothing() {
    let mut db = small();
    assert_eq!(ids(&mut db, "x < 2"), vec![1]);
    assert_eq!(ids(&mut db, "x <= 2"), vec![1]);
    assert_eq!(ids(&mut db, "x > 0"), vec![1, 3]);
    assert_eq!(ids(&mut db, "x <> 2"), vec![1, 3]);
    assert_eq!(ids(&mut db, "x != 2"), vec![1, 3]);
    assert_eq!(ids(&mut db, "x = y"), vec![3]);
    assert_eq!(ids(&mut db, "x <> y"), Vec::<i64>::new());
    assert_eq!(ids(&mut db, "s <> 'a'"), vec![3]);
    assert_eq!(ids(&mut db, "s < 'z'"), vec![1, 3]);
    assert_eq!(ids(&mut db, "x IS NULL"), vec![2]);
    assert_eq!(ids(&mut db, "x IS NOT NULL"), vec![1, 3]);
}

#[test]
fn not_keeps_unknown_unknown() {
    let mut db = small();
    assert_eq!(ids(&mut db, "NOT (x = 2)"), vec![1, 3]);
    assert_eq!(ids(&mut db, "NOT (x < 2)"), vec![3]);
    assert_eq!(ids(&mut db, "NOT (x IN (1))"), vec![3]);
    assert_eq!(ids(&mut db, "NOT (x BETWEEN 0 AND 2)"), vec![3]);
    assert_eq!(ids(&mut db, "NOT NOT (x = 1)"), vec![1]);
}

#[test]
fn not_binds_looser_than_a_comparison() {
    let mut db = small();
    assert_eq!(ids(&mut db, "NOT x = 1"), vec![3]);
    assert_eq!(ids(&mut db, "NOT x > 1"), vec![1]);
    assert_eq!(ids(&mut db, "NOT x = 1 AND id > 2"), vec![3]);
    assert_eq!(ids(&mut db, "NOT x = 1 OR id = 2"), vec![2, 3]);
    assert_eq!(ids(&mut db, "NOT x BETWEEN 2 AND 9"), vec![1]);
    // a Bool column still works bare
    run(&mut db, "DATASET b COLUMNS (id: Int, ok: Bool)");
    run(&mut db, "INSERT INTO b VALUES (1, true)");
    run(&mut db, "INSERT INTO b VALUES (2, false)");
    assert_eq!(ints(&mut db, "SELECT id FROM b WHERE NOT ok"), vec![2]);
    assert_eq!(
        ints(&mut db, "SELECT id FROM b WHERE NOT ok AND id = 2"),
        vec![2]
    );
}

#[test]
fn and_or_follow_three_valued_logic() {
    let mut db = small();
    // unknown OR true is true; unknown AND false is false (so NOT is true)
    assert_eq!(ids(&mut db, "x < 2 OR id = 2"), vec![1, 2]);
    assert_eq!(ids(&mut db, "NOT (x < 2 AND id = 3)"), vec![1, 2, 3]);
    // unknown AND true stays unknown, and so does its NOT
    assert_eq!(ids(&mut db, "NOT (x < 2 AND id = 2)"), vec![1, 3]);
}

#[test]
fn in_and_between_with_nulls() {
    let mut db = small();
    assert_eq!(ids(&mut db, "x IN (1, 3)"), vec![1, 3]);
    assert_eq!(ids(&mut db, "x BETWEEN 0 AND 2"), vec![1]);
    assert_eq!(ids(&mut db, "x IN (y, 1)"), vec![1, 3]);
    // 1 IN (NULL, 2) is unknown, so its NOT is too
    assert_eq!(ids(&mut db, "NOT (x IN (y, 2))"), Vec::<i64>::new());
}

#[test]
fn case_when_with_null_takes_else() {
    let mut db = small();
    let got = strings(
        &mut db,
        "SELECT CASE WHEN x < 2 THEN 'lt' ELSE 'other' END AS c FROM t",
    );
    assert_eq!(got, vec!["lt", "other", "other"]);
    let got = strings(
        &mut db,
        "SELECT CASE x WHEN y THEN 'same' ELSE 'differ' END AS c FROM t",
    );
    assert_eq!(got, vec!["differ", "differ", "same"]);
}

#[test]
fn update_and_delete_skip_null_rows() {
    let mut db = small();
    run(&mut db, "UPDATE t SET s = \"low\" WHERE x < 2");
    assert_eq!(ids(&mut db, "s = 'low'"), vec![1]);
    run(&mut db, "DELETE FROM t WHERE x <> 3");
    assert_eq!(ids(&mut db, "id > 0"), vec![2, 3]);
}

#[test]
fn hash_join_never_matches_null_keys() {
    let mut db = TensorDb::new();
    run(&mut db, "DATASET l COLUMNS (id: Int, k: Int?)");
    run(&mut db, "INSERT INTO l VALUES (1, NULL)");
    run(&mut db, "INSERT INTO l VALUES (2, 7)");
    run(&mut db, "DATASET r COLUMNS (rid: Int, k2: Int?)");
    run(&mut db, "INSERT INTO r VALUES (10, NULL)");
    run(&mut db, "INSERT INTO r VALUES (20, 7)");
    run(&mut db, "INSERT INTO r VALUES (30, NULL)");
    assert_eq!(
        ints(&mut db, "SELECT id FROM l JOIN r ON l.k = r.k2"),
        vec![2]
    );
    let mut left = ints(&mut db, "SELECT id FROM l LEFT JOIN r ON l.k = r.k2");
    left.sort();
    assert_eq!(
        left,
        vec![1, 2],
        "a LEFT JOIN keeps the NULL-key row once, unmatched"
    );
}

/// 3,000 rows (three partitions), every tenth x NULL: the scan, partition
/// pruning and a sorted index must all agree.
#[test]
fn scan_pruning_and_sorted_index_agree() {
    let mut db = TensorDb::new();
    for name in ["scan", "sorted"] {
        run(
            &mut db,
            &format!("DATASET {} COLUMNS (id: Int, x: Int?)", name),
        );
        for i in 0..3000 {
            let x = if i % 10 == 0 {
                "NULL".to_string()
            } else {
                i.to_string()
            };
            run(
                &mut db,
                &format!("INSERT INTO {} VALUES ({}, {})", name, i, x),
            );
        }
    }
    run(&mut db, "CREATE SORTED INDEX ON sorted(x)");
    for predicate in [
        "x < 100",
        "x <= 2500",
        "x > 2900",
        "x BETWEEN 5 AND 1500",
        "NOT x >= 100",
        "x <> 51",
        "x < 100 OR id = 0",
    ] {
        let mut scan = ints(&mut db, &format!("SELECT id FROM scan WHERE {}", predicate));
        let mut sorted = ints(
            &mut db,
            &format!("SELECT id FROM sorted WHERE {}", predicate),
        );
        scan.sort();
        sorted.sort();
        assert_eq!(
            scan, sorted,
            "scan and sorted index differ on `{}`",
            predicate
        );
    }
    assert_eq!(
        ints(&mut db, "SELECT COUNT(*) FROM scan WHERE x < 100"),
        vec![90]
    );
    assert_eq!(
        ints(&mut db, "SELECT COUNT(*) FROM scan WHERE x <> 51"),
        vec![2699]
    );
}
