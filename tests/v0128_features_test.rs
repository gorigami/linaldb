// tests/v0128_features_test.rs
// Integration test coverage for v0.1.28 features (CONSISTENCY_PLAN.md Track
// C / C3). These shipped with parser-level unit tests (11 new tests per the
// CHANGELOG) but no end-to-end integration coverage verifying actual query
// results — a higher silent-regression risk than later versions, which got
// dedicated integration test files. RIGHT JOIN / FULL OUTER JOIN row
// correctness is already covered by tests/correctness_integration.rs, so
// this file focuses on what wasn't: subqueries, IN, BETWEEN, LIMIT+OFFSET,
// multi-column ORDER BY, and the FILTER boolean-literal regression guard.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn setup(db: &mut TensorDb) {
    exec(
        db,
        "DATASET items COLUMNS (id: Int, price: Float, active: Bool)",
        1,
    );
    exec(db, "INSERT INTO items VALUES (1, 10.0, true)", 2);
    exec(db, "INSERT INTO items VALUES (2, 20.0, false)", 3);
    exec(db, "INSERT INTO items VALUES (3, 30.0, true)", 4);
    exec(db, "INSERT INTO items VALUES (4, 40.0, true)", 5);
    exec(db, "INSERT INTO items VALUES (5, 50.0, false)", 6);
}

fn ids(ds: &linal::core::dataset_legacy::Dataset) -> Vec<i64> {
    ds.rows
        .iter()
        .map(|r| match r.get("id").unwrap() {
            Value::Int(n) => *n,
            other => panic!("expected Int id, got {other:?}"),
        })
        .collect()
}

// ── Subqueries ────────────────────────────────────────────────────────────

#[test]
fn test_subquery_from_select() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT * FROM (SELECT id, price FROM items WHERE price > 15) AS sub",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = ids(&ds);
    result.sort();
    assert_eq!(result, vec![2, 3, 4, 5]);
}

#[test]
fn test_nested_subquery_two_levels() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT * FROM (SELECT * FROM (SELECT id, price FROM items WHERE price > 15) AS inner_sub WHERE price < 45) AS outer_sub",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = ids(&ds);
    result.sort();
    assert_eq!(result, vec![2, 3, 4]);
}

// ── IN / BETWEEN ────────────────────────────────────────────────────────────

#[test]
fn test_where_in() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(&mut db, "SELECT id FROM items WHERE id IN (1, 3, 5)", 7);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = ids(&ds);
    result.sort();
    assert_eq!(result, vec![1, 3, 5]);
}

#[test]
fn test_where_between() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id FROM items WHERE price BETWEEN 15 AND 35",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = ids(&ds);
    result.sort();
    assert_eq!(result, vec![2, 3]);
}

#[test]
fn test_compound_between_and() {
    let mut db = TensorDb::new();
    setup(&mut db);

    // BETWEEN uses a restricted precedence (parse_pratt(4)) specifically so
    // a trailing `AND active = true` binds to the outer predicate, not into
    // the BETWEEN's upper bound.
    let out = exec(
        &mut db,
        "SELECT id FROM items WHERE price BETWEEN 15 AND 35 AND active = true",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ids(&ds), vec![3]);
}

// ── LIMIT + OFFSET ──────────────────────────────────────────────────────────

#[test]
fn test_limit_with_offset() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id FROM items ORDER BY id LIMIT 2 OFFSET 1",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ids(&ds), vec![2, 3]);
}

// ── Multi-column ORDER BY ───────────────────────────────────────────────────

#[test]
fn test_multi_column_order_by() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id FROM items ORDER BY active DESC, price ASC",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    // active=true rows first (by price asc: 1,3,4), then active=false (2,5)
    assert_eq!(ids(&ds), vec![1, 3, 4, 2, 5]);
}

// ── Compound FILTER / boolean literal regression ────────────────────────────

#[test]
fn test_filter_boolean_literal_not_parsed_as_column_ref() {
    // CHANGELOG v0.1.28: `FILTER active = true` used to parse `true` as a
    // column reference instead of a boolean literal (Expr::Bool added to
    // fix it). Guards against that regression recurring.
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(&mut db, "SELECT id FROM items FILTER active = true", 7);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let mut result = ids(&ds);
    result.sort();
    assert_eq!(result, vec![1, 3, 4]);
}
