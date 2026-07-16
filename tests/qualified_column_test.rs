// tests/qualified_column_test.rs
// Regression tests for CONSISTENCY_PLAN.md Track F / F1: table-qualified
// columns (`table.col`) and table aliasing (`FROM t alias` / `FROM t AS
// alias` / `JOIN t alias ON ...`).
//
// Two independent bugs compounded to make this silently broken:
//
//   1. Any *unaliased* Computed SELECT expression (not just `table.col` —
//      also e.g. `price * 2` with no `AS`) was silently dropped from the
//      output entirely. `apply_window_and_computed_exprs` names unaliased
//      Computed columns `__cmp_{idx}`, but the final SELECT-order
//      projection step looked them up under the unrelated literal string
//      "expr" — the lookup failed, so `filter_map` silently dropped the
//      column. Reproduced without any qualified column at all:
//      `SELECT id, x * 2 FROM t` used to return only the `id` column.
//   2. `table.col` parsed into `Expr::Field { base, field }`, which the
//      SQL row evaluator (`dsl_expr_to_logical_expr`) had no case for —
//      it fell through to a `_ => LogicalExpr::Literal(Value::Null)`
//      catch-all, so even once bug #1 was fixed, `a.id` still evaluated
//      to `NULL` instead of the actual value.
//   3. `FROM table alias` / `JOIN table alias ON ...` didn't parse at all
//      (`"Unknown command"`) — no alias-parsing support existed.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

// ── Bug #1: unaliased computed expressions were dropped entirely ─────────

#[test]
fn test_unaliased_computed_expr_not_dropped() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: Int, price: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 100.0)", 2);

    let out = exec(&mut db, "SELECT id, price * 2 FROM t", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.schema.fields.len(), 2, "both columns must be present");
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
    assert_eq!(ds.rows[0].values[1], Value::Float(200.0));
}

#[test]
fn test_multiple_unaliased_computed_exprs() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int, b: Int)", 1);
    exec(&mut db, "INSERT INTO t VALUES (2, 3)", 2);

    let out = exec(&mut db, "SELECT a * 2, b * 2 FROM t", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.schema.fields.len(), 2);
    assert_eq!(ds.rows[0].values[0], Value::Int(4));
    assert_eq!(ds.rows[0].values[1], Value::Int(6));
}

// ── Bug #2: table.col resolved to NULL instead of the real value ─────────

#[test]
fn test_qualified_column_resolves_to_real_value() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET a COLUMNS (id: Int, x: Int)", 1);
    exec(&mut db, "INSERT INTO a VALUES (1, 100)", 2);

    let out = exec(&mut db, "SELECT a.id FROM a", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
}

#[test]
fn test_qualified_columns_in_join_select() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET orders COLUMNS (id: Int, user_id: Int)", 1);
    exec(&mut db, "INSERT INTO orders VALUES (1, 5)", 2);
    exec(&mut db, "DATASET users COLUMNS (uid: Int, name: String)", 3);
    exec(&mut db, "INSERT INTO users VALUES (5, \"bob\")", 4);

    let out = exec(
        &mut db,
        "SELECT orders.id, users.name FROM orders JOIN users ON orders.user_id = users.uid",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
    assert_eq!(ds.rows[0].values[1], Value::String("bob".to_string()));
}

// ── Bug #3: FROM/JOIN table aliasing didn't parse ─────────────────────────

#[test]
fn test_join_with_table_aliases_and_qualified_select() {
    // The exact shape of the Track B doc example that shipped broken.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET orders COLUMNS (id: Int, user_id: Int)", 1);
    exec(&mut db, "INSERT INTO orders VALUES (1, 5)", 2);
    exec(&mut db, "DATASET users COLUMNS (uid: Int, name: String)", 3);
    exec(&mut db, "INSERT INTO users VALUES (5, \"bob\")", 4);

    let out = exec(
        &mut db,
        "SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.uid",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
    assert_eq!(ds.rows[0].values[1], Value::String("bob".to_string()));
}

#[test]
fn test_from_as_alias_with_qualified_where() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET orders COLUMNS (id: Int, total: Float)", 1);
    exec(&mut db, "INSERT INTO orders VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO orders VALUES (2, 20.0)", 3);

    let out = exec(&mut db, "SELECT id FROM orders AS o WHERE o.id = 1", 4);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
}

#[test]
fn test_bare_from_alias_no_as_keyword() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET orders COLUMNS (id: Int)", 1);
    exec(&mut db, "INSERT INTO orders VALUES (7)", 2);

    let out = exec(&mut db, "SELECT o.id FROM orders o", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[0], Value::Int(7));
}
