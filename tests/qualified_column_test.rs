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

// ── Bug #4: a qualified reference silently returned the WRONG table's
//    value when both sides of a JOIN share a bare column name ────────────
//
// Found 2026-09-13 while building a real-data notebook outside this repo
// (`linal-hub/notebooks/07_single_cell_pca_pipeline.ipynb`): a nearest-
// centroid classification query joining two datasets that both had a
// `cell_type` column reported a suspicious 100% accuracy -- the qualified
// `centroids.cell_type` reference was silently returning `cells.cell_type`
// instead, because `dsl_expr_to_logical_expr`'s `Expr::Field` arm dropped
// the table qualifier entirely and resolved by bare column name into a
// merged row where only the *left* side's field kept that bare name (the
// right side's colliding field is renamed to `r_<name>` by
// `LogicalPlan::Join::schema()`, a rename this conversion never consulted).
// Root cause and blast radius (SELECT, WHERE, aggregates; not `ON`, which
// resolves separately, pre-merge) verified directly in `src/dsl/executor/query.rs`
// before this fix.

#[test]
fn test_join_colliding_column_name_select_resolves_correct_side() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET left_t COLUMNS (id: Int, tag: String)", 1);
    exec(&mut db, "INSERT INTO left_t VALUES (1, \"LEFT-A\")", 2);
    exec(&mut db, "INSERT INTO left_t VALUES (2, \"LEFT-B\")", 3);
    exec(&mut db, "DATASET right_t COLUMNS (id: Int, tag: String)", 4);
    exec(&mut db, "INSERT INTO right_t VALUES (1, \"RIGHT-X\")", 5);
    exec(&mut db, "INSERT INTO right_t VALUES (2, \"RIGHT-Y\")", 6);

    let out = exec(
        &mut db,
        "SELECT left_t.id AS id, left_t.tag AS a, right_t.tag AS b \
         FROM left_t JOIN right_t ON left_t.id = right_t.id",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 2);
    for row in &ds.rows {
        let Value::Int(id) = row.values[0] else {
            panic!("expected int id")
        };
        assert_eq!(
            row.values[1],
            Value::String(format!("LEFT-{}", if id == 1 { "A" } else { "B" }))
        );
        assert_eq!(
            row.values[2],
            Value::String(format!("RIGHT-{}", if id == 1 { "X" } else { "Y" }))
        );
    }
}

#[test]
fn test_join_colliding_column_name_select_resolves_correct_side_with_alias() {
    // Same collision, but the qualifier is a JOIN alias rather than the
    // literal dataset name -- the alias must be recognized as naming the
    // right side too (`JoinClause::alias`), not just the dataset's real name.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET left_t COLUMNS (id: Int, tag: String)", 1);
    exec(&mut db, "INSERT INTO left_t VALUES (1, \"LEFT-A\")", 2);
    exec(&mut db, "DATASET right_t COLUMNS (id: Int, tag: String)", 3);
    exec(&mut db, "INSERT INTO right_t VALUES (1, \"RIGHT-X\")", 4);

    let out = exec(
        &mut db,
        "SELECT l.id AS id, l.tag AS a, r.tag AS b FROM left_t l JOIN right_t r ON l.id = r.id",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[1], Value::String("LEFT-A".to_string()));
    assert_eq!(ds.rows[0].values[2], Value::String("RIGHT-X".to_string()));
}

#[test]
fn test_join_colliding_column_name_where_resolves_correct_side() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET left_t COLUMNS (id: Int, tag: String)", 1);
    exec(&mut db, "INSERT INTO left_t VALUES (1, \"LEFT-A\")", 2);
    exec(&mut db, "INSERT INTO left_t VALUES (2, \"LEFT-B\")", 3);
    exec(&mut db, "DATASET right_t COLUMNS (id: Int, tag: String)", 4);
    exec(&mut db, "INSERT INTO right_t VALUES (1, \"RIGHT-X\")", 5);
    exec(&mut db, "INSERT INTO right_t VALUES (2, \"RIGHT-Y\")", 6);

    let out = exec(
        &mut db,
        "SELECT left_t.id AS id, right_t.tag AS b FROM left_t JOIN right_t \
         ON left_t.id = right_t.id WHERE right_t.tag = \"RIGHT-Y\"",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.rows[0].values[0], Value::Int(2));
    assert_eq!(ds.rows[0].values[1], Value::String("RIGHT-Y".to_string()));
}

#[test]
fn test_join_colliding_column_name_aggregate_resolves_correct_side() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET left_t COLUMNS (id: Int, grp: Int)", 1);
    exec(&mut db, "INSERT INTO left_t VALUES (1, 1)", 2);
    exec(&mut db, "INSERT INTO left_t VALUES (2, 1)", 3);
    exec(&mut db, "DATASET right_t COLUMNS (id: Int, grp: Float)", 4);
    exec(&mut db, "INSERT INTO right_t VALUES (1, 100.0)", 5);
    exec(&mut db, "INSERT INTO right_t VALUES (2, 200.0)", 6);

    let out = exec(
        &mut db,
        "SELECT grp, AVG(right_t.grp) AS avg_right FROM left_t JOIN right_t \
         ON left_t.id = right_t.id GROUP BY grp",
        7,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 1);
    // If this silently averaged left_t.grp (1, 1) instead, it would read 1.0.
    assert_eq!(ds.rows[0].values[1], Value::Float(150.0));
}

// ── Bug #5: an un-aliased qualified SELECT column was labeled `__cmp_0`
//    instead of its real name ─────────────────────────────────────────────
//
// Found 2026-09-13 building a real-data notebook outside this repo
// (`linal-hub/notebooks/08_production_network_systemic_risk.ipynb`):
// `SELECT t.col FROM t` (no `AS`) returned the correct value but reported
// the output column as `__cmp_0` -- the SELECT-list classifier only treated
// a *bare* `Expr::Ref` as a plain `Column`; a qualified `Expr::Field`
// reference with no alias fell into the generic `Computed` catch-all, whose
// unaliased-naming fallback (`__cmp_{idx}`, meant for a genuine expression
// like `price * 2`) fired even though this is just a column reference.

#[test]
fn test_unaliased_qualified_select_column_keeps_its_real_name() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int, b: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 2.0)", 2);

    let out = exec(&mut db, "SELECT t.a, t.b FROM t", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let names: Vec<&str> = ds.schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
    assert_eq!(ds.rows[0].values[0], Value::Int(1));
    assert_eq!(ds.rows[0].values[1], Value::Float(2.0));
}

#[test]
fn test_unaliased_qualified_select_column_keeps_its_real_name_across_join() {
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
    let names: Vec<&str> = ds.schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["id", "name"]);
}

#[test]
fn test_qualified_select_column_with_alias_still_uses_the_alias() {
    // Guard against over-broadening the new Column-classification arm: an
    // explicit `AS` alias must still win, exactly as it already does for a
    // bare (unqualified) column reference.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1)", 2);

    let out = exec(&mut db, "SELECT t.a AS renamed FROM t", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let names: Vec<&str> = ds.schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["renamed"]);
}

// ── Bug #6: a qualified column failed to parse at all in GROUP BY,
//    ORDER BY (plain or windowed), window PARTITION BY, and LAG/LEAD's
//    column argument ────────────────────────────────────────────────────
//
// Found 2026-09-13, same real economics notebook as bug #5. Root cause:
// unlike the general expression parser (used by SELECT/WHERE, which
// handles `t.col` naturally via `Expr::Field`), each of these clauses
// parsed its column name via a raw single `eat_ident()` that never checked
// for a following `.` -- an oversight, not a deliberate restriction (no
// existing test exercised a qualified column in any of these clauses
// before this fix). The qualifier is parsed and discarded (not threaded
// through), matching how every qualified-column reference in this engine
// already resolves -- by final field name only, never true table
// disambiguation (`Schema::get_field_index` is a flat exact-string
// lookup) -- so this changes zero existing behavior, it only accepts
// syntax that previously failed to parse.

#[test]
fn test_qualified_column_in_plain_order_by() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int, b: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 2.0)", 2);
    exec(&mut db, "INSERT INTO t VALUES (2, 1.0)", 3);

    let out = exec(&mut db, "SELECT a, b FROM t ORDER BY t.b ASC", 4);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[0], Value::Int(2));
    assert_eq!(ds.rows[1].values[0], Value::Int(1));
}

#[test]
fn test_qualified_column_in_group_by() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (cat: String, val: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (\"x\", 1.0)", 2);
    exec(&mut db, "INSERT INTO t VALUES (\"x\", 3.0)", 3);
    exec(&mut db, "INSERT INTO t VALUES (\"y\", 10.0)", 4);

    let out = exec(
        &mut db,
        "SELECT cat, AVG(val) AS m FROM t GROUP BY t.cat",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 2);
    let row_x = ds
        .rows
        .iter()
        .find(|r| r.values[0] == Value::String("x".to_string()))
        .expect("group for cat=x");
    assert_eq!(row_x.values[1], Value::Float(2.0));
}

#[test]
fn test_qualified_column_in_window_partition_by_and_order_by() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (cat: String, val: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (\"x\", 1.0)", 2);
    exec(&mut db, "INSERT INTO t VALUES (\"x\", 3.0)", 3);
    exec(&mut db, "INSERT INTO t VALUES (\"y\", 10.0)", 4);

    let out = exec(
        &mut db,
        "SELECT cat, val, RANK() OVER (PARTITION BY t.cat ORDER BY t.val DESC) AS r FROM t",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let row = ds
        .rows
        .iter()
        .find(|r| r.values[0] == Value::String("x".to_string()) && r.values[1] == Value::Float(3.0))
        .expect("x/3.0 row");
    assert_eq!(row.values[2], Value::Int(1));
}

#[test]
fn test_qualified_column_in_lag() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: Int, val: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO t VALUES (2, 20.0)", 3);

    let out = exec(
        &mut db,
        "SELECT id, LAG(t.val) OVER (ORDER BY t.id) AS prev FROM t",
        4,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[1], Value::Null);
    assert_eq!(ds.rows[1].values[1], Value::Float(10.0));
}
