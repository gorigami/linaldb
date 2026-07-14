// tests/window_functions_test.rs
// Regression tests for Track E (E1) of CONSISTENCY_PLAN.md, plus general
// window function coverage overlapping Track C (C1) — window functions had
// almost no dedicated test coverage before this file.
//
// E1: combining multiple window functions with *different* `OVER (...)`
// specs in one SELECT — especially mixing LAG/LEAD with a differently
// specced window function — silently produced wrong values or an outright
// schema error. Root cause: apply_window_func (src/dsl/executor/query.rs)
// built the new window-result column's Field without `.nullable()`, so
// Tuple::new's schema validation rejected any row where the window function
// produced NULL (which LAG/LEAD do for boundary rows), and the code
// silently fell back to the pre-window row via `.unwrap_or(row)` — leaving
// a Vec<Tuple> with inconsistent per-row schemas.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn setup(db: &mut TensorDb) {
    exec(
        db,
        "DATASET items COLUMNS (id: Int, price: Float, category: String)",
        1,
    );
    exec(db, "INSERT INTO items VALUES (1, 10.0, 'x')", 2);
    exec(db, "INSERT INTO items VALUES (2, 20.0, 'x')", 3);
    exec(db, "INSERT INTO items VALUES (3, 5.0, 'y')", 4);
}

// ── Individual window functions ───────────────────────────────────────────

#[test]
fn test_row_number() {
    let mut db = TensorDb::new();
    setup(&mut db);
    let out = exec(
        &mut db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY price DESC) AS rn FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 3);
    // price desc: id2(20)=1, id1(10)=2, id3(5)=3
    let rn = |row: usize| ds.rows[row].get("rn").unwrap().clone();
    let id = |row: usize| ds.rows[row].get("id").unwrap().clone();
    for i in 0..3 {
        let expected = match id(i) {
            Value::Int(2) => 1,
            Value::Int(1) => 2,
            Value::Int(3) => 3,
            other => panic!("unexpected id {other:?}"),
        };
        assert_eq!(rn(i), Value::Int(expected));
    }
}

#[test]
fn test_rank_and_dense_rank_with_partition_by() {
    let mut db = TensorDb::new();
    setup(&mut db);
    exec(&mut db, "INSERT INTO items VALUES (4, 20.0, 'x')", 5); // tie with id2

    let out = exec(
        &mut db,
        "SELECT id, category, RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rk, DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS drk FROM items",
        6,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 4);
    for row in &ds.rows {
        let id = row.get("id").unwrap().clone();
        let (rk, drk) = (
            row.get("rk").unwrap().clone(),
            row.get("drk").unwrap().clone(),
        );
        match id {
            // category 'x': price 20(id2)=1, 20(id4)=1(tie), 10(id1)=3 (RANK has gap) / 2 (DENSE_RANK no gap)
            Value::Int(2) | Value::Int(4) => {
                assert_eq!(rk, Value::Int(1));
                assert_eq!(drk, Value::Int(1));
            }
            Value::Int(1) => {
                assert_eq!(
                    rk,
                    Value::Int(3),
                    "RANK should skip to 3 after a tie at 1-1"
                );
                assert_eq!(drk, Value::Int(2), "DENSE_RANK should not skip");
            }
            // category 'y': single row
            Value::Int(3) => {
                assert_eq!(rk, Value::Int(1));
                assert_eq!(drk, Value::Int(1));
            }
            other => panic!("unexpected id {other:?}"),
        }
    }
}

#[test]
fn test_lag_and_lead() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id, LAG(price) OVER (ORDER BY id) AS prev, LEAD(price) OVER (ORDER BY id) AS next FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 3);
    // ordered by id: 1(10), 2(20), 3(5)
    assert_eq!(ds.rows[0].get("prev").unwrap().clone(), Value::Null);
    assert_eq!(ds.rows[0].get("next").unwrap().clone(), Value::Float(20.0));
    assert_eq!(ds.rows[1].get("prev").unwrap().clone(), Value::Float(10.0));
    assert_eq!(ds.rows[1].get("next").unwrap().clone(), Value::Float(5.0));
    assert_eq!(ds.rows[2].get("prev").unwrap().clone(), Value::Float(20.0));
    assert_eq!(ds.rows[2].get("next").unwrap().clone(), Value::Null);
}

#[test]
fn test_windowed_sum_is_cumulative() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id, SUM(price) OVER (PARTITION BY category ORDER BY id) AS running FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    // category 'x': id1(10)->10, id2(20)->30; category 'y': id3(5)->5
    for row in &ds.rows {
        let id = row.get("id").unwrap().clone();
        let running = row.get("running").unwrap().clone();
        match id {
            Value::Int(1) => assert_eq!(running, Value::Float(10.0)),
            Value::Int(2) => assert_eq!(running, Value::Float(30.0)),
            Value::Int(3) => assert_eq!(running, Value::Float(5.0)),
            other => panic!("unexpected id {other:?}"),
        }
    }
}

// ── E1 regression: combining differently-specced window functions ────────

#[test]
fn test_lag_then_sum_with_different_specs() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = exec(
        &mut db,
        "SELECT id, LAG(price) OVER (ORDER BY id) AS prev, SUM(price) OVER (PARTITION BY category ORDER BY id) AS running FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 3, "no rows should be dropped");
    for row in &ds.rows {
        let id = row.get("id").unwrap().clone();
        let running = row.get("running").unwrap().clone();
        // Before the fix, `running` was wrongly identical across rows in
        // the same partition (LAG's NULL corrupted the row set).
        match id {
            Value::Int(1) => assert_eq!(running, Value::Float(10.0)),
            Value::Int(2) => assert_eq!(
                running,
                Value::Float(30.0),
                "running total for id=2 must be cumulative (10+20), not stuck at 10"
            ),
            Value::Int(3) => assert_eq!(running, Value::Float(5.0)),
            other => panic!("unexpected id {other:?}"),
        }
    }
}

#[test]
fn test_lag_then_row_number_different_specs_no_schema_error() {
    let mut db = TensorDb::new();
    setup(&mut db);

    // Before the fix this errored: Parse { msg: "Row 1 has incompatible schema" }
    let out = exec(
        &mut db,
        "SELECT id, LAG(price) OVER (ORDER BY id) AS prev, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table, got an error — this was the E1 schema-mismatch bug")
    };
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.rows[0].get("prev").unwrap().clone(), Value::Null);
    assert_eq!(ds.rows[0].get("rn").unwrap().clone(), Value::Int(1));
}

#[test]
fn test_row_number_then_lag_column_not_silently_dropped() {
    let mut db = TensorDb::new();
    setup(&mut db);

    // Before the fix, the `prev` column silently disappeared from the
    // output entirely when LAG followed a window function with a
    // different spec.
    let out = exec(
        &mut db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn, LAG(price) OVER (ORDER BY id) AS prev FROM items",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert!(
        ds.schema.get_field("prev").is_some(),
        "prev column must be present in schema"
    );
    assert_eq!(ds.rows[0].get("prev").unwrap().clone(), Value::Null);
    assert_eq!(ds.rows[1].get("prev").unwrap().clone(), Value::Float(10.0));
}

#[test]
fn test_six_window_functions_combined_with_differing_specs() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET wide COLUMNS (id: Int, price: Float, category: String)",
        1,
    );
    exec(&mut db, "INSERT INTO wide VALUES (1, 10.0, 'x')", 2);
    exec(&mut db, "INSERT INTO wide VALUES (2, 20.0, 'x')", 3);

    let out = exec(
        &mut db,
        "SELECT id, price, \
         ROW_NUMBER() OVER (ORDER BY price DESC) AS rn, \
         RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rk, \
         DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS drk, \
         LAG(price) OVER (ORDER BY id) AS prev_price, \
         LEAD(price, 2) OVER (ORDER BY id) AS next2_price, \
         SUM(price) OVER (PARTITION BY category ORDER BY id) AS running_total \
         FROM wide",
        4,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 2);

    let row1 = &ds.rows[0]; // id=1, price=10.0
    assert_eq!(row1.get("rn").unwrap().clone(), Value::Int(2));
    assert_eq!(row1.get("rk").unwrap().clone(), Value::Int(2));
    assert_eq!(row1.get("drk").unwrap().clone(), Value::Int(2));
    assert_eq!(row1.get("prev_price").unwrap().clone(), Value::Null);
    assert_eq!(row1.get("next2_price").unwrap().clone(), Value::Null);
    assert_eq!(
        row1.get("running_total").unwrap().clone(),
        Value::Float(10.0)
    );

    let row2 = &ds.rows[1]; // id=2, price=20.0
    assert_eq!(row2.get("rn").unwrap().clone(), Value::Int(1));
    assert_eq!(row2.get("rk").unwrap().clone(), Value::Int(1));
    assert_eq!(row2.get("drk").unwrap().clone(), Value::Int(1));
    assert_eq!(row2.get("prev_price").unwrap().clone(), Value::Float(10.0));
    assert_eq!(row2.get("next2_price").unwrap().clone(), Value::Null);
    assert_eq!(
        row2.get("running_total").unwrap().clone(),
        Value::Float(30.0),
        "running total must be cumulative (10+20), the original bug symptom"
    );
}
