// tests/avg_aggregation_test.rs
//
// Tests for AVG aggregation function

use linal::core::value::Value;
use linal::dsl::execute_line;
use linal::{execute_script, TensorDb};

#[test]
fn test_avg_basic_int() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET scores COLUMNS (id: Int, score: Int)
        INSERT INTO scores VALUES (1, 10)
        INSERT INTO scores VALUES (2, 20)
        INSERT INTO scores VALUES (3, 30)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG(score) should be (10 + 20 + 30) / 3 = 20.0
    let query = r#"DATASET result FROM scores SELECT AVG(score)"#;
    execute_line(&mut db, query, 0).expect("AVG query failed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows[0].values[0], Value::Float(20.0));
}

#[test]
fn test_avg_basic_float() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET prices COLUMNS (id: Int, price: Float)
        INSERT INTO prices VALUES (1, 10.5)
        INSERT INTO prices VALUES (2, 20.5)
        INSERT INTO prices VALUES (3, 30.5)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG(price) should be (10.5 + 20.5 + 30.5) / 3 = 20.5
    let query = r#"DATASET result FROM prices SELECT AVG(price)"#;
    execute_line(&mut db, query, 0).expect("AVG query failed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows[0].values[0], Value::Float(20.5));
}

#[test]
fn test_avg_with_group_by() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET sales COLUMNS (region: String, amount: Int)
        INSERT INTO sales VALUES ("North", 100)
        INSERT INTO sales VALUES ("South", 200)
        INSERT INTO sales VALUES ("North", 150)
        INSERT INTO sales VALUES ("South", 250)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG(amount) GROUP BY region
    // North: (100 + 150) / 2 = 125.0
    // South: (200 + 250) / 2 = 225.0
    let query = r#"DATASET result FROM sales GROUP BY region SELECT region, AVG(amount)"#;
    execute_line(&mut db, query, 0).expect("AVG GROUP BY query failed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 2);

    // Find North and South rows
    let find_row = |region: &str| {
        result
            .rows
            .iter()
            .find(|r| r.values[0] == Value::String(region.to_string()))
    };

    let north = find_row("North").expect("North not found");
    assert_eq!(north.values[1], Value::Float(125.0));

    let south = find_row("South").expect("South not found");
    assert_eq!(south.values[1], Value::Float(225.0));
}

#[test]
fn test_avg_computed_column() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET items COLUMNS (id: Int, price: Int, qty: Int)
        INSERT INTO items VALUES (1, 10, 2)
        INSERT INTO items VALUES (2, 20, 3)
        INSERT INTO items VALUES (3, 30, 1)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG(price * qty)
    // Row 1: 10 * 2 = 20
    // Row 2: 20 * 3 = 60
    // Row 3: 30 * 1 = 30
    // Average: (20 + 60 + 30) / 3 = 36.666...
    let query = r#"DATASET result FROM items SELECT AVG(price * qty)"#;
    execute_line(&mut db, query, 0).expect("AVG computed column query failed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 1);
    // Allow small floating point differences
    if let Value::Float(avg) = result.rows[0].values[0] {
        assert!(
            (avg - 36.666666).abs() < 0.01,
            "Expected ~36.67, got {}",
            avg
        );
    } else {
        panic!("Expected Float value");
    }
}

#[test]
fn test_avg_mixed_int_float() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET values COLUMNS (id: Int, value: Int)
        INSERT INTO values VALUES (1, 10)
        INSERT INTO values VALUES (2, 20)
        INSERT INTO values VALUES (3, 30)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG should convert Int to Float
    let query = r#"DATASET result FROM values SELECT AVG(value)"#;
    execute_line(&mut db, query, 0).expect("AVG query failed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows[0].values[0], Value::Float(20.0));
}

#[test]
fn test_avg_empty_dataset() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET empty COLUMNS (id: Int, value: Int)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // AVG over no rows: one row, NULL (SQL semantics since v0.1.93; it used
    // to return no rows).
    let query = r#"DATASET result FROM empty SELECT AVG(value)"#;
    execute_line(&mut db, query, 0).expect("AVG query should succeed");

    let result = db.get_dataset("result").expect("Result dataset not found");
    assert_eq!(result.len(), 1, "a global aggregate always returns one row");
    assert_eq!(result.rows[0].values[0], Value::Null);
}
