// tests/dsl_dataset_complete.rs
//
// Comprehensive integration test for the complete dataset DSL implementation
// This test defines the expected behavior for all three phases:
// - Phase 1: Engine integration (DONE)
// - Phase 2: Dataset definition and insertion
// - Phase 3: Query language

use linal::{execute_script, TensorDb};

#[test]
fn test_complete_dataset_workflow() {
    let mut db = TensorDb::new();

    let script = r#"
        # Phase 2: Dataset Definition and Insertion
        DATASET users COLUMNS (id: INT, name: STRING, age: INT, score: FLOAT)
        
        INSERT INTO users VALUES (1, "Alice", 30, 0.95)
        INSERT INTO users VALUES (2, "Bob", 25, 0.87)
        INSERT INTO users VALUES (3, "Carol", 35, 0.92)
        INSERT INTO users VALUES (4, "Dave", 28, 0.88)
        INSERT INTO users VALUES (5, "Eve", 32, 0.91)
        
        # Show all datasets
        SHOW ALL DATASETS
        
        # Phase 3: Query Language
        # Filter: Get users older than 28
        DATASET adults FROM users FILTER age > 28
        
        # Select: Project specific columns
        DATASET names_scores FROM users SELECT name, score
        
        # Order and Limit: Top 3 users by score
        DATASET top_users FROM users ORDER BY score DESC LIMIT 3
        
        # Combined query: Adults with high scores
        DATASET high_performers FROM users FILTER age > 28 SELECT name, age, score ORDER BY score DESC
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify original dataset
    let users = db.get_dataset("users").unwrap();
    assert_eq!(users.len(), 5);
    assert_eq!(users.schema.len(), 4);

    // Verify filtered dataset
    let adults = db.get_dataset("adults").unwrap();
    assert_eq!(adults.len(), 3); // Alice(30), Carol(35), Eve(32) - Dave(28) excluded

    // Verify projected dataset
    let names_scores = db.get_dataset("names_scores").unwrap();
    assert_eq!(names_scores.schema.len(), 2);
    assert_eq!(names_scores.len(), 5);

    // Verify ordered and limited dataset
    let top_users = db.get_dataset("top_users").unwrap();
    assert_eq!(top_users.len(), 3);
    // Should be Alice(0.95), Carol(0.92), Eve(0.91)

    // Verify combined query
    let high_performers = db.get_dataset("high_performers").unwrap();
    assert_eq!(high_performers.schema.len(), 3);
    assert!(high_performers.len() <= 4);
}

#[test]
fn test_dataset_creation_basic() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET products COLUMNS (id: INT, name: STRING, price: FLOAT)
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify dataset was created
    let products = db.get_dataset("products").unwrap();
    assert_eq!(products.len(), 0);
    assert_eq!(products.schema.len(), 3);
}

#[test]
fn test_dataset_insertion_basic() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET items COLUMNS (id: INT, name: STRING)
        INSERT INTO items VALUES (1, "Item A")
        INSERT INTO items VALUES (2, "Item B")
    "#;

    execute_script(&mut db, script).unwrap();

    let items = db.get_dataset("items").unwrap();
    assert_eq!(items.len(), 2);
}

#[test]
fn test_dataset_filter() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET numbers COLUMNS (id: INT, value: INT)
        INSERT INTO numbers VALUES (1, 10)
        INSERT INTO numbers VALUES (2, 20)
        INSERT INTO numbers VALUES (3, 30)
        INSERT INTO numbers VALUES (4, 40)
        
        DATASET big_numbers FROM numbers FILTER value > 20
    "#;

    execute_script(&mut db, script).unwrap();

    let big_numbers = db.get_dataset("big_numbers").unwrap();
    assert_eq!(big_numbers.len(), 2); // 30 and 40
}

#[test]
fn test_dataset_select() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET people COLUMNS (id: INT, name: STRING, age: INT, city: STRING)
        INSERT INTO people VALUES (1, "Alice", 30, "NYC")
        INSERT INTO people VALUES (2, "Bob", 25, "LA")
        
        DATASET names_only FROM people SELECT name
    "#;

    execute_script(&mut db, script).unwrap();

    let names_only = db.get_dataset("names_only").unwrap();
    assert_eq!(names_only.schema.len(), 1);
    assert_eq!(names_only.len(), 2);
}

#[test]
fn test_dataset_order_by() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET scores COLUMNS (name: STRING, score: FLOAT)
        INSERT INTO scores VALUES ("Alice", 0.95)
        INSERT INTO scores VALUES ("Bob", 0.87)
        INSERT INTO scores VALUES ("Carol", 0.92)
        
        DATASET sorted FROM scores ORDER BY score DESC
    "#;

    execute_script(&mut db, script).unwrap();

    let sorted = db.get_dataset("sorted").unwrap();
    assert_eq!(sorted.len(), 3);
    // First row should be Alice with 0.95
}

#[test]
fn test_dataset_limit() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET numbers COLUMNS (value: INT)
        INSERT INTO numbers VALUES (1)
        INSERT INTO numbers VALUES (2)
        INSERT INTO numbers VALUES (3)
        INSERT INTO numbers VALUES (4)
        INSERT INTO numbers VALUES (5)
        
        DATASET first_three FROM numbers LIMIT 3
    "#;

    execute_script(&mut db, script).unwrap();

    let first_three = db.get_dataset("first_three").unwrap();
    assert_eq!(first_three.len(), 3);
}

#[test]
fn test_dataset_combined_query() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET employees COLUMNS (id: INT, name: STRING, dept: STRING, salary: FLOAT)
        INSERT INTO employees VALUES (1, "Alice", "Engineering", 95000.0)
        INSERT INTO employees VALUES (2, "Bob", "Sales", 75000.0)
        INSERT INTO employees VALUES (3, "Carol", "Engineering", 98000.0)
        INSERT INTO employees VALUES (4, "Dave", "Sales", 72000.0)
        INSERT INTO employees VALUES (5, "Eve", "Engineering", 92000.0)
        
        DATASET top_engineers FROM employees FILTER dept = "Engineering" SELECT name, salary ORDER BY salary DESC LIMIT 2
    "#;

    execute_script(&mut db, script).unwrap();

    let top_engineers = db.get_dataset("top_engineers").unwrap();
    assert_eq!(top_engineers.len(), 2);
    assert_eq!(top_engineers.schema.len(), 2);
    // Should be Carol (98000) and Alice (95000)
}

#[test]
fn test_show_all_datasets() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET dataset1 COLUMNS (id: INT)
        DATASET dataset2 COLUMNS (name: STRING)
        DATASET dataset3 COLUMNS (value: FLOAT)
        
        SHOW ALL DATASETS
    "#;

    execute_script(&mut db, script).unwrap();

    let names = db.list_dataset_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"dataset1".to_string()));
    assert!(names.contains(&"dataset2".to_string()));
    assert!(names.contains(&"dataset3".to_string()));
}

#[test]
fn test_dataset_type_validation() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET typed COLUMNS (int_col: INT, float_col: FLOAT, string_col: STRING, bool_col: BOOL)
        INSERT INTO typed VALUES (42, 3.14, "hello", true)
    "#;

    execute_script(&mut db, script).unwrap();

    let typed = db.get_dataset("typed").unwrap();
    assert_eq!(typed.len(), 1);
    assert_eq!(typed.schema.len(), 4);
}

#[test]
#[should_panic] // Should fail due to type mismatch
fn test_dataset_type_mismatch() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET strict COLUMNS (id: INT, name: STRING)
        INSERT INTO strict VALUES ("not_an_int", "Alice")
    "#;

    execute_script(&mut db, script).unwrap();
}

#[test]
fn test_dataset_derived_column_and_persistence_round_trip() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        LET ds = dataset("test_ds")
        ds.add_column("vec", v1)

        LET v2 = ds.vec * 2.0

        ds.add_column("vec_doubled", v2)

        SAVE DATASET test_ds TO "test_output.parquet"

        SHOW ds
    "#;

    execute_script(&mut db, script).unwrap();

    let ds = db.get_tensor_dataset("test_ds").unwrap();
    assert_eq!(ds.columns.len(), 2);

    // Check v2 value
    let v2_tensor = db.get("v2").unwrap();
    assert_eq!(*v2_tensor.data, vec![2.0, 4.0, 6.0]);

    // Clear and load back
    let mut db2 = TensorDb::new();
    let load_script = r#"
        LOAD DATASET test_ds FROM "test_output.parquet"
        SHOW test_ds
    "#;
    execute_script(&mut db2, load_script).unwrap();

    let loaded_ds = db2.get_dataset("test_ds").unwrap();
    assert_eq!(loaded_ds.len(), 3);
}

#[test]
#[should_panic] // Should fail due to duplicate dataset name
fn test_dataset_duplicate_name() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET users COLUMNS (id: INT)
        DATASET users COLUMNS (name: STRING)
    "#;

    execute_script(&mut db, script).unwrap();
}
