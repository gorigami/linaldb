// tests/tensor_dataset_dsl_test.rs
//
// Tests for the tensor-dataset builder DSL: `LET ds = dataset("name")` plus
// `ds.add_column(...)`. Distinct from the record-style `DATASET ... COLUMNS`
// dataset feature covered in dsl_dataset_complete.rs.

use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn test_tensor_dataset_dsl() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        VECTOR v2 = [4.0, 5.0, 6.0]

        LET ds = dataset("my_dataset")
        ds.add_column("vec1", v1)
        ds.add_column("vec2", v2)

        SHOW ds
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify dataset in registry
    let ds = db.get_tensor_dataset("my_dataset").unwrap();
    assert_eq!(ds.name, "my_dataset");
    assert_eq!(ds.columns.len(), 2);
    assert!(ds.columns.contains_key("vec1"));
    assert!(ds.columns.contains_key("vec2"));
}

#[test]
fn test_comprehensive_tensor_dataset() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        VECTOR v2 = [10.0, 20.0, 30.0]
        MATRIX m1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        VECTOR s1 = [42.0, 43.0, 44.0]

        LET ds = dataset("ml_ready")
        ds.add_column("features_a", v1)
        ds.add_column("features_b", v2)
        ds.add_column("weights", m1)
        ds.add_column("target", s1)

        SHOW ds

        LET v3 = ADD v1 v2
        ds.add_column("combined", v3)

        SHOW ds
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify results via engine API
    let ds = db
        .get_tensor_dataset("ml_ready")
        .expect("Dataset ml_ready should exist");
    assert_eq!(ds.name, "ml_ready");
    assert_eq!(ds.columns.len(), 5);

    // Check specific columns
    assert!(ds.columns.contains_key("features_a"));
    assert!(ds.columns.contains_key("combined"));

    // Verify Column Schema
    let col_m1 = ds
        .schema
        .columns
        .iter()
        .find(|c| c.name == "weights")
        .unwrap();
    // MATRIX(3, 2)
    assert!(format!("{:?}", col_m1.value_type).contains("Matrix(3, 2)"));
}
