use linal::dsl::execute_line;
use linal::TensorDb;

#[test]
fn test_dataset_integrity_health_warning() {
    let mut db = TensorDb::new();

    // 1. Create a tensor and a dataset
    execute_line(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]", 1).unwrap();
    execute_line(&mut db, r#"LET ds = dataset("test_ds")"#, 2).unwrap();
    execute_line(&mut db, "ds.add_column('col1', v1)", 3).unwrap();

    // 2. Verify it's healthy
    let output = execute_line(&mut db, "SHOW ds", 4).unwrap();
    let output_str = output.to_string();
    assert!(output_str.contains("✅ Dataset verified"));
    assert!(output_str.contains("col1"));

    // 3. Break integrity
    db.remove_tensor("v1");

    // 4. Verify it's unhealthy
    let output = execute_line(&mut db, "SHOW ds", 5).unwrap();
    let output_str = output.to_string();
    assert!(output_str.contains("⚠️  HEALTH WARNING"));
    assert!(output_str.contains("[!] Column 'col1' depends on a deleted or missing tensor"));
}

#[test]
fn test_row_count_validation_error_message() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]", 1).unwrap();
    execute_line(&mut db, "VECTOR v2 = [10.0, 20.0]", 2).unwrap();
    execute_line(&mut db, r#"LET ds = dataset("test_ds")"#, 3).unwrap();
    execute_line(&mut db, "ds.add_column('col1', v1)", 4).unwrap();

    let result = execute_line(&mut db, "ds.add_column('col2', v2)", 5);
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Column 'col2' has 2 rows, but dataset 'test_ds' has 3 rows"));
}

#[test]
fn test_matrix_row_count_validation() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "MATRIX m1 = [[1.0, 2.0], [3.0, 4.0]]", 1).unwrap(); // 2 rows
    execute_line(
        &mut db,
        "MATRIX m2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]",
        2,
    )
    .unwrap(); // 3 rows
    execute_line(&mut db, r#"LET ds = dataset("matrix_ds")"#, 3).unwrap();
    execute_line(&mut db, "ds.add_column('col1', m1)", 4).unwrap();

    let result = execute_line(&mut db, "ds.add_column('col2', m2)", 5);
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Column 'col2' has 3 rows, but dataset 'matrix_ds' has 2 rows"));
}
