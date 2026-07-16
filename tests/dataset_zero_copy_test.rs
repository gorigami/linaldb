use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn test_zero_copy_guarantee() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        LET ds = dataset("test_ds")
        ds.add_column("c1", v1)
    "#;
    execute_script(&mut db, script).unwrap();

    let ds = db.get_tensor_dataset("test_ds").unwrap();
    let col_ref = ds.get_reference("c1").unwrap();
    let col_tensor_id = match col_ref {
        linal::core::dataset::ResourceReference::Tensor { id } => *id,
        _ => panic!("Expected tensor reference"),
    };

    // Get tensor from names
    let entry = db.active_instance().get_tensor_id("v1").unwrap();

    assert_eq!(col_tensor_id, entry);

    // Verify deref points to same data
    let t1 = db.get("v1").unwrap();
    let ds_t1 = db.active_instance().store.get(col_tensor_id).unwrap();

    assert!(std::sync::Arc::ptr_eq(&t1.data, &ds_t1.data));
}
