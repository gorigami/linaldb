use linal::core::tensor::{Shape, Tensor, TensorId};
use std::sync::Arc;

#[test]
fn test_from_shared() {
    let data = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
    let shape = Shape::new(vec![2, 2]);
    let id = TensorId::new();
    let metadata = Arc::new(linal::core::tensor::TensorMetadata::new(id, None));

    let tensor = Tensor::from_shared(id, shape.clone(), data.clone(), metadata)
        .expect("Should create tensor");

    assert_eq!(tensor.len(), 4);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.data_ref(), &[1.0, 2.0, 3.0, 4.0]);

    // Verify it is actually using the shared memory
    assert!(Arc::ptr_eq(&tensor.data, &data));
}

#[test]
fn test_share_from_owned() {
    let data = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(vec![3]);
    let id = TensorId::new();
    let metadata = linal::core::tensor::TensorMetadata::new(id, None);
    let tensor = Tensor::new(id, shape, data, metadata).expect("Should create tensor");

    // calling share on tensor should return reference to same Arc
    let shared = tensor.share();
    assert_eq!(*shared, vec![1.0, 2.0, 3.0]);

    // In new implementation, tensor.share() just clones the Arc
    assert!(Arc::ptr_eq(&tensor.data, &shared));
}

#[test]
fn test_share_from_shared() {
    let data = Arc::new(vec![10.0, 20.0]);
    let shape = Shape::new(vec![2]);
    let id = TensorId::new();
    let metadata = Arc::new(linal::core::tensor::TensorMetadata::new(id, None));
    let tensor =
        Tensor::from_shared(id, shape, data.clone(), metadata).expect("Should create tensor");

    // calling share on already shared tensor should return clone of Arc (cheap)
    let shared_again = tensor.share();
    assert!(Arc::ptr_eq(&data, &shared_again));
}

// test_copy_on_write removed because Tensor is now strictly immutable

#[test]
fn test_dataset_column_zero_copy_guarantee() {
    let mut db = linal::TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        LET ds = dataset("test_ds")
        ds.add_column("c1", v1)
    "#;
    linal::dsl::execute_script(&mut db, script).unwrap();

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

    assert!(Arc::ptr_eq(&t1.data, &ds_t1.data));
}
