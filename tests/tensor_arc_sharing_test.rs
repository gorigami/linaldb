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
