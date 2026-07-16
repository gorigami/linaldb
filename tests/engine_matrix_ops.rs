use linal::engine::context::ExecutionContext;
use linal::{Shape, TensorDb};

#[test]
fn test_engine_matmul() {
    let mut db = TensorDb::new();

    // Create two matrices: A (2x3) and B (3x2)
    db.insert_named(
        "A",
        Shape::new(vec![2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    db.insert_named(
        "B",
        Shape::new(vec![3, 2]),
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Perform matrix multiplication: C = A * B
    db.eval_matmul(&mut ctx, "C", "A", "B").unwrap();

    // Verify result
    let c = db.get("C").unwrap();
    assert_eq!(c.shape.dims, vec![2, 2]);

    // Expected: [[58, 64], [139, 154]]
    assert_eq!(c.data_ref(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_engine_matmul_identity() {
    let mut db = TensorDb::new();

    // Create a 2x2 matrix
    db.insert_named("A", Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap();

    // Create identity matrix
    db.insert_named("I", Shape::new(vec![2, 2]), vec![1.0, 0.0, 0.0, 1.0])
        .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // A * I should equal A
    db.eval_matmul(&mut ctx, "result", "A", "I").unwrap();

    let result = db.get("result").unwrap();
    let original = db.get("A").unwrap();

    assert_eq!(result.data, original.data);
}

#[test]
fn test_engine_matmul_dimension_mismatch() {
    let mut db = TensorDb::new();

    // Create incompatible matrices
    db.insert_named(
        "A",
        Shape::new(vec![2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    db.insert_named("B", Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // This should fail - dimensions don't match
    let result = db.eval_matmul(&mut ctx, "C", "A", "B");
    assert!(result.is_err());
}

#[test]
fn test_engine_reshape_basic() {
    let mut db = TensorDb::new();

    // Create a 2x3 matrix
    db.insert_named(
        "A",
        Shape::new(vec![2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Reshape to 3x2
    db.eval_reshape(&mut ctx, "B", "A", Shape::new(vec![3, 2]))
        .unwrap();

    let b = db.get("B").unwrap();
    assert_eq!(b.shape.dims, vec![3, 2]);
    assert_eq!(b.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_engine_reshape_to_vector() {
    let mut db = TensorDb::new();

    // Create a 2x3 matrix
    db.insert_named(
        "M",
        Shape::new(vec![2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Reshape to 1D vector
    db.eval_reshape(&mut ctx, "V", "M", Shape::new(vec![6]))
        .unwrap();

    let v = db.get("V").unwrap();
    assert_eq!(v.shape.dims, vec![6]);
    assert_eq!(v.shape.rank(), 1);
    assert_eq!(v.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_engine_reshape_invalid_size() {
    let mut db = TensorDb::new();

    // Create a 2x3 matrix (6 elements)
    db.insert_named(
        "A",
        Shape::new(vec![2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Try to reshape to incompatible size (2x2 = 4 elements)
    let result = db.eval_reshape(&mut ctx, "B", "A", Shape::new(vec![2, 2]));
    assert!(result.is_err());
}

#[test]
fn test_engine_matmul_chain() {
    let mut db = TensorDb::new();

    // Create three matrices for chained multiplication
    db.insert_named("A", Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap();

    db.insert_named("B", Shape::new(vec![2, 2]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap();

    db.insert_named("C", Shape::new(vec![2, 2]), vec![1.0, 0.0, 0.0, 1.0])
        .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Compute A * B
    db.eval_matmul(&mut ctx, "AB", "A", "B").unwrap();

    // Compute (A * B) * C
    db.eval_matmul(&mut ctx, "ABC", "AB", "C").unwrap();

    let abc = db.get("ABC").unwrap();
    assert_eq!(abc.shape.dims, vec![2, 2]);

    // (A * B) * I = A * B
    let ab = db.get("AB").unwrap();
    assert_eq!(abc.data, ab.data);
}

#[test]
fn test_engine_reshape_then_matmul() {
    let mut db = TensorDb::new();

    // Create a vector
    db.insert_named("V", Shape::new(vec![6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Reshape to 2x3 matrix
    db.eval_reshape(&mut ctx, "A", "V", Shape::new(vec![2, 3]))
        .unwrap();

    // Create another matrix 3x2
    db.insert_named(
        "B",
        Shape::new(vec![3, 2]),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Multiply reshaped matrix
    db.eval_matmul(&mut ctx, "C", "A", "B").unwrap();

    let c = db.get("C").unwrap();
    assert_eq!(c.shape.dims, vec![2, 2]);
}

#[test]
fn test_engine_strict_mode_matmul() {
    let mut db = TensorDb::new();

    // Create matrices in STRICT mode
    db.insert_named_with_kind(
        "A",
        Shape::new(vec![2, 2]),
        vec![1.0, 2.0, 3.0, 4.0],
        linal::TensorKind::Strict,
    )
    .unwrap();

    db.insert_named_with_kind(
        "B",
        Shape::new(vec![2, 2]),
        vec![5.0, 6.0, 7.0, 8.0],
        linal::TensorKind::Strict,
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Matmul should work in strict mode
    db.eval_matmul(&mut ctx, "C", "A", "B").unwrap();

    let c = db.get("C").unwrap();
    assert_eq!(c.shape.dims, vec![2, 2]);
    // Expected: [[19, 22], [43, 50]]
    assert_eq!(c.data_ref(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_engine_multiple_reshapes() {
    let mut db = TensorDb::new();

    // Start with a vector
    db.insert_named(
        "V",
        Shape::new(vec![12]),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    // Create context
    let mut ctx = ExecutionContext::new();

    // Reshape to 3x4
    db.eval_reshape(&mut ctx, "M1", "V", Shape::new(vec![3, 4]))
        .unwrap();
    let m1 = db.get("M1").unwrap();
    assert_eq!(m1.shape.dims, vec![3, 4]);

    // Reshape to 4x3
    db.eval_reshape(&mut ctx, "M2", "M1", Shape::new(vec![4, 3]))
        .unwrap();
    let m2 = db.get("M2").unwrap();
    assert_eq!(m2.shape.dims, vec![4, 3]);

    // Reshape to 2x6
    db.eval_reshape(&mut ctx, "M3", "M2", Shape::new(vec![2, 6]))
        .unwrap();
    let m3 = db.get("M3").unwrap();
    assert_eq!(m3.shape.dims, vec![2, 6]);

    // Reshape back to vector
    db.eval_reshape(&mut ctx, "V2", "M3", Shape::new(vec![12]))
        .unwrap();
    let v2 = db.get("V2").unwrap();

    // Data should be preserved through all reshapes
    let original = db.get("V").unwrap();
    assert_eq!(v2.data, original.data);
}

#[test]
fn test_engine_binary_and_unary_scenario() {
    use linal::{BinaryOp, UnaryOp};

    let mut db = TensorDb::new();

    let shape_vec3 = Shape::new(vec![3]);

    db.insert_named("a", shape_vec3.clone(), vec![1.0, 0.0, 0.0])
        .unwrap();
    db.insert_named("b", shape_vec3.clone(), vec![0.0, 1.0, 0.0])
        .unwrap();
    db.insert_named("c", shape_vec3.clone(), vec![1.0, 1.0, 0.0])
        .unwrap();

    let mut ctx = ExecutionContext::new();

    db.eval_binary(&mut ctx, "c1", "a", "b", BinaryOp::Add)
        .unwrap();
    db.eval_binary(&mut ctx, "corr_ac", "a", "c", BinaryOp::Correlate)
        .unwrap();
    db.eval_binary(&mut ctx, "sim_ac", "a", "c", BinaryOp::Similarity)
        .unwrap();
    db.eval_unary(&mut ctx, "a_half", "a", UnaryOp::Scale(0.5))
        .unwrap();

    let c1 = db.get("c1").unwrap();
    let corr_ac = db.get("corr_ac").unwrap();
    let sim_ac = db.get("sim_ac").unwrap();
    let a_half = db.get("a_half").unwrap();

    assert_eq!(c1.shape.dims, vec![3]);
    assert_eq!(c1.data_ref(), vec![1.0, 1.0, 0.0]);

    assert_eq!(corr_ac.shape.rank(), 0);
    assert!((corr_ac.data[0] - 1.0).abs() < 1e-6);

    let expected = 1.0 / 2f32.sqrt();
    assert!((sim_ac.data[0] - expected).abs() < 1e-6);

    assert_eq!(a_half.data_ref(), vec![0.5, 0.0, 0.0]);
}
