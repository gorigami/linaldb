// tests/lazy_expression_test.rs
//
// Tests for lazy tensor expressions: the core evaluation kernel
// (evaluate_expression) and its DSL surface (`LAZY LET c = ADD a b`).
// Distinct from lazy *columns* on record-style datasets, which are covered
// in lazy_columns_test.rs.

use linal::core::tensor::{Expression, Shape, Tensor, TensorId, TensorMetadata};
use linal::dsl::execute_script;
use linal::engine::kernels::evaluate_expression;
use linal::engine::TensorDb;

#[test]
fn test_lazy_add_multiply() {
    let id_a = TensorId::new();
    let id_b = TensorId::new();
    let shape = Shape::new(vec![2]);
    let a = Tensor::new(
        id_a,
        shape.clone(),
        vec![1.0, 2.0],
        TensorMetadata::new(id_a, None),
    )
    .unwrap();
    let b = Tensor::new(
        id_b,
        shape.clone(),
        vec![10.0, 20.0],
        TensorMetadata::new(id_b, None),
    )
    .unwrap();

    // Expression: (A + B) * 2.0
    let expr = Expression::ScalarMul(
        Box::new(Expression::Add(
            Box::new(Expression::Literal(a)),
            Box::new(Expression::Literal(b)),
        )),
        2.0,
    );

    let res = evaluate_expression(&expr, chrono::Utc::now()).unwrap();

    assert_eq!(res.data_ref(), &[22.0, 44.0]);
}

#[test]
fn test_lazy_matmul() {
    let id_a = TensorId::new();
    let id_b = TensorId::new();

    // 2x1 * 1x2 -> 2x2
    let a = Tensor::new(
        id_a,
        Shape::new(vec![2, 1]),
        vec![2.0, 3.0],
        TensorMetadata::new(id_a, None),
    )
    .unwrap();
    let b = Tensor::new(
        id_b,
        Shape::new(vec![1, 2]),
        vec![4.0, 5.0],
        TensorMetadata::new(id_b, None),
    )
    .unwrap();

    let expr = Expression::MatMul(
        Box::new(Expression::Literal(a)),
        Box::new(Expression::Literal(b)),
    );

    let res = evaluate_expression(&expr, chrono::Utc::now()).unwrap();

    assert_eq!(res.shape.dims, vec![2, 2]);
    // [2 * 4, 2 * 5]
    // [3 * 4, 3 * 5]
    assert_eq!(res.data_ref(), &[8.0, 10.0, 12.0, 15.0]);
}

#[test]
fn test_lazy_dsl_flow() {
    let mut db = TensorDb::new();

    // 1. Define base tensors
    let script = "
        VECTOR a = [1.0, 2.0, 3.0]
        VECTOR b = [4.0, 5.0, 6.0]
        # Define a lazy addition
        LAZY LET c = ADD a b
        # Define a lazy multiplication on top of lazy addition
        LAZY LET d = SCALE c BY 2.0
    ";

    execute_script(&mut db, script).expect("Script execution failed");

    // 2. Verify variables are registered as lazy
    let names = db.list_names();
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"c".to_string()));
    assert!(names.contains(&"d".to_string()));

    // 3. SHOW should trigger evaluation
    let show_script = "SHOW d";
    execute_script(&mut db, show_script).expect("SHOW d failed");
}
