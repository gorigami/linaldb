// tests/dsl_matrix_ops.rs

use linal::{execute_script, TensorDb};

#[test]
fn test_dsl_vector_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3]
        SHOW v
    "#;

    execute_script(&mut db, script).unwrap();

    let v = db.get("v").unwrap();
    assert_eq!(v.shape.dims, vec![3]);
    assert_eq!(v.data_ref(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dsl_matrix_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2], [3, 4]]
        SHOW A
    "#;

    execute_script(&mut db, script).unwrap();

    let a = db.get("A").unwrap();
    assert_eq!(a.shape.dims, vec![2, 2]);
    assert_eq!(a.data_ref(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dsl_matrix_3x2() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX M = [[1, 2], [3, 4], [5, 6]]
    "#;

    execute_script(&mut db, script).unwrap();

    let m = db.get("M").unwrap();
    assert_eq!(m.shape.dims, vec![3, 2]);
    assert_eq!(m.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_matmul() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        MATRIX B = [[7, 8], [9, 10], [11, 12]]
        LET C = MATMUL A B
        SHOW C
    "#;

    execute_script(&mut db, script).unwrap();

    let c = db.get("C").unwrap();
    assert_eq!(c.shape.dims, vec![2, 2]);
    // Expected: [[58, 64], [139, 154]]
    assert_eq!(c.data_ref(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_dsl_transpose() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        LET A_T = TRANSPOSE A
        SHOW A_T
    "#;

    execute_script(&mut db, script).unwrap();

    let a_t = db.get("A_T").unwrap();
    assert_eq!(a_t.shape.dims, vec![3, 2]);

    // With zero-copy transpose, the internal data buffer is shared and thus
    // has the original layout [1, 2, 3, 4, 5, 6].
    // We cannot just inspect `data_ref()`. We must verify logical indexing.
    // Original A: [[1, 2, 3], [4, 5, 6]]
    // Transposed A: [[1, 4], [2, 5], [3, 6]]

    use linal::engine::kernels::index;
    assert_eq!(index(a_t, &[0, 0]).unwrap(), 1.0);
    assert_eq!(index(a_t, &[0, 1]).unwrap(), 4.0);
    assert_eq!(index(a_t, &[1, 0]).unwrap(), 2.0);
    assert_eq!(index(a_t, &[1, 1]).unwrap(), 5.0);
    assert_eq!(index(a_t, &[2, 0]).unwrap(), 3.0);
    assert_eq!(index(a_t, &[2, 1]).unwrap(), 6.0);
}

#[test]
fn test_dsl_flatten() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX M = [[1, 2, 3], [4, 5, 6]]
        LET flat = FLATTEN M
        SHOW flat
    "#;

    execute_script(&mut db, script).unwrap();

    let flat = db.get("flat").unwrap();
    assert_eq!(flat.shape.dims, vec![6]);
    assert_eq!(flat.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_reshape() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4, 5, 6]
        LET M = RESHAPE v TO [2, 3]
        SHOW M
    "#;

    execute_script(&mut db, script).unwrap();

    let m = db.get("M").unwrap();
    assert_eq!(m.shape.dims, vec![2, 3]);
    assert_eq!(m.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_matrix_chain_operations() {
    let mut db = TensorDb::new();

    let script = r#"
        # Create two matrices
        MATRIX A = [[1, 2], [3, 4]]
        MATRIX B = [[5, 6], [7, 8]]
        
        # Multiply them
        LET C = MATMUL A B
        
        # Transpose the result
        LET C_T = TRANSPOSE C
        
        # Flatten it
        LET flat = FLATTEN C_T
        
        SHOW flat
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify the chain worked
    let flat = db.get("flat").unwrap();
    assert_eq!(flat.shape.dims, vec![4]);
    assert_eq!(flat.shape.rank(), 1);
}

#[test]
fn test_dsl_vector_operations() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR a = [1, 2, 3]
        VECTOR b = [4, 5, 6]

        LET sum = ADD a b
        LET corr = CORRELATE a WITH b
        LET sim = SIMILARITY a WITH b

        SHOW sum
        SHOW corr
        SHOW sim
    "#;

    execute_script(&mut db, script).unwrap();

    let sum = db.get("sum").unwrap();
    assert_eq!(sum.data_ref(), vec![5.0, 7.0, 9.0]);

    // b = a + 3 is a perfect positive affine relationship -> Pearson correlation of 1.0.
    let corr = db.get("corr").unwrap();
    assert!((corr.data[0] - 1.0).abs() < 1e-5);
}

/// Regression test for a silent-correctness bug found via a real end-to-end example (leukemia
/// gene-expression classification in linal-hub): `CORRELATE` is documented
/// (docs/DSL_REFERENCE.md) as computing Pearson correlation, but was actually wired to a raw,
/// unnormalized dot product (`dot([1,2,3,4],[2,4,5,9]) = 61`, visibly not a correlation
/// coefficient). This pair is not perfectly (anti-)correlated, so the two statistics give
/// clearly different numbers -- the perfectly-correlated pair in `dsl_scenarios.rs` alone
/// wouldn't have caught a dot-product-in-disguise, since a perfect correlation is 1.0 either way.
#[test]
fn test_correlate_is_pearson_not_dot_product() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR a = [1, 2, 3, 4]
        VECTOR b = [2, 4, 5, 9]
        LET corr = CORRELATE a WITH b
    "#;
    execute_script(&mut db, script).unwrap();

    let corr = db.get("corr").unwrap();
    // Hand-computed Pearson correlation for this pair -- nowhere near 61.0 (the raw dot product).
    assert!((corr.data[0] - 0.9647638).abs() < 1e-5);
}

#[test]
fn test_dsl_reshape_multiple_times() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        LET M1 = RESHAPE v TO [3, 4]
        LET M2 = RESHAPE M1 TO [4, 3]
        LET M3 = RESHAPE M2 TO [2, 6]
        LET back = RESHAPE M3 TO [12]
    "#;

    execute_script(&mut db, script).unwrap();

    let back = db.get("back").unwrap();
    let original = db.get("v").unwrap();
    assert_eq!(back.data, original.data);
}

#[test]
fn test_dsl_matmul_with_transpose() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        LET A_T = TRANSPOSE A
        LET result = MATMUL A A_T
    "#;

    execute_script(&mut db, script).unwrap();

    let result = db.get("result").unwrap();
    assert_eq!(result.shape.dims, vec![2, 2]);
    // A * A^T where A is 2x3 and A^T is 3x2 = 2x2 result
    assert_eq!(result.data_ref(), vec![14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn test_dsl_mixed_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        # Mix old and new syntax
        DEFINE old AS TENSOR [2, 2] VALUES [1, 2, 3, 4]
        MATRIX new = [[5, 6], [7, 8]]
        
        LET result = MATMUL old new
        SHOW result
    "#;

    execute_script(&mut db, script).unwrap();

    let result = db.get("result").unwrap();
    assert_eq!(result.shape.dims, vec![2, 2]);
}

#[test]
fn test_dsl_vector_to_matrix_reshape() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4]
        LET M = RESHAPE v TO [2, 2]
        LET v2 = FLATTEN M
    "#;

    execute_script(&mut db, script).unwrap();

    let v2 = db.get("v2").unwrap();
    let v = db.get("v").unwrap();
    assert_eq!(v2.data, v.data);
}
