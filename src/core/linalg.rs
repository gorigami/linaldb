//! Classical linear algebra (trace/determinant/rank/inverse/solve/
//! eigendecomposition/decompositions) built on `nalgebra` -- see
//! `LINEAGE_AND_LINALG_PLAN.md` Phase 8 for the full operator list and the
//! "why nalgebra, not LAPACK/BLAS" rationale (pure Rust, matches the
//! `realfft`/`rustfft` precedent this codebase already has for numerical
//! crates over binding to a system library).
//!
//! Deliberately kept separate from `engine/kernels.rs` (existing real-space
//! elementwise/reduction tensor math) and `core::signal` (frequency-domain),
//! since this is a third, distinct numerical domain built on a third crate.
//!
//! **Precision**: every operation here promotes the `Tensor`'s native `f32`
//! data to `f64` for the actual `nalgebra` computation, then narrows the
//! result back to `f32` -- classical linear algebra (determinants,
//! eigenvalues, LU pivoting) accumulates error fast enough in `f32` that
//! computing in `f64` throughout meaningfully improves accuracy even though
//! the engine's `Tensor` storage stays `f32`-only by design.
//!
//! **Error philosophy**: every fallible operation here returns a real
//! `Err(String)` on a singular/near-singular/non-square/non-symmetric
//! input -- never a silent `NaN`/`Inf` in the output. This is a locked
//! design decision for Phase 8 (`INVERSE`/`SOLVE`'s "loud-error-on-singular"
//! philosophy), not an incidental choice.

use crate::core::tensor::{Shape, Tensor};
use nalgebra::DMatrix;

/// A matrix's flat row-major `f32` data plus its `Shape` -- the return
/// shape every decomposition output here takes, ready for `Tensor::new`.
pub type MatrixData = (Vec<f32>, Shape);

/// Converts a rank-2 `Tensor` into an `f64` `nalgebra` matrix. Errors on
/// anything that isn't a genuine matrix (rank 0/1, or rank 3+) -- every
/// operator in this module needs a real 2D matrix, not a vector or scalar.
fn tensor_to_matrix(tensor: &Tensor) -> Result<DMatrix<f64>, String> {
    if tensor.shape.rank() != 2 {
        return Err(format!(
            "expected a rank-2 Matrix, got rank {} (shape {:?})",
            tensor.shape.rank(),
            tensor.shape.dims
        ));
    }
    let rows = tensor.shape.dims[0];
    let cols = tensor.shape.dims[1];
    let data = tensor.to_logical_vec();
    let data_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    // `to_logical_vec` is row-major; `from_row_slice` expects the same.
    Ok(DMatrix::from_row_slice(rows, cols, &data_f64))
}

/// Converts an `f64` `nalgebra` matrix back into row-major `f32` tensor data
/// plus its `Shape`, ready for `Tensor::new`.
fn matrix_to_tensor_data(m: &DMatrix<f64>) -> (Vec<f32>, Shape) {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(m[(i, j)] as f32);
        }
    }
    (data, Shape::new(vec![rows, cols]))
}

fn require_square(m: &DMatrix<f64>, op: &str) -> Result<(), String> {
    if m.nrows() != m.ncols() {
        return Err(format!(
            "{op} requires a square matrix, got {}x{}",
            m.nrows(),
            m.ncols()
        ));
    }
    Ok(())
}

/// `TRACE a` -- sum of the diagonal. Requires a square matrix.
pub fn trace(tensor: &Tensor) -> Result<f32, String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "TRACE")?;
    Ok(m.trace() as f32)
}

/// `DETERMINANT a`. Requires a square matrix. `0.0` is a legitimate
/// (singular-matrix) result here, not an error -- unlike `INVERSE`/`SOLVE`,
/// there's nothing unsafe about reporting a real zero determinant.
pub fn determinant(tensor: &Tensor) -> Result<f32, String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "DETERMINANT")?;
    Ok(m.determinant() as f32)
}

/// Numerical tolerance for rank/singularity/symmetry checks, scaled by the
/// matrix's own largest singular value / magnitude where relevant --
/// an absolute epsilon would be meaningless across the wildly different
/// magnitudes real data brings (LIGO-strain-sized values vs. pixel counts).
const REL_EPSILON: f64 = 1e-10;

/// `RANK a` -- numerical rank via singular value decomposition (the
/// standard, numerically stable way to compute rank; row-reduction is not
/// used because it's not numerically stable for floating-point input).
/// Returned as `f32` like every other scalar reduction (`SUM`/`MEAN`/...) --
/// no new scalar `Value` variant needed for an integer-valued result.
pub fn rank(tensor: &Tensor) -> Result<f32, String> {
    let m = tensor_to_matrix(tensor)?;
    let svd = m.clone().svd(false, false);
    let max_singular = svd.singular_values.iter().cloned().fold(0.0, f64::max);
    let eps = REL_EPSILON * max_singular.max(1.0);
    let r = svd.singular_values.iter().filter(|&&s| s > eps).count();
    Ok(r as f32)
}

/// `INVERSE a`. Requires a square matrix. Errors loudly (never returns a
/// silent `NaN`-filled result) if the matrix is singular or numerically too
/// close to singular to invert reliably -- detected via `nalgebra`'s own
/// `try_inverse`, which internally uses LU decomposition with partial
/// pivoting and reports failure rather than dividing by a near-zero pivot.
pub fn inverse(tensor: &Tensor) -> Result<MatrixData, String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "INVERSE")?;
    let inv = m
        .try_inverse()
        .ok_or_else(|| "INVERSE: matrix is singular (not invertible)".to_string())?;
    Ok(matrix_to_tensor_data(&inv))
}

/// `SOLVE a b` -- solves `Ax = b` for `x` via LU decomposition with partial
/// pivoting. `a` must be square; `b` a vector (rank-1) with length matching
/// `a`'s row count. Errors loudly on a singular `a`, same philosophy as
/// `INVERSE` -- never a silent `NaN` vector.
pub fn solve(a_tensor: &Tensor, b_tensor: &Tensor) -> Result<Vec<f32>, String> {
    let a = tensor_to_matrix(a_tensor)?;
    require_square(&a, "SOLVE")?;

    if b_tensor.shape.rank() != 1 {
        return Err(format!(
            "SOLVE: b must be a rank-1 Vector, got rank {} (shape {:?})",
            b_tensor.shape.rank(),
            b_tensor.shape.dims
        ));
    }
    if b_tensor.shape.dims[0] != a.nrows() {
        return Err(format!(
            "SOLVE: b has length {} but a is {}x{} -- lengths must match",
            b_tensor.shape.dims[0],
            a.nrows(),
            a.ncols()
        ));
    }

    let b_data: Vec<f64> = b_tensor
        .to_logical_vec()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let b = nalgebra::DVector::from_vec(b_data);

    let lu = a.lu();
    let x = lu
        .solve(&b)
        .ok_or_else(|| "SOLVE: matrix `a` is singular -- no unique solution".to_string())?;
    Ok(x.iter().map(|&v| v as f32).collect())
}

/// `EIGENVALUES a` -- real eigenvalues of a **symmetric** matrix only
/// (guarantees real eigenvalues, no complex-number `Value`/`ValueType`
/// support needed -- see `LINEAGE_AND_LINALG_PLAN.md` Phase 8.3). Errors if
/// the matrix isn't square or isn't symmetric within `REL_EPSILON`
/// (relative to its largest entry) rather than silently treating it as
/// symmetric and returning a wrong answer.
pub fn eigenvalues_symmetric(tensor: &Tensor) -> Result<Vec<f32>, String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "EIGENVALUES")?;
    require_symmetric(&m, "EIGENVALUES")?;
    let eigen = nalgebra::linalg::SymmetricEigen::new(m);
    Ok(eigen.eigenvalues.iter().map(|&v| v as f32).collect())
}

fn require_symmetric(m: &DMatrix<f64>, op: &str) -> Result<(), String> {
    let max_abs = m.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let eps = REL_EPSILON * max_abs.max(1.0);
    for i in 0..m.nrows() {
        for j in (i + 1)..m.ncols() {
            if (m[(i, j)] - m[(j, i)]).abs() > eps {
                return Err(format!(
                    "{op}: matrix is not symmetric (entries [{i}][{j}]={} vs [{j}][{i}]={} differ) -- only symmetric matrices are supported today",
                    m[(i, j)],
                    m[(j, i)]
                ));
            }
        }
    }
    Ok(())
}

/// `QR a` -- QR decomposition (`a = Q * R`), any rectangular matrix.
/// Returns `(q_data, q_shape, r_data, r_shape)`.
pub fn qr(tensor: &Tensor) -> Result<(MatrixData, MatrixData), String> {
    let m = tensor_to_matrix(tensor)?;
    let decomp = m.qr();
    let q = decomp.q();
    let r = decomp.r();
    Ok((matrix_to_tensor_data(&q), matrix_to_tensor_data(&r)))
}

/// `LU a` -- LU decomposition with partial pivoting (`P * a = L * U`).
/// Requires a square matrix. Returns `(p_data, p_shape, l_data, l_shape,
/// u_data, u_shape)` -- `P` is included (not just `L`/`U`) specifically so
/// `P @ a == L @ U` actually holds for a caller that checks it; dropping
/// the permutation (a tempting simplification) would silently make that
/// property false for any matrix that needs row pivoting.
pub fn lu(tensor: &Tensor) -> Result<(MatrixData, MatrixData, MatrixData), String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "LU")?;
    let n = m.nrows();
    let decomp = m.lu();
    let l = decomp.l();
    let u = decomp.u();
    let mut p_mat = DMatrix::<f64>::identity(n, n);
    decomp.p().permute_rows(&mut p_mat);
    Ok((
        matrix_to_tensor_data(&p_mat),
        matrix_to_tensor_data(&l),
        matrix_to_tensor_data(&u),
    ))
}

/// `CHOLESKY a` -- Cholesky decomposition (`a = L * L^T`) of a symmetric
/// **positive-definite** matrix. Single output (`L`), unlike `LU`/`QR`/
/// `SVD`/`EIGEN` -- there's only one matrix to return, so this doesn't need
/// Phase 8.4's multi-output `LET` binding. Errors loudly (never a silent
/// `NaN`-filled result) if `a` isn't symmetric positive-definite, same
/// philosophy as `INVERSE`/`SOLVE`.
pub fn cholesky(tensor: &Tensor) -> Result<MatrixData, String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "CHOLESKY")?;
    require_symmetric(&m, "CHOLESKY")?;
    let decomp = m
        .cholesky()
        .ok_or_else(|| "CHOLESKY: matrix is not positive-definite".to_string())?;
    Ok(matrix_to_tensor_data(&decomp.l()))
}

/// `EIGEN a` -- full eigendecomposition (eigenvalues + eigenvectors) of a
/// **symmetric** matrix. Deliberately narrower than the plan's original
/// "general case" wording: a truly general (non-symmetric) eigendecomposition
/// can have complex eigenvalues/eigenvectors, and this engine has no
/// `Value`/`ValueType::Complex` (a locked constraint from `EIGENVALUES`,
/// Phase 8.3, that a "general case" here would silently contradict) --
/// so this stays symmetric-only, consistent with `EIGENVALUES`, and returns
/// real eigenvectors as columns of the second output matrix.
pub fn eigen_symmetric(tensor: &Tensor) -> Result<(Vec<f32>, MatrixData), String> {
    let m = tensor_to_matrix(tensor)?;
    require_square(&m, "EIGEN")?;
    require_symmetric(&m, "EIGEN")?;
    let eigen = nalgebra::linalg::SymmetricEigen::new(m);
    let values: Vec<f32> = eigen.eigenvalues.iter().map(|&v| v as f32).collect();
    Ok((values, matrix_to_tensor_data(&eigen.eigenvectors)))
}

/// `QR a` as a uniform `Vec` of (data, shape) pairs, in bind order (`q`,
/// `r`) -- the shape `engine/db.rs`'s generic multi-output eval helper
/// wants, so it can stay generic over which decomposition it's wrapping.
pub fn qr_outputs(tensor: &Tensor) -> Result<Vec<MatrixData>, String> {
    let (q, r) = qr(tensor)?;
    Ok(vec![q, r])
}

/// `LU a` as a uniform `Vec`, in bind order (`p`, `l`, `u`).
pub fn lu_outputs(tensor: &Tensor) -> Result<Vec<MatrixData>, String> {
    let (p, l, u) = lu(tensor)?;
    Ok(vec![p, l, u])
}

/// `EIGEN a` as a uniform `Vec`, in bind order (eigenvalues, eigenvectors).
pub fn eigen_outputs(tensor: &Tensor) -> Result<Vec<MatrixData>, String> {
    let (values, vectors) = eigen_symmetric(tensor)?;
    let n = values.len();
    Ok(vec![(values, Shape::new(vec![n])), vectors])
}

/// `SVD a` as a uniform `Vec`, in bind order (`u`, `s`, `vt`).
pub fn svd_outputs(tensor: &Tensor) -> Result<Vec<MatrixData>, String> {
    let (u, s, v_t) = svd(tensor)?;
    let n = s.len();
    Ok(vec![u, (s, Shape::new(vec![n])), v_t])
}

/// `SVD a` -- singular value decomposition (`a = U * diag(s) * Vt`), any
/// rectangular matrix. Returns `(u_data, u_shape, s, (vt_data, vt_shape))`.
pub fn svd(tensor: &Tensor) -> Result<(MatrixData, Vec<f32>, MatrixData), String> {
    let m = tensor_to_matrix(tensor)?;
    let decomp = m.svd(true, true);
    let u = decomp
        .u
        .ok_or_else(|| "SVD: failed to compute U (unexpected)".to_string())?;
    let v_t = decomp
        .v_t
        .ok_or_else(|| "SVD: failed to compute V^T (unexpected)".to_string())?;
    let s: Vec<f32> = decomp.singular_values.iter().map(|&v| v as f32).collect();
    Ok((matrix_to_tensor_data(&u), s, matrix_to_tensor_data(&v_t)))
}

/// `PCA a COMPONENTS k` -- projects `a`'s rows (samples) onto their top-`k`
/// principal components: mean-center each column, then keep the first `k`
/// columns of `U * diag(s)` from `a`'s SVD (equivalently, the projection
/// onto the top-`k` right singular vectors). Single output (the projected
/// data), built directly on `svd` above -- the capstone Phase 8.6 operator.
pub fn pca(tensor: &Tensor, k: usize) -> Result<MatrixData, String> {
    let m = tensor_to_matrix(tensor)?;
    let (rows, cols) = (m.nrows(), m.ncols());
    if k == 0 || k > cols {
        return Err(format!(
            "PCA: components ({k}) must be between 1 and the input's column count ({cols})"
        ));
    }

    // Mean-center each column (each feature) -- PCA is defined on centered data.
    let mut centered = m.clone();
    for j in 0..cols {
        let mean: f64 = centered.column(j).iter().sum::<f64>() / rows as f64;
        for i in 0..rows {
            centered[(i, j)] -= mean;
        }
    }

    let decomp = centered.svd(true, true);
    let u = decomp
        .u
        .ok_or_else(|| "PCA: failed to compute U (unexpected)".to_string())?;
    let s = decomp.singular_values;

    // Projection onto the top-k components: (U * diag(s))'s first k columns.
    let mut projected = DMatrix::<f64>::zeros(rows, k);
    for j in 0..k {
        let scale = s[j];
        for i in 0..rows {
            projected[(i, j)] = u[(i, j)] * scale;
        }
    }
    Ok(matrix_to_tensor_data(&projected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{Shape, TensorId, TensorMetadata};

    fn matrix_tensor(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
        let id = TensorId::new();
        let meta = TensorMetadata::new(id, None);
        Tensor::new(id, Shape::new(vec![rows, cols]), data, meta).unwrap()
    }

    fn vector_tensor(data: Vec<f32>) -> Tensor {
        let id = TensorId::new();
        let meta = TensorMetadata::new(id, None);
        let n = data.len();
        Tensor::new(id, Shape::new(vec![n]), data, meta).unwrap()
    }

    #[test]
    fn trace_of_identity_is_n() {
        let t = matrix_tensor(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(trace(&t).unwrap(), 3.0);
    }

    #[test]
    fn trace_requires_square() {
        let t = matrix_tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(trace(&t).is_err());
    }

    #[test]
    fn determinant_2x2_hand_computable() {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        let t = matrix_tensor(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!((determinant(&t).unwrap() - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn determinant_of_singular_matrix_is_zero_not_an_error() {
        let t = matrix_tensor(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        assert!(determinant(&t).unwrap().abs() < 1e-4);
    }

    #[test]
    fn rank_of_full_rank_identity() {
        let t = matrix_tensor(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(rank(&t).unwrap(), 3.0);
    }

    #[test]
    fn rank_of_rank_deficient_matrix() {
        // row 2 = 2 * row 1 -> rank 1
        let t = matrix_tensor(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        assert_eq!(rank(&t).unwrap(), 1.0);
    }

    #[test]
    fn inverse_round_trips_to_identity() {
        let t = matrix_tensor(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let (data, shape) = inverse(&t).unwrap();
        let inv = matrix_tensor(shape.dims[0], shape.dims[1], data);
        // A * A^-1 ~= I
        let a = tensor_to_matrix(&t).unwrap();
        let a_inv = tensor_to_matrix(&inv).unwrap();
        let product = a * a_inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn inverse_of_singular_matrix_errors_loudly() {
        let t = matrix_tensor(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let err = inverse(&t).unwrap_err();
        assert!(err.contains("singular"));
    }

    #[test]
    fn solve_hand_computable_system() {
        // [[2,0],[0,2]] x = [4, 6] -> x = [2, 3]
        let a = matrix_tensor(2, 2, vec![2.0, 0.0, 0.0, 2.0]);
        let b = vector_tensor(vec![4.0, 6.0]);
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-4);
        assert!((x[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn solve_of_singular_system_errors_loudly() {
        let a = matrix_tensor(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let b = vector_tensor(vec![1.0, 2.0]);
        assert!(solve(&a, &b).is_err());
    }

    #[test]
    fn eigenvalues_of_diagonal_matrix_are_the_diagonal() {
        let t = matrix_tensor(2, 2, vec![5.0, 0.0, 0.0, 3.0]);
        let mut eigs = eigenvalues_symmetric(&t).unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 3.0).abs() < 1e-4);
        assert!((eigs[1] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn eigenvalues_rejects_non_symmetric_matrix() {
        let t = matrix_tensor(2, 2, vec![1.0, 2.0, 0.0, 1.0]);
        assert!(eigenvalues_symmetric(&t).is_err());
    }

    fn matrix_from(shape: &Shape, data: &[f32]) -> DMatrix<f64> {
        DMatrix::from_row_slice(
            shape.dims[0],
            shape.dims[1],
            &data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        )
    }

    fn assert_matrices_close(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() < tol,
                    "mismatch at [{i}][{j}]: {} vs {}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn qr_reconstructs_a() {
        let a_t = matrix_tensor(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (q, r) = qr(&a_t).unwrap();
        let q_m = matrix_from(&q.1, &q.0);
        let r_m = matrix_from(&r.1, &r.0);
        let a_m = tensor_to_matrix(&a_t).unwrap();
        assert_matrices_close(&(q_m * r_m), &a_m, 1e-3);
    }

    #[test]
    fn lu_reconstructs_pa_as_lu() {
        // A matrix that needs row pivoting under partial-pivoting LU.
        let a_t = matrix_tensor(3, 3, vec![0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0]);
        let (p, l, u) = lu(&a_t).unwrap();
        let p_m = matrix_from(&p.1, &p.0);
        let l_m = matrix_from(&l.1, &l.0);
        let u_m = matrix_from(&u.1, &u.0);
        let a_m = tensor_to_matrix(&a_t).unwrap();
        assert_matrices_close(&(p_m * a_m), &(l_m * u_m), 1e-3);
    }

    #[test]
    fn lu_requires_square() {
        let t = matrix_tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(lu(&t).is_err());
    }

    #[test]
    fn cholesky_reconstructs_a_as_l_lt() {
        // Symmetric positive-definite: [[4,2],[2,3]]
        let a_t = matrix_tensor(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let (data, shape) = cholesky(&a_t).unwrap();
        let l_m = matrix_from(&shape, &data);
        let a_m = tensor_to_matrix(&a_t).unwrap();
        assert_matrices_close(&(l_m.clone() * l_m.transpose()), &a_m, 1e-3);
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let a_t = matrix_tensor(2, 2, vec![1.0, 2.0, 2.0, 1.0]); // not PSD
        assert!(cholesky(&a_t).is_err());
    }

    #[test]
    fn eigen_symmetric_reconstructs_a_as_v_diag_vt() {
        let a_t = matrix_tensor(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (values, (vec_data, vec_shape)) = eigen_symmetric(&a_t).unwrap();
        let v = matrix_from(&vec_shape, &vec_data);
        let diag = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(
            values.iter().map(|&x| x as f64).collect(),
        ));
        let a_m = tensor_to_matrix(&a_t).unwrap();
        assert_matrices_close(&(v.clone() * diag * v.transpose()), &a_m, 1e-3);
    }

    #[test]
    fn svd_reconstructs_a_as_u_diag_s_vt() {
        let a_t = matrix_tensor(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let (u, s, v_t) = svd(&a_t).unwrap();
        let u_m = matrix_from(&u.1, &u.0);
        let vt_m = matrix_from(&v_t.1, &v_t.0);
        let mut diag = DMatrix::<f64>::zeros(u_m.ncols(), vt_m.nrows());
        for (i, &sv) in s.iter().enumerate() {
            diag[(i, i)] = sv as f64;
        }
        let a_m = tensor_to_matrix(&a_t).unwrap();
        assert_matrices_close(&(u_m * diag * vt_m), &a_m, 1e-3);
    }

    #[test]
    fn pca_projects_to_requested_component_count() {
        // 4 samples, 3 features -> project to 2 components.
        let a_t = matrix_tensor(
            4,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0, 1.0],
        );
        let (data, shape) = pca(&a_t, 2).unwrap();
        assert_eq!(shape.dims, vec![4, 2]);
        assert_eq!(data.len(), 8);
    }

    #[test]
    fn pca_rejects_too_many_components() {
        let a_t = matrix_tensor(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(pca(&a_t, 5).is_err());
    }
}
