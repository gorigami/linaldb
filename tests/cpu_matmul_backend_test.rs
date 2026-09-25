// tests/cpu_matmul_backend_test.rs
//
// `CpuBackend::matmul` -- the path the DSL's `MATMUL` takes. With the
// default `faer-matmul` feature it runs faer's GEMM, and without it the
// hand-rolled SIMD/scalar kernels. Either way it must match a reference
// product, honor zero-copy views, report shape errors, and be deterministic
// run to run (ARCHITECTURE.md "Semantic Invariants": bit-deterministic on the
// same backend).

use linal::core::backend::{ComputeBackend, CpuBackend};
use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::dsl::{execute_line, DslOutput};
use linal::engine::context::ExecutionContext;
use linal::engine::TensorDb;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

fn tensor(dims: Vec<usize>, data: Vec<f32>) -> Tensor {
    let id = TensorId::new();
    Tensor::new(id, Shape::new(dims), data, TensorMetadata::new(id, None)).unwrap()
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
        }
    }
    c.into_iter().map(|v| v as f32).collect()
}

fn assert_close(expected: &[f32], got: &[f32], what: &str) {
    assert_eq!(expected.len(), got.len(), "{what}: length");
    for (i, (e, g)) in expected.iter().zip(got).enumerate() {
        let tol = 1e-4 * e.abs().max(1.0);
        assert!((e - g).abs() <= tol, "{what}[{i}]: expected {e}, got {g}");
    }
}

#[test]
fn matches_reference_across_shapes() {
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();
    // Tiny (scalar path without faer), odd, rectangular, and large (SIMD
    // path without faer) shapes.
    for &(m, k, n) in &[
        (1, 1, 1),
        (2, 3, 4),
        (7, 13, 5),
        (33, 64, 17),
        (128, 96, 200),
    ] {
        let a = values(m * k, 1.0);
        let b = values(k * n, 2.0);
        let c = cpu
            .matmul(
                &mut ctx,
                &tensor(vec![m, k], a.clone()),
                &tensor(vec![k, n], b.clone()),
                TensorId::new(),
            )
            .unwrap();
        assert_eq!(c.shape.dims, vec![m, n]);
        assert_close(
            &reference(&a, &b, m, k, n),
            &c.to_logical_vec(),
            &format!("{m}x{k}x{n}"),
        );
    }
}

#[test]
fn honors_transposed_views() {
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();
    let n = 64;
    let a = tensor(vec![n, n], values(n * n, 3.0));
    let b = tensor(vec![n, n], values(n * n, 4.0));
    let at = cpu.transpose(&mut ctx, &a, TensorId::new()).unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&at.data, &a.data),
        "transpose is a view"
    );

    let got = cpu.matmul(&mut ctx, &at, &b, TensorId::new()).unwrap();
    let expected = reference(&at.to_logical_vec(), &b.to_logical_vec(), n, n, n);
    assert_close(&expected, &got.to_logical_vec(), "Aᵀ·B");
}

#[test]
fn reports_shape_errors() {
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();
    let a = tensor(vec![2, 3], values(6, 1.0));
    let v = tensor(vec![6], values(6, 1.0));
    assert!(cpu.matmul(&mut ctx, &a, &a, TensorId::new()).is_err());
    assert!(cpu.matmul(&mut ctx, &a, &v, TensorId::new()).is_err());
}

#[test]
fn is_bit_deterministic_run_to_run() {
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();
    // Large enough that the multithreaded path is used.
    let n = 512;
    let a = tensor(vec![n, n], values(n * n, 5.0));
    let b = tensor(vec![n, n], values(n * n, 6.0));
    let first = cpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap();
    let first_bits: Vec<u32> = first.data.iter().map(|v| v.to_bits()).collect();
    for run in 1..8 {
        let again = cpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap();
        let bits: Vec<u32> = again.data.iter().map(|v| v.to_bits()).collect();
        assert!(bits == first_bits, "run {run} differs bitwise from run 0");
    }
}

#[test]
fn dsl_matmul_uses_the_backend_and_is_correct() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = linal::core::config::EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db = TensorDb::with_config(config);

    let n = 40;
    let lit = |v: &[f32]| {
        let rows: Vec<String> = v
            .chunks(n)
            .map(|r| {
                format!(
                    "[{}]",
                    r.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();
        format!("[{}]", rows.join(", "))
    };
    let a = values(n * n, 7.0);
    let b = values(n * n, 8.0);
    execute_line(&mut db, &format!("MATRIX a = {}", lit(&a)), 1).unwrap();
    execute_line(&mut db, &format!("MATRIX b = {}", lit(&b)), 1).unwrap();
    execute_line(&mut db, "LET c = MATMUL a b", 1).unwrap();
    let DslOutput::Tensor(c) = execute_line(&mut db, "SHOW c", 1).unwrap() else {
        panic!("expected a tensor")
    };
    // `MATRIX` literals round-trip through the DSL's own float parsing, so
    // compare against the stored inputs rather than the generator.
    let a_stored = db.get("a").unwrap().to_logical_vec();
    let b_stored = db.get("b").unwrap().to_logical_vec();
    assert_close(
        &reference(&a_stored, &b_stored, n, n, n),
        &c.to_logical_vec(),
        "DSL MATMUL",
    );
}

#[test]
fn honors_offset_and_submatrix_views() {
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();
    // A 6x8 buffer the views below borrow from.
    let (rows, cols) = (6, 8);
    let buf = std::sync::Arc::new(values(rows * cols, 9.0));
    let view = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
        let id = TensorId::new();
        Tensor::from_shared_strided(
            id,
            Shape::new(dims),
            buf.clone(),
            std::sync::Arc::new(TensorMetadata::new(id, None)),
            strides,
            offset,
        )
        .unwrap()
    };
    let b = tensor(vec![8, 5], values(40, 10.0));
    let c = tensor(vec![3, 4], values(12, 11.0));

    // Rows 2..5 of the buffer: row-major with an offset (the zero-copy path).
    let rows_view = view(vec![3, 8], vec![cols, 1], 2 * cols);
    let got = cpu
        .matmul(&mut ctx, &rows_view, &b, TensorId::new())
        .unwrap();
    let expected = reference(&rows_view.to_logical_vec(), &b.to_logical_vec(), 3, 8, 5);
    assert_close(&expected, &got.to_logical_vec(), "row-offset view");

    // Rows 1..3 x cols 2..6: a genuine submatrix (the copying path).
    let sub = view(vec![2, 3], vec![cols, 1], cols + 2);
    let got = cpu.matmul(&mut ctx, &sub, &c, TensorId::new()).unwrap();
    let expected = reference(&sub.to_logical_vec(), &c.to_logical_vec(), 2, 3, 4);
    assert_close(&expected, &got.to_logical_vec(), "submatrix view");
}
