// tests/gpu_backend_test.rs
//
// CPU <-> GPU parity for the `gpu-wgpu` backend (SCALING_AND_GPU_PLAN.md
// Track C). Only built with `--features gpu-wgpu`; every test skips cleanly
// (with a note on stderr) on a machine with no usable GPU adapter.
#![cfg(feature = "gpu-wgpu")]

use linal::core::backend::gpu::{GpuBackend, GpuContext, MATMUL_GPU_MIN_FLOPS};
use linal::core::backend::{ComputeBackend, CpuBackend};
use linal::core::config::{ComputeBackendKind, EngineConfig};
use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::dsl::{execute_line, DslOutput};
use linal::engine::context::ExecutionContext;
use linal::engine::TensorDb;
use std::sync::Arc;

fn gpu() -> Option<Arc<GpuContext>> {
    match GpuContext::shared() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: {}", e);
            None
        }
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tensor(dims: Vec<usize>, data: Vec<f32>) -> Tensor {
    let id = TensorId::new();
    Tensor::new(id, Shape::new(dims), data, TensorMetadata::new(id, None)).unwrap()
}

fn assert_close(cpu: &[f32], gpu: &[f32], what: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{what}: length");
    for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
        let tol = 1e-4 * c.abs().max(1.0);
        assert!((c - g).abs() <= tol, "{what}[{i}]: cpu {c} vs gpu {g}");
    }
}

#[test]
fn matmul_matches_cpu_across_shapes() {
    let Some(ctx) = gpu() else { return };
    for &(m, k, n) in &[
        (1, 1, 1),
        (3, 5, 2),
        (16, 16, 16),
        (17, 33, 9),
        (64, 300, 129),
    ] {
        let a = values(m * k, 1);
        let b = values(k * n, 2);
        let mut cpu = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                cpu[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
        let gpu = ctx.matmul(&a, &b, m, k, n).unwrap();
        assert_close(&cpu, &gpu, &format!("matmul {m}x{k}x{n}"));
    }
}

#[test]
fn backend_matmul_matches_cpu_including_views() {
    let Some(_) = gpu() else { return };
    let backend = GpuBackend::new().unwrap();
    let cpu = CpuBackend::new();
    let mut ctx = ExecutionContext::new();

    // Big enough to take the GPU path.
    let n = 160;
    assert!(n * n * n >= MATMUL_GPU_MIN_FLOPS);
    let a = tensor(vec![n, n], values(n * n, 3));
    let b = tensor(vec![n, n], values(n * n, 4));
    let expected = cpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap();
    let got = backend.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap();
    assert_eq!(got.shape.dims, vec![n, n]);
    assert_close(&expected.to_logical_vec(), &got.to_logical_vec(), "backend");

    // A zero-copy transposed view must multiply by its logical values.
    let at = cpu.transpose(&mut ctx, &a, TensorId::new()).unwrap();
    let expected = cpu.matmul(&mut ctx, &at, &b, TensorId::new()).unwrap();
    let got = backend.matmul(&mut ctx, &at, &b, TensorId::new()).unwrap();
    assert_close(
        &expected.to_logical_vec(),
        &got.to_logical_vec(),
        "transposed",
    );

    // Shape errors still come from the CPU path.
    let bad = tensor(vec![3, 4], values(12, 5));
    assert!(backend
        .matmul(&mut ctx, &bad, &bad, TensorId::new())
        .is_err());
}

#[test]
fn batch_cosine_matches_cpu_and_flags_zero_norms() {
    let Some(ctx) = gpu() else { return };
    let (rows, dim) = (1000, 384);
    let mut x = values(rows * dim, 6);
    x[..dim].fill(0.0); // row 0 has zero norm
    let q = values(dim, 7);
    let got = ctx.batch_cosine(&q, &x, dim).unwrap();
    assert_eq!(got.len(), rows);
    assert!(got[0].is_nan());
    let norm_q = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    let expected: Vec<f32> = (1..rows)
        .map(|r| {
            let row = &x[r * dim..(r + 1) * dim];
            let dot: f32 = row.iter().zip(&q).map(|(a, b)| a * b).sum();
            dot / (row.iter().map(|v| v * v).sum::<f32>().sqrt() * norm_q)
        })
        .collect();
    assert_close(&expected, &got[1..], "cosine");
}

#[test]
fn gpu_backend_is_selectable_from_config_and_reported() {
    let Some(_) = gpu() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    config.compute.backend = ComputeBackendKind::Gpu;
    let mut db = TensorDb::with_config(config);

    let DslOutput::Message(msg) = execute_line(&mut db, "SHOW BACKEND", 1).unwrap() else {
        panic!()
    };
    assert!(msg.contains("Gpu (wgpu"), "{msg}");

    // The DSL's MATMUL goes through the backend.
    let n = 160;
    let rows = |seed| {
        let v = values(n * n, seed);
        let body: Vec<String> = v
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
        (v, format!("[{}]", body.join(", ")))
    };
    let (a, a_lit) = rows(8);
    let (b, b_lit) = rows(9);
    execute_line(&mut db, &format!("MATRIX a = {a_lit}"), 1).unwrap();
    execute_line(&mut db, &format!("MATRIX b = {b_lit}"), 1).unwrap();
    execute_line(&mut db, "LET c = MATMUL a b", 1).unwrap();
    let DslOutput::Tensor(c) = execute_line(&mut db, "SHOW c", 1).unwrap() else {
        panic!()
    };
    let mut expected = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            expected[i * n + j] = (0..n).map(|p| a[i * n + p] * b[p * n + j]).sum();
        }
    }
    assert_close(&expected, &c.to_logical_vec(), "DSL MATMUL");
}
