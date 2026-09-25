//! Track C benchmark spike (`docs/SCALING_AND_GPU_ROADMAP.md`): does the `gpu-wgpu`
//! backend pay off at this engine's realistic sizes? Every GPU number
//! includes the per-call host->device upload and device->host readback,
//! since `Tensor.data` stays in host memory in this spike.
//!
//! - `matmul`: the CPU backend (`CpuBackend::matmul`, the path the DSL's
//!   `MATMUL` actually takes), `faer` directly, and `GpuBackend::matmul`.
//! - `batch_cosine`: scoring N vectors against one query, the shape of an
//!   exact brute-force vector search -- a parallel CPU scan (Rayon) vs
//!   `GpuContext::batch_cosine`.
//!
//! Run with `cargo bench --features gpu-wgpu --bench gpu_backend`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linal::core::backend::gpu::{GpuBackend, GpuContext};
use linal::core::backend::{ComputeBackend, CpuBackend};
use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::context::ExecutionContext;
use rayon::prelude::*;
use std::time::Duration;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.001).sin()).collect()
}

fn square(n: usize, seed: f32) -> Tensor {
    let id = TensorId::new();
    Tensor::new(
        id,
        Shape::new(vec![n, n]),
        values(n * n, seed),
        TensorMetadata::new(id, None),
    )
    .unwrap()
}

fn faer_matmul(a: &faer::Mat<f32>, b: &faer::Mat<f32>, n: usize) -> faer::Mat<f32> {
    let mut dst = faer::Mat::<f32>::zeros(n, n);
    faer::linalg::matmul::matmul(
        &mut dst,
        faer::Accum::Replace,
        a,
        b,
        1.0f32,
        faer::Par::rayon(0),
    );
    dst
}

fn cpu_batch_cosine(q: &[f32], rows: &[f32], dim: usize) -> Vec<f32> {
    let norm_q = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    rows.par_chunks(dim)
        .map(|row| {
            let (mut dot, mut norm) = (0.0f32, 0.0f32);
            for (x, y) in row.iter().zip(q) {
                dot += x * y;
                norm += x * x;
            }
            dot / (norm.sqrt() * norm_q)
        })
        .collect()
}

fn matmul(c: &mut Criterion) {
    let gpu = GpuBackend::new().expect("no GPU adapter");
    let cpu = CpuBackend::new();
    let mut group = c.benchmark_group("matmul");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));

    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let a = square(n, 1.0);
        let b = square(n, 2.0);
        let a_faer = faer::Mat::<f32>::from_fn(n, n, |i, j| a.data[i * n + j]);
        let b_faer = faer::Mat::<f32>::from_fn(n, n, |i, j| b.data[i * n + j]);

        group.bench_with_input(BenchmarkId::new("cpu_backend", n), &n, |bench, _| {
            let mut ctx = ExecutionContext::new();
            bench.iter(|| black_box(cpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("faer", n), &n, |bench, _| {
            bench.iter(|| black_box(faer_matmul(&a_faer, &b_faer, n)));
        });
        group.bench_with_input(BenchmarkId::new("gpu_backend", n), &n, |bench, _| {
            let mut ctx = ExecutionContext::new();
            bench.iter(|| black_box(gpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap()));
        });
    }
    group.finish();
}

fn batch_cosine(c: &mut Criterion) {
    let gpu = GpuContext::shared().expect("no GPU adapter");
    let mut group = c.benchmark_group("batch_cosine");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));

    for &dim in &[384usize, 768] {
        for &rows in &[10_000usize, 100_000, 1_000_000] {
            let x = values(rows * dim, 3.0);
            let q = values(dim, 4.0);
            let id = format!("{rows}x{dim}");
            group.bench_with_input(BenchmarkId::new("cpu_rayon", &id), &rows, |bench, _| {
                bench.iter(|| black_box(cpu_batch_cosine(&q, &x, dim)));
            });
            group.bench_with_input(BenchmarkId::new("gpu", &id), &rows, |bench, _| {
                bench.iter(|| black_box(gpu.batch_cosine(&q, &x, dim).unwrap()));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, matmul, batch_cosine);
criterion_main!(benches);
