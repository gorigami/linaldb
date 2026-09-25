//! Phase 4 benchmark spike (`PERFORMANCE_OPTIMIZATION_PLAN.md`): measures
//! this engine's current hand-rolled, Rayon-parallelized dense matmul
//! kernel (`engine::kernels::matmul`) against `faer`'s dense GEMM, at
//! square-matrix sizes representative of real usage on this engine
//! (fits comfortably within its existing memory-limited execution model,
//! not ML-training scale). `faer` is a dev-dependency only at this point --
//! not wired into the engine -- this bench exists purely to produce a real
//! number to decide Phase 4's scope with, per design decision #1.
//!
//! `cpu_backend` measures `CpuBackend::matmul`, the path the DSL's `MATMUL`
//! actually takes. With the default `faer-matmul` feature it's faer; run with
//! `--no-default-features` to measure the hand-rolled SIMD kernel it used
//! before (see docs/SCALING_AND_GPU_ROADMAP.md).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linal::core::backend::{ComputeBackend, CpuBackend};
use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::context::ExecutionContext;
use linal::engine::kernels::matmul;

fn make_square_tensor(n: usize, seed: f32) -> Tensor {
    let data: Vec<f32> = (0..n * n)
        .map(|i| ((i as f32 + seed) * 0.001).sin())
        .collect();
    let id = TensorId::new();
    let meta = TensorMetadata::new(id, None);
    Tensor::new(id, Shape::new(vec![n, n]), data, meta).unwrap()
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

fn matmul_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_backend");
    group.sample_size(10);

    let cpu = CpuBackend::new();
    for &n in &[50usize, 200, 500, 1000, 2048] {
        let a = make_square_tensor(n, 1.0);
        let b = make_square_tensor(n, 2.0);

        let a_faer = faer::Mat::<f32>::from_fn(n, n, |i, j| a.data[i * n + j]);
        let b_faer = faer::Mat::<f32>::from_fn(n, n, |i, j| b.data[i * n + j]);

        group.bench_with_input(BenchmarkId::new("current_kernel", n), &n, |bench, _| {
            bench.iter(|| black_box(matmul(&a, &b, TensorId::new()).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("cpu_backend", n), &n, |bench, _| {
            let mut ctx = ExecutionContext::new();
            bench.iter(|| black_box(cpu.matmul(&mut ctx, &a, &b, TensorId::new()).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("faer", n), &n, |bench, _| {
            bench.iter(|| black_box(faer_matmul(&a_faer, &b_faer, n)));
        });
    }

    group.finish();
}

criterion_group!(benches, matmul_backend_comparison);
criterion_main!(benches);
