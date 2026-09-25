use super::{ComputeBackend, ScalarBackend, SimdBackend};
use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;

#[derive(Debug)]
pub struct CpuBackend {
    scalar: ScalarBackend,
    simd: SimdBackend,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

const SIMD_THRESHOLD: usize = 1024;

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            scalar: ScalarBackend::new(),
            simd: SimdBackend::new(),
        }
    }

    fn use_simd(&self, len: usize) -> bool {
        // Use SIMD only for tensors larger than the threshold
        len >= SIMD_THRESHOLD
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        #[cfg(feature = "faer-matmul")]
        {
            "Cpu (Hybrid Scalar/SIMD; matmul via faer)"
        }
        #[cfg(not(feature = "faer-matmul"))]
        {
            "Cpu (Hybrid Scalar/SIMD)"
        }
    }

    fn add(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if self.use_simd(a.len()) {
            self.simd.add(ctx, a, b, new_id)
        } else {
            self.scalar.add(ctx, a, b, new_id)
        }
    }

    fn sub(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if self.use_simd(a.len()) {
            self.simd.sub(ctx, a, b, new_id)
        } else {
            self.scalar.sub(ctx, a, b, new_id)
        }
    }

    fn multiply(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if self.use_simd(a.len()) {
            self.simd.multiply(ctx, a, b, new_id)
        } else {
            self.scalar.multiply(ctx, a, b, new_id)
        }
    }

    fn divide(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        // Division is often slower in SIMD or needs careful handling, fallback to scalar for now
        self.scalar.divide(ctx, a, b, new_id)
    }

    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        // With `faer-matmul` (a default feature), every matmul goes through
        // `kernels::matmul_with_timestamp`, which uses faer's cache-blocked
        // GEMM. It honors strides and offsets and validates shapes itself.
        // Measured on an Apple M4: 169 ms -> 5 ms at 1024^2, versus the
        // hand-rolled SIMD kernel below (see docs/SCALING_AND_GPU_ROADMAP.md).
        // Before this, the DSL's `MATMUL` took the SIMD path for any large
        // contiguous input, so the feature never reached it.
        #[cfg(feature = "faer-matmul")]
        {
            crate::engine::kernels::matmul_with_timestamp(a, b, new_id, ctx.created_at)
        }

        // Without the feature: SIMD for large contiguous inputs, scalar otherwise.
        #[cfg(not(feature = "faer-matmul"))]
        {
            if self.use_simd(a.len()) {
                self.simd.matmul(ctx, a, b, new_id)
            } else {
                self.scalar.matmul(ctx, a, b, new_id)
            }
        }
    }

    fn dot(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if self.use_simd(a.len()) {
            self.simd.dot(ctx, a, b)
        } else {
            self.scalar.dot(ctx, a, b)
        }
    }

    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        // Cosine similarity involves dot product and magnitude.
        // We let scalar handle it for now, which uses dot_1d (not yet specialized in cpu dispatcher)
        // Optimization: specialize this to use SIMD dot product if beneficial.
        self.scalar.cosine_similarity(ctx, a, b)
    }

    fn distance(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if self.use_simd(a.len()) {
            self.simd.distance(ctx, a, b)
        } else {
            self.scalar.distance(ctx, a, b)
        }
    }

    fn scale(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.scale(ctx, a, factor, new_id)
    }

    fn normalize(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.normalize(ctx, a, new_id)
    }

    fn transpose(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.transpose(ctx, a, new_id)
    }

    fn flatten(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.flatten(ctx, a, new_id)
    }

    fn sum(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.sum(ctx, a, new_id)
    }

    fn mean(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.mean(ctx, a, new_id)
    }

    fn stdev(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.stdev(ctx, a, new_id)
    }

    fn reshape(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.reshape(ctx, a, new_shape, new_id)
    }

    fn stack(
        &self,
        ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.stack(ctx, tensors, axis, new_id)
    }
}
