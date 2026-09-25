use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;
use crate::engine::kernels;

pub mod pool;
pub use pool::{PoolStats, TensorPool};

pub trait ComputeBackend: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;

    /// Allocate an output buffer for a tensor.
    /// Uses the execution context's tensor pool for reuse when possible.
    /// For tiny tensors (≤16 elements), uses stack allocation.
    /// For small tensors (<256 elements), uses direct allocation to avoid pool overhead.
    fn alloc_output(&self, ctx: &mut ExecutionContext, len: usize) -> Vec<f32> {
        use smallvec::{smallvec, SmallVec};

        // Stack allocation for tiny tensors (≤16 elements)
        const STACK_THRESHOLD: usize = 16;
        // Pool overhead (~40ns) is significant for small allocations (~100ns)
        const POOL_THRESHOLD: usize = 256;

        if len <= STACK_THRESHOLD {
            // Stack allocation - zero heap allocation!
            let small: SmallVec<[f32; 16]> = smallvec![0.0; len];
            small.to_vec()
        } else if len < POOL_THRESHOLD {
            // Direct allocation for small tensors
            let mut vec = Vec::with_capacity(len);
            vec.resize(len, 0.0);
            vec
        } else {
            // Pool for medium/large tensors
            ctx.acquire_vec(len)
        }
    }

    // Binary operations
    fn add(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn sub(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn multiply(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn divide(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Matrix operations
    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Reductions / Metrics
    fn dot(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String>;
    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String>;
    fn distance(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String>;
    /// Pearson correlation coefficient between two rank-1 tensors. Default implementation
    /// shared by every backend (scalar/cpu/simd) -- this is a small O(n) scan with no
    /// SIMD-specific fast path today, unlike `dot`/`cosine_similarity`.
    fn correlate(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        kernels::pearson_correlation_1d(a, b)
    }

    // Unary operations
    fn scale(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn normalize(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn transpose(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn flatten(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Statistical Aggregations
    fn sum(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn mean(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn stdev(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    /// Population variance -- default implementation shared by every backend
    /// (like `correlate` above), no SIMD-specific fast path.
    fn variance(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::variance(a, new_id)
    }
    /// Median (sorts all elements) -- default implementation shared by
    /// every backend, no SIMD-specific fast path.
    fn median(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::median(a, new_id)
    }
    /// `p`-th quantile (sorts all elements) -- default implementation
    /// shared by every backend, no SIMD-specific fast path.
    fn quantile(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        p: f64,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::quantile(a, p, new_id)
    }
    /// Population covariance between two same-shape tensors -- default
    /// implementation shared by every backend, no SIMD-specific fast path.
    fn covariance(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        kernels::covariance(a, b)
    }

    // Layout operations
    fn reshape(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn stack(
        &self,
        ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
}

pub mod cpu;
#[cfg(feature = "gpu-wgpu")]
pub mod gpu;
pub mod scalar;
pub mod simd;

/// The backend `[compute] backend` asks for (see `ComputeConfig`). Asking
/// for `gpu` without a usable GPU -- or in a build without the `gpu-wgpu`
/// feature -- warns once and falls back to the CPU backend.
pub fn from_config(config: &crate::core::config::ComputeConfig) -> Box<dyn ComputeBackend> {
    use crate::core::config::ComputeBackendKind;
    match config.backend {
        ComputeBackendKind::Cpu => Box::new(CpuBackend::new()),
        ComputeBackendKind::Gpu => gpu_or_cpu(),
    }
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_or_cpu() -> Box<dyn ComputeBackend> {
    match gpu::GpuBackend::new() {
        Ok(b) => Box::new(b),
        Err(e) => {
            warn_gpu_fallback(&e);
            Box::new(CpuBackend::new())
        }
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
fn gpu_or_cpu() -> Box<dyn ComputeBackend> {
    warn_gpu_fallback("this build doesn't include the `gpu-wgpu` feature");
    Box::new(CpuBackend::new())
}

fn warn_gpu_fallback(reason: &str) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "Warning: [compute] backend = \"gpu\" requested, using the CPU backend instead: {}",
            reason
        )
    });
}

pub use cpu::CpuBackend;
pub use scalar::ScalarBackend;
pub use simd::SimdBackend;
