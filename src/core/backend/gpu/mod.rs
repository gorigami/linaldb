//! Optional GPU compute backend over `wgpu` (feature `gpu-wgpu`) -- the
//! Track C spike of `docs/SCALING_AND_GPU_ROADMAP.md`.
//!
//! Scope is deliberately narrow: measure whether a GPU pays off for this
//! engine's workloads before committing to device-resident tensors.
//! `Tensor.data` stays in host memory; every GPU call uploads its inputs,
//! runs one kernel (`kernels.wgsl`) and reads the result back.
//!
//! - `GpuContext` owns the device and the compiled pipelines. It's created
//!   once per process (`GpuContext::shared`).
//! - `GpuBackend` implements `ComputeBackend`. It only sends `matmul` to the
//!   GPU, and only above `MATMUL_GPU_MIN_FLOPS`; every other operation, and
//!   any small or non-2D matmul, delegates to `CpuBackend`. A single `dot` or
//!   `cosine_similarity` is one vector pair, so the upload alone would cost
//!   more than the CPU computing it.
//! - `GpuContext::batch_cosine` scores many vectors against one query in a
//!   single dispatch, the shape a brute-force exact vector search has. It's
//!   exposed for the benchmark (`benches/gpu_backend.rs`) and isn't wired into
//!   the query planner yet: that's gated on the benchmark's results.

use super::{ComputeBackend, CpuBackend};
use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use crate::engine::context::ExecutionContext;
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

/// Below this many multiply-adds (m*k*n), a matmul stays on the CPU: upload,
/// dispatch and readback overhead dominate. For scale, 128^3 is ~2.1M.
pub const MATMUL_GPU_MIN_FLOPS: usize = 2_000_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend_name: String,
    matmul: wgpu::ComputePipeline,
    cosine: wgpu::ComputePipeline,
    max_binding_bytes: u64,
    max_workgroups: u32,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.adapter_name)
            .field("backend", &self.backend_name)
            .finish()
    }
}

static SHARED: OnceLock<Result<Arc<GpuContext>, String>> = OnceLock::new();

impl GpuContext {
    /// The process-wide context, created on first use. `Err` when no usable
    /// GPU adapter exists (headless CI, no drivers).
    pub fn shared() -> Result<Arc<GpuContext>, String> {
        SHARED
            .get_or_init(|| pollster::block_on(Self::init()).map(Arc::new))
            .clone()
    }

    async fn init() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter available: {}", e))?;
        let info = adapter.get_info();
        // Ask for everything the adapter supports: the default limits cap a
        // storage binding at 128 MiB, which a 1M x 768 f32 matrix exceeds.
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("linal"),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("cannot open GPU device: {}", e))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("linal kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let matmul = pipeline("matmul");
        let cosine = pipeline("cosine");

        Ok(Self {
            adapter_name: info.name,
            backend_name: format!("{:?}", info.backend),
            matmul,
            cosine,
            max_binding_bytes: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            device,
            queue,
        })
    }

    /// e.g. "Apple M2 (Metal)".
    pub fn description(&self) -> String {
        format!("{} ({})", self.adapter_name, self.backend_name)
    }

    /// Runs `pipeline` over two input buffers and returns `out_len` floats.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        b: &[f32],
        dims: Dims,
        out_len: usize,
        workgroups: (u32, u32),
    ) -> Result<Vec<f32>, String> {
        for (what, bytes) in [
            ("first input", std::mem::size_of_val(a) as u64),
            ("second input", std::mem::size_of_val(b) as u64),
            ("output", (out_len * 4) as u64),
        ] {
            if bytes > self.max_binding_bytes {
                return Err(format!(
                    "{} is {} bytes, over this GPU's {}-byte binding limit",
                    what, bytes, self.max_binding_bytes
                ));
            }
        }
        if workgroups.0 > self.max_workgroups || workgroups.1 > self.max_workgroups {
            return Err("problem too large for one dispatch".into());
        }

        let storage = |label, data: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buf_a = storage("in_a", a);
        let buf_b = storage("in_b", b);
        let out_bytes = (out_len.max(1) * 4) as u64;
        let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_dims = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &readback, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {}", e))?;
        rx.recv()
            .map_err(|e| format!("GPU readback never completed: {}", e))?
            .map_err(|e| format!("GPU readback failed: {}", e))?;
        let out = {
            let view = readback
                .get_mapped_range(..)
                .map_err(|e| format!("GPU readback failed: {}", e))?;
            bytemuck::cast_slice::<u8, f32>(&view)[..out_len].to_vec()
        };
        readback.unmap();
        Ok(out)
    }

    /// C[m, n] = A[m, k] * B[k, n], all row-major and contiguous.
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        if a.len() != m * k || b.len() != k * n {
            return Err("matmul input lengths don't match their dimensions".into());
        }
        let dims = Dims {
            m: u32::try_from(m).map_err(|_| "m too large")?,
            k: u32::try_from(k).map_err(|_| "k too large")?,
            n: u32::try_from(n).map_err(|_| "n too large")?,
            _pad: 0,
        };
        let groups = (n.div_ceil(16) as u32, m.div_ceil(16) as u32);
        self.run(&self.matmul, a, b, dims, m * n, groups)
    }

    /// Cosine similarity of each of the `rows.len() / dim` vectors in `rows`
    /// (row-major) against `query`. A zero-norm row (or query) gives NaN.
    /// Chunked so any number of rows fits the GPU's binding-size limit.
    pub fn batch_cosine(
        &self,
        query: &[f32],
        rows: &[f32],
        dim: usize,
    ) -> Result<Vec<f32>, String> {
        if dim == 0 || query.len() != dim || !rows.len().is_multiple_of(dim) {
            return Err("batch_cosine: query/rows don't match dim".into());
        }
        let total = rows.len() / dim;
        let by_binding = (self.max_binding_bytes / (dim as u64 * 4)) as usize;
        let by_dispatch = self.max_workgroups as usize * 256;
        let chunk = by_binding.min(by_dispatch).max(1);

        let mut out = Vec::with_capacity(total);
        for start in (0..total).step_by(chunk) {
            let count = chunk.min(total - start);
            let slice = &rows[start * dim..(start + count) * dim];
            let dims = Dims {
                m: count as u32,
                k: dim as u32,
                n: 0,
                _pad: 0,
            };
            let groups = (count.div_ceil(256) as u32, 1);
            out.extend(self.run(&self.cosine, slice, query, dims, count, groups)?);
        }
        Ok(out)
    }
}

/// `ComputeBackend` that sends large dense matmuls to the GPU and
/// everything else to the CPU.
#[derive(Debug)]
pub struct GpuBackend {
    ctx: Arc<GpuContext>,
    cpu: CpuBackend,
    name: String,
}

impl GpuBackend {
    /// `Err` when no GPU is available; callers fall back to `CpuBackend`.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::shared()?;
        let name = format!(
            "Gpu (wgpu: {}; matmul >= {} multiply-adds, CPU otherwise)",
            ctx.description(),
            MATMUL_GPU_MIN_FLOPS
        );
        Ok(Self {
            ctx,
            cpu: CpuBackend::new(),
            name,
        })
    }
}

macro_rules! delegate_binary {
    ($($op:ident),*) => {$(
        fn $op(
            &self,
            ctx: &mut ExecutionContext,
            a: &Tensor,
            b: &Tensor,
            new_id: TensorId,
        ) -> Result<Tensor, String> {
            self.cpu.$op(ctx, a, b, new_id)
        }
    )*};
}

macro_rules! delegate_unary {
    ($($op:ident),*) => {$(
        fn $op(
            &self,
            ctx: &mut ExecutionContext,
            a: &Tensor,
            new_id: TensorId,
        ) -> Result<Tensor, String> {
            self.cpu.$op(ctx, a, new_id)
        }
    )*};
}

impl ComputeBackend for GpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    delegate_binary!(add, sub, multiply, divide);
    delegate_unary!(normalize, transpose, flatten, sum, mean, stdev);

    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        let eligible = a.shape.rank() == 2
            && b.shape.rank() == 2
            && a.shape.dims[1] == b.shape.dims[0]
            && a.shape.dims[0] * a.shape.dims[1] * b.shape.dims[1] >= MATMUL_GPU_MIN_FLOPS;
        if !eligible {
            // Includes every shape error: the CPU path reports them.
            return self.cpu.matmul(ctx, a, b, new_id);
        }
        let (m, k, n) = (a.shape.dims[0], a.shape.dims[1], b.shape.dims[1]);
        // Logical (stride-aware) values, so a transposed/sliced zero-copy
        // view multiplies correctly -- same contract as the faer path.
        let a_data = a.to_logical_vec();
        let b_data = b.to_logical_vec();
        match self.ctx.matmul(&a_data, &b_data, m, k, n) {
            Ok(data) => {
                let metadata = TensorMetadata::new_with_timestamp(new_id, None, ctx.created_at);
                Tensor::new(new_id, Shape::new(vec![m, n]), data, metadata)
            }
            // e.g. over the device's buffer limits: the CPU can still do it.
            Err(_) => self.cpu.matmul(ctx, a, b, new_id),
        }
    }

    fn dot(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        self.cpu.dot(ctx, a, b)
    }

    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        self.cpu.cosine_similarity(ctx, a, b)
    }

    fn distance(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        self.cpu.distance(ctx, a, b)
    }

    fn scale(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.cpu.scale(ctx, a, factor, new_id)
    }

    fn reshape(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.cpu.reshape(ctx, a, new_shape, new_id)
    }

    fn stack(
        &self,
        ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.cpu.stack(ctx, tensors, axis, new_id)
    }
}
