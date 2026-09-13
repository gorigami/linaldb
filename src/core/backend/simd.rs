use super::{ComputeBackend, ScalarBackend};
use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;

#[derive(Debug, Default)]
pub struct SimdBackend {
    scalar: ScalarBackend, // Fallback for unsupported ops or small tensors
}

impl SimdBackend {
    pub fn new() -> Self {
        Self {
            scalar: ScalarBackend::new(),
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn is_aligned(ptr: *const f32, alignment: usize) -> bool {
        (ptr as usize).is_multiple_of(alignment)
    }

    #[inline(always)]
    fn add_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            // NEON vld1q_f32 works for both aligned and unaligned
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vaddq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_add_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_add_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        // Remaining elements
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn sub_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vsubq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_sub_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_sub_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn mul_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vmulq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_mul_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_mul_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn dot_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut i = 0;
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            unsafe {
                let mut vsum = vdupq_n_f32(0.0);
                while i + 4 <= n {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    vsum = vfmaq_f32(vsum, va, vb);
                    i += 4;
                }
                sum = vaddvq_f32(vsum);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut vsum = _mm_setzero_ps();
                if Self::is_aligned(a_ptr, 16) && Self::is_aligned(b_ptr, 16) {
                    while i + 4 <= n {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
                        i += 4;
                    }
                } else {
                    while i + 4 <= n {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
                        i += 4;
                    }
                }
                // Horizontal add
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), vsum);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
        }

        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    #[inline(always)]
    fn distance_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut i = 0;
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            unsafe {
                let mut vsum = vdupq_n_f32(0.0);
                while i + 4 <= n {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vdiff = vsubq_f32(va, vb);
                    vsum = vfmaq_f32(vsum, vdiff, vdiff);
                    i += 4;
                }
                sum = vaddvq_f32(vsum);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut vsum = _mm_setzero_ps();
                if Self::is_aligned(a_ptr, 16) && Self::is_aligned(b_ptr, 16) {
                    while i + 4 <= n {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vdiff = _mm_sub_ps(va, vb);
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(vdiff, vdiff));
                        i += 4;
                    }
                } else {
                    while i + 4 <= n {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vdiff = _mm_sub_ps(va, vb);
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(vdiff, vdiff));
                        i += 4;
                    }
                }
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), vsum);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
        }

        while i < n {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        sum.sqrt()
    }

    #[allow(unused_variables)]
    fn matmul_simd(&self, a: &[f32], b: &[f32], m: usize, n: usize, p: usize, out: &mut [f32]) {
        // Naive explicit SIMD matmul: C[i, j..j+4] += A[i, k] * B[k, j..j+4]
        // This requires B to be row-major contiguous.

        // Zero out result first
        for x in out.iter_mut() {
            *x = 0.0;
        }

        for i in 0..m {
            for k in 0..n {
                let a_val = a[i * n + k];

                // Broadcast A[i, k]
                #[cfg(target_arch = "aarch64")]
                {
                    use std::arch::aarch64::*;
                    unsafe {
                        let va = vdupq_n_f32(a_val);
                        let mut j = 0;
                        while j + 4 <= p {
                            let vb = vld1q_f32(b.as_ptr().add(k * p + j));
                            // C[i, j] accumulator (load, fma, store)
                            // We need to accumulate into existing C value?
                            // Current loop structure:
                            // C[i, j] = sum(A[i,k]*B[k,j]).
                            // We are doing outer product style here:
                            // iterating k inside.
                            // So we load C[i, j], add A*B, store C[i, j].

                            let c_ptr = out.as_mut_ptr().add(i * p + j);
                            let vc = vld1q_f32(c_ptr);
                            let vres = vfmaq_f32(vc, va, vb);
                            vst1q_f32(c_ptr, vres);

                            j += 4;
                        }
                        // Residual
                        while j < p {
                            out[i * p + j] += a_val * b[k * p + j];
                            j += 1;
                        }
                    }
                }

                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::*;
                    unsafe {
                        let va = _mm_set1_ps(a_val);
                        let mut j = 0;
                        // Determine alignment of rows of B and C
                        // P might not be multiple of 4.
                        // Alignments might vary per row if P is not aligned.
                        // Safe to use unaligned loads.

                        while j + 4 <= p {
                            let vb = _mm_loadu_ps(b.as_ptr().add(k * p + j));
                            let c_ptr = out.as_mut_ptr().add(i * p + j);
                            let vc = _mm_loadu_ps(c_ptr);
                            let vres = _mm_add_ps(vc, _mm_mul_ps(va, vb));
                            _mm_storeu_ps(c_ptr, vres);
                            j += 4;
                        }
                        // Residual
                        while j < p {
                            out[i * p + j] += a_val * b[k * p + j];
                            j += 1;
                        }
                    }
                }

                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    for j in 0..p {
                        out[i * p + j] += a_val * b[k * p + j];
                    }
                }
            }
        }
    }
}

impl ComputeBackend for SimdBackend {
    fn name(&self) -> &str {
        #[cfg(target_arch = "aarch64")]
        return "SIMD (NEON Optimized)";
        #[cfg(target_arch = "x86_64")]
        return "SIMD (SSE/AVX Optimized)";
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return "SIMD (Portable Fallback)";
    }

    /// Allocate an output buffer for a tensor.
    fn alloc_output(&self, _ctx: &mut ExecutionContext, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn add(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        // SIMD only works correctly on physically contiguous tensors with matching
        // layout AND matching shape -- anything else (including a legitimate
        // scalar/shape broadcast) must fall through to the scalar backend below,
        // which is the only path that actually implements broadcasting.
        if a.shape == b.shape && a.is_contiguous() && b.is_contiguous() {
            let len = a.len();
            let mut data = self.alloc_output(ctx, len);
            self.add_simd(a.data_ref(), b.data_ref(), &mut data);
            let metadata = crate::core::tensor::TensorMetadata::new_with_timestamp(
                new_id,
                None,
                ctx.created_at,
            );
            return Tensor::new(new_id, a.shape.clone(), data, metadata).map_err(|e| e.to_string());
        }

        // Fallback to scalar for strided/broadcast/complex cases
        self.scalar.add(ctx, a, b, new_id)
    }

    fn sub(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape == b.shape && a.is_contiguous() && b.is_contiguous() {
            let len = a.len();
            let mut data = self.alloc_output(ctx, len);
            self.sub_simd(a.data_ref(), b.data_ref(), &mut data);
            let metadata = crate::core::tensor::TensorMetadata::new_with_timestamp(
                new_id,
                None,
                ctx.created_at,
            );
            return Tensor::new(new_id, a.shape.clone(), data, metadata).map_err(|e| e.to_string());
        }

        self.scalar.sub(ctx, a, b, new_id)
    }

    fn multiply(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape == b.shape && a.is_contiguous() && b.is_contiguous() {
            let len = a.len();
            let mut data = self.alloc_output(ctx, len);
            self.mul_simd(a.data_ref(), b.data_ref(), &mut data);
            let metadata = crate::core::tensor::TensorMetadata::new_with_timestamp(
                new_id,
                None,
                ctx.created_at,
            );
            return Tensor::new(new_id, a.shape.clone(), data, metadata).map_err(|e| e.to_string());
        }

        self.scalar.multiply(ctx, a, b, new_id)
    }

    fn divide(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        // TODO: Implement SIMD divide with zero-check or relax safety?
        // For now, use scalar to ensure zero-check correctness
        self.scalar.divide(ctx, a, b, new_id)
    }

    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape.rank() != 2 || b.shape.rank() != 2 {
            return Err("matmul expects rank-2 tensors".into());
        }

        // Use SIMD if contiguous
        if a.is_contiguous() && b.is_contiguous() {
            let m = a.shape.dims[0];
            let n = a.shape.dims[1];
            let n2 = b.shape.dims[0];
            let p = b.shape.dims[1];

            if n != n2 {
                return Err("Dimension mismatch".into());
            }

            // Allocate output C[m, p]
            let len = m * p;
            let mut data = self.alloc_output(ctx, len);

            self.matmul_simd(a.data_ref(), b.data_ref(), m, n, p, &mut data);

            let shape = crate::core::tensor::Shape::new(vec![m, p]);
            let metadata = crate::core::tensor::TensorMetadata::new_with_timestamp(
                new_id,
                None,
                ctx.created_at,
            );
            return Tensor::new(new_id, shape, data, metadata).map_err(|e| e.to_string());
        }

        self.scalar.matmul(ctx, a, b, new_id)
    }

    fn dot(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err("Length mismatch".into());
        }
        Ok(self.dot_simd(a.data_ref(), b.data_ref()))
    }

    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        self.scalar.cosine_similarity(ctx, a, b)
    }

    fn distance(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err("Length mismatch".into());
        }
        Ok(self.distance_simd(a.data_ref(), b.data_ref()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{Shape, TensorMetadata};

    // Regression tests for a real bug found building a production-network
    // economics showcase (a 45x45 real-data matrix): `SimdBackend::add`/`sub`/
    // `multiply` returned a bare "Shape mismatch" error for ANY shape mismatch,
    // including a legitimate scalar broadcast (e.g. `matrix * 0.5`), as soon as
    // the left operand crossed `CpuBackend`'s SIMD_THRESHOLD (1024 elements) --
    // below that threshold `ScalarBackend` (which correctly broadcasts) was used
    // instead, so the bug was invisible on small fixtures. These construct
    // above-threshold tensors specifically to exercise the SIMD path.
    const ABOVE_SIMD_THRESHOLD: usize = 1200;

    fn big_vector(fill: f32) -> Tensor {
        let id = TensorId::new();
        let shape = Shape::new(vec![ABOVE_SIMD_THRESHOLD]);
        let data = vec![fill; ABOVE_SIMD_THRESHOLD];
        Tensor::new(id, shape, data, TensorMetadata::new(id, None)).unwrap()
    }

    fn scalar_tensor(v: f32) -> Tensor {
        let id = TensorId::new();
        Tensor::new(
            id,
            Shape::new(vec![]),
            vec![v],
            TensorMetadata::new(id, None),
        )
        .unwrap()
    }

    #[test]
    fn add_broadcasts_scalar_above_simd_threshold() {
        let backend = SimdBackend::new();
        let mut ctx = ExecutionContext::new();
        let a = big_vector(2.0);
        let b = scalar_tensor(3.0);
        let result = backend.add(&mut ctx, &a, &b, TensorId::new()).unwrap();
        assert_eq!(result.shape.dims, vec![ABOVE_SIMD_THRESHOLD]);
        assert!(result.data_ref().iter().all(|&v| v == 5.0));
    }

    #[test]
    fn sub_broadcasts_scalar_above_simd_threshold() {
        let backend = SimdBackend::new();
        let mut ctx = ExecutionContext::new();
        let a = big_vector(5.0);
        let b = scalar_tensor(3.0);
        let result = backend.sub(&mut ctx, &a, &b, TensorId::new()).unwrap();
        assert_eq!(result.shape.dims, vec![ABOVE_SIMD_THRESHOLD]);
        assert!(result.data_ref().iter().all(|&v| v == 2.0));
    }

    #[test]
    fn multiply_broadcasts_scalar_above_simd_threshold() {
        let backend = SimdBackend::new();
        let mut ctx = ExecutionContext::new();
        let a = big_vector(4.0);
        let b = scalar_tensor(0.5);
        let result = backend.multiply(&mut ctx, &a, &b, TensorId::new()).unwrap();
        assert_eq!(result.shape.dims, vec![ABOVE_SIMD_THRESHOLD]);
        assert!(result.data_ref().iter().all(|&v| v == 2.0));
    }

    #[test]
    fn multiply_still_uses_fast_path_for_matching_shapes_above_threshold() {
        let backend = SimdBackend::new();
        let mut ctx = ExecutionContext::new();
        let a = big_vector(4.0);
        let b = big_vector(2.0);
        let result = backend.multiply(&mut ctx, &a, &b, TensorId::new()).unwrap();
        assert_eq!(result.shape.dims, vec![ABOVE_SIMD_THRESHOLD]);
        assert!(result.data_ref().iter().all(|&v| v == 8.0));
    }
}
