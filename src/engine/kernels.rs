// src/engine/kernels.rs

use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use rayon::prelude::*;

// Threshold for switching to parallel execution
const PARALLEL_THRESHOLD: usize = 50_000;

/// Estrategia para combinar dos tensores en una operación elemento a elemento.
/// Soporta:
/// - escalar con cualquier shape (broadcast)
/// - vectores (rank 1) de distinta longitud (padding con neutros)
fn elementwise_binary_op(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    neutral_a: f32,
    neutral_b: f32,
    op: impl Fn(f32, f32) -> Result<f32, String> + Sync + Send,
) -> Result<Tensor, String> {
    elementwise_binary_op_with_timestamp(a, b, new_id, neutral_a, neutral_b, op, chrono::Utc::now())
}

/// Optimized version for operations that cannot fail (Add, Sub, Mul)
/// Removes Result wrapping per element to allow better optimization/vectorization
fn elementwise_binary_op_infallible_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    neutral_a: f32,
    neutral_b: f32,
    op: impl Fn(f32, f32) -> f32 + Sync + Send,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    if a.shape == b.shape {
        // Fast path for same shape
        let len = a.shape.num_elements();
        let mut data = Vec::with_capacity(len);

        if let (Some(a_slice), Some(b_slice)) = (a.as_contiguous_slice(), b.as_contiguous_slice()) {
            if len >= PARALLEL_THRESHOLD {
                // Parallel execution for large tensors
                a_slice
                    .par_iter()
                    .zip(b_slice.par_iter())
                    .map(|(x, y)| op(*x, *y))
                    .collect_into_vec(&mut data);
            } else {
                // Serial execution for smaller tensors
                data.extend(a_slice.iter().zip(b_slice.iter()).map(|(x, y)| op(*x, *y)));
            }
        } else {
            // Strided path (currently serial fallback)
            // TODO: Parallelize strided path if needed
            match a.shape.rank() {
                1 => {
                    let len = a.shape.dims[0];
                    let a_stride = a.strides[0];
                    let b_stride = b.strides[0];
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();

                    for i in 0..len {
                        let val_a = a_data[a.offset + i * a_stride];
                        let val_b = b_data[b.offset + i * b_stride];
                        data.push(op(val_a, val_b));
                    }
                }
                2 => {
                    let rows = a.shape.dims[0];
                    let cols = a.shape.dims[1];
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();

                    for i in 0..rows {
                        let a_row = a.offset + i * a.strides[0];
                        let b_row = b.offset + i * b.strides[0];
                        for j in 0..cols {
                            let val_a = a_data[a_row + j * a.strides[1]];
                            let val_b = b_data[b_row + j * b.strides[1]];
                            data.push(op(val_a, val_b));
                        }
                    }
                }
                _ => {
                    // Efficient N-dim traversal for matching shapes
                    let rank = a.shape.rank();
                    let mut current_indices = vec![0; rank];
                    let mut a_off = a.offset;
                    let mut b_off = b.offset;
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();

                    for _ in 0..len {
                        data.push(op(a_data[a_off], b_data[b_off]));

                        // Advance indices and update offsets incrementally
                        for j in (0..rank).rev() {
                            current_indices[j] += 1;
                            if current_indices[j] < a.shape.dims[j] {
                                a_off += a.strides[j];
                                b_off += b.strides[j];
                                break;
                            } else {
                                a_off -= (current_indices[j] - 1) * a.strides[j];
                                b_off -= (current_indices[j] - 1) * b.strides[j];
                                current_indices[j] = 0;
                            }
                        }
                    }
                }
            }
        }

        let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
        return Tensor::new(new_id, a.shape.clone(), data, metadata);
    }

    // Reuse fallback logic but wrapping op
    elementwise_binary_op_with_timestamp(
        a,
        b,
        new_id,
        neutral_a,
        neutral_b,
        |x, y| Ok(op(x, y)),
        timestamp,
    )
}

fn elementwise_binary_op_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    neutral_a: f32,
    neutral_b: f32,
    op: impl Fn(f32, f32) -> Result<f32, String> + Sync + Send,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    if a.shape == b.shape {
        let len = a.len();
        let mut data = Vec::with_capacity(len);

        if let (Some(a_slice), Some(b_slice)) = (a.as_contiguous_slice(), b.as_contiguous_slice()) {
            if len >= PARALLEL_THRESHOLD {
                // Parallel logic with Result is tricky because `par_iter` doesn't easily collect Results into Result<Vec>
                // safely without mutexes or custom reduction.
                // For now, let's keep the fallible path SERIAL unless we really feel the need to optimize it.
                // Most hot path ops (add/sub/mul) are handled by the infallible version above.
                for i in 0..len {
                    data.push(op(a_slice[i], b_slice[i])?);
                }
            } else {
                for i in 0..len {
                    data.push(op(a_slice[i], b_slice[i])?);
                }
            }
        } else {
            // Strided path (Rank 1 & 2 specialized)
            match a.shape.rank() {
                1 => {
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();
                    for i in 0..len {
                        let val_a = a_data[a.offset + i * a.strides[0]];
                        let val_b = b_data[b.offset + i * b.strides[0]];
                        data.push(op(val_a, val_b)?);
                    }
                }
                2 => {
                    let rows = a.shape.dims[0];
                    let cols = a.shape.dims[1];
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();
                    for i in 0..rows {
                        let a_row = a.offset + i * a.strides[0];
                        let b_row = b.offset + i * b.strides[0];
                        for j in 0..cols {
                            let val_a = a_data[a_row + j * a.strides[1]];
                            let val_b = b_data[b_row + j * b.strides[1]];
                            data.push(op(val_a, val_b)?);
                        }
                    }
                }
                _ => {
                    // Efficient N-dim traversal for matching shapes (fallible)
                    let rank = a.shape.rank();
                    let mut current_indices = vec![0; rank];
                    let mut a_off = a.offset;
                    let mut b_off = b.offset;
                    let a_data = a.data_ref();
                    let b_data = b.data_ref();

                    for _ in 0..len {
                        data.push(op(a_data[a_off], b_data[b_off])?);

                        // Advance indices and update offsets incrementally
                        for j in (0..rank).rev() {
                            current_indices[j] += 1;
                            if current_indices[j] < a.shape.dims[j] {
                                a_off += a.strides[j];
                                b_off += b.strides[j];
                                break;
                            } else {
                                a_off -= (current_indices[j] - 1) * a.strides[j];
                                b_off -= (current_indices[j] - 1) * b.strides[j];
                                current_indices[j] = 0;
                            }
                        }
                    }
                }
            }
        }

        let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
        return Tensor::new(new_id, a.shape.clone(), data, metadata);
    }

    match (a.shape.rank(), b.shape.rank()) {
        // Escalar con cualquier shape (broadcast)
        (0, _) => {
            let shape = b.shape.clone();
            let mut data = Vec::with_capacity(b.shape.num_elements());
            let scalar = a.data_ref()[a.offset];

            if let Some(b_slice) = b.as_contiguous_slice() {
                for &y in b_slice {
                    data.push(op(scalar, y)?);
                }
            } else {
                // Strided fallback
                // Just use scalar_mul-like logic but with op
                let b_data = b.data_ref();
                match b.shape.rank() {
                    1 => {
                        let len = b.shape.dims[0];
                        for i in 0..len {
                            data.push(op(scalar, b_data[b.offset + i * b.strides[0]])?);
                        }
                    }
                    2 => {
                        let rows = b.shape.dims[0];
                        let cols = b.shape.dims[1];
                        for i in 0..rows {
                            let row_start = b.offset + i * b.strides[0];
                            for j in 0..cols {
                                data.push(op(scalar, b_data[row_start + j * b.strides[1]])?);
                            }
                        }
                    }
                    _ => {
                        // Generalized strided broadcast for scalars
                        let rank = b.shape.rank();
                        let mut current_indices = vec![0; rank];
                        let mut b_off = b.offset;
                        let b_data = b.data_ref();

                        for _ in 0..b.shape.num_elements() {
                            data.push(op(scalar, b_data[b_off])?);

                            for j in (0..rank).rev() {
                                current_indices[j] += 1;
                                if current_indices[j] < b.shape.dims[j] {
                                    b_off += b.strides[j];
                                    break;
                                } else {
                                    b_off -= (current_indices[j] - 1) * b.strides[j];
                                    current_indices[j] = 0;
                                }
                            }
                        }
                    }
                }
            }
            let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
            Tensor::new(new_id, shape, data, metadata)
        }
        (_, 0) => {
            let shape = a.shape.clone();
            let mut data = Vec::with_capacity(a.shape.num_elements());
            let scalar = b.data_ref()[b.offset];

            if let Some(a_slice) = a.as_contiguous_slice() {
                for &x in a_slice {
                    data.push(op(x, scalar)?);
                }
            } else {
                let a_data = a.data_ref();
                match a.shape.rank() {
                    1 => {
                        let len = a.shape.dims[0];
                        for i in 0..len {
                            data.push(op(a_data[a.offset + i * a.strides[0]], scalar)?);
                        }
                    }
                    2 => {
                        let rows = a.shape.dims[0];
                        let cols = a.shape.dims[1];
                        for i in 0..rows {
                            let row_start = a.offset + i * a.strides[0];
                            for j in 0..cols {
                                data.push(op(a_data[row_start + j * a.strides[1]], scalar)?);
                            }
                        }
                    }
                    _ => {
                        // Generalized strided broadcast for scalars (scalar on right)
                        let rank = a.shape.rank();
                        let mut current_indices = vec![0; rank];
                        let mut a_off = a.offset;
                        let a_data = a.data_ref();

                        for _ in 0..a.shape.num_elements() {
                            data.push(op(a_data[a_off], scalar)?);

                            for j in (0..rank).rev() {
                                current_indices[j] += 1;
                                if current_indices[j] < a.shape.dims[j] {
                                    a_off += a.strides[j];
                                    break;
                                } else {
                                    a_off -= (current_indices[j] - 1) * a.strides[j];
                                    current_indices[j] = 0;
                                }
                            }
                        }
                    }
                }
            }
            let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
            Tensor::new(new_id, shape, data, metadata)
        }
        // Vectores (rank 1) – posiblemente longitudes distintas (padding)
        (1, 1) => {
            let len_a = a.shape.dims[0]; // use dims, logic len
            let len_b = b.shape.dims[0];
            let len = len_a.max(len_b);

            let mut data = Vec::with_capacity(len);

            // Strided access
            let a_stride = a.strides[0];
            let b_stride = b.strides[0];
            let a_data = a.data_ref();
            let b_data = b.data_ref();

            for i in 0..len {
                let x = if i < len_a {
                    a_data[a.offset + i * a_stride]
                } else {
                    neutral_a
                };
                let y = if i < len_b {
                    b_data[b.offset + i * b_stride]
                } else {
                    neutral_b
                };
                data.push(op(x, y)?);
            }

            let shape = Shape::new(vec![len]);
            let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
            Tensor::new(new_id, shape, data, metadata)
        }
        // Otros casos: de momento, error
        _ => Err(format!(
            "Unsupported shapes for element-wise op: {:?} vs {:?}",
            a.shape.dims, b.shape.dims
        )),
    }
}

/// Verifica que dos shapes sean exactamente iguales
fn ensure_same_shape(a: &Shape, b: &Shape) -> Result<(), String> {
    if a.dims != b.dims {
        Err(format!(
            "Shape mismatch: requested operation requires identical shapes, but got {:?} vs {:?}",
            a.dims, b.dims
        ))
    } else {
        Ok(())
    }
}

/// Verifica que dos tensores sean vectores (rank 1)
fn ensure_rank_1(a: &Tensor, name: &str) -> Result<(), String> {
    if a.shape.rank() != 1 {
        Err(format!(
            "Dimension error: {} expects a rank-1 tensor (vector), but got rank-{} with shape {:?}",
            name,
            a.shape.rank(),
            a.shape.dims
        ))
    } else {
        Ok(())
    }
}

/// Verifica que dos tensores sean matrices (rank 2)
fn ensure_rank_2(a: &Tensor, name: &str) -> Result<(), String> {
    if a.shape.rank() != 2 {
        Err(format!(
            "Dimension error: {} expects a rank-2 tensor (matrix), but got rank-{} with shape {:?}",
            name,
            a.shape.rank(),
            a.shape.dims
        ))
    } else {
        Ok(())
    }
}

/// Suma elemento a elemento: a + b
pub fn add(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    add_with_timestamp(a, b, new_id, chrono::Utc::now())
}

pub fn add_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    // Addition is safe in f32 (no panic)
    elementwise_binary_op_infallible_with_timestamp(a, b, new_id, 0.0, 0.0, |x, y| x + y, timestamp)
}

/// Resta elemento a elemento: a - b
pub fn sub(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    sub_with_timestamp(a, b, new_id, chrono::Utc::now())
}

pub fn sub_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    // Subtraction is safe in f32
    elementwise_binary_op_infallible_with_timestamp(a, b, new_id, 0.0, 0.0, |x, y| x - y, timestamp)
}

/// Multiplicación elemento a elemento: a * b
pub fn multiply(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    multiply_with_timestamp(a, b, new_id, chrono::Utc::now())
}

pub fn multiply_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    // Multiplication is safe in f32
    elementwise_binary_op_infallible_with_timestamp(a, b, new_id, 1.0, 1.0, |x, y| x * y, timestamp)
}

/// División elemento a elemento: a / b
pub fn divide(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    divide_with_timestamp(a, b, new_id, chrono::Utc::now())
}

pub fn divide_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    elementwise_binary_op_with_timestamp(
        a,
        b,
        new_id,
        1.0,
        1.0,
        |x, y| {
            if y == 0.0 {
                return Err("Division by zero in element-wise divide".into());
            }
            Ok(x / y)
        },
        timestamp,
    )
}

/// Multiplicación por escalar: s * a
pub fn scalar_mul(a: &Tensor, s: f32, new_id: TensorId) -> Result<Tensor, String> {
    scalar_mul_with_timestamp(a, s, new_id, chrono::Utc::now())
}

pub fn scalar_mul_with_timestamp(
    a: &Tensor,
    s: f32,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(a.shape.num_elements());

    if let Some(slice) = a.as_contiguous_slice() {
        if slice.len() >= PARALLEL_THRESHOLD {
            // Parallel map
            slice.par_iter().map(|x| s * x).collect_into_vec(&mut data);
        } else {
            data.extend(slice.iter().map(|x| s * x));
        }
    } else {
        // Strided fallback (serial for now)
        let a_data = a.data_ref();
        match a.shape.rank() {
            1 => {
                let len = a.shape.dims[0];
                let stride = a.strides[0];
                for i in 0..len {
                    data.push(s * a_data[a.offset + i * stride]);
                }
            }
            2 => {
                let rows = a.shape.dims[0];
                let cols = a.shape.dims[1];
                for i in 0..rows {
                    let row_start = a.offset + i * a.strides[0];
                    for j in 0..cols {
                        data.push(s * a_data[row_start + j * a.strides[1]]);
                    }
                }
            }
            _ => {
                // Generalized strided scalar_mul
                let rank = a.shape.rank();
                let mut current_indices = vec![0; rank];
                let mut a_off = a.offset;
                let a_data = a.data_ref();

                for _ in 0..a.shape.num_elements() {
                    data.push(s * a_data[a_off]);

                    for j in (0..rank).rev() {
                        current_indices[j] += 1;
                        if current_indices[j] < a.shape.dims[j] {
                            a_off += a.strides[j];
                            break;
                        } else {
                            a_off -= (current_indices[j] - 1) * a.strides[j];
                            current_indices[j] = 0;
                        }
                    }
                }
            }
        }
    }
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, a.shape.clone(), data, metadata)
}

/// Producto punto entre dos tensores rank-1 (vectores)
pub fn dot_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    ensure_rank_1(a, "dot_1d")?;
    ensure_rank_1(b, "dot_1d")?;
    ensure_same_shape(&a.shape, &b.shape)?;

    let sum = a
        .data_ref()
        .iter()
        .zip(b.data_ref().iter())
        .map(|(x, y)| x * y)
        .sum();

    Ok(sum)
}

/// Pearson correlation coefficient between two rank-1 tensors (vectors).
///
/// This is the actual statistic `CORRELATE a WITH b` is documented to compute
/// (docs/DSL_REFERENCE.md) — mean-centered and normalized by each vector's own spread, unlike
/// `dot_1d` (raw, unnormalized) which `BinaryOp::Correlate` was previously wired to by mistake.
pub fn pearson_correlation_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    ensure_rank_1(a, "pearson_correlation_1d")?;
    ensure_rank_1(b, "pearson_correlation_1d")?;
    ensure_same_shape(&a.shape, &b.shape)?;

    let a_data = a.data_ref();
    let b_data = b.data_ref();
    let n = a_data.len();
    if n == 0 {
        return Err("Cannot compute Pearson correlation of an empty tensor".into());
    }
    let n_f = n as f32;

    let mean_a: f32 = a_data.iter().sum::<f32>() / n_f;
    let mean_b: f32 = b_data.iter().sum::<f32>() / n_f;

    let (mut cov, mut var_a, mut var_b) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a_data.iter().zip(b_data.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return Err("Cannot compute Pearson correlation with a zero-variance vector".into());
    }

    Ok(cov / (var_a.sqrt() * var_b.sqrt()))
}

/// L2 Norm of a tensor (any rank)
pub fn l2_norm_1d(a: &Tensor) -> Result<f32, String> {
    // Treat as a flat vector of elements
    Ok(a.data_ref().iter().map(|x| x * x).sum::<f32>().sqrt())
}

/// Distancia L2 entre dos tensores rank-1
pub fn distance_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    if a.shape.rank() != 1 || b.shape.rank() != 1 {
        return Err("distance_1d expects rank-1 tensors".into());
    }
    ensure_same_shape(&a.shape, &b.shape)?;

    let sum_sq: f32 = a
        .data_ref()
        .iter()
        .zip(b.data_ref().iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();

    Ok(sum_sq.sqrt())
}

/// Similitud coseno entre dos tensores rank-1
pub fn cosine_similarity_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    let dot_ab = dot_1d(a, b)?;
    let norm_a = l2_norm_1d(a)?;
    let norm_b = l2_norm_1d(b)?;

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Cannot compute cosine similarity with zero-norm vector".into());
    }

    Ok(dot_ab / (norm_a * norm_b))
}

/// Sum of all elements in a tensor
pub fn sum(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    sum_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn sum_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let s: f32 = if let Some(slice) = a.as_contiguous_slice() {
        slice.iter().sum()
    } else {
        let mut total = 0.0;
        let a_data = a.data_ref();
        let len = a.shape.num_elements();
        let rank = a.shape.rank();

        if rank == 1 {
            let stride = a.strides[0];
            for i in 0..len {
                total += a_data[a.offset + i * stride];
            }
        } else {
            let mut current_indices = vec![0; rank];
            let mut a_off = a.offset;
            for _ in 0..len {
                total += a_data[a_off];
                for j in (0..rank).rev() {
                    current_indices[j] += 1;
                    if current_indices[j] < a.shape.dims[j] {
                        a_off += a.strides[j];
                        break;
                    } else {
                        a_off -= (current_indices[j] - 1) * a.strides[j];
                        current_indices[j] = 0;
                    }
                }
            }
        }
        total
    };

    // A true scalar (rank-0), not a rank-1 Vector(1) -- see CORRELATE/SIMILARITY/DISTANCE for
    // the same convention. A rank-1 [1]-shaped result here used to make `vector - SUM(vector)`
    // (and other elementwise ops) silently fall into the differing-length-vector padding branch
    // in `elementwise_binary_op_with_timestamp` instead of the correct scalar-broadcast branch,
    // corrupting every element past index 0.
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, Shape::new(Vec::new()), vec![s], metadata)
}

/// Mean of all elements in a tensor
pub fn mean(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    mean_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn mean_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let total_elements = a.shape.num_elements() as f32;
    if total_elements == 0.0 {
        return Err("Cannot compute mean of empty tensor".into());
    }

    let s_tensor = sum_with_timestamp(a, TensorId::new(), timestamp)?;
    let m = s_tensor.data_ref()[0] / total_elements;

    // True scalar (rank-0) -- see the comment on `sum_with_timestamp` above for why this matters.
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, Shape::new(Vec::new()), vec![m], metadata)
}

/// Standard deviation of all elements in a tensor
pub fn stdev(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    stdev_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn stdev_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let total_elements = a.shape.num_elements() as f32;
    if total_elements == 0.0 {
        return Err("Cannot compute stdev of empty tensor".into());
    }

    let mean_val = mean_with_timestamp(a, TensorId::new(), timestamp)?.data_ref()[0];

    let sq_diff_sum: f32 = if let Some(slice) = a.as_contiguous_slice() {
        slice.iter().map(|&x| (x - mean_val).powi(2)).sum()
    } else {
        let mut total = 0.0;
        let a_data = a.data_ref();
        let len = a.shape.num_elements();
        let rank = a.shape.rank();

        let mut current_indices = vec![0; rank];
        let mut a_off = a.offset;
        for _ in 0..len {
            total += (a_data[a_off] - mean_val).powi(2);
            for j in (0..rank).rev() {
                current_indices[j] += 1;
                if current_indices[j] < a.shape.dims[j] {
                    a_off += a.strides[j];
                    break;
                } else {
                    a_off -= (current_indices[j] - 1) * a.strides[j];
                    current_indices[j] = 0;
                }
            }
        }
        total
    };

    let variance = sq_diff_sum / total_elements;
    let stdev = variance.sqrt();

    // True scalar (rank-0) -- see the comment on `sum_with_timestamp` above for why this matters.
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, Shape::new(Vec::new()), vec![stdev], metadata)
}

/// Normaliza un tensor rank-1 a norma 1 (L2)
pub fn normalize_1d(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    normalize_1d_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn normalize_1d_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let norm = l2_norm_1d(a)?;
    if norm == 0.0 {
        return Err("Cannot normalize a zero tensor".into());
    }
    let factor = 1.0 / norm;
    scalar_mul_with_timestamp(a, factor, new_id, timestamp)
}

/// Suma elemento a elemento (RELAXED):
/// - escalar con cualquier shape
/// - vectores de distinta longitud → padding con 0.0
pub fn add_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        0.0, // neutral_a
        0.0, // neutral_b
        |x, y| Ok(x + y),
    )
}

/// Resta elemento a elemento (RELAXED)
pub fn sub_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        0.0, // si falta a → 0 - y
        0.0, // si falta b → x - 0
        |x, y| Ok(x - y),
    )
}

/// Multiplicación elemento a elemento (RELAXED)
pub fn multiply_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        1.0, // si falta a → 1 * y = y
        1.0, // si falta b → x * 1 = x
        |x, y| Ok(x * y),
    )
}

/// División elemento a elemento (RELAXED)
pub fn divide_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        1.0, // si falta a → 1 / y
        1.0, // si falta b → x / 1 = x
        |x, y| {
            if y == 0.0 {
                Err("Division by zero in element-wise divide".into())
            } else {
                Ok(x / y)
            }
        },
    )
}

// ============================================================================
// MATRIX OPERATIONS (Rank-2 Tensors)
// ============================================================================

/// Matrix multiplication: C = A * B
/// A: [m, n], B: [n, p] → C: [m, p]
pub fn matmul(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    matmul_with_timestamp(a, b, new_id, chrono::Utc::now())
}

pub fn matmul_with_timestamp(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    ensure_rank_2(a, "matmul")?;
    ensure_rank_2(b, "matmul")?;

    let m = a.shape.dims[0];
    let n = a.shape.dims[1];
    let n2 = b.shape.dims[0];
    let p = b.shape.dims[1];

    if n != n2 {
        return Err(format!(
            "Matrix dimension mismatch in matmul: A is [{}x{}], B is [{}x{}]. Inner dimensions ({} vs {}) must match.",
            m, n, n2, p, n, n2
        ));
    }

    let mut data = vec![0.0; m * p];

    let a_strides = &a.strides;
    let b_strides = &b.strides;
    let a_data = a.data_ref();
    let b_data = b.data_ref();
    let a_offset = a.offset;
    let b_offset = b.offset;

    // Cost estimation: m * p output elements, each taking n muls and adds.
    // Total ops ~ 2 * m * n * p.
    let total_ops = m * n * p;

    if total_ops >= PARALLEL_THRESHOLD {
        // Parallelize over rows of C (chunks of size p)
        data.par_chunks_mut(p)
            .enumerate()
            .for_each(|(i, row_slice)| {
                let a_row_base = a_offset + i * a_strides[0];
                for (j, row_val) in row_slice.iter_mut().enumerate().take(p) {
                    let mut sum = 0.0;
                    let b_col_base = b_offset + j * b_strides[1];
                    for k in 0..n {
                        let a_idx = a_row_base + k * a_strides[1];
                        let b_idx = b_col_base + k * b_strides[0];
                        // Unchecked access for performance?
                        // Currently checked:
                        // sum += a_data[a_idx] * b_data[b_idx];
                        // Let's stick to checked or minimal risk for now.
                        // But we should use safe indexing.
                        if a_idx < a_data.len() && b_idx < b_data.len() {
                            sum += a_data[a_idx] * b_data[b_idx];
                        }
                    }
                    *row_val = sum;
                }
            });
    } else {
        // Serial execution
        for i in 0..m {
            let a_row_base = a_offset + i * a_strides[0];
            for j in 0..p {
                let mut sum = 0.0;
                let b_col_base = b_offset + j * b_strides[1];
                for k in 0..n {
                    let a_idx = a_row_base + k * a_strides[1];
                    let b_idx = b_col_base + k * b_strides[0];
                    if a_idx < a_data.len() && b_idx < b_data.len() {
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                }
                data[i * p + j] = sum;
            }
        }
    }

    let shape = Shape::new(vec![m, p]);
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, shape, data, metadata)
}

/// Transpose a rank-2 tensor (matrix)
/// A: [m, n] → A^T: [n, m]
pub fn transpose(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    transpose_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn transpose_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    if a.shape.rank() != 2 {
        return Err("transpose expects rank-2 tensor (matrix)".into());
    }

    let m = a.shape.dims[0];
    let n = a.shape.dims[1];

    // New shape: [n, m]
    let new_shape = Shape::new(vec![n, m]);

    // Swap strides: [stride_1, stride_0]
    let new_strides = vec![a.strides[1], a.strides[0]];

    // Zero-copy: share the underlying data Arc
    let data = a.data.clone();
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);

    Tensor::from_shared_strided(
        new_id,
        new_shape,
        data,
        std::sync::Arc::new(metadata),
        new_strides,
        a.offset,
    )
}

/// Reshape a tensor to a new shape (total elements must match)
pub fn reshape(a: &Tensor, new_shape: Shape, new_id: TensorId) -> Result<Tensor, String> {
    reshape_with_timestamp(a, new_shape, new_id, chrono::Utc::now())
}

pub fn reshape_with_timestamp(
    a: &Tensor,
    new_shape: Shape,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let old_elements = a.shape.num_elements();
    let new_elements = new_shape.num_elements();

    if old_elements != new_elements {
        return Err(format!(
            "Cannot reshape: old shape {:?} has {} elements, new shape {:?} has {} elements",
            a.shape.dims, old_elements, new_shape.dims, new_elements
        ));
    }

    // Zero-copy: share the underlying data Arc
    let data = a.data.clone();
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    // Note: We wrap metadata in Arc here since from_shared expects it
    Tensor::from_shared(new_id, new_shape, data, std::sync::Arc::new(metadata))
}

/// Flatten a tensor to rank-1 (vector)
pub fn flatten(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    flatten_with_timestamp(a, new_id, chrono::Utc::now())
}

pub fn flatten_with_timestamp(
    a: &Tensor,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let total_elements = a.shape.num_elements();

    let data = if let Some(slice) = a.as_contiguous_slice() {
        slice.to_vec()
    } else {
        // Materialize strided tensor into contiguous vector
        // This is necessary because flatten produces a 1D contiguous tensor
        let mut collected = Vec::with_capacity(total_elements);

        // Use rank-specific iterators for efficiency, similar to other kernels
        // Or generic iteration since flatten is less critical for perf than matmul?
        // Let's use a simple generic approach for now to ensure correctness
        // We can optimize with specific rank handling later if needed.

        match a.shape.rank() {
            2 => {
                let rows = a.shape.dims[0];
                let cols = a.shape.dims[1];
                let a_data = a.data_ref();
                for i in 0..rows {
                    let row_base = a.offset + i * a.strides[0];
                    for j in 0..cols {
                        collected.push(a_data[row_base + j * a.strides[1]]);
                    }
                }
            }
            _ => {
                // Generalized N-dims materialization for flatten
                let rank = a.shape.rank();
                let mut current_indices = vec![0; rank];
                let mut a_off = a.offset;
                let a_data = a.data_ref();

                for _ in 0..total_elements {
                    collected.push(a_data[a_off]);

                    for j in (0..rank).rev() {
                        current_indices[j] += 1;
                        if current_indices[j] < a.shape.dims[j] {
                            a_off += a.strides[j];
                            break;
                        } else {
                            a_off -= (current_indices[j] - 1) * a.strides[j];
                            current_indices[j] = 0;
                        }
                    }
                }
            }
        }
        collected
    };

    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    let shape = Shape::new(vec![total_elements]);
    Tensor::new(new_id, shape, data, metadata)
}

/// Slice a tensor along a dimension
/// Returns a new tensor with elements from start (inclusive) to end (exclusive)
pub fn slice(
    a: &Tensor,
    dim: usize,
    start: usize,
    end: usize,
    new_id: TensorId,
) -> Result<Tensor, String> {
    slice_with_timestamp(a, dim, start, end, new_id, chrono::Utc::now())
}

pub fn slice_with_timestamp(
    a: &Tensor,
    dim: usize,
    start: usize,
    end: usize,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    if dim >= a.shape.rank() {
        return Err(format!(
            "Dimension {} out of bounds for tensor with rank {}",
            dim,
            a.shape.rank()
        ));
    }

    if start >= end {
        return Err(format!(
            "Invalid slice range: start {} >= end {}",
            start, end
        ));
    }

    if end > a.shape.dims[dim] {
        return Err(format!(
            "Slice end {} exceeds dimension size {}",
            end, a.shape.dims[dim]
        ));
    }

    let mut new_dims = a.shape.dims.clone();
    new_dims[dim] = end - start;
    let new_shape = Shape::new(new_dims);

    // Calculate new offset
    // offset += start * stride[dim]
    let new_offset = a.offset + start * a.strides[dim];

    // Strides remain the same!
    // Even if we stick to a subset of columns, the stride to get to the next row
    // is still the full width of the ORIGINAL allocated row.

    // Zero-copy: share data
    let data = a.data.clone();
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);

    Tensor::from_shared_strided(
        new_id,
        new_shape,
        data,
        std::sync::Arc::new(metadata),
        a.strides.clone(),
        new_offset,
    )
}

/// Index into a tensor to get a single element
pub fn index(a: &Tensor, indices: &[usize]) -> Result<f32, String> {
    if indices.len() != a.shape.rank() {
        return Err(format!(
            "Index dimension mismatch: tensor has rank {}, got {} indices",
            a.shape.rank(),
            indices.len()
        ));
    }

    // Check bounds
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= a.shape.dims[i] {
            return Err(format!(
                "Index {} out of bounds for dimension {} (size {})",
                idx, i, a.shape.dims[i]
            ));
        }
    }

    // Compute flat index using explicit strides and offset
    let mut flat_idx = a.offset;
    for (i, &idx) in indices.iter().enumerate() {
        flat_idx += idx * a.strides[i];
    }

    Ok(a.data_ref()[flat_idx])
}

/// Index into a tensor and return result as a scalar tensor
pub fn index_to_scalar(a: &Tensor, indices: &[usize], new_id: TensorId) -> Result<Tensor, String> {
    index_to_scalar_with_timestamp(a, indices, new_id, chrono::Utc::now())
}

pub fn index_to_scalar_with_timestamp(
    a: &Tensor,
    indices: &[usize],
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    let value = index(a, indices)?;
    // Create scalar tensor (rank 0)
    let shape = Shape::new(vec![]);
    let data = vec![value];
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, shape, data, metadata)
}

/// Slice specification for a single dimension
#[derive(Debug, Clone)]
pub enum SliceSpec {
    /// Single index: reduces dimension
    Index(usize),
    /// Range: start..end (inclusive start, exclusive end)
    Range(usize, usize),
    /// Wildcard: entire dimension
    All,
}

/// Multi-dimensional slicing
/// Supports: m[0, *], m[0:2, :], m[*, 1], etc.
pub fn slice_multi(a: &Tensor, specs: &[SliceSpec], new_id: TensorId) -> Result<Tensor, String> {
    if specs.len() != a.shape.rank() {
        return Err(format!(
            "Slice spec dimension mismatch: tensor has rank {}, got {} specs",
            a.shape.rank(),
            specs.len()
        ));
    }

    // For rank-1 (vectors)
    if a.shape.rank() == 1 {
        match &specs[0] {
            SliceSpec::Index(idx) => {
                // Single element -> scalar
                index_to_scalar(a, &[*idx], new_id)
            }
            SliceSpec::Range(start, end) => {
                // Range -> vector slice
                slice(a, 0, *start, *end, new_id)
            }
            SliceSpec::All => {
                // Entire vector -> copy
                Ok(a.clone())
            }
        }
    }
    // For rank-2 (matrices)
    else if a.shape.rank() == 2 {
        let rows = a.shape.dims[0];
        let cols = a.shape.dims[1];

        match (&specs[0], &specs[1]) {
            // Single element: m[i, j]
            (SliceSpec::Index(i), SliceSpec::Index(j)) => index_to_scalar(a, &[*i, *j], new_id),
            // Row slice: m[i, *] or m[i, :]
            (SliceSpec::Index(i), SliceSpec::All) => {
                // Extract row as vector
                let mut data = Vec::with_capacity(cols);
                for j in 0..cols {
                    data.push(a.data_ref()[i * cols + j]);
                }
                let shape = Shape::new(vec![cols]);
                let metadata = TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata)
            }
            // Column slice: m[*, j] or m[:, j]
            (SliceSpec::All, SliceSpec::Index(j)) => {
                // Extract column as vector
                let mut data = Vec::with_capacity(rows);
                for i in 0..rows {
                    data.push(a.data_ref()[i * cols + j]);
                }
                let shape = Shape::new(vec![rows]);
                let metadata = TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata)
            }
            // Row range: m[i:k, *]
            (SliceSpec::Range(start, end), SliceSpec::All) => slice(a, 0, *start, *end, new_id),
            // Column range: m[*, j:k]
            (SliceSpec::All, SliceSpec::Range(start, end)) => slice(a, 1, *start, *end, new_id),
            // Submatrix: m[i:k, j:l]
            (SliceSpec::Range(row_start, row_end), SliceSpec::Range(col_start, col_end)) => {
                let new_rows = row_end - row_start;
                let new_cols = col_end - col_start;
                let mut data = Vec::with_capacity(new_rows * new_cols);

                for i in *row_start..*row_end {
                    for j in *col_start..*col_end {
                        data.push(a.data_ref()[i * cols + j]);
                    }
                }

                let shape = Shape::new(vec![new_rows, new_cols]);
                let metadata = TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata)
            }
            // Full matrix: m[*, *]
            (SliceSpec::All, SliceSpec::All) => Ok(a.clone()),
            // Mixed cases with ranges
            (SliceSpec::Index(i), SliceSpec::Range(start, end)) => {
                // Row i, columns start:end -> vector
                let new_cols = end - start;
                let mut data = Vec::with_capacity(new_cols);
                for j in *start..*end {
                    data.push(a.data_ref()[i * cols + j]);
                }
                let shape = Shape::new(vec![new_cols]);
                let metadata = TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata)
            }
            (SliceSpec::Range(start, end), SliceSpec::Index(j)) => {
                // Rows start:end, column j -> vector
                let new_rows = end - start;
                let mut data = Vec::with_capacity(new_rows);
                for i in *start..*end {
                    data.push(a.data_ref()[i * cols + j]);
                }
                let shape = Shape::new(vec![new_rows]);
                let metadata = TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata)
            }
        }
    } else {
        Err(format!(
            "Multi-dimensional slicing not yet implemented for rank-{} tensors",
            a.shape.rank()
        ))
    }
}

/// Stack a list of tensors along a new axis (0 for now)
/// All tensors must have the same shape.
/// Result rank = Input rank + 1
pub fn stack(tensors: &[&Tensor], axis: usize, new_id: TensorId) -> Result<Tensor, String> {
    stack_with_timestamp(tensors, axis, new_id, chrono::Utc::now())
}

pub fn stack_with_timestamp(
    tensors: &[&Tensor],
    axis: usize,
    new_id: TensorId,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("Cannot stack empty list of tensors".into());
    }

    let first_shape = &tensors[0].shape;
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape.dims != first_shape.dims {
            return Err(format!(
                "Tensor at index {} has different shape {:?} compared to first tensor {:?}",
                i, t.shape.dims, first_shape.dims
            ));
        }
    }

    // New shape: insert 'tensors.len()' at 'axis' position
    // For MVP, simplified: only axis 0 (stacking rows)
    if axis != 0 {
        return Err("Stack only supported on axis 0 for now".into());
    }

    let mut new_dims = vec![tensors.len()];
    new_dims.extend_from_slice(&first_shape.dims);

    let mut new_data = Vec::with_capacity(new_dims.iter().product());

    for t in tensors {
        new_data.extend_from_slice(t.data_ref());
    }

    let new_shape = Shape::new(new_dims);
    let metadata = TensorMetadata::new_with_timestamp(new_id, None, timestamp);
    Tensor::new(new_id, new_shape, new_data, metadata)
}

// ============================================================================
// LAZY EVALUATION ENGINE
// ============================================================================

use crate::core::tensor::Expression;

/// Evalúa recursivamente una expresión para producir un Tensor materializado.
/// Este es el núcleo del sistema de evaluación perezosa de LINAL.
pub fn evaluate_expression(
    expr: &Expression,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<Tensor, String> {
    match expr {
        Expression::Literal(t) => Ok(t.clone()),
        Expression::Add(left, right) => {
            let lt = evaluate_expression(left, timestamp)?;
            let rt = evaluate_expression(right, timestamp)?;
            add_with_timestamp(&lt, &rt, TensorId::new(), timestamp)
        }
        Expression::Sub(left, right) => {
            let lt = evaluate_expression(left, timestamp)?;
            let rt = evaluate_expression(right, timestamp)?;
            sub_with_timestamp(&lt, &rt, TensorId::new(), timestamp)
        }
        Expression::Multiply(left, right) => {
            let lt = evaluate_expression(left, timestamp)?;
            let rt = evaluate_expression(right, timestamp)?;
            multiply_with_timestamp(&lt, &rt, TensorId::new(), timestamp)
        }
        Expression::MatMul(left, right) => {
            let lt = evaluate_expression(left, timestamp)?;
            let rt = evaluate_expression(right, timestamp)?;
            matmul_with_timestamp(&lt, &rt, TensorId::new(), timestamp)
        }
        Expression::ScalarMul(inner, s) => {
            let t = evaluate_expression(inner, timestamp)?;
            scalar_mul_with_timestamp(&t, *s, TensorId::new(), timestamp)
        }
        Expression::Divide(left, right) => {
            let lt = evaluate_expression(left, timestamp)?;
            let rt = evaluate_expression(right, timestamp)?;
            divide_with_timestamp(&lt, &rt, TensorId::new(), timestamp)
        }
        Expression::Normalize(inner) => {
            let t = evaluate_expression(inner, timestamp)?;
            normalize_1d_with_timestamp(&t, TensorId::new(), timestamp)
        }
        Expression::Sum(inner) => {
            let t = evaluate_expression(inner, timestamp)?;
            sum_with_timestamp(&t, TensorId::new(), timestamp)
        }
        Expression::Mean(inner) => {
            let t = evaluate_expression(inner, timestamp)?;
            mean_with_timestamp(&t, TensorId::new(), timestamp)
        }
        Expression::Stdev(inner) => {
            let t = evaluate_expression(inner, timestamp)?;
            stdev_with_timestamp(&t, TensorId::new(), timestamp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{Shape, Tensor, TensorId};

    fn tensor_1d(_id: u64, vals: Vec<f32>) -> Tensor {
        let id = TensorId::new();
        let shape = Shape::new(vec![vals.len()]);
        let metadata = TensorMetadata::new(id, None);
        Tensor::new(id, shape, vals, metadata).unwrap()
    }

    #[test]
    fn test_add_simple() {
        let a = tensor_1d(1, vec![1.0, 2.0, 3.0]);
        let b = tensor_1d(2, vec![4.0, 5.0, 6.0]);
        let result = add(&a, &b, TensorId::new()).unwrap();

        assert_eq!(result.shape.dims, vec![3]);
        assert_eq!(result.data_ref(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_multiply_and_divide() {
        let a = tensor_1d(1, vec![1.0, 2.0, 3.0]);
        let b = tensor_1d(2, vec![4.0, 5.0, 6.0]);

        let prod = multiply(&a, &b, TensorId::new()).unwrap();

        assert_eq!(prod.data_ref(), &[4.0, 10.0, 18.0]);

        let ratio = divide(&b, &a, TensorId::new()).unwrap();

        assert_eq!(ratio.data_ref(), &[4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_cosine_and_distance_and_normalize() {
        let a = tensor_1d(1, vec![1.0, 0.0, 0.0]);
        let b = tensor_1d(2, vec![1.0, 1.0, 0.0]);

        let sim = cosine_similarity_1d(&a, &b).unwrap();
        let expected_sim = 1.0 / 2f32.sqrt();
        assert!((sim - expected_sim).abs() < 1e-6);

        let dist = distance_1d(&a, &b).unwrap();
        let expected_dist =
            ((1.0_f32 - 1.0_f32).powi(2) + (0.0_f32 - 1.0_f32).powi(2) + 0.0f32).sqrt();
        assert!((dist - expected_dist).abs() < 1e-6);

        let n = normalize_1d(&b, TensorId::new()).unwrap();

        let norm_n = l2_norm_1d(&n).unwrap();
        assert!((norm_n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stack() {
        let a = tensor_1d(1, vec![1.0, 2.0]);
        let b = tensor_1d(2, vec![3.0, 4.0]);

        // Stack [2] and [2] -> [2, 2] matrix
        let stacked = stack(&[&a, &b], 0, TensorId::new()).unwrap();

        assert_eq!(stacked.shape.dims, vec![2, 2]);
        assert_eq!(stacked.data_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
