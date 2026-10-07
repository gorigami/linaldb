//! Contiguous vector storage shared by the vector indexes (`VectorIndex`,
//! `HnswIndex`): every indexed vector lives once, back to back in a single
//! `Vec<f32>`, with its row id and L2 norm alongside. Replaces the earlier
//! one-`Tensor`-per-vector layout, which cost a heap allocation plus tensor
//! metadata (a UUID, a timestamp, lineage) for every vector.

use crate::core::quant::{f16_to_f32, i8_to_f32, QuantVec};
use crate::core::value::Value;
use std::borrow::Cow;
use std::sync::Arc;

/// Cosine similarity with precomputed norms. Computes exactly what
/// `query::physical`'s `COSINE_SIM` does -- same element order for the dot
/// product, same norm formula -- so an index-accelerated predicate and a
/// plain `WHERE COSINE_SIM(...)` agree bit for bit at the threshold.
#[inline]
pub(crate) fn cosine_with_norms(a: &[f32], norm_a: f32, b: &[f32], norm_b: f32) -> f32 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

#[inline]
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// How the stored vectors are encoded: plain `f32`, or quantized for a
/// `Vector(d, F16|I8)` column (`core::quant`). Decided by the first vector
/// added.
#[derive(Debug, Clone)]
enum Storage {
    F32(Arc<Vec<f32>>),
    F16(Arc<Vec<u16>>),
    I8 {
        data: Arc<Vec<i8>>,
        scales: Arc<Vec<f32>>,
    },
}

impl Default for Storage {
    fn default() -> Self {
        Storage::F32(Arc::default())
    }
}

/// Cloning is O(1): the buffers are shared (`Arc`) and copied only when a
/// clone is then written to, so cloning a dataset (`SAVE DATASET`, `SHOW`)
/// doesn't copy every indexed vector.
#[derive(Debug, Clone, Default)]
pub(crate) struct FlatVectors {
    dim: usize,
    storage: Storage,
    row_ids: Arc<Vec<usize>>,
    norms: Arc<Vec<f32>>,
}

impl FlatVectors {
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Adds a row's value. `Null` is skipped (not indexed); anything other
    /// than a `Vector` (or quantized vector) of the store's dimension is an
    /// error.
    pub fn add_value(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Vector(v) => self.push(row_id, v),
            Value::QVector(q) => self.push_quantized(row_id, q),
            Value::Null => Ok(()),
            Value::SparseVector(_) => Err(
                "vector indexes (IVF, HNSW) need a dense Vector column; search a SparseVector \
                 column exactly with SEARCH ... PREFILTER or COSINE_SIM in SELECT/WHERE"
                    .to_string(),
            ),
            other => Err(format!("Cannot index {:?} as Vector", other.value_type())),
        }
    }

    fn check_dim(&mut self, len: usize) -> Result<(), String> {
        if self.row_ids.is_empty() && self.dim == 0 {
            self.dim = len;
        } else if len != self.dim {
            return Err(format!(
                "vector index: dimension mismatch -- the index holds {}-dimensional vectors, got {}",
                self.dim, len
            ));
        }
        Ok(())
    }

    pub fn push(&mut self, row_id: usize, v: &[f32]) -> Result<(), String> {
        self.check_dim(v.len())?;
        let Storage::F32(data) = &mut self.storage else {
            return Err("vector index: this index holds quantized vectors".to_string());
        };
        Arc::make_mut(data).extend_from_slice(v);
        Arc::make_mut(&mut self.row_ids).push(row_id);
        Arc::make_mut(&mut self.norms).push(l2_norm(v));
        Ok(())
    }

    fn push_quantized(&mut self, row_id: usize, q: &QuantVec) -> Result<(), String> {
        self.check_dim(q.len())?;
        if self.row_ids.is_empty() {
            self.storage = match q {
                QuantVec::F16(_) => Storage::F16(Arc::default()),
                QuantVec::I8 { .. } => Storage::I8 {
                    data: Arc::default(),
                    scales: Arc::default(),
                },
            };
        }
        match (&mut self.storage, q) {
            (Storage::F16(data), QuantVec::F16(bits)) => {
                Arc::make_mut(data).extend_from_slice(bits)
            }
            (Storage::I8 { data, scales }, QuantVec::I8 { scale, data: q }) => {
                Arc::make_mut(data).extend_from_slice(q);
                Arc::make_mut(scales).push(*scale);
            }
            _ => return Err("vector index: mixed vector encodings in one column".to_string()),
        }
        Arc::make_mut(&mut self.row_ids).push(row_id);
        Arc::make_mut(&mut self.norms).push(l2_norm(&q.dequantize()));
        Ok(())
    }

    /// Stored vector `i` as `f32` -- borrowed for an `f32` store, decoded
    /// (with the same formula as `QuantVec::dequantize`) otherwise.
    pub fn values(&self, i: usize) -> Cow<'_, [f32]> {
        let r = i * self.dim..(i + 1) * self.dim;
        match &self.storage {
            Storage::F32(d) => Cow::Borrowed(&d[r]),
            Storage::F16(d) => Cow::Owned(d[r].iter().map(|h| f16_to_f32(*h)).collect()),
            Storage::I8 { data, scales } => {
                let s = scales[i];
                Cow::Owned(data[r].iter().map(|q| i8_to_f32(*q, s)).collect())
            }
        }
    }

    /// `query · stored[i]`, in the order `COSINE_SIM` multiplies and sums.
    #[inline]
    fn dot_query(&self, i: usize, query: &[f32]) -> f32 {
        let r = i * self.dim..(i + 1) * self.dim;
        match &self.storage {
            Storage::F32(d) => query.iter().zip(&d[r]).map(|(a, b)| a * b).sum(),
            Storage::F16(d) => query
                .iter()
                .zip(&d[r])
                .map(|(a, b)| a * f16_to_f32(*b))
                .sum(),
            Storage::I8 { data, scales } => {
                let s = scales[i];
                query
                    .iter()
                    .zip(&data[r])
                    .map(|(a, b)| a * i8_to_f32(*b, s))
                    .sum()
            }
        }
    }

    #[inline]
    fn dot_between(&self, i: usize, j: usize) -> f32 {
        let (ri, rj) = (
            i * self.dim..(i + 1) * self.dim,
            j * self.dim..(j + 1) * self.dim,
        );
        match &self.storage {
            Storage::F32(d) => d[ri].iter().zip(&d[rj]).map(|(a, b)| a * b).sum(),
            Storage::F16(d) => d[ri]
                .iter()
                .zip(&d[rj])
                .map(|(a, b)| f16_to_f32(*a) * f16_to_f32(*b))
                .sum(),
            Storage::I8 { data, scales } => {
                let (si, sj) = (scales[i], scales[j]);
                data[ri]
                    .iter()
                    .zip(&data[rj])
                    .map(|(a, b)| i8_to_f32(*a, si) * i8_to_f32(*b, sj))
                    .sum()
            }
        }
    }

    /// Adds stored vector `i` into `sums` element-wise (k-means centroids).
    pub fn add_to(&self, i: usize, sums: &mut [f32]) {
        for (s, v) in sums.iter_mut().zip(self.values(i).iter()) {
            *s += v;
        }
    }

    pub fn row_id(&self, i: usize) -> usize {
        self.row_ids[i]
    }

    #[inline]
    pub fn norm(&self, i: usize) -> f32 {
        self.norms[i]
    }

    /// Cosine similarity between stored vector `i` and `query` (whose norm
    /// the caller precomputed).
    #[inline]
    pub fn cosine(&self, i: usize, query: &[f32], query_norm: f32) -> f32 {
        let norm = self.norms[i];
        if query_norm == 0.0 || norm == 0.0 {
            return 0.0;
        }
        self.dot_query(i, query) / (query_norm * norm)
    }

    /// Cosine similarity between two stored vectors.
    #[inline]
    pub fn cosine_between(&self, i: usize, j: usize) -> f32 {
        let (ni, nj) = (self.norms[i], self.norms[j]);
        if ni == 0.0 || nj == 0.0 {
            return 0.0;
        }
        self.dot_between(i, j) / (ni * nj)
    }

    /// Errors unless `query` has the store's dimension (an empty store
    /// accepts anything: there is nothing to compare against).
    pub fn check_query(&self, query: &[f32]) -> Result<(), String> {
        if !self.is_empty() && query.len() != self.dim {
            return Err(format!(
                "query vector has dimension {}, but the index holds {}-dimensional vectors",
                query.len(),
                self.dim
            ));
        }
        Ok(())
    }

    /// Bytes held, by capacity.
    pub fn memory_bytes(&self) -> usize {
        let data = match &self.storage {
            Storage::F32(d) => d.capacity() * std::mem::size_of::<f32>(),
            Storage::F16(d) => d.capacity() * std::mem::size_of::<u16>(),
            Storage::I8 { data, scales } => data.capacity() + scales.capacity() * 4,
        };
        data + self.row_ids.capacity() * std::mem::size_of::<usize>()
            + self.norms.capacity() * std::mem::size_of::<f32>()
    }

    /// Exact top-`k` over positions `candidates` by cosine similarity to
    /// `query`, as `(row_id, score)`, highest first. Ties keep candidate
    /// order (stable sort), so results are deterministic.
    pub fn top_k(
        &self,
        candidates: impl Iterator<Item = usize>,
        query: &[f32],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let qn = l2_norm(query);
        let mut scores: Vec<(usize, f32)> = candidates
            .map(|i| (self.row_ids[i], self.cosine(i, query, qn)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
}
