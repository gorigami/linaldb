//! Contiguous vector storage shared by the vector indexes (`VectorIndex`,
//! `HnswIndex`): every indexed vector lives once, back to back in a single
//! `Vec<f32>`, with its row id and L2 norm alongside. Replaces the earlier
//! one-`Tensor`-per-vector layout, which cost a heap allocation plus tensor
//! metadata (a UUID, a timestamp, lineage) for every vector.

use crate::core::value::Value;

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

/// Cloning is O(1): the buffers are shared (`Arc`) and copied only when a
/// clone is then written to, so cloning a dataset (`SAVE DATASET`, `SHOW`)
/// doesn't copy every indexed vector.
#[derive(Debug, Clone, Default)]
pub(crate) struct FlatVectors {
    dim: usize,
    data: std::sync::Arc<Vec<f32>>,
    row_ids: std::sync::Arc<Vec<usize>>,
    norms: std::sync::Arc<Vec<f32>>,
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
    /// than a `Vector` of the store's dimension is an error.
    pub fn add_value(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Vector(v) => self.push(row_id, v),
            Value::Null => Ok(()),
            other => Err(format!("Cannot index {:?} as Vector", other.value_type())),
        }
    }

    pub fn push(&mut self, row_id: usize, v: &[f32]) -> Result<(), String> {
        if self.row_ids.is_empty() && self.dim == 0 {
            self.dim = v.len();
        } else if v.len() != self.dim {
            return Err(format!(
                "vector index: dimension mismatch -- the index holds {}-dimensional vectors, got {}",
                self.dim,
                v.len()
            ));
        }
        std::sync::Arc::make_mut(&mut self.data).extend_from_slice(v);
        std::sync::Arc::make_mut(&mut self.row_ids).push(row_id);
        std::sync::Arc::make_mut(&mut self.norms).push(l2_norm(v));
        Ok(())
    }

    #[inline]
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
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
        cosine_with_norms(query, query_norm, self.vector(i), self.norms[i])
    }

    /// Cosine similarity between two stored vectors.
    #[inline]
    pub fn cosine_between(&self, i: usize, j: usize) -> f32 {
        cosine_with_norms(self.vector(i), self.norms[i], self.vector(j), self.norms[j])
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
        self.data.capacity() * std::mem::size_of::<f32>()
            + self.row_ids.capacity() * std::mem::size_of::<usize>()
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
