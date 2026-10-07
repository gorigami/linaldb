//! Sparse vectors (`SparseVector(dim)` columns): only the nonzero entries,
//! as strictly increasing indices with their `f32` values -- e.g. a finely
//! binned spectrum that is mostly zeros.
//!
//! Every operation visits entries in index order and skips zeros, which is
//! exactly what the dense loops compute (adding a `0.0` product doesn't
//! change an f32 sum), so `COSINE_SIM`/`DOT`/`L2_NORM` on a `SparseVector`
//! give the same bits as on its dense equivalent.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVec {
    dim: usize,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl PartialEq for SparseVec {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.indices == other.indices
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for SparseVec {}

impl std::hash::Hash for SparseVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dim.hash(state);
        self.indices.hash(state);
        for v in &self.values {
            v.to_bits().hash(state);
        }
    }
}

impl SparseVec {
    /// Validates: same number of indices and values, indices strictly
    /// increasing (no duplicates) and below `dim`, values finite.
    pub fn new(dim: usize, indices: Vec<u32>, values: Vec<f32>) -> Result<Self, String> {
        if indices.len() != values.len() {
            return Err(format!(
                "sparse vector has {} indices but {} values",
                indices.len(),
                values.len()
            ));
        }
        for (k, w) in indices.windows(2).enumerate() {
            if w[1] == w[0] {
                return Err(format!(
                    "sparse vector has duplicate index {} (entry {})",
                    w[0],
                    k + 1
                ));
            }
            if w[1] < w[0] {
                return Err(format!(
                    "sparse vector indices must be increasing: {} comes after {} (entry {})",
                    w[1],
                    w[0],
                    k + 1
                ));
            }
        }
        if let Some(&last) = indices.last() {
            if last as usize >= dim {
                return Err(format!(
                    "sparse vector index {} is out of range for dimension {}",
                    last, dim
                ));
            }
        }
        if let Some(k) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!("sparse vector value at entry {} is not finite", k));
        }
        Ok(Self {
            dim,
            indices,
            values,
        })
    }

    /// The nonzero entries of a dense vector.
    pub fn from_dense(v: &[f32]) -> Self {
        let (indices, values) = v
            .iter()
            .enumerate()
            .filter(|(_, x)| **x != 0.0)
            .map(|(i, x)| (i as u32, *x))
            .unzip();
        Self {
            dim: v.len(),
            indices,
            values,
        }
    }

    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.dim];
        for (&i, &v) in self.indices.iter().zip(&self.values) {
            out[i as usize] = v;
        }
        out
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    pub fn dot(&self, other: &Self) -> f32 {
        let (mut i, mut j, mut sum) = (0, 0, 0.0f32);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    pub fn dot_dense(&self, dense: &[f32]) -> f32 {
        self.indices
            .iter()
            .zip(&self.values)
            .map(|(&i, &v)| v * dense[i as usize])
            .sum()
    }

    /// Each value divided by `divisor` -- `NORMALIZE` divides like the dense
    /// path does (multiplying by `1 / norm` would round differently).
    pub fn divide(&self, divisor: f32) -> Self {
        Self {
            dim: self.dim,
            indices: self.indices.clone(),
            values: self.values.iter().map(|v| v / divisor).collect(),
        }
    }

    pub fn scale(&self, factor: f32) -> Self {
        Self {
            dim: self.dim,
            indices: self.indices.clone(),
            values: self.values.iter().map(|v| v * factor).collect(),
        }
    }
}

impl std::fmt::Display for SparseVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sparse({}) {{", self.dim)?;
        for (k, (i, v)) in self.indices.iter().zip(&self.values).enumerate() {
            if k > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", i, v)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_dense_bit_for_bit() {
        let a_dense = vec![0.0, 0.3, 0.0, 0.0, -1.25, 0.0, 7.0, 0.0];
        let b_dense = vec![1.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.1, 0.0];
        let (a, b) = (
            SparseVec::from_dense(&a_dense),
            SparseVec::from_dense(&b_dense),
        );
        let dense_dot: f32 = a_dense.iter().zip(&b_dense).map(|(x, y)| x * y).sum();
        assert_eq!(a.dot(&b).to_bits(), dense_dot.to_bits());
        assert_eq!(a.dot_dense(&b_dense).to_bits(), dense_dot.to_bits());
        let dense_norm = a_dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(a.l2_norm().to_bits(), dense_norm.to_bits());
        assert_eq!(a.to_dense(), a_dense);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(SparseVec::new(4, vec![1, 1], vec![1.0, 2.0]).is_err());
        assert!(SparseVec::new(4, vec![2, 1], vec![1.0, 2.0]).is_err());
        assert!(SparseVec::new(4, vec![4], vec![1.0]).is_err());
        assert!(SparseVec::new(4, vec![1], vec![f32::NAN]).is_err());
        assert!(SparseVec::new(4, vec![1], vec![]).is_err());
        assert!(SparseVec::new(4, vec![0, 3], vec![1.0, 2.0]).is_ok());
    }
}
