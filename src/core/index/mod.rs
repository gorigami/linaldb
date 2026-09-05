use crate::core::tensor::Tensor;
use crate::core::value::Value;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Types of supported indices
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    /// Exact match index (hash map based)
    Hash,
    /// Vector similarity index (linear scan for MVP, HNSW later)
    Vector,
}

/// A persistable record of "column X has an index of type Y", independent of
/// the index's in-memory contents. Indices themselves (`Box<dyn Index>`) are
/// never serialized directly (`Dataset.indices` is `#[serde(skip)]`) because
/// their row-id contents are only meaningful alongside the exact row vector
/// they were built from. Instead, `SAVE DATASET` persists these definitions
/// and `LOAD DATASET` rebuilds each index from the freshly loaded rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub column: String,
    pub index_type: IndexType,
}

/// Core trait for all index implementations
/// Indices store a mapping from values/vectors to row IDs
pub trait Index: Send + Sync + Debug {
    /// Add a new entry to the index
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String>;

    /// Find row IDs that exactly match the given value
    /// Returns empty vector if no match or if index doesn't support exact lookup
    fn lookup(&self, value: &Value) -> Result<Vec<usize>, String>;

    /// Find k nearest neighbors to the query vector
    /// Returns vector of (row_id, score) tuples
    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String>;

    /// Find every entry whose similarity to `query` passes `threshold`
    /// (`>` if `strict`, `>=` otherwise). Unlike `search`, this must be
    /// exact: it answers a boolean predicate (`WHERE COSINE_SIM(...) >
    /// threshold`), not a top-k ranking, so an implementation may only skip
    /// work it can prove cannot contain a passing entry. The default here is
    /// the safe baseline (score everything via `search`, then filter) and is
    /// only overridden by index types that can safely skip more.
    fn search_threshold(
        &self,
        query: &Tensor,
        threshold: f32,
        strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        let all = self.search(query, usize::MAX)?;
        Ok(all
            .into_iter()
            .filter(|(_, score)| {
                if strict {
                    *score > threshold
                } else {
                    *score >= threshold
                }
            })
            .collect())
    }

    /// Get the type of this index
    fn index_type(&self) -> IndexType;

    /// Clone the index box
    fn box_clone(&self) -> Box<dyn Index>;

    /// Called once after a batch of `add()` calls completes (full backfill on
    /// `CREATE INDEX`, or rebuild on `LOAD DATASET`) so an index can compute
    /// any batch-derived structure (e.g. k-means clusters) from everything
    /// added so far. No-op by default.
    fn build(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Clone for Box<dyn Index> {
    fn clone(&self) -> Box<dyn Index> {
        self.box_clone()
    }
}

// Re-export specific implementations
pub mod hash;
pub mod vector;
