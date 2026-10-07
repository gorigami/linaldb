use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Statistics for a dataset, including row counts and column-level summaries.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetStats {
    pub row_count: u64,
    /// stats per column name, in name order (a `BTreeMap`, so `stats.json`
    /// is byte-identical for identical data; it was a `HashMap` before
    /// v0.1.93, whose order changed run to run)
    pub columns: BTreeMap<String, ColumnStats>,
}

/// Statistics for a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f32>,
    pub sparsity: Option<f32>,
    pub null_count: u64,
    /// For tensor columns: batch size, etc.
    pub tensor_shape: Option<Vec<usize>>,
}

impl DatasetStats {
    pub fn new(row_count: u64) -> Self {
        Self {
            row_count,
            columns: BTreeMap::new(),
        }
    }

    pub fn add_column_stats(&mut self, name: String, stats: ColumnStats) {
        self.columns.insert(name, stats);
    }
}
