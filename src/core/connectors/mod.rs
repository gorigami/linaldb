pub mod csv_connector;
pub mod hdf5_connector;
pub mod numpy_connector;
pub mod zarr_connector;

use crate::core::dataset::{DatasetLineage, DatasetSchema};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use thiserror::Error;

/// Arrow `Field` metadata key connectors use to stash a column's original
/// N-D array shape (comma-joined dims), so `record_batch_to_tensors` can
/// rebuild the real `Shape` instead of assuming a flat `Vector`.
pub const SHAPE_METADATA_KEY: &str = "linal.shape";

/// Build an Arrow `Field`, attaching the original array shape as metadata
/// when it's genuinely multi-dimensional (`dims.len() > 1`). Rank-0/1 data
/// gets a plain `Field` byte-identical to before this existed, since a 1D
/// shape is already exactly what `record_batch_to_tensors` assumes by
/// default.
pub fn field_with_shape(name: &str, data_type: DataType, nullable: bool, dims: &[usize]) -> Field {
    let field = Field::new(name, data_type, nullable);
    if dims.len() > 1 {
        let shape_str = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut metadata = HashMap::new();
        metadata.insert(SHAPE_METADATA_KEY.to_string(), shape_str);
        field.with_metadata(metadata)
    } else {
        field
    }
}

/// Parse the shape metadata `field_with_shape` attaches, if present and
/// well-formed. Returns `None` on anything missing or malformed so callers
/// can safely fall back to the flat-`Vector` assumption.
pub fn read_shape_metadata(field: &Field) -> Option<Vec<usize>> {
    let raw = field.metadata().get(SHAPE_METADATA_KEY)?;
    let dims: Option<Vec<usize>> = raw.split(',').map(|s| s.parse::<usize>().ok()).collect();
    dims.filter(|d| !d.is_empty())
}

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Feature not supported: {0}")]
    Unsupported(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Trait for dataset connectors that can read various scientific and standard formats.
pub trait Connector: Send + Sync {
    /// Unique name of the connector (e.g., "csv", "hdf5")
    fn name(&self) -> &str;

    /// Check if this connector can handle the given path/URI
    fn can_handle(&self, path: &str) -> bool;

    /// Read the dataset from the given path
    fn read_dataset(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError>;

    /// Inspect the dataset to get its schema without reading all data
    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError>;
}

/// Registry for managing available connectors
pub struct ConnectorRegistry {
    connectors: Vec<Box<dyn Connector>>,
}

impl ConnectorRegistry {
    pub fn new() -> Self {
        Self {
            connectors: Vec::new(),
        }
    }

    pub fn register(&mut self, connector: Box<dyn Connector>) {
        self.connectors.push(connector);
    }

    pub fn find_connector(&self, path: &str) -> Option<&dyn Connector> {
        self.connectors
            .iter()
            .find(|c| c.can_handle(path))
            .map(|c| c.as_ref())
    }

    pub fn list_connectors(&self) -> Vec<&str> {
        self.connectors.iter().map(|c| c.name()).collect()
    }
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
