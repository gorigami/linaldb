use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a node in the dataset derivation DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: Uuid,
    pub dataset_name: String,
    pub dataset_hash: String,
    pub operation: String,
    pub parents: Vec<Uuid>,
    pub engine_version: String,
}

/// A DAG representing the full derivation and dependency history of a dataset.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetLineage {
    pub nodes: Vec<LineageNode>,
    /// Non-fatal issues raised while building this lineage — e.g. a
    /// connector silently skipping a field it couldn't read (unsupported
    /// dtype) or reconcile (shape/length mismatch against the rest of the
    /// file). `#[serde(default)]` so lineage.json files persisted before
    /// this field existed still deserialize cleanly.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl DatasetLineage {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: LineageNode) {
        self.nodes.push(node);
    }

    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}
