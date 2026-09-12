//! Unified provenance model shared by tensors and datasets.
//!
//! See `LINEAGE_AND_LINALG_PLAN.md`'s "Phase 0 outcome" for the locked design
//! this module implements: a `ProvenanceRecord` describes one operation
//! ("this execution consumed these entities, produced those"). The
//! unification point is this event type, not the underlying data model --
//! `Tensor` and `Dataset` stay structurally different, but both are
//! referenced through the same `ProvenanceEntity` enum.
//!
//! Ancestry resolution (`ProvenanceStore::resolve_ancestry`) walks the log by
//! content hash, not by `TensorId`/dataset-instance id -- those are
//! process-local UUIDs regenerated on `LOAD`, so a hash is the only key that
//! survives a restart.

use crate::core::tensor::{ExecutionId, TensorId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use uuid::Uuid;

/// Identifies one `ProvenanceRecord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProvenanceId(pub Uuid);

impl Default for ProvenanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl ProvenanceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// A tensor or dataset referenced as an input/output of a `ProvenanceRecord`.
/// Carries a content hash so the reference survives process restarts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceEntity {
    Tensor {
        id: TensorId,
        name: Option<String>,
        content_hash: String,
    },
    Dataset {
        name: String,
        content_hash: String,
    },
}

impl ProvenanceEntity {
    pub fn tensor(id: TensorId, name: Option<String>, content_hash: impl Into<String>) -> Self {
        Self::Tensor {
            id,
            name,
            content_hash: content_hash.into(),
        }
    }

    pub fn dataset(name: impl Into<String>, content_hash: impl Into<String>) -> Self {
        Self::Dataset {
            name: name.into(),
            content_hash: content_hash.into(),
        }
    }

    pub fn content_hash(&self) -> &str {
        match self {
            Self::Tensor { content_hash, .. } => content_hash,
            Self::Dataset { content_hash, .. } => content_hash,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Tensor { name, .. } => name.as_deref(),
            Self::Dataset { name, .. } => Some(name.as_str()),
        }
    }

    /// Best-effort human-readable label for text-tree output. A short tensor
    /// id is the fallback only for an *unnamed* tensor -- when a name is
    /// available (the common case), it alone is enough: the content hash
    /// `format_lineage_tree` already prints alongside this would make
    /// `"name (tensor <id>)"` a redundant second identifier for the same
    /// entity.
    pub fn display_name(&self) -> String {
        match self {
            Self::Tensor { name: Some(n), .. } => n.clone(),
            Self::Tensor { name: None, id, .. } => format!("tensor {}", short_uuid(&id.0)),
            Self::Dataset { name, .. } => name.clone(),
        }
    }
}

fn short_uuid(id: &Uuid) -> String {
    id.to_string()[..8].to_string()
}

/// One provenance event: an operation that consumed `inputs` and produced
/// `outputs`. Every field here is required, not an optional add-on (see
/// LINEAGE_AND_LINALG_PLAN.md's locked design decisions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub id: ProvenanceId,
    pub operation: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub inputs: Vec<ProvenanceEntity>,
    pub outputs: Vec<ProvenanceEntity>,
    pub timestamp: DateTime<Utc>,
    pub execution_id: ExecutionId,
    pub engine_version: String,
}

impl ProvenanceRecord {
    pub fn new(operation: impl Into<String>, execution_id: ExecutionId) -> Self {
        Self {
            id: ProvenanceId::new(),
            operation: operation.into(),
            parameters: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            timestamp: Utc::now(),
            execution_id,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn with_param(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    pub fn with_inputs(mut self, inputs: Vec<ProvenanceEntity>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<ProvenanceEntity>) -> Self {
        self.outputs = outputs;
        self
    }
}

/// Real SHA256 over arbitrary bytes -- generalizes
/// `TensorMetadata::compute_hash`'s pattern so datasets get the same real
/// hash tensors already had (fixes the `format!("{name}:{rowcount}")`
/// placeholder in `core/storage.rs` and `dsl/persistence.rs`).
pub fn compute_content_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Real content hash for an Arrow `RecordBatch` -- used by connectors before
/// their data becomes a `core::dataset_legacy::Dataset`. Replaces the
/// empty-string `dataset_hash: "".to_string()` placeholder every connector
/// used to write into its `lineage.json` node.
pub fn record_batch_content_hash(batch: &arrow::record_batch::RecordBatch) -> String {
    use arrow::ipc::writer::StreamWriter;
    let mut buf = Vec::new();
    let result: Result<(), arrow::error::ArrowError> = (|| {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()
    })();
    match result {
        Ok(()) => compute_content_hash(&buf),
        // A batch that fails to serialize to IPC is not a real failure mode
        // for any connector in this engine today; fall back to a
        // deterministic-but-weaker fingerprint over the schema rather than
        // panicking or returning an empty string.
        Err(_) => compute_content_hash(format!("{:?}", batch.schema()).as_bytes()),
    }
}

/// A materialized ancestry tree -- the shape `EXPLAIN LINEAGE`/`SHOW LINEAGE`
/// render, identical whether the root entity is a tensor or a dataset
/// (Phase 2.3's unification proof).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceTree {
    pub entity: ProvenanceEntity,
    pub operation: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub timestamp: Option<DateTime<Utc>>,
    pub execution_id: Option<ExecutionId>,
    pub inputs: Vec<ProvenanceTree>,
}

impl ProvenanceTree {
    /// A leaf node with no known producer -- either genuinely the root of
    /// the ancestry chain, or an entity that predates this feature and has
    /// no recorded history.
    pub fn root(entity: ProvenanceEntity) -> Self {
        Self {
            entity,
            operation: "ROOT".to_string(),
            parameters: BTreeMap::new(),
            timestamp: None,
            execution_id: None,
            inputs: Vec::new(),
        }
    }
}

/// Guards `resolve_ancestry` against a corrupt/cyclic log; real ancestry
/// graphs in this engine are nowhere near this deep.
const MAX_ANCESTRY_DEPTH: usize = 256;

/// Append-only provenance log, one per `DatabaseInstance`, shared across all
/// tensors and datasets in that DB -- the unification point is this single
/// store, not one log per dataset package.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceStore {
    records: Vec<ProvenanceRecord>,
}

impl ProvenanceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, record: ProvenanceRecord) {
        self.records.push(record);
    }

    pub fn records(&self) -> &[ProvenanceRecord] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Most recent record whose `outputs` include an entity with this content
    /// hash (most-recent-wins in case of a hash collision across distinct
    /// operations, which SHA256 makes practically impossible anyway).
    pub fn find_producer(&self, content_hash: &str) -> Option<&ProvenanceRecord> {
        self.records
            .iter()
            .rev()
            .find(|r| r.outputs.iter().any(|o| o.content_hash() == content_hash))
    }

    /// Same idea as `find_producer`, but resolves a specific *entity* (hash
    /// plus optional name) and only considers records at index `< before`,
    /// causally prior to whatever record is currently being resolved.
    /// `resolve_ancestry` uses this (not plain `find_producer`) to walk
    /// inputs.
    ///
    /// Two independent problems this guards against, both real and
    /// reproduced via an actual `DATASET ... FROM ... FILTER` that removes
    /// no rows.
    ///
    /// Self-reference: an operation that doesn't change its data's content
    /// hash (e.g. that no-op `FILTER`) makes a record's own input carry the
    /// *same* hash as its output. An unbounded search would match the
    /// record against itself as its own producer and recurse forever.
    /// `before` strictly decreases on every recursive call (an input can
    /// only be produced earlier in the append-only log than the record
    /// consuming it), which also makes `resolve_ancestry` provably
    /// terminating on its own -- `MAX_ANCESTRY_DEPTH` is a secondary safety
    /// net, not the only thing preventing an infinite walk.
    ///
    /// Cross-entity hash collision: two *differently-named* entities can
    /// legitimately share a content hash (that same no-op `FILTER`: the
    /// filtered dataset and its source are byte-identical). Hash alone
    /// can't tell "the record that produced the dataset I'm resolving" from
    /// "some other, later record that coincidentally produced identical
    /// bytes for a different name" -- so when the entity being resolved has
    /// a name, a record whose matching output shares that name is preferred
    /// over a more-recent hash-only match. Content hash stays the primary,
    /// restart-survivable key (per the locked design); name is only a
    /// tiebreaker among candidates that already match on hash.
    fn find_producer_before(
        &self,
        entity: &ProvenanceEntity,
        before: usize,
    ) -> Option<(usize, &ProvenanceRecord)> {
        let hash = entity.content_hash();
        let slice = &self.records[..before.min(self.records.len())];

        if let Some(name) = entity.name() {
            let by_name = slice.iter().enumerate().rev().find(|(_, r)| {
                r.outputs
                    .iter()
                    .any(|o| o.content_hash() == hash && o.name() == Some(name))
            });
            if by_name.is_some() {
                return by_name;
            }
        }

        slice
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.outputs.iter().any(|o| o.content_hash() == hash))
    }

    pub fn resolve_ancestry(&self, entity: &ProvenanceEntity) -> ProvenanceTree {
        self.resolve_ancestry_bounded(entity, self.records.len(), 0)
    }

    fn resolve_ancestry_bounded(
        &self,
        entity: &ProvenanceEntity,
        before: usize,
        depth: usize,
    ) -> ProvenanceTree {
        if depth >= MAX_ANCESTRY_DEPTH {
            return ProvenanceTree {
                entity: entity.clone(),
                operation: "TRUNCATED (max ancestry depth reached)".to_string(),
                parameters: BTreeMap::new(),
                timestamp: None,
                execution_id: None,
                inputs: Vec::new(),
            };
        }

        match self.find_producer_before(entity, before) {
            Some((idx, record)) => ProvenanceTree {
                entity: entity.clone(),
                operation: record.operation.clone(),
                parameters: record.parameters.clone(),
                timestamp: Some(record.timestamp),
                execution_id: Some(record.execution_id),
                inputs: record
                    .inputs
                    .iter()
                    .map(|i| self.resolve_ancestry_bounded(i, idx, depth + 1))
                    .collect(),
            },
            None => ProvenanceTree::root(entity.clone()),
        }
    }

    /// Overwrite the whole log at `path` (JSON Lines: one record per line).
    pub fn save_jsonl(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut buf = String::new();
        for record in &self.records {
            buf.push_str(
                &serde_json::to_string(record).expect("ProvenanceRecord always serializes"),
            );
            buf.push('\n');
        }
        std::fs::write(path, buf)
    }

    /// Append one record to `path` without rewriting the whole file.
    pub fn append_jsonl(path: impl AsRef<Path>, record: &ProvenanceRecord) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(
            file,
            "{}",
            serde_json::to_string(record).expect("ProvenanceRecord always serializes")
        )
    }

    /// Missing file means an empty, freshly-created DB -- not an error.
    pub fn load_jsonl(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        let mut records = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: ProvenanceRecord = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            records.push(record);
        }
        Ok(Self { records })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_entity(hash: &str) -> ProvenanceEntity {
        ProvenanceEntity::tensor(TensorId::new(), Some("t".to_string()), hash)
    }

    #[test]
    fn content_hash_is_deterministic_and_sensitive_to_input() {
        let a = compute_content_hash(b"hello");
        let b = compute_content_hash(b"hello");
        let c = compute_content_hash(b"hellp");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64); // hex-encoded SHA256
    }

    #[test]
    fn record_serialization_round_trips() {
        let record = ProvenanceRecord::new("SCALE", ExecutionId::new())
            .with_param("factor", 2.5)
            .with_inputs(vec![tensor_entity("h1")])
            .with_outputs(vec![tensor_entity("h2")]);
        let json = serde_json::to_string(&record).unwrap();
        let back: ProvenanceRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.operation, "SCALE");
        assert_eq!(back.parameters.get("factor"), Some(&serde_json::json!(2.5)));
        assert_eq!(back.inputs[0].content_hash(), "h1");
        assert_eq!(back.outputs[0].content_hash(), "h2");
    }

    #[test]
    fn resolve_ancestry_walks_multi_step_chain_by_content_hash() {
        let mut store = ProvenanceStore::new();
        store.append(
            ProvenanceRecord::new("IMPORT", ExecutionId::new())
                .with_outputs(vec![tensor_entity("h1")]),
        );
        store.append(
            ProvenanceRecord::new("SCALE", ExecutionId::new())
                .with_inputs(vec![tensor_entity("h1")])
                .with_outputs(vec![tensor_entity("h2")]),
        );
        store.append(
            ProvenanceRecord::new("NORMALIZE", ExecutionId::new())
                .with_inputs(vec![tensor_entity("h2")])
                .with_outputs(vec![tensor_entity("h3")]),
        );

        let tree = store.resolve_ancestry(&tensor_entity("h3"));
        assert_eq!(tree.operation, "NORMALIZE");
        assert_eq!(tree.inputs[0].operation, "SCALE");
        assert_eq!(tree.inputs[0].inputs[0].operation, "IMPORT");
        assert!(tree.inputs[0].inputs[0].inputs.is_empty());
    }

    #[test]
    fn resolve_ancestry_terminates_when_a_record_preserves_content_hash() {
        // A no-op-content transform (e.g. FILTER matching every row) whose
        // input and output share a content hash used to make `find_producer`
        // match the record against its own input, recursing forever. The
        // real producer of that hash (IMPORT, at an earlier position) must
        // still be found instead.
        let mut store = ProvenanceStore::new();
        store.append(
            ProvenanceRecord::new("IMPORT", ExecutionId::new())
                .with_outputs(vec![tensor_entity("h1")]),
        );
        store.append(
            ProvenanceRecord::new("FILTER (matches all)", ExecutionId::new())
                .with_inputs(vec![tensor_entity("h1")])
                .with_outputs(vec![tensor_entity("h1")]), // same hash as its own input
        );

        let tree = store.resolve_ancestry(&tensor_entity("h1"));
        assert_eq!(tree.operation, "FILTER (matches all)");
        assert_eq!(tree.inputs.len(), 1);
        assert_eq!(tree.inputs[0].operation, "IMPORT");
        assert!(tree.inputs[0].inputs.is_empty());
    }

    #[test]
    fn resolve_ancestry_disambiguates_by_name_on_hash_collision() {
        // `DATASET big FROM raw FILTER val > 5` where the filter removes no
        // rows leaves `big` byte-identical to `raw` -- a real case, not a
        // contrived one. Resolving `raw` by name must not get attributed to
        // the later "DATASET FROM" record that happens to share its hash.
        let mut store = ProvenanceStore::new();
        store.append(
            ProvenanceRecord::new("IMPORT csv", ExecutionId::new()).with_outputs(vec![
                crate::core::provenance::ProvenanceEntity::dataset("raw", "h1"),
            ]),
        );
        store.append(
            ProvenanceRecord::new("DATASET FROM", ExecutionId::new())
                .with_inputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
                    "raw", "h1",
                )])
                .with_outputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
                    "big", "h1", // identical content to raw -- no-op filter
                )]),
        );

        let raw_tree = store.resolve_ancestry(&crate::core::provenance::ProvenanceEntity::dataset(
            "raw", "h1",
        ));
        assert_eq!(raw_tree.operation, "IMPORT csv");
        assert!(raw_tree.inputs.is_empty());

        let big_tree = store.resolve_ancestry(&crate::core::provenance::ProvenanceEntity::dataset(
            "big", "h1",
        ));
        assert_eq!(big_tree.operation, "DATASET FROM");
        assert_eq!(big_tree.inputs[0].operation, "IMPORT csv");
    }

    #[test]
    fn resolve_ancestry_of_unknown_hash_is_root() {
        let store = ProvenanceStore::new();
        let tree = store.resolve_ancestry(&tensor_entity("never-produced"));
        assert_eq!(tree.operation, "ROOT");
        assert!(tree.inputs.is_empty());
    }

    #[test]
    fn jsonl_round_trip_through_disk() {
        let dir = std::env::temp_dir().join(format!("linal_prov_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("provenance.jsonl");

        let r1 = ProvenanceRecord::new("IMPORT", ExecutionId::new())
            .with_outputs(vec![tensor_entity("h1")]);
        let r2 = ProvenanceRecord::new("SCALE", ExecutionId::new())
            .with_inputs(vec![tensor_entity("h1")])
            .with_outputs(vec![tensor_entity("h2")]);

        ProvenanceStore::append_jsonl(&path, &r1).unwrap();
        ProvenanceStore::append_jsonl(&path, &r2).unwrap();

        let loaded = ProvenanceStore::load_jsonl(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        let tree = loaded.resolve_ancestry(&tensor_entity("h2"));
        assert_eq!(tree.operation, "SCALE");
        assert_eq!(tree.inputs[0].operation, "IMPORT");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_jsonl_file_loads_as_empty_store() {
        let path =
            std::env::temp_dir().join(format!("linal_prov_missing_{}.jsonl", Uuid::new_v4()));
        let store = ProvenanceStore::load_jsonl(&path).unwrap();
        assert!(store.is_empty());
    }
}
