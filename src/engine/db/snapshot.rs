//! WAL checkpoints and startup recovery for `TensorDb` (see `engine::wal`).
//!
//! A checkpoint is a private snapshot of one database's in-memory state at
//! `{db_dir}/checkpoint/`:
//!
//! - `manifest.json`: the WAL sequence number it covers, every tensor's
//!   header (id, shape, strides, offset, metadata), the name table (name ->
//!   id + kind, so aliases stay aliases), lazy expressions, tensor-first
//!   datasets, dataset variables, and pipeline sources;
//! - `buffers/{i}.f32`: tensor data, one little-endian file per distinct
//!   `Arc` buffer, so zero-copy views still share storage after a restore;
//! - `datasets/`: record datasets, written through the same package
//!   SAVE/LOAD code path as user data (`persistence::save_dataset_to_dir`).
//!
//! It never touches the user's own `SAVE`d packages or their versions.
//! Tensor ids are preserved, because tensor-first datasets reference
//! tensors by id.
//!
//! It's written to `checkpoint.tmp/` and swapped in by rename; the previous
//! checkpoint is kept as `checkpoint.old/` until the swap completes, so a
//! crash at any point leaves one complete checkpoint to restore from.

use super::{NameEntry, TensorDb};
use crate::core::tensor::{Expression, Shape, Tensor, TensorId, TensorMetadata};
use crate::dsl::ast::Statement;
use crate::engine::error::EngineError;
use crate::engine::operations::TensorKind;
use crate::engine::wal::{self, Wal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const CHECKPOINT_DIR: &str = "checkpoint";
const CHECKPOINT_TMP: &str = "checkpoint.tmp";
const CHECKPOINT_OLD: &str = "checkpoint.old";
const MANIFEST: &str = "manifest.json";
const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: u32,
    /// Every WAL record with `seq <= this` is reflected in the checkpoint.
    seq: u64,
    created_at: DateTime<Utc>,
    tensors: Vec<TensorEntry>,
    names: BTreeMap<String, NameSnapshot>,
    lazy: Vec<(TensorId, Expression)>,
    tensor_datasets: crate::core::dataset::DatasetRegistry,
    dataset_vars: BTreeMap<String, String>,
    record_datasets: Vec<String>,
    pipelines: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct TensorEntry {
    id: TensorId,
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
    metadata: TensorMetadata,
    buffer: usize,
}

#[derive(Serialize, Deserialize)]
struct NameSnapshot {
    id: TensorId,
    kind: TensorKind,
}

fn io_err(context: &str, e: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidOp(format!("{}: {}", context, e))
}

fn write_buffer(path: &Path, data: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn read_buffer(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!("{}: length is not a multiple of 4", path.display()));
    }
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// The checkpoint to restore from: `checkpoint/`, or `checkpoint.old/` if a
/// crash interrupted the swap between the two renames.
fn existing_checkpoint(db_dir: &Path) -> Option<PathBuf> {
    [CHECKPOINT_DIR, CHECKPOINT_OLD]
        .iter()
        .map(|d| db_dir.join(d))
        .find(|d| d.join(MANIFEST).exists())
}

fn checkpoint_seq(db_dir: &Path) -> Result<u64, String> {
    let Some(dir) = existing_checkpoint(db_dir) else {
        return Ok(0);
    };
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(dir.join(MANIFEST)).map_err(|e| format!("{}: {}", dir.display(), e))?,
    )
    .map_err(|e| format!("{}: {}", dir.join(MANIFEST).display(), e))?;
    Ok(manifest.seq)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

impl TensorDb {
    fn wal_enabled(&self) -> bool {
        self.config.wal.enabled && !self.active_instance().replaying
    }

    /// Opens the active database's WAL if it isn't open yet, continuing the
    /// sequence after whatever the log and checkpoint on disk already hold.
    fn ensure_wal_open(&mut self) -> Result<(), String> {
        if self.active_instance().wal.is_some() {
            return Ok(());
        }
        let sync = self.config.wal.sync;
        let inst = self.active_instance_mut();
        let (records, _) = wal::read_records(&inst.db_dir)?;
        let last = records
            .last()
            .map(|r| r.seq)
            .unwrap_or(0)
            .max(checkpoint_seq(&inst.db_dir)?);
        inst.wal = Some(
            Wal::open(&inst.db_dir, last + 1, sync)
                .map_err(|e| format!("cannot open WAL in {}: {}", inst.db_dir.display(), e))?,
        );
        Ok(())
    }

    /// Checked before executing a statement: a database whose recovery
    /// failed refuses everything, and one whose WAL broke refuses further
    /// mutations (anything except a `CHECKPOINT`).
    pub(crate) fn wal_precheck(&self, stmt: &Statement) -> Result<(), EngineError> {
        let inst = self.active_instance();
        if inst.replaying {
            return Ok(());
        }
        if let Some(err) = &inst.recovery_error {
            return Err(EngineError::InvalidOp(format!(
                "Database '{}' failed WAL recovery on startup and is unavailable: {}",
                inst.name, err
            )));
        }
        if let Some(err) = &inst.wal_broken {
            if stmt.is_mutating() {
                return Err(EngineError::InvalidOp(format!(
                    "Database '{}' has un-logged changes because a WAL write failed ({}); \
                     run CHECKPOINT to snapshot the current state and resume logging",
                    inst.name, err
                )));
            }
        }
        Ok(())
    }

    /// Object names used to find what a name-less `IMPORT` created.
    pub(crate) fn object_names(&self) -> BTreeSet<String> {
        let inst = self.active_instance();
        let mut names: BTreeSet<String> = inst.names.keys().cloned().collect();
        names.extend(inst.list_dataset_names());
        names.extend(inst.tensor_datasets.list_names());
        names
    }

    /// Content fingerprint of a named object: a tensor's content hash, a
    /// record dataset's content hash, a tensor-first dataset's materialized
    /// content hash, or (`pipeline:<name>`) a pipeline's source.
    pub(crate) fn object_fingerprint(&self, name: &str) -> Option<String> {
        if let Some(pipeline) = name.strip_prefix("pipeline:") {
            return self
                .pipelines
                .get(pipeline)
                .map(|p| sha256_hex(p.source.as_bytes()));
        }
        let inst = self.active_instance();
        if let Ok(t) = inst.get(name) {
            return Some(t.data_hash().to_string());
        }
        if let Ok(ds) = inst.get_dataset(name) {
            return Some(ds.content_hash());
        }
        inst.materialize_tensor_dataset(name)
            .ok()
            .map(|ds| ds.content_hash())
    }

    /// Whether a successful `stmt` must be appended to the WAL.
    pub(crate) fn wal_should_log(&self, stmt: &Statement) -> bool {
        self.wal_enabled() && stmt.is_mutating()
    }

    /// Whether a successful `stmt` should be followed by an automatic
    /// checkpoint: after a `SAVE`, so a later replay never re-runs a `LOAD`
    /// against a package this `SAVE` has since overwritten.
    pub(crate) fn wal_checkpoint_after(&self, stmt: &Statement) -> bool {
        self.wal_enabled() && matches!(stmt, Statement::Save(_))
    }

    /// Appends a successfully executed statement to the active database's
    /// WAL, then checkpoints if the log has outgrown `[wal]
    /// checkpoint_bytes`.
    pub(crate) fn wal_append(
        &mut self,
        statement: &str,
        fingerprints: BTreeMap<String, String>,
    ) -> Result<(), EngineError> {
        let result = self.ensure_wal_open().and_then(|_| {
            self.active_instance_mut()
                .wal
                .as_mut()
                .expect("opened above")
                .append(statement, fingerprints)
                .map_err(|e| e.to_string())
        });
        if let Err(e) = result {
            self.active_instance_mut().wal_broken = Some(e.clone());
            return Err(EngineError::InvalidOp(format!(
                "The statement was applied but could not be written to the WAL ({}); \
                 further changes to this database are refused until CHECKPOINT succeeds",
                e
            )));
        }

        let too_big = self
            .active_instance()
            .wal
            .as_ref()
            .is_some_and(|w| w.size_bytes() > self.config.wal.checkpoint_bytes);
        if too_big {
            if let Err(e) = self.checkpoint() {
                eprintln!("Warning: automatic WAL checkpoint failed: {}", e);
            }
        }
        Ok(())
    }

    /// `CHECKPOINT`: snapshots the active database and truncates its WAL.
    pub fn checkpoint(&mut self) -> Result<String, EngineError> {
        if !self.config.wal.enabled {
            return Err(EngineError::InvalidOp(
                "CHECKPOINT requires the write-ahead log: set `[wal] enabled = true` in linal.toml"
                    .to_string(),
            ));
        }
        self.ensure_wal_open().map_err(EngineError::InvalidOp)?;
        self.active_instance_mut().replaying = true;
        let result = self.write_checkpoint();
        self.active_instance_mut().replaying = false;
        let (tensors, datasets, seq) = result?;

        let inst = self.active_instance_mut();
        inst.wal
            .as_mut()
            .expect("opened above")
            .truncate()
            .map_err(|e| io_err("cannot truncate WAL", e))?;
        inst.wal_broken = None;
        Ok(format!(
            "Checkpoint written for database '{}': {} tensor(s), {} dataset(s); WAL truncated after seq {}",
            inst.name, tensors, datasets, seq
        ))
    }

    fn write_checkpoint(&mut self) -> Result<(usize, usize, u64), EngineError> {
        // Absolute: `persistence::save_dataset_to_dir` resolves a relative
        // path against the database directory, which would nest a relative
        // `data_dir` (the default, `./data`) inside itself.
        let db_dir = std::path::absolute(&self.active_instance().db_dir)
            .map_err(|e| io_err("cannot resolve database directory", e))?;
        let tmp = db_dir.join(CHECKPOINT_TMP);
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp).map_err(|e| io_err("cannot clear checkpoint.tmp", e))?;
        }
        let buffers_dir = tmp.join("buffers");
        std::fs::create_dir_all(&buffers_dir)
            .map_err(|e| io_err("cannot create checkpoint directory", e))?;

        let inst = self.active_instance();
        let seq = inst.wal.as_ref().map(|w| w.last_seq()).unwrap_or(0);

        // Tensors, sharing one buffer file per distinct Arc.
        let mut buffer_ids: HashMap<usize, usize> = HashMap::new();
        let mut tensors = Vec::new();
        for tensor in inst.store.tensors() {
            let key = Arc::as_ptr(&tensor.data) as usize;
            let buffer = match buffer_ids.get(&key) {
                Some(&i) => i,
                None => {
                    let i = buffer_ids.len();
                    write_buffer(&buffers_dir.join(format!("{}.f32", i)), &tensor.data)
                        .map_err(|e| io_err("cannot write checkpoint buffer", e))?;
                    buffer_ids.insert(key, i);
                    i
                }
            };
            tensors.push(TensorEntry {
                id: tensor.id,
                shape: tensor.shape.clone(),
                strides: tensor.strides.clone(),
                offset: tensor.offset,
                metadata: (*tensor.metadata).clone(),
                buffer,
            });
        }

        let names = inst
            .names
            .iter()
            .map(|(name, e)| {
                (
                    name.clone(),
                    NameSnapshot {
                        id: e.id,
                        kind: e.kind,
                    },
                )
            })
            .collect();
        let lazy = inst
            .lazy_store
            .iter()
            .map(|(id, expr)| (*id, expr.clone()))
            .collect();
        let tensor_datasets = serde_json::from_value(
            serde_json::to_value(&inst.tensor_datasets)
                .map_err(|e| io_err("cannot snapshot tensor datasets", e))?,
        )
        .map_err(|e| io_err("cannot snapshot tensor datasets", e))?;
        let dataset_vars = inst
            .dataset_vars
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let record_datasets = inst.list_dataset_names();
        let pipelines = self
            .pipelines
            .summary()
            .into_iter()
            .filter_map(|(name, _)| self.pipelines.get(&name).map(|p| (name, p.source)))
            .collect();

        let datasets_dir = tmp.join("datasets");
        for name in &record_datasets {
            crate::dsl::persistence::save_dataset_to_dir(self, name, &datasets_dir)
                .map_err(|e| io_err(&format!("cannot snapshot dataset '{}'", name), e))?;
        }

        let manifest = Manifest {
            format: FORMAT_VERSION,
            seq,
            created_at: Utc::now(),
            tensors,
            names,
            lazy,
            tensor_datasets,
            dataset_vars,
            record_datasets,
            pipelines,
        };
        let counts = (manifest.names.len(), manifest.record_datasets.len(), seq);
        let manifest_path = tmp.join(MANIFEST);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).map_err(|e| io_err("cannot encode manifest", e))?,
        )
        .map_err(|e| io_err("cannot write manifest", e))?;
        std::fs::File::open(&manifest_path)
            .and_then(|f| f.sync_all())
            .map_err(|e| io_err("cannot sync manifest", e))?;

        // Swap: checkpoint -> checkpoint.old, checkpoint.tmp -> checkpoint.
        let current = db_dir.join(CHECKPOINT_DIR);
        let old = db_dir.join(CHECKPOINT_OLD);
        if old.exists() {
            std::fs::remove_dir_all(&old).map_err(|e| io_err("cannot remove checkpoint.old", e))?;
        }
        if current.exists() {
            std::fs::rename(&current, &old).map_err(|e| io_err("cannot rotate checkpoint", e))?;
        }
        std::fs::rename(&tmp, &current).map_err(|e| io_err("cannot install checkpoint", e))?;
        if old.exists() {
            let _ = std::fs::remove_dir_all(&old);
        }
        Ok(counts)
    }

    /// Restores the active database from its checkpoint, if any, and
    /// returns the WAL sequence number the checkpoint covers.
    fn restore_checkpoint(&mut self) -> Result<u64, String> {
        // Absolute for the same reason as in `write_checkpoint`.
        let db_dir = std::path::absolute(&self.active_instance().db_dir)
            .map_err(|e| format!("cannot resolve database directory: {}", e))?;
        let Some(dir) = existing_checkpoint(&db_dir) else {
            return Ok(0);
        };
        let manifest_path = dir.join(MANIFEST);
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(&manifest_path)
                .map_err(|e| format!("{}: {}", manifest_path.display(), e))?,
        )
        .map_err(|e| format!("{}: {}", manifest_path.display(), e))?;
        if manifest.format != FORMAT_VERSION {
            return Err(format!(
                "{}: unsupported checkpoint format {}",
                manifest_path.display(),
                manifest.format
            ));
        }

        let mut buffers: HashMap<usize, Arc<Vec<f32>>> = HashMap::new();
        {
            let inst = self.active_instance_mut();
            for entry in manifest.tensors {
                let data = match buffers.get(&entry.buffer) {
                    Some(d) => d.clone(),
                    None => {
                        let d = Arc::new(read_buffer(
                            &dir.join("buffers").join(format!("{}.f32", entry.buffer)),
                        )?);
                        buffers.insert(entry.buffer, d.clone());
                        d
                    }
                };
                let tensor = Tensor {
                    id: entry.id,
                    shape: entry.shape,
                    data,
                    metadata: Arc::new(entry.metadata),
                    strides: entry.strides,
                    offset: entry.offset,
                };
                inst.store
                    .insert_existing_tensor(tensor)
                    .map_err(|e| format!("restoring tensor: {}", e))?;
            }
            for (name, n) in manifest.names {
                inst.names.insert(
                    name,
                    NameEntry {
                        id: n.id,
                        kind: n.kind,
                    },
                );
            }
            inst.lazy_store.extend(manifest.lazy);
            inst.tensor_datasets = manifest.tensor_datasets;
            inst.dataset_vars.extend(manifest.dataset_vars);
        }

        for (name, source) in manifest.pipelines {
            if let Ok(Statement::DefinePipeline(def)) = crate::dsl::parser::parse(&source) {
                self.pipelines.insert(
                    name,
                    crate::dsl::ast::StoredPipeline {
                        steps: def.steps,
                        source,
                    },
                );
            }
        }

        let datasets_dir = dir.join("datasets");
        for name in &manifest.record_datasets {
            crate::dsl::persistence::load_dataset_from_dir(self, name, &datasets_dir)
                .map_err(|e| format!("restoring dataset '{}' from checkpoint: {}", name, e))?;
        }
        Ok(manifest.seq)
    }

    /// Restores the active database's checkpoint and replays its WAL,
    /// returning the last sequence number now reflected in memory.
    fn replay_active(&mut self) -> Result<u64, String> {
        let db_dir = self.active_instance().db_dir.clone();
        let checkpoint_seq = self.restore_checkpoint()?;
        let (records, warning) = wal::read_records(&db_dir)?;
        if let Some(w) = warning {
            eprintln!("Warning: {}", w);
            wal::rewrite(&db_dir, &records)
                .map_err(|e| format!("cannot repair truncated WAL: {}", e))?;
        }
        let mut last = checkpoint_seq;
        for record in records {
            if record.seq <= checkpoint_seq {
                // Already in the checkpoint: a crash landed between writing
                // it and truncating the log.
                continue;
            }
            crate::dsl::execute_line(self, &record.statement, 0).map_err(|e| {
                format!(
                    "replaying WAL record {} (`{}`) failed: {}",
                    record.seq, record.statement, e
                )
            })?;
            for (name, expected) in &record.fingerprints {
                let actual = self.object_fingerprint(name);
                if actual.as_deref() != Some(expected.as_str()) {
                    return Err(format!(
                        "WAL record {} (`{}`) loaded '{}' from a file whose content has changed \
                         since it was logged, so replay can't reproduce it; restore that file or \
                         move {} aside (losing changes since the last checkpoint)",
                        record.seq,
                        record.statement,
                        name,
                        db_dir.join(wal::WAL_FILE).display()
                    ));
                }
            }
            last = record.seq;
        }
        Ok(last)
    }

    /// Startup recovery for every database (`[wal] enabled`). A database
    /// that fails to recover is left in a refusing state
    /// (`recovery_error`) instead of running on partial data.
    pub(crate) fn recover_all_from_wal(&mut self) {
        let active = self.active_db.clone();
        let mut names: Vec<String> = self.databases.keys().cloned().collect();
        names.sort();
        for name in names {
            self.active_db = name.clone();
            self.active_instance_mut().replaying = true;
            let result = self.replay_active();
            self.active_instance_mut().replaying = false;
            match result {
                Ok(last) => {
                    let sync = self.config.wal.sync;
                    let inst = self.active_instance_mut();
                    match Wal::open(&inst.db_dir, last + 1, sync) {
                        Ok(w) => inst.wal = Some(w),
                        Err(e) => {
                            inst.recovery_error = Some(format!("cannot open WAL: {}", e));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: WAL recovery failed for database '{}': {}", name, e);
                    self.active_instance_mut().recovery_error = Some(e);
                }
            }
        }
        self.active_db = active;
    }
}
