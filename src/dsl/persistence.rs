use crate::core::connectors::{
    csv_connector::CsvConnector, hdf5_connector::Hdf5Connector, numpy_connector::NumpyConnector,
    zarr_connector::ZarrConnector, ConnectorRegistry,
};
use crate::core::dataset::{Dataset, DatasetMetadata, DatasetOrigin, ResourceReference};
use crate::core::storage::{record_batch_to_tensors, CsvStorage, ParquetStorage, StorageEngine};
use crate::dsl::ast::{ListTarget, PersistKind, StoredPipeline};
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

// ─── Path helpers ─────────────────────────────────────────────────────────────

fn instance_base_dir(db: &TensorDb) -> PathBuf {
    let mut path = db.config.storage.data_dir.clone();
    path.push(&db.active_instance().name);
    path
}

fn resolve_persistence_path(db: &TensorDb, path: &str) -> String {
    let path_buf = PathBuf::from(path);
    if path_buf.is_absolute() {
        return path.to_string();
    }
    let mut resolved = instance_base_dir(db);
    if !path.is_empty() {
        resolved.push(path);
    }
    if let Some(parent) = resolved.parent() {
        let _ = fs::create_dir_all(parent);
    }
    resolved.to_string_lossy().into_owned()
}

pub fn get_connector_registry() -> ConnectorRegistry {
    let mut registry = ConnectorRegistry::new();
    registry.register(Box::new(CsvConnector::new()));
    registry.register(Box::new(NumpyConnector));
    registry.register(Box::new(Hdf5Connector));
    registry.register(Box::new(ZarrConnector));
    registry
}

// ─── Save ─────────────────────────────────────────────────────────────────────

fn save_dataset_core(
    db: &mut TensorDb,
    dataset_name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let (disk_name, storage_path) = if let Some(p_str) = explicit_path {
        let p_path = Path::new(p_str);
        if p_path.extension().is_some() {
            let dn = p_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(dataset_name)
                .to_string();
            let parent = p_path.parent().and_then(|p| p.to_str()).unwrap_or("");
            (dn, resolve_persistence_path(db, parent))
        } else {
            (
                dataset_name.to_string(),
                resolve_persistence_path(db, p_str),
            )
        }
    } else {
        (dataset_name.to_string(), resolve_persistence_path(db, ""))
    };

    let mut dataset = match db.get_dataset(dataset_name) {
        Ok(ds) => ds.clone(),
        Err(_) => db
            .materialize_tensor_dataset(dataset_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?,
    };

    if disk_name != dataset_name {
        dataset.metadata.name = Some(disk_name.clone());
    }

    let storage = ParquetStorage::new(&storage_path);
    storage
        .save_dataset(&dataset)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset: {}", e),
        })?;

    let mut metadata = if storage.metadata_exists(&disk_name) {
        let mut meta = storage
            .load_dataset_metadata(&disk_name)
            .unwrap_or_else(|_| DatasetMetadata::new(disk_name.clone(), DatasetOrigin::Created));
        meta.increment_version();
        meta
    } else {
        DatasetMetadata::new(disk_name.clone(), DatasetOrigin::Created)
    };

    let content_hash = format!("{}:{}", dataset_name, dataset.rows.len());
    metadata.update_hash(content_hash);
    metadata.record_schema(dataset.schema.as_ref().clone().into());
    storage
        .save_dataset_metadata(&metadata)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save metadata: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Saved dataset '{}' (v{}) to '{}'",
        dataset_name, metadata.version, storage_path
    )))
}

fn save_tensor_core(
    db: &mut TensorDb,
    tensor_name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let storage_path = explicit_path
        .map(|p| resolve_persistence_path(db, p))
        .unwrap_or_else(|| resolve_persistence_path(db, ""));

    let tensor = db
        .active_instance()
        .get(tensor_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let storage = ParquetStorage::new(&storage_path);
    storage
        .save_tensor(tensor_name, tensor)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save tensor: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Saved tensor '{}' to '{}'",
        tensor_name, storage_path
    )))
}

// ─── Load ─────────────────────────────────────────────────────────────────────

fn load_dataset_core(
    db: &mut TensorDb,
    dataset_name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let (disk_name, storage_path) = if let Some(p_str) = explicit_path {
        let p_path = Path::new(p_str);
        if p_path.extension().is_some() {
            let dn = p_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(p_str)
                .to_string();
            let parent = p_path.parent().and_then(|p| p.to_str()).unwrap_or("");
            (dn, resolve_persistence_path(db, parent))
        } else {
            (
                dataset_name.to_string(),
                resolve_persistence_path(db, p_str),
            )
        }
    } else {
        (dataset_name.to_string(), resolve_persistence_path(db, ""))
    };

    let storage = ParquetStorage::new(&storage_path);
    if let Ok(mut dataset) = storage.load_reference_dataset(&disk_name) {
        let metadata_info = if storage.metadata_exists(&disk_name) {
            if let Ok(meta) = storage.load_dataset_metadata(&disk_name) {
                let info = format!(
                    " (v{}, {})",
                    meta.version,
                    match meta.origin {
                        DatasetOrigin::Created => "Created",
                        DatasetOrigin::Imported { .. } => "Imported",
                        DatasetOrigin::Derived { .. } => "Derived",
                        DatasetOrigin::Bound { .. } => "Bound",
                        DatasetOrigin::Attached { .. } => "Attached",
                    }
                );
                dataset.metadata = Some(meta);
                info
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if dataset_name != disk_name {
            dataset.name = dataset_name.to_string();
            if let Some(meta) = &mut dataset.metadata {
                meta.name = dataset_name.to_string();
            }
        }

        db.active_instance_mut().register_tensor_dataset(dataset);

        return Ok(DslOutput::Message(format!(
            "Loaded reference dataset '{}'{} from '{}'",
            dataset_name, metadata_info, storage_path
        )));
    }

    let mut dataset = storage
        .load_dataset(&disk_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!(
                "Failed to load dataset '{}' from '{}': {}",
                disk_name, storage_path, e
            ),
        })?;

    if dataset_name != disk_name {
        dataset.metadata.name = Some(dataset_name.to_string());
    }

    let schema = dataset.schema.clone();
    match db.create_dataset(dataset_name.to_string(), schema) {
        Ok(_) => {}
        Err(crate::engine::EngineError::DatasetError(
            crate::core::store::DatasetStoreError::NameAlreadyExists(_),
        )) => {
            return Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::DatasetError(
                    crate::core::store::DatasetStoreError::NameAlreadyExists(
                        dataset_name.to_string(),
                    ),
                ),
            });
        }
        Err(e) => {
            return Err(DslError::Engine {
                line: line_no,
                source: e,
            })
        }
    }

    let row_count = dataset.len();
    for row in dataset.rows {
        db.insert_row(dataset_name, row)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Loaded dataset '{}' from '{}' ({} rows)",
        dataset_name, storage_path, row_count
    )))
}

fn load_tensor_core(
    db: &mut TensorDb,
    tensor_name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let storage_path = explicit_path
        .map(|p| resolve_persistence_path(db, p))
        .unwrap_or_else(|| resolve_persistence_path(db, ""));

    let storage = ParquetStorage::new(&storage_path);
    let tensor = storage
        .load_tensor(tensor_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load tensor: {}", e),
        })?;

    db.active_instance_mut()
        .insert_tensor_object(tensor_name, tensor)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Loaded tensor '{}' from '{}'",
        tensor_name, storage_path
    )))
}

// ─── List ─────────────────────────────────────────────────────────────────────

fn list_versions_core(
    db: &TensorDb,
    dataset_name: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let path = format!(
        "{}/{}",
        db.config.storage.data_dir.to_string_lossy(),
        db.active_instance().name
    );
    let storage = ParquetStorage::new(&path);

    if !storage.metadata_exists(dataset_name) {
        return Ok(DslOutput::Message(format!(
            "No metadata found for dataset '{}'",
            dataset_name
        )));
    }

    let metadata = storage
        .load_dataset_metadata(dataset_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load metadata: {}", e),
        })?;

    let mut output = format!("=== Version History for Dataset: {} ===\n", dataset_name);
    output.push_str(&format!("Current Version: {}\n", metadata.version));
    output.push_str(&format!(
        "Current Schema Version: {}\n",
        metadata.schema_version
    ));
    output.push_str("\nSchema History:\n");

    if metadata.schema_history.is_empty() {
        output.push_str("  (Initial schema only)\n");
    } else {
        for v in &metadata.schema_history {
            output.push_str(&format!(
                "  - v{}: {} columns, migration: {:?}\n",
                v.version,
                v.schema.columns.len(),
                v.migration
            ));
        }
    }
    output.push_str("================================");
    Ok(DslOutput::Message(output))
}

fn list_datasets_core(
    db: &TensorDb,
    from_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let path = from_path
        .map(|p| resolve_persistence_path(db, p))
        .unwrap_or_else(|| resolve_persistence_path(db, ""));

    let storage = ParquetStorage::new(&path);
    let datasets = storage.list_datasets().map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Failed to list datasets: {}", e),
    })?;

    let message = if datasets.is_empty() {
        format!("No datasets found in '{}'", path)
    } else {
        format!("Datasets in '{}':\n  - {}", path, datasets.join("\n  - "))
    };
    Ok(DslOutput::Message(message))
}

fn list_tensors_core(
    db: &TensorDb,
    from_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let path = from_path
        .map(|p| resolve_persistence_path(db, p))
        .unwrap_or_else(|| resolve_persistence_path(db, ""));

    let storage = ParquetStorage::new(&path);
    let tensors = storage.list_tensors().map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Failed to list tensors: {}", e),
    })?;

    let message = if tensors.is_empty() {
        format!("No tensors found in '{}'", path)
    } else {
        format!("Tensors in '{}':\n  - {}", path, tensors.join("\n  - "))
    };
    Ok(DslOutput::Message(message))
}

// ─── Import / Export ──────────────────────────────────────────────────────────

fn use_dataset_core(
    db: &mut TensorDb,
    path_str: &str,
    name_override: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let registry = get_connector_registry();
    let connector = registry
        .find_connector(path_str)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("No connector found for path: {}", path_str),
        })?;

    let (batch, _lineage) = connector
        .read_dataset(path_str)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Connector failed: {}", e),
        })?;

    let tensors = record_batch_to_tensors(&batch).map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Failed to convert to tensors: {}", e),
    })?;

    let ds_name = name_override.unwrap_or_else(|| {
        Path::new(path_str)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("ephemeral_ds")
    });

    let mut ds = Dataset::new(ds_name);
    for (col_name, tensor) in tensors {
        let tensor_id = tensor.id;
        let tensor_shape = tensor.shape.clone();

        db.active_instance_mut()
            .insert_tensor_object(format!("{}_{}", ds_name, col_name), tensor)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;

        let value_type = match tensor_shape.rank() {
            1 => crate::core::value::ValueType::Vector(tensor_shape.dims[0]),
            2 => crate::core::value::ValueType::Matrix(tensor_shape.dims[0], tensor_shape.dims[1]),
            0 => crate::core::value::ValueType::Float,
            _ => crate::core::value::ValueType::Vector(tensor_shape.num_elements()),
        };

        let schema =
            crate::core::dataset::ColumnSchema::new(col_name.clone(), value_type, tensor_shape);
        ds.add_column(col_name, ResourceReference::tensor(tensor_id), schema);
    }

    db.active_instance_mut().register_tensor_dataset(ds);

    Ok(DslOutput::Table(
        db.active_instance()
            .materialize_tensor_dataset(ds_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?,
    ))
}

fn import_dataset_core(
    db: &mut TensorDb,
    path_str: &str,
    name_override: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let registry = get_connector_registry();
    let connector = registry
        .find_connector(path_str)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("No connector found for path: {}", path_str),
        })?;

    let (batch, lineage) = connector
        .read_dataset(path_str)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Connector failed: {}", e),
        })?;

    let ds_name = name_override.unwrap_or_else(|| {
        Path::new(path_str)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("imported_ds")
    });

    let storage_path = resolve_persistence_path(db, "");
    let storage = ParquetStorage::new(&storage_path);

    let metadata = DatasetMetadata::new(
        ds_name.to_string(),
        DatasetOrigin::Imported {
            source: path_str.to_string(),
        },
    );

    storage
        .save_dataset_package(ds_name, &batch, &metadata, &lineage)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset package: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Imported dataset '{}' and persisted to {}",
        ds_name, storage_path
    )))
}

fn export_csv_core(
    db: &mut TensorDb,
    dataset_name: &str,
    export_path: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let dataset = db.get_dataset(dataset_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    let resolved_path = resolve_persistence_path(db, export_path);
    let csv_storage = CsvStorage::new(&resolved_path);
    csv_storage
        .export_dataset(dataset, &resolved_path)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to export CSV: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Exported dataset '{}' to '{}'",
        dataset_name, export_path
    )))
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

fn pipeline_dir(db: &TensorDb) -> std::path::PathBuf {
    let mut path = instance_base_dir(db);
    path.push("pipelines");
    path
}

fn save_pipeline_core(
    db: &TensorDb,
    name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let stored = db.pipelines.get(name).ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: format!("Pipeline '{}' not found", name),
    })?;

    let save_path = if let Some(p) = explicit_path {
        PathBuf::from(resolve_persistence_path(db, p))
    } else {
        pipeline_dir(db).join(format!("{}.json", name))
    };

    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent).map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to create pipeline directory: {}", e),
        })?;
    }

    let json = serde_json::json!({
        "name": name,
        "source": stored.source,
        "version": env!("CARGO_PKG_VERSION")
    });

    fs::write(&save_path, serde_json::to_string_pretty(&json).unwrap()).map_err(|e| {
        DslError::Parse {
            line: line_no,
            msg: format!("Failed to write pipeline file: {}", e),
        }
    })?;

    Ok(DslOutput::Message(format!(
        "Saved pipeline '{}' to '{}'",
        name,
        save_path.display()
    )))
}

fn load_pipeline_core(
    db: &mut TensorDb,
    name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let load_path = if let Some(p) = explicit_path {
        PathBuf::from(resolve_persistence_path(db, p))
    } else {
        pipeline_dir(db).join(format!("{}.json", name))
    };

    let content = fs::read_to_string(&load_path).map_err(|_| DslError::Parse {
        line: line_no,
        msg: format!("Pipeline file not found: '{}'", load_path.display()),
    })?;

    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Invalid pipeline JSON: {}", e),
    })?;

    let source = json["source"].as_str().ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Pipeline file missing 'source' field".into(),
    })?;

    // Re-parse the DSL source to reconstruct steps.
    let stmt = crate::dsl::parser::parse(source).map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Failed to re-parse pipeline source: {:?}", e),
    })?;

    match stmt {
        crate::dsl::ast::Statement::DefinePipeline(def) => {
            db.pipelines.insert(
                name.to_string(),
                StoredPipeline {
                    steps: def.steps,
                    source: source.to_string(),
                },
            );
            Ok(DslOutput::Message(format!(
                "Loaded pipeline '{}' from '{}'",
                name,
                load_path.display()
            )))
        }
        _ => Err(DslError::Parse {
            line: line_no,
            msg: "Pipeline file source is not a DEFINE PIPELINE statement".into(),
        }),
    }
}

// ─── Typed public API (called from executor) ──────────────────────────────────

pub fn save_typed(
    db: &mut TensorDb,
    kind: PersistKind,
    name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match kind {
        PersistKind::Dataset => save_dataset_core(db, name, explicit_path, line_no),
        PersistKind::Tensor => save_tensor_core(db, name, explicit_path, line_no),
        PersistKind::Pipeline => save_pipeline_core(db, name, explicit_path, line_no),
    }
}

pub fn load_typed(
    db: &mut TensorDb,
    kind: PersistKind,
    name: &str,
    explicit_path: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match kind {
        PersistKind::Dataset => load_dataset_core(db, name, explicit_path, line_no),
        PersistKind::Tensor => load_tensor_core(db, name, explicit_path, line_no),
        PersistKind::Pipeline => load_pipeline_core(db, name, explicit_path, line_no),
    }
}

pub fn list_typed(
    db: &TensorDb,
    target: &ListTarget,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match target {
        ListTarget::Datasets => list_datasets_core(db, None, line_no),
        ListTarget::Tensors => list_tensors_core(db, None, line_no),
        ListTarget::DatasetVersions(name) => list_versions_core(db, name, line_no),
        ListTarget::DatasetPackages => list_datasets_core(db, None, line_no),
        ListTarget::Pipelines => crate::dsl::executor::execute_show_pipelines(db),
    }
}

pub fn import_typed(
    db: &mut TensorDb,
    ephemeral: bool,
    path: &str,
    name_override: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    if ephemeral {
        use_dataset_core(db, path, name_override, line_no)
    } else {
        import_dataset_core(db, path, name_override, line_no)
    }
}

pub fn export_typed(
    db: &mut TensorDb,
    name: &str,
    path: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    export_csv_core(db, name, path, line_no)
}

pub fn import_csv_typed(
    db: &mut TensorDb,
    path: &str,
    name_override: Option<&str>,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let resolved_path = resolve_persistence_path(db, path);
    let csv_storage = CsvStorage::new(&resolved_path);
    let dataset = csv_storage
        .import_dataset(&resolved_path)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to import CSV: {}", e),
        })?;
    let final_name =
        name_override.unwrap_or(dataset.metadata.name.as_deref().unwrap_or("imported_csv"));
    let schema = dataset.schema.clone();
    db.create_dataset(final_name.to_string(), schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let row_count = dataset.len();
    for row in dataset.rows {
        db.insert_row(final_name, row)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }
    Ok(DslOutput::Message(format!(
        "Imported {} rows from '{}' into dataset '{}'",
        row_count, path, final_name
    )))
}
