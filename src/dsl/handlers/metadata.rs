use crate::core::storage::ParquetStorage;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Typed entry point — called directly from the executor without a string round-trip.
pub fn set_metadata_typed(
    db: &mut TensorDb,
    dataset_name: &str,
    raw_key: &str,
    raw_value: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let key = raw_key.to_lowercase();
    let value = raw_value.trim_matches('"').to_string();

    db.set_dataset_metadata(dataset_name, key.clone(), value.clone())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let path = format!(
        "{}/{}",
        db.config.storage.data_dir.to_string_lossy(),
        db.active_instance().name
    );
    let storage = ParquetStorage::new(&path);

    if storage.metadata_exists(dataset_name) {
        let mut metadata =
            storage
                .load_dataset_metadata(dataset_name)
                .map_err(|e| DslError::Parse {
                    line: line_no,
                    msg: format!("Failed to load metadata: {}", e),
                })?;

        match key.as_str() {
            "author" => metadata.author = Some(value.clone()),
            "description" => metadata.description = Some(value.clone()),
            "tag" => metadata.add_tag(value.clone()),
            _ => {}
        }

        metadata.increment_version();
        storage
            .save_dataset_metadata(&metadata)
            .map_err(|e| DslError::Parse {
                line: line_no,
                msg: format!("Failed to save metadata: {}", e),
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Updated metadata for dataset '{}': {} = {}",
        dataset_name, key, value
    )))
}

/// Handle `SET DATASET <name> METADATA <key> = <value>` (string-based, for the legacy fallback chain).
pub fn handle_set_metadata(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("SET DATASET ").unwrap().trim();

    let parts: Vec<&str> = rest.splitn(2, " METADATA ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: SET DATASET <name> METADATA <key> = <value>".to_string(),
        });
    }

    let dataset_name = parts[0].trim();
    let kv_part = parts[1].trim();

    let kv: Vec<&str> = kv_part.splitn(2, '=').collect();
    if kv.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: <key> = <value> after METADATA".to_string(),
        });
    }

    set_metadata_typed(db, dataset_name, kv[0].trim(), kv[1].trim(), line_no)
}
