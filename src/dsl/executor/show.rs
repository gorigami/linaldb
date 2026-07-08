use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

pub(super) fn execute_show(
    db: &mut TensorDb,
    target: ShowTarget,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    match target {
        ShowTarget::All => {
            let mut names = db.list_names();
            names.sort();
            let mut output = String::from("--- ALL TENSORS ---\n");
            for name in names {
                if let Ok(t) = db.get(&name) {
                    output.push_str(&format!(
                        "{}: shape {:?}, len {}, data = {:?}\n",
                        name,
                        t.shape.dims,
                        t.data.len(),
                        t.data
                    ));
                }
            }
            output.push_str("-------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::AllDatasets => {
            let mut names = db.list_dataset_names();
            names.sort();
            let mut output = String::from("--- ALL DATASETS ---\n");
            for name in names {
                if let Ok(dataset) = db.get_dataset(&name) {
                    output.push_str(&format!(
                        "Dataset: {} (rows: {}, columns: {})\n",
                        name,
                        dataset.len(),
                        dataset.schema.len()
                    ));
                    for field in &dataset.schema.fields {
                        output.push_str(&format!("  - {}: {}\n", field.name, field.value_type));
                    }
                }
            }
            output.push_str("--------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::AllDatabases => {
            let mut names = db.list_databases();
            names.sort();
            let mut output = String::from("--- ALL DATABASES ---\n");
            for name in names {
                output.push_str(&format!("  - {}\n", name));
            }
            output.push_str("---------------------");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::Indexes(filter) => {
            let indices = db.list_indices();
            let mut output = if let Some(ref ds_name) = filter {
                format!("--- INDICES FOR {} ---\n", ds_name)
            } else {
                String::from("--- ALL INDICES ---\n")
            };
            output.push_str(&format!(
                "{:<20} {:<20} {:<10}\n",
                "Dataset", "Column", "Type"
            ));
            output.push_str(&format!("{:-<52}\n", ""));
            let mut count = 0;
            for (ds, col, type_str) in indices {
                if let Some(ref ds_filter) = filter {
                    if &ds != ds_filter {
                        continue;
                    }
                }
                output.push_str(&format!("{:<20} {:<20} {:<10}\n", ds, col, type_str));
                count += 1;
            }
            output.push_str("-------------------");
            if count == 0 {
                if let Some(ref ds_name) = filter {
                    if db.get_dataset(ds_name).is_err() {
                        return Err(DslError::Engine {
                            line: line_no,
                            source: crate::engine::EngineError::NameNotFound(ds_name.clone()),
                        });
                    }
                }
            }
            Ok(DslOutput::Message(output))
        }

        ShowTarget::Lineage(name) => {
            let tree = db.get_lineage_tree(&name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let mut output = format!("Lineage for tensor '{}':\n", name);
            output.push_str(&format_lineage_tree(&tree, 0));
            Ok(DslOutput::Message(output))
        }

        ShowTarget::DatasetMetadata(dataset_name) => {
            if let Ok(dataset) = db.get_dataset(&dataset_name) {
                let metadata = &dataset.metadata;
                let mut output = format!(
                    "=== Dataset Metadata: {} (In-Memory/Legacy) ===\n",
                    dataset_name
                );
                output.push_str(&format!("Version: {}\n", metadata.version));
                output.push_str("Origin: Created\n");
                output.push_str(&format!("Created: {:?}\n", metadata.created_at));
                output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
                output.push_str(&format!("Rows: {}\n", metadata.row_count));
                if !metadata.extra.is_empty() {
                    output.push_str("\nExtra Metadata:\n");
                    for (k, v) in &metadata.extra {
                        output.push_str(&format!("  {}: {}\n", k, v));
                    }
                }
                output.push_str("================================");
                return Ok(DslOutput::Message(output));
            }
            if let Some(dataset) = db.get_tensor_dataset(&dataset_name) {
                if let Some(metadata) = &dataset.metadata {
                    let mut output = format!(
                        "=== Dataset Metadata: {} (In-Memory/Tensor) ===\n",
                        dataset_name
                    );
                    output.push_str(&format!("Version: {}\n", metadata.version));
                    output.push_str(&format!("Hash: {}\n", metadata.hash));
                    output.push_str(&format!("Origin: {:?}\n", metadata.origin));
                    if let Some(author) = &metadata.author {
                        output.push_str(&format!("Author: {}\n", author));
                    }
                    if let Some(desc) = &metadata.description {
                        output.push_str(&format!("Description: {}\n", desc));
                    }
                    if !metadata.tags.is_empty() {
                        output.push_str(&format!("Tags: {}\n", metadata.tags.join(", ")));
                    }
                    output.push_str(&format!("Created: {:?}\n", metadata.created_at));
                    output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
                    output.push_str("================================");
                    return Ok(DslOutput::Message(output));
                }
            }
            let path = format!(
                "{}/{}",
                db.config.storage.data_dir.to_string_lossy(),
                db.active_instance().name
            );
            let storage = crate::core::storage::ParquetStorage::new(&path);
            if !storage.metadata_exists(&dataset_name) {
                return Ok(DslOutput::Message(format!(
                    "No metadata found for dataset '{}'",
                    dataset_name
                )));
            }
            let metadata =
                storage
                    .load_dataset_metadata(&dataset_name)
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: format!("Failed to load metadata: {}", e),
                    })?;
            let mut output = format!("=== Dataset Metadata: {} ===\n", metadata.name);
            output.push_str(&format!("Version: {}\n", metadata.version));
            output.push_str(&format!("Schema Version: {}\n", metadata.schema_version));
            output.push_str(&format!("Hash: {}\n", metadata.hash));
            output.push_str(&format!("Origin: {:?}\n", metadata.origin));
            if let Some(author) = &metadata.author {
                output.push_str(&format!("Author: {}\n", author));
            }
            if let Some(desc) = &metadata.description {
                output.push_str(&format!("Description: {}\n", desc));
            }
            if !metadata.tags.is_empty() {
                output.push_str(&format!("Tags: {}\n", metadata.tags.join(", ")));
            }
            output.push_str(&format!("Created: {:?}\n", metadata.created_at));
            output.push_str(&format!("Updated: {:?}\n", metadata.updated_at));
            output.push_str("================================");
            Ok(DslOutput::Message(output))
        }

        ShowTarget::DatasetVersions(dataset_name) => {
            let path = format!(
                "{}/{}",
                db.config.storage.data_dir.to_string_lossy(),
                db.active_instance().name
            );
            let storage = crate::core::storage::ParquetStorage::new(&path);
            if !storage.metadata_exists(&dataset_name) {
                return Ok(DslOutput::Message(format!(
                    "No metadata found for dataset '{}'",
                    dataset_name
                )));
            }
            let metadata =
                storage
                    .load_dataset_metadata(&dataset_name)
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

        ShowTarget::Shape(name) => {
            let t = db.get(&name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!(
                "SHAPE {}: {:?}\n",
                name, t.shape.dims
            )))
        }

        ShowTarget::Schema(name) => {
            if let Ok(dataset) = db.get_dataset(&name) {
                let mut output = format!("Schema for dataset '{}' (Legacy):\n", name);
                output.push_str(&format!(
                    "{:<20} {:<20} {:<10}\n",
                    "Field", "Type", "Nullable"
                ));
                output.push_str(&format!("{:-<52}\n", ""));
                for field in &dataset.schema.fields {
                    output.push_str(&format!(
                        "{:<20} {:<20} {:<10}\n",
                        field.name,
                        format!("{:?}", field.value_type),
                        field.nullable
                    ));
                }
                return Ok(DslOutput::Message(output));
            }
            if let Some(ds) = db.get_tensor_dataset(&name) {
                let mut output = format!("Schema for dataset '{}' (Tensor-First):\n", name);
                output.push_str(&format!(
                    "{:<20} {:<20} {:<10} {:<10}\n",
                    "Column", "Type", "Role", "Nullable"
                ));
                output.push_str(&format!("{:-<62}\n", ""));
                for col in &ds.schema.columns {
                    output.push_str(&format!(
                        "{:<20} {:<20} {:<10} {:<10}\n",
                        col.name,
                        format!("{}", col.value_type),
                        format!("{:?}", col.role),
                        col.nullable
                    ));
                }
                return Ok(DslOutput::Message(output));
            }
            Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::DatasetNotFound(name),
            })
        }

        ShowTarget::StringLiteral(s) => Ok(DslOutput::Message(s)),

        ShowTarget::Named(name) => {
            let _ = db.evaluate(&name);
            if let Ok(t) = db.get(&name) {
                return Ok(DslOutput::Tensor(t.clone()));
            }
            if let Ok(dataset) = db.get_dataset(&name) {
                return Ok(DslOutput::Table(dataset.clone()));
            }
            if let Some(ds) = db.get_tensor_dataset(&name) {
                let health_info = db.verify_tensor_dataset(&name).unwrap_or_default();
                return Ok(DslOutput::TensorTable(ds.clone(), health_info));
            }
            Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::NameNotFound(name),
            })
        }
    }
}

fn format_lineage_tree(node: &crate::engine::LineageNode, indent: usize) -> String {
    let mut out = String::new();
    let indent_str = "  ".repeat(indent);
    let name_part = if let Some(name) = &node.name {
        format!(" ({})", name)
    } else {
        String::new()
    };
    out.push_str(&format!(
        "{}{}{} [{}]\n",
        indent_str, node.operation, name_part, node.tensor_id.0
    ));
    for input in &node.inputs {
        out.push_str(&format_lineage_tree(input, indent + 1));
    }
    out
}
