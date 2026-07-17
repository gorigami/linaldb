use crate::core::connectors::{field_with_shape, resolve_shape_dims, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

pub struct ZarrConnector;

/// Accumulates results across a recursive `visit_group` walk. Bundled into
/// one struct (rather than several separate `&mut` params) to keep
/// `visit_group`/`process_array`'s arg count under clippy's threshold, and
/// to carry the optional `FIELDS (...)` selection through the traversal.
struct IngestAccumulator<'a> {
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    num_rows: usize,
    warnings: Vec<String>,
    /// `Some` when the caller passed an explicit `FIELDS (...)` list:
    /// unlisted arrays are skipped silently (the user chose not to
    /// include them), and a listed array that can't be read or doesn't
    /// share the other listed arrays' shape is a hard error rather than a
    /// warned skip, since the caller has no fallback expectation once
    /// they've named exactly what they want.
    requested: Option<&'a [String]>,
    found: std::collections::HashSet<String>,
}

impl<'a> IngestAccumulator<'a> {
    fn new(requested: Option<&'a [String]>) -> Self {
        Self {
            fields: Vec::new(),
            columns: Vec::new(),
            num_rows: 0,
            warnings: Vec::new(),
            requested,
            found: std::collections::HashSet::new(),
        }
    }
}

impl Connector for ZarrConnector {
    fn name(&self) -> &str {
        "zarr"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path_obj = Path::new(path);
        path_obj.extension().and_then(|s| s.to_str()) == Some("zarr")
            || path_obj.join("zarr.json").exists()
            || path_obj.join(".zgroup").exists()
    }

    fn read_dataset(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let store = Arc::new(
            FilesystemStore::new(path)
                .map_err(|e| ConnectorError::Io(std::io::Error::other(e.to_string())))?,
        );

        let mut acc = IngestAccumulator::new(fields);

        self.visit_group(store.clone(), "/", "", &mut acc)?;

        if let Some(requested) = fields {
            let missing: Vec<&String> = requested
                .iter()
                .filter(|n| !acc.found.contains(*n))
                .collect();
            if !missing.is_empty() {
                return Err(ConnectorError::Parse(format!(
                    "FIELDS: array(s) not found in Zarr store '{path}': {}",
                    missing
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }

        if acc.fields.is_empty() {
            return Err(ConnectorError::Parse(
                "No arrays found in Zarr store".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(acc.fields));
        let batch = RecordBatch::try_new(schema, acc.columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "zarr_import".to_string(),
            dataset_hash: "".to_string(),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });
        lineage.warnings = acc.warnings;

        Ok((batch, lineage))
    }

    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError> {
        let (batch, _) = self.read_dataset(path, None)?;

        let fields = batch
            .schema()
            .fields()
            .iter()
            .map(|f| {
                let dims = resolve_shape_dims(f, batch.num_rows());
                ColumnSchema::new(f.name().clone(), ValueType::Float, Shape::new(dims))
            })
            .collect();

        Ok(DatasetSchema::new(fields))
    }
}

impl ZarrConnector {
    fn visit_group(
        &self,
        store: Arc<FilesystemStore>,
        group_path: &str,
        prefix: &str,
        acc: &mut IngestAccumulator,
    ) -> Result<(), ConnectorError> {
        // In zarrs, arrays and groups are distinct. We can try to open as group.
        if let Ok(group) = Group::open(store.clone(), group_path) {
            let children = group
                .children(false)
                .map_err(|e| ConnectorError::Other(e.to_string()))?;

            for member in children.iter() {
                let member_name = member.name();
                let member_path = member.path().to_string();
                let member_prefix: String = if prefix.is_empty() {
                    member_name.to_string()
                } else {
                    format!("{}/{}", prefix, member_name)
                };

                // Try as array first
                if let Ok(array) = Array::open(store.clone(), &member_path) {
                    self.process_array(&array, &member_prefix, acc)?;
                } else {
                    // Try as group (recursive)
                    self.visit_group(
                        store.clone(),
                        &format!("{}/", member_path),
                        &member_prefix,
                        acc,
                    )?;
                }
            }
        } else if let Ok(array) = Array::open(store.clone(), group_path) {
            // Root might be an array
            self.process_array(&array, "data", acc)?;
        }

        Ok(())
    }

    fn process_array(
        &self,
        array: &Array<FilesystemStore>,
        name: &str,
        acc: &mut IngestAccumulator,
    ) -> Result<(), ConnectorError> {
        let requested = matches!(acc.requested, Some(names) if names.iter().any(|n| n == name));
        if acc.requested.is_some() && !requested {
            // Not one of the explicitly-requested arrays -- skip silently,
            // the caller chose not to include it.
            return Ok(());
        }

        let subset = array.subset_all();
        // zarrs 0.19 uses retrieve_array_subset_elements for sync reading.
        // A dtype this can't read as f32 skips just this array (with a
        // warning) rather than aborting the whole store's read via `?` --
        // unless it was explicitly requested, in which case it's a hard
        // error instead.
        let data: Vec<f32> = match array.retrieve_array_subset_elements::<f32>(&subset) {
            Ok(d) => d,
            Err(e) => {
                let msg = format!("Zarr array '{name}': not readable as an f32 array ({e})");
                if requested {
                    return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
                }
                acc.warnings.push(format!("Skipped {msg}"));
                return Ok(());
            }
        };

        if acc.num_rows == 0 {
            acc.num_rows = data.len();
        } else if data.len() != acc.num_rows {
            let msg = format!(
                "Zarr array '{name}': has {} element(s), expected {} \
                 (doesn't match other arrays already ingested from this store)",
                data.len(),
                acc.num_rows
            );
            if requested {
                // Explicitly requested alongside others that don't share
                // its shape -- no fallback expectation once fields are
                // named explicitly.
                return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
            }
            // Inconsistent flattened length vs. the other arrays already
            // ingested from this store -- can't combine into one
            // RecordBatch, so this array is skipped. Zarr groups commonly
            // bundle arrays of different shapes, so warn loudly rather than
            // silently dropping real data.
            acc.warnings.push(format!("Skipped {msg}"));
            return Ok(());
        }

        acc.found.insert(name.to_string());
        let dims: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
        acc.fields
            .push(field_with_shape(name, DataType::Float32, false, &dims));
        acc.columns.push(Arc::new(Float32Array::from(data)));

        Ok(())
    }
}
