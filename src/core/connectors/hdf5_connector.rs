use crate::core::connectors::{field_with_shape, resolve_shape_dims, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hdf5::{Dataset, File, Group};
use std::path::Path;
use std::sync::Arc;

pub struct Hdf5Connector;

/// Accumulates results across a recursive `visit_group` walk. Bundled into
/// one struct (rather than several separate `&mut` params) to keep
/// `visit_group`/`process_dataset`'s arg count reasonable, and to carry the
/// optional `FIELDS (...)` selection through the traversal.
struct IngestAccumulator<'a> {
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    num_rows: usize,
    warnings: Vec<String>,
    /// `Some` when the caller passed an explicit `FIELDS (...)` list:
    /// unlisted datasets are skipped silently (the user chose not to
    /// include them), and a listed dataset that can't be read or doesn't
    /// share the other listed datasets' shape is a hard error rather than
    /// a warned skip, since the caller has no fallback expectation once
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

impl Connector for Hdf5Connector {
    fn name(&self) -> &str {
        "hdf5"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path = Path::new(path);
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("h5") | Some("hdf5") | Some("h5ad") | Some("nc")
        )
    }

    fn read_dataset(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = File::open(path)
            .map_err(|e| ConnectorError::Io(std::io::Error::other(e.to_string())))?;

        let mut acc = IngestAccumulator::new(fields);
        self.visit_group(&file, "", &mut acc)?;

        if let Some(requested) = fields {
            let missing: Vec<&String> = requested
                .iter()
                .filter(|n| !acc.found.contains(*n))
                .collect();
            if !missing.is_empty() {
                return Err(ConnectorError::Parse(format!(
                    "FIELDS: dataset(s) not found in HDF5 file '{path}': {}",
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
                "No datasets found in HDF5 file".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(acc.fields));
        let batch = RecordBatch::try_new(schema, acc.columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "hdf5_import".to_string(),
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

impl Hdf5Connector {
    fn visit_group(
        &self,
        group: &Group,
        prefix: &str,
        acc: &mut IngestAccumulator,
    ) -> Result<(), ConnectorError> {
        // Visit datasets in this group
        for member_name in group
            .member_names()
            .map_err(|e| ConnectorError::Other(e.to_string()))?
        {
            let name = if prefix.is_empty() {
                member_name.clone()
            } else {
                format!("{}_{}", prefix, member_name)
            };

            // Check if it's a dataset or a group
            if let Ok(ds) = group.dataset(&member_name) {
                self.process_dataset(&ds, &name, acc)?;
            } else if let Ok(subgroup) = group.group(&member_name) {
                self.visit_group(&subgroup, &name, acc)?;
            }
        }
        Ok(())
    }

    fn process_dataset(
        &self,
        ds: &Dataset,
        name: &str,
        acc: &mut IngestAccumulator,
    ) -> Result<(), ConnectorError> {
        let requested = matches!(acc.requested, Some(names) if names.iter().any(|n| n == name));
        if acc.requested.is_some() && !requested {
            // Not one of the explicitly-requested fields -- skip silently,
            // the caller chose not to include it.
            return Ok(());
        }

        // We only support numeric datasets for now. The underlying Arrow
        // column is always a flat 1D array (LINAL's connector output
        // convention), but we stash the dataset's original shape as field
        // metadata so record_batch_to_tensors can rebuild the real
        // Matrix/Tensor shape instead of assuming a flat Vector.
        let shape = ds.shape();

        let data: Vec<f32> = match ds.read_raw::<f32>() {
            Ok(v) => v,
            Err(_) => {
                // Try reading as f64 and casting
                match ds.read_raw::<f64>() {
                    Ok(v) => v.into_iter().map(|x| x as f32).collect(),
                    Err(e) => {
                        let msg = format!(
                            "HDF5 dataset '{name}': not readable as a numeric \
                             (float-convertible) array ({e})"
                        );
                        if requested {
                            return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
                        }
                        // Not explicitly requested: skip, but say so --
                        // silently dropping a real dataset with no
                        // indication at all is worse than a loud skip.
                        acc.warnings.push(format!("Skipped {msg}"));
                        return Ok(());
                    }
                }
            }
        };

        if acc.num_rows == 0 {
            acc.num_rows = data.len();
        } else if data.len() != acc.num_rows {
            let msg = format!(
                "HDF5 dataset '{name}': has {} element(s), expected {} \
                 (doesn't match other datasets already ingested from this file)",
                data.len(),
                acc.num_rows
            );
            if requested {
                // The caller explicitly asked for this field alongside
                // others that don't share its shape -- can't silently
                // drop it since there's no fallback expectation once
                // fields are named explicitly.
                return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
            }
            // Inconsistent flattened length vs. the other datasets already
            // ingested from this file -- can't combine into one RecordBatch
            // (Arrow columns in a batch must share a row count), so this
            // dataset is skipped. HDF5 files commonly bundle arrays of
            // different shapes (data + labels + metadata), so warn loudly
            // rather than silently dropping real data.
            acc.warnings.push(format!("Skipped {msg}"));
            return Ok(());
        }

        acc.found.insert(name.to_string());
        acc.fields
            .push(field_with_shape(name, DataType::Float32, false, &shape));
        acc.columns.push(Arc::new(Float32Array::from(data)));

        Ok(())
    }
}
