use crate::core::connectors::{field_with_shape, resolve_shape_dims, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hdf5::types::{FloatSize, TypeDescriptor};
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
            dataset_hash: crate::core::provenance::record_batch_content_hash(&batch),
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
                let vt = match f.data_type() {
                    DataType::Float64 => ValueType::Float64,
                    _ => ValueType::Float,
                };
                ColumnSchema::new(f.name().clone(), vt, Shape::new(dims))
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

        // HDF5's C library does implicit numeric type conversion on read,
        // so `read_raw::<f32>()` on a genuinely double-precision dataset
        // usually *succeeds* (silently narrowing) rather than erroring --
        // the on-disk dtype has to be checked up front to actually preserve
        // f64 precision; relying on a read error as the f64 signal would
        // almost never fire for real double datasets.
        let is_declared_f64 = ds
            .dtype()
            .ok()
            .and_then(|dt| dt.to_descriptor().ok())
            .is_some_and(|td| matches!(td, TypeDescriptor::Float(FloatSize::U8)));

        let data = if is_declared_f64 {
            match ds.read_raw::<f64>() {
                Ok(v) => HdfNumericColumn::F64(v),
                Err(e) => return self.skip_or_error(name, requested, acc, &e.to_string()),
            }
        } else {
            match ds.read_raw::<f32>() {
                Ok(v) => HdfNumericColumn::F32(v),
                Err(_) => match ds.read_raw::<f64>() {
                    // Dtype detection didn't flag this as f64 up front (or
                    // the dataset's declared type just isn't f32-readable),
                    // but an f64 read still worked -- keep full precision
                    // rather than narrowing, same policy as the declared
                    // case above.
                    Ok(v) => HdfNumericColumn::F64(v),
                    Err(e) => return self.skip_or_error(name, requested, acc, &e.to_string()),
                },
            }
        };

        let len = data.len();
        if acc.num_rows == 0 {
            acc.num_rows = len;
        } else if len != acc.num_rows {
            let msg = format!(
                "HDF5 dataset '{name}': has {} element(s), expected {} \
                 (doesn't match other datasets already ingested from this file)",
                len, acc.num_rows
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
        match data {
            HdfNumericColumn::F32(v) => {
                acc.fields
                    .push(field_with_shape(name, DataType::Float32, false, &shape));
                acc.columns.push(Arc::new(Float32Array::from(v)));
            }
            HdfNumericColumn::F64(v) => {
                acc.fields
                    .push(field_with_shape(name, DataType::Float64, false, &shape));
                acc.columns.push(Arc::new(Float64Array::from(v)));
            }
        }

        Ok(())
    }

    /// Shared "explicitly-requested field errors loudly, otherwise skip with
    /// a warning" policy for a dataset that couldn't be read numerically at
    /// all (used by both the declared-f64 and f32-then-f64-fallback paths).
    fn skip_or_error(
        &self,
        name: &str,
        requested: bool,
        acc: &mut IngestAccumulator,
        err: &str,
    ) -> Result<(), ConnectorError> {
        let msg = format!(
            "HDF5 dataset '{name}': not readable as a numeric (float-convertible) array ({err})"
        );
        if requested {
            return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
        }
        acc.warnings.push(format!("Skipped {msg}"));
        Ok(())
    }
}

/// A numeric HDF5 dataset read at either its declared precision (f64) or
/// LINAL's default (f32).
enum HdfNumericColumn {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl HdfNumericColumn {
    fn len(&self) -> usize {
        match self {
            HdfNumericColumn::F32(v) => v.len(),
            HdfNumericColumn::F64(v) => v.len(),
        }
    }
}
