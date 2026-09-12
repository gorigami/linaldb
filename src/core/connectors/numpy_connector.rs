use crate::core::connectors::{field_with_shape, resolve_shape_dims, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use ndarray::ArrayD;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub struct NumpyConnector;

impl Connector for NumpyConnector {
    fn name(&self) -> &str {
        "numpy"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path = Path::new(path);
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("npy") | Some("npz")
        )
    }

    fn read_dataset(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ConnectorError::Parse("Missing file extension".to_string()))?;

        if ext == "npz" {
            self.read_npz(path, fields)
        } else {
            self.read_npy(path, fields)
        }
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

impl NumpyConnector {
    fn read_npy(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        if let Some(names) = fields {
            if names.iter().any(|n| n != "array") {
                return Err(ConnectorError::Parse(format!(
                    "FIELDS: a single .npy file only has one array, named \"array\" \
                     (requested: {})",
                    names.join(", ")
                )));
            }
        }

        // The NPY header encodes the array's exact on-disk dtype, so unlike
        // HDF5 (which does implicit numeric conversion), ndarray_npy's
        // ArrayD<f32> read genuinely errors on a real float64 file rather
        // than silently narrowing -- the error-triggered fallback is a
        // reliable f64 signal here.
        let arr: NumpyArray = match ndarray_npy::read_npy::<_, ArrayD<f32>>(path) {
            Ok(v) => NumpyArray::F32(v),
            Err(_) => {
                let v: ArrayD<f64> = ndarray_npy::read_npy(path)
                    .map_err(|e| ConnectorError::Parse(format!("Failed to read NPY: {}", e)))?;
                NumpyArray::F64(v)
            }
        };

        let (batch, lineage) = self.array_to_batch("array", arr)?;
        Ok((batch, lineage))
    }

    fn read_npz(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = File::open(path)?;
        let mut npz = ndarray_npy::NpzReader::new(file)
            .map_err(|e| ConnectorError::Parse(format!("Failed to open NPZ: {}", e)))?;

        let mut out_fields = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        let mut num_rows = 0;
        let mut warnings = Vec::new();
        let mut found = std::collections::HashSet::new();

        let names: Vec<String> = npz
            .names()
            .map_err(|e| ConnectorError::Parse(format!("Failed to list NPZ names: {}", e)))?
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        for name in names {
            let requested = matches!(fields, Some(names) if names.iter().any(|n| n == &name));
            if fields.is_some() && !requested {
                // Not one of the explicitly-requested arrays -- skip
                // silently, the caller chose not to include it.
                continue;
            }

            // f32 first, f64 fallback on error -- reliable here since NPZ
            // (like NPY) encodes the exact on-disk dtype and ndarray_npy
            // doesn't silently convert.
            let f32_result: Result<ArrayD<f32>, _> = npz.by_name(&name);
            let arr: NumpyArray = match f32_result {
                Ok(arr) => NumpyArray::F32(arr),
                Err(_) => {
                    let f64_result: Result<ArrayD<f64>, _> = npz.by_name(&name);
                    match f64_result {
                        Ok(arr) => NumpyArray::F64(arr),
                        Err(e) => {
                            let msg = format!(
                                "NPZ array '{name}': not readable as an f32 or f64 array ({e})"
                            );
                            if requested {
                                return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
                            }
                            // Not explicitly requested: skip, but say so --
                            // silently dropping a real array with no
                            // indication at all is worse than a loud skip.
                            warnings.push(format!("Skipped {msg}"));
                            continue;
                        }
                    }
                }
            };

            let len = arr.len();
            if num_rows == 0 {
                num_rows = len;
            } else if len != num_rows {
                let msg = format!(
                    "NPZ array '{name}': has {len} element(s), expected {num_rows} \
                     (doesn't match other arrays already ingested from this file)"
                );
                if requested {
                    // Explicitly requested alongside others that don't
                    // share its shape -- no fallback expectation once
                    // fields are named explicitly.
                    return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
                }
                // Inconsistent flattened length vs. the other arrays already
                // ingested from this file -- can't combine into one
                // RecordBatch, so this array is skipped. NPZ archives
                // commonly bundle arrays of different shapes, so warn
                // loudly rather than silently dropping real data.
                warnings.push(format!("Skipped {msg}"));
                continue;
            }

            found.insert(name.clone());
            let dims = arr.dims();
            match arr {
                NumpyArray::F32(arr) => {
                    out_fields.push(field_with_shape(&name, DataType::Float32, false, &dims));
                    let data: Vec<f32> = arr.iter().cloned().collect();
                    columns.push(Arc::new(Float32Array::from(data)));
                }
                NumpyArray::F64(arr) => {
                    out_fields.push(field_with_shape(&name, DataType::Float64, false, &dims));
                    let data: Vec<f64> = arr.iter().cloned().collect();
                    columns.push(Arc::new(Float64Array::from(data)));
                }
            }
        }

        if let Some(requested) = fields {
            let missing: Vec<&String> = requested.iter().filter(|n| !found.contains(*n)).collect();
            if !missing.is_empty() {
                return Err(ConnectorError::Parse(format!(
                    "FIELDS: array(s) not found in NPZ file '{path}': {}",
                    missing
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }

        if out_fields.is_empty() {
            return Err(ConnectorError::Parse(
                "No valid f32/f64 arrays found in NPZ".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(out_fields));
        let batch = RecordBatch::try_new(schema, columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "numpy_import".to_string(),
            dataset_hash: crate::core::provenance::record_batch_content_hash(&batch),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });
        lineage.warnings = warnings;

        Ok((batch, lineage))
    }

    fn array_to_batch(
        &self,
        name: &str,
        arr: NumpyArray,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let dims = arr.dims();
        let (data_type, array): (DataType, ArrayRef) = match arr {
            NumpyArray::F32(arr) => {
                let data: Vec<f32> = arr.iter().cloned().collect();
                (DataType::Float32, Arc::new(Float32Array::from(data)))
            }
            NumpyArray::F64(arr) => {
                let data: Vec<f64> = arr.iter().cloned().collect();
                (DataType::Float64, Arc::new(Float64Array::from(data)))
            }
        };

        let schema = Arc::new(Schema::new(vec![field_with_shape(
            name, data_type, false, &dims,
        )]));

        let batch = RecordBatch::try_new(schema, vec![array])?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "numpy_import".to_string(),
            dataset_hash: crate::core::provenance::record_batch_content_hash(&batch),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((batch, lineage))
    }
}

/// A NumPy array read at either its declared precision (f64) or LINAL's
/// default (f32) -- NPY/NPZ encode the exact on-disk dtype, so which variant
/// is produced is determined by a real read attempt, not a heuristic.
enum NumpyArray {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

impl NumpyArray {
    fn len(&self) -> usize {
        match self {
            NumpyArray::F32(a) => a.len(),
            NumpyArray::F64(a) => a.len(),
        }
    }

    fn dims(&self) -> Vec<usize> {
        match self {
            NumpyArray::F32(a) => a.shape().to_vec(),
            NumpyArray::F64(a) => a.shape().to_vec(),
        }
    }
}
