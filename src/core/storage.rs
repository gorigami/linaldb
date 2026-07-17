use crate::core::dataset::{
    ColumnSchema, ColumnStats, DatasetLineage, DatasetManifest, DatasetMetadata, DatasetOrigin,
    DatasetSchema, DatasetStats, LineageNode,
};
use crate::core::dataset_legacy::{Dataset, DatasetMetadata as DatasetMetadataLegacy};
use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use crate::core::tuple::{Schema, Tuple};
use crate::core::value::{Value, ValueType};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_csv as csv;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

/// Storage engine trait for persisting datasets and tensors
pub trait StorageEngine {
    /// Save a dataset to storage
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError>;

    /// Load a dataset from storage
    fn load_dataset(&self, name: &str) -> Result<Dataset, StorageError>;

    /// Save a reference-based dataset
    fn save_reference_dataset(
        &self,
        dataset: &crate::core::dataset::Dataset,
    ) -> Result<(), StorageError>;

    /// Load a reference-based dataset
    fn load_reference_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::Dataset, StorageError>;

    /// Check if a dataset exists
    fn dataset_exists(&self, name: &str) -> bool;

    /// Delete a dataset
    fn delete_dataset(&self, name: &str) -> Result<(), StorageError>;

    /// List all datasets
    fn list_datasets(&self) -> Result<Vec<String>, StorageError>;

    /// Save a tensor to storage
    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError>;

    /// Load a tensor from storage
    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError>;

    /// Check if a tensor exists
    fn tensor_exists(&self, name: &str) -> bool;

    /// Delete a tensor
    fn delete_tensor(&self, name: &str) -> Result<(), StorageError>;

    /// List all tensors
    fn list_tensors(&self) -> Result<Vec<String>, StorageError>;
}

/// Parquet-based storage implementation
pub struct ParquetStorage {
    base_path: String,
}

impl ParquetStorage {
    pub fn new(base_path: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    pub fn base_path(&self) -> &str {
        &self.base_path
    }

    fn dataset_dir(&self, name: &str) -> String {
        format!("{}/datasets/{}", self.base_path, name)
    }

    fn dataset_data_path(&self, name: &str) -> String {
        format!("{}/datasets/{}/data.parquet", self.base_path, name)
    }

    fn dataset_schema_path(&self, name: &str) -> String {
        format!("{}/datasets/{}/schema.json", self.base_path, name)
    }

    fn dataset_stats_path(&self, name: &str) -> String {
        format!("{}/datasets/{}/stats.json", self.base_path, name)
    }

    fn dataset_lineage_path(&self, name: &str) -> String {
        format!("{}/datasets/{}/lineage.json", self.base_path, name)
    }

    fn dataset_manifest_path(&self, name: &str) -> String {
        format!("{}/datasets/{}/manifest.json", self.base_path, name)
    }

    fn legacy_metadata_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.meta.json", self.base_path, name)
    }

    fn legacy_persistent_metadata_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.metadata.json", self.base_path, name)
    }

    fn tensor_path(&self, name: &str) -> String {
        format!("{}/tensors/{}.json", self.base_path, name)
    }

    fn ensure_directories(&self, name: Option<&str>) -> Result<(), StorageError> {
        let datasets_dir = format!("{}/datasets", self.base_path);
        let tensors_dir = format!("{}/tensors", self.base_path);
        fs::create_dir_all(&datasets_dir)?;
        fs::create_dir_all(&tensors_dir)?;

        if let Some(n) = name {
            let ds_dir = self.dataset_dir(n);
            fs::create_dir_all(ds_dir)?;
        }
        Ok(())
    }

    /// Save dataset metadata to JSON file
    pub fn save_dataset_metadata(
        &self,
        metadata: &crate::core::dataset::DatasetMetadata,
    ) -> Result<(), StorageError> {
        self.ensure_directories(Some(&metadata.name))?;
        let path = self.legacy_persistent_metadata_path(&metadata.name);
        let json = serde_json::to_string_pretty(metadata).map_err(|e| {
            StorageError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load dataset metadata from JSON file
    pub fn load_dataset_metadata(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::DatasetMetadata, StorageError> {
        let path = self.legacy_persistent_metadata_path(name);
        if !Path::new(&path).exists() {
            return Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Metadata file not found: {}", path),
            )));
        }
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json).map_err(|e| {
            StorageError::Serialization(format!("Failed to deserialize metadata: {}", e))
        })?;
        Ok(metadata)
    }

    /// Check if metadata exists for a dataset
    pub fn metadata_exists(&self, name: &str) -> bool {
        Path::new(&self.legacy_persistent_metadata_path(name)).exists()
    }

    /// Check if a delivery manifest exists for a dataset — i.e. it has been
    /// persisted as a full dataset package (data.parquet + schema.json +
    /// stats.json + lineage.json + manifest.json), which is what the
    /// `/delivery` HTTP routes serve.
    pub fn manifest_exists(&self, name: &str) -> bool {
        Path::new(&self.dataset_manifest_path(name)).exists()
    }
    /// Save a complete dataset package (folder-based)
    pub fn save_dataset_package(
        &self,
        name: &str,
        batch: &RecordBatch,
        metadata: &DatasetMetadata,
        lineage: &DatasetLineage,
    ) -> Result<(), StorageError> {
        self.ensure_directories(Some(name))?;

        // 1. Save data.parquet
        let data_path = self.dataset_data_path(name);
        let file = fs::File::create(&data_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(batch)?;
        writer.close()?;

        // 2. Save schema.json
        let schema_path = self.dataset_schema_path(name);
        let schema: DatasetSchema = batch.schema().clone().into();
        let schema_json = serde_json::to_string_pretty(&schema)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(schema_path, schema_json)?;

        // 3. Save stats.json
        let stats_path = self.dataset_stats_path(name);
        let stats = self.compute_stats(batch);
        let stats_json = serde_json::to_string_pretty(&stats)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(stats_path, stats_json)?;

        // 4. Save lineage.json
        let lineage_path = self.dataset_lineage_path(name);
        let lineage_json = serde_json::to_string_pretty(lineage)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(lineage_path, lineage_json)?;

        // 5. Save manifest.json
        let manifest_path = self.dataset_manifest_path(name);
        let mut manifest =
            DatasetManifest::new(name.to_string(), "1.0".to_string(), metadata.hash.clone());
        manifest
            .formats
            .insert("parquet".to_string(), "data.parquet".to_string());
        manifest
            .entrypoints
            .insert("default".to_string(), "data.parquet".to_string());

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(manifest_path, manifest_json)?;

        Ok(())
    }

    /// Write the legacy `.meta.json` sidecar `StorageEngine::load_dataset`
    /// hard-requires to exist. Shared by `StorageEngine::save_dataset` (which
    /// already has a fully populated metadata object with real per-column
    /// stats accumulated over the dataset's lifetime) and
    /// `save_legacy_metadata_for_batch` below (which doesn't).
    pub fn save_legacy_metadata(
        &self,
        name: &str,
        metadata: &DatasetMetadataLegacy,
    ) -> Result<(), StorageError> {
        self.ensure_directories(Some(name))?;
        let meta_path = self.legacy_metadata_path(name);
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&meta_path, metadata_json)?;
        Ok(())
    }

    /// Build a minimal legacy metadata object (schema + row_count only, no
    /// per-column `ColumnStats` — those need row-level Tuple scanning that
    /// connector-sourced imports never do, and `load_dataset` never reads
    /// them) directly from an Arrow `RecordBatch`, and write it as the
    /// `.meta.json` sidecar. Used by `import_dataset_core` for
    /// connector-sourced imports (HDF5/Numpy/Zarr/CSV-via-IMPORT), which
    /// otherwise write a `save_dataset_package` output that `load_dataset`
    /// can never find (missing this exact file).
    pub fn save_legacy_metadata_for_batch(
        &self,
        name: &str,
        batch: &RecordBatch,
    ) -> Result<(), StorageError> {
        let schema = arrow_schema_to_tuple_schema(batch.schema().as_ref());
        let mut metadata = DatasetMetadataLegacy::new(Some(name.to_string()), schema);
        metadata.row_count = batch.num_rows();
        self.save_legacy_metadata(name, &metadata)
    }

    fn compute_stats(&self, batch: &RecordBatch) -> DatasetStats {
        let mut stats = DatasetStats::new(batch.num_rows() as u64);
        for field in batch.schema().fields() {
            let col_name = field.name();
            let array = batch.column_by_name(col_name).unwrap();
            let null_count = array.null_count() as u64;

            let (min, max, mean) = match field.data_type() {
                DataType::Float32 => {
                    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
                        let vals: Vec<f32> = arr.iter().flatten().collect();
                        if vals.is_empty() {
                            (None, None, None)
                        } else {
                            let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                            let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let avg = vals.iter().sum::<f32>() / vals.len() as f32;
                            (Some(mn), Some(mx), Some(avg))
                        }
                    } else {
                        (None, None, None)
                    }
                }
                DataType::Float64 => {
                    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                        let vals: Vec<f64> = arr.iter().flatten().collect();
                        if vals.is_empty() {
                            (None, None, None)
                        } else {
                            let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                            let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
                            (Some(mn as f32), Some(mx as f32), Some(avg as f32))
                        }
                    } else {
                        (None, None, None)
                    }
                }
                DataType::Int64 => {
                    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                        let vals: Vec<i64> = arr.iter().flatten().collect();
                        if vals.is_empty() {
                            (None, None, None)
                        } else {
                            let mn = *vals.iter().min().unwrap();
                            let mx = *vals.iter().max().unwrap();
                            let avg = vals.iter().sum::<i64>() as f32 / vals.len() as f32;
                            (Some(mn as f32), Some(mx as f32), Some(avg))
                        }
                    } else {
                        (None, None, None)
                    }
                }
                DataType::Int32 => {
                    if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                        let vals: Vec<i32> = arr.iter().flatten().collect();
                        if vals.is_empty() {
                            (None, None, None)
                        } else {
                            let mn = *vals.iter().min().unwrap();
                            let mx = *vals.iter().max().unwrap();
                            let avg = vals.iter().sum::<i32>() as f32 / vals.len() as f32;
                            (Some(mn as f32), Some(mx as f32), Some(avg))
                        }
                    } else {
                        (None, None, None)
                    }
                }
                _ => (None, None, None),
            };

            stats.add_column_stats(
                col_name.to_string(),
                ColumnStats {
                    min,
                    max,
                    mean,
                    sparsity: Some(1.0 - (null_count as f32 / batch.num_rows() as f32)),
                    null_count,
                    tensor_shape: None,
                },
            );
        }
        stats
    }
}

impl From<Arc<ArrowSchema>> for DatasetSchema {
    fn from(arrow_schema: Arc<ArrowSchema>) -> Self {
        let columns = arrow_schema
            .fields()
            .iter()
            .map(|f| {
                let vt = match f.data_type() {
                    DataType::Int64 => ValueType::Int,
                    DataType::Float32 | DataType::Float64 => ValueType::Float,
                    DataType::Boolean => ValueType::Bool,
                    _ => ValueType::String,
                };
                ColumnSchema::new(
                    f.name().clone(),
                    vt,
                    crate::core::tensor::Shape::new(vec![]),
                )
                .with_nullable(f.is_nullable())
            })
            .collect();
        Self { columns }
    }
}

/// Convert an Arrow schema into LINAL's legacy relational `core::tuple::Schema`,
/// used as the authoritative schema `load_dataset` reconstructs rows against.
/// Mirrors `dataset_to_record_batch`'s reverse mapping (Int, Float, String,
/// Bool); anything else falls back to `String` — not reachable by any current
/// connector (HDF5/Numpy/Zarr only ever emit Float32; CSV import produces
/// Int64/Float64/Utf8/Boolean), and `load_dataset` will surface a clean
/// `StorageError` rather than silently corrupt data if it ever is.
fn arrow_schema_to_tuple_schema(arrow_schema: &ArrowSchema) -> Schema {
    let fields = arrow_schema
        .fields()
        .iter()
        .map(|f| {
            let value_type = match f.data_type() {
                DataType::Int64 | DataType::Int32 => ValueType::Int,
                DataType::Float32 | DataType::Float64 => ValueType::Float,
                DataType::Utf8 | DataType::LargeUtf8 => ValueType::String,
                DataType::Boolean => ValueType::Bool,
                _ => ValueType::String,
            };
            let mut field = crate::core::tuple::Field::new(f.name().clone(), value_type);
            if f.is_nullable() {
                field = field.nullable();
            }
            field
        })
        .collect();
    Schema::new(fields)
}

/// Convert Arrow RecordBatch to LINAL Rows
pub fn record_batch_to_rows(
    batch: &RecordBatch,
    schema: &Arc<Schema>,
) -> Result<Vec<Tuple>, StorageError> {
    let num_rows = batch.num_rows();
    let mut tuples = Vec::with_capacity(num_rows);

    let mut columns_data: Vec<Vec<Value>> = Vec::new();

    for field in &schema.fields {
        let arrow_col = batch.column_by_name(&field.name).ok_or_else(|| {
            StorageError::Serialization(format!("Column {} missing in Arrow batch", field.name))
        })?;

        let values = arrow_array_to_values(arrow_col, &field.value_type, num_rows)?;
        columns_data.push(values);
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..num_rows {
        let mut row_values = Vec::with_capacity(schema.fields.len());
        for col_idx in 0..schema.fields.len() {
            row_values.push(columns_data[col_idx][i].clone());
        }
        tuples.push(Tuple::new(schema.clone(), row_values).map_err(StorageError::Serialization)?);
    }

    Ok(tuples)
}

fn arrow_array_to_values(
    array: &ArrayRef,
    target_type: &ValueType,
    num_rows: usize,
) -> Result<Vec<Value>, StorageError> {
    match target_type {
        ValueType::Int => {
            if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if int_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Int(int_array.value(i))
                        }
                    })
                    .collect())
            } else if let Some(int_array) = array.as_any().downcast_ref::<Int32Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if int_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Int(int_array.value(i) as i64)
                        }
                    })
                    .collect())
            } else {
                Err(StorageError::Serialization(
                    "Expected Int64Array or Int32Array for Int type".to_string(),
                ))
            }
        }
        ValueType::Float => {
            if let Some(float_array) = array.as_any().downcast_ref::<Float32Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if float_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Float(float_array.value(i))
                        }
                    })
                    .collect())
            } else if let Some(double_array) = array.as_any().downcast_ref::<Float64Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if double_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Float(double_array.value(i) as f32)
                        }
                    })
                    .collect())
            } else {
                Err(StorageError::Serialization(
                    "Expected Float32Array or Float64Array for Float type".to_string(),
                ))
            }
        }
        ValueType::String => {
            let string_array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| StorageError::Serialization("Expected StringArray".to_string()))?;
            Ok((0..num_rows)
                .map(|i| {
                    if string_array.is_null(i) {
                        Value::Null
                    } else {
                        Value::String(string_array.value(i).to_string())
                    }
                })
                .collect())
        }
        ValueType::Bool => {
            let bool_array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| StorageError::Serialization("Expected BooleanArray".to_string()))?;
            Ok((0..num_rows)
                .map(|i| {
                    if bool_array.is_null(i) {
                        Value::Null
                    } else {
                        Value::Bool(bool_array.value(i))
                    }
                })
                .collect())
        }
        ValueType::Vector(_) | ValueType::Matrix(_, _) => {
            let string_array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StorageError::Serialization("Expected StringArray for complex type".to_string())
                })?;
            Ok((0..num_rows)
                .map(|i| {
                    if string_array.is_null(i) {
                        Value::Null
                    } else {
                        let json_str = string_array.value(i);
                        serde_json::from_str(json_str).unwrap_or(Value::Null)
                    }
                })
                .collect())
        }
        ValueType::Null => Ok(vec![Value::Null; num_rows]),
    }
}

/// Convert Arrow ArrayRef to f32 vector for LINAL tensors
pub fn arrow_array_to_f32_vec(array: &ArrayRef) -> Result<Vec<f32>, StorageError> {
    let num_rows = array.len();
    if let Some(float_array) = array.as_any().downcast_ref::<Float32Array>() {
        Ok((0..num_rows)
            .map(|i| {
                if float_array.is_null(i) {
                    0.0
                } else {
                    float_array.value(i)
                }
            })
            .collect())
    } else if let Some(double_array) = array.as_any().downcast_ref::<Float64Array>() {
        Ok((0..num_rows)
            .map(|i| {
                if double_array.is_null(i) {
                    0.0
                } else {
                    double_array.value(i) as f32
                }
            })
            .collect())
    } else if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
        Ok((0..num_rows)
            .map(|i| {
                if int_array.is_null(i) {
                    0.0
                } else {
                    int_array.value(i) as f32
                }
            })
            .collect())
    } else if let Some(int_array) = array.as_any().downcast_ref::<Int32Array>() {
        Ok((0..num_rows)
            .map(|i| {
                if int_array.is_null(i) {
                    0.0
                } else {
                    int_array.value(i) as f32
                }
            })
            .collect())
    } else {
        Err(StorageError::Serialization(format!(
            "Unsupported Arrow type for tensor conversion: {:?}",
            array.data_type()
        )))
    }
}

/// Convert RecordBatch to a list of named Tensors
pub fn record_batch_to_tensors(batch: &RecordBatch) -> Result<Vec<(String, Tensor)>, StorageError> {
    let mut tensors = Vec::new();
    let num_rows = batch.num_rows();

    for field in batch.schema().fields() {
        let col_name = field.name();
        let array = batch.column_by_name(col_name).ok_or_else(|| {
            StorageError::Serialization(format!("Column {} missing in batch", col_name))
        })?;

        let data = arrow_array_to_f32_vec(array)?;
        // Connectors stash the original N-D shape as field metadata (see
        // core::connectors::field_with_shape); use it when present and
        // consistent with the actual data length, otherwise fall back to
        // the flat-vector assumption (also covers relational/CSV imports,
        // which never attach shape metadata at all).
        let dims = crate::core::connectors::read_shape_metadata(field)
            .filter(|dims| dims.iter().product::<usize>() == data.len())
            .unwrap_or_else(|| vec![num_rows]);
        let shape = Shape::new(dims);
        let id = TensorId::new();
        let metadata = TensorMetadata::new(id, Some("ingestion".to_string()));
        let tensor = Tensor::new(id, shape, data, metadata).map_err(StorageError::Serialization)?;

        tensors.push((col_name.clone(), tensor));
    }

    Ok(tensors)
}

/// Convert Dataset to Arrow RecordBatch
pub fn dataset_to_record_batch(dataset: &Dataset) -> Result<RecordBatch, StorageError> {
    // Build Arrow schema from dataset schema
    let arrow_fields: Vec<ArrowField> = dataset
        .schema
        .fields
        .iter()
        .map(|f| {
            let data_type = match &f.value_type {
                ValueType::Int => DataType::Int64,
                ValueType::Float => DataType::Float32,
                ValueType::String => DataType::Utf8,
                ValueType::Bool => DataType::Boolean,
                _ => DataType::Utf8, // Fallback for complex types (serialize as JSON string)
            };
            ArrowField::new(&f.name, data_type, f.nullable)
        })
        .collect();

    let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));

    // Convert rows to Arrow arrays
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for field in &dataset.schema.fields {
        let column_data: Vec<&Value> = dataset
            .rows
            .iter()
            .map(|row| {
                row.values
                    .iter()
                    .zip(&row.schema.fields)
                    .find(|(_, f)| f.name == field.name)
                    .map(|(v, _)| v)
                    .unwrap_or(&Value::Null)
            })
            .collect();

        let array: ArrayRef = match &field.value_type {
            ValueType::Int => {
                let values: Vec<Option<i64>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Int(i) => Some(*i),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(Int64Array::from(values))
            }
            ValueType::Float => {
                let values: Vec<Option<f32>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Float(f) => Some(*f),
                        Value::Int(i) => Some(*i as f32),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(Float32Array::from(values))
            }
            ValueType::String => {
                let values: Vec<Option<&str>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => Some(s.as_str()),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(StringArray::from(values))
            }
            ValueType::Bool => {
                let values: Vec<Option<bool>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Bool(b) => Some(*b),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(BooleanArray::from(values))
            }
            _ => {
                // For complex types (Vector, Matrix), serialize as JSON strings
                let values: Vec<Option<String>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Null => None,
                        v => Some(serde_json::to_string(v).unwrap_or_else(|_| "null".to_string())),
                    })
                    .collect();
                Arc::new(StringArray::from(values))
            }
        };

        arrays.push(array);
    }

    RecordBatch::try_new(arrow_schema, arrays).map_err(StorageError::Arrow)
}

impl StorageEngine for ParquetStorage {
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError> {
        let dataset_name =
            dataset.metadata.name.as_ref().ok_or_else(|| {
                StorageError::Serialization("Dataset must have a name".to_string())
            })?;

        // 1. Convert to RecordBatch
        let record_batch = dataset_to_record_batch(dataset)?;

        // 2. Generate new Metadata
        let mut metadata = DatasetMetadata::new(dataset_name.to_string(), DatasetOrigin::Created);

        // Compute content hash (simple hash based on row count and name)
        let content_hash = format!("{}:{}", dataset_name, dataset.rows.len());
        metadata.update_hash(content_hash);

        // 3. Generate minimal Lineage
        let mut lineage = DatasetLineage::new();
        lineage.add_node(LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: dataset_name.to_string(),
            dataset_hash: metadata.hash.clone(),
            operation: "SAVE (Legacy)".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        // 4. Save as full package
        self.save_dataset_package(dataset_name, &record_batch, &metadata, &lineage)?;

        // 5. Also save legacy metadata for backward compatibility (optional but safer for now)
        self.save_legacy_metadata(dataset_name, &dataset.metadata)?;

        Ok(())
    }

    fn load_dataset(&self, name: &str) -> Result<Dataset, StorageError> {
        let meta_path = self.legacy_metadata_path(name);
        if !Path::new(&meta_path).exists() {
            return Err(StorageError::DatasetNotFound(name.to_string()));
        }

        let metadata_json = fs::read_to_string(&meta_path)?;
        let metadata: DatasetMetadataLegacy = serde_json::from_str(&metadata_json)
            .map_err(|e| StorageError::Serialization(format!("Metadata error: {}", e)))?;

        let data_path = self.dataset_data_path(name);
        if !Path::new(&data_path).exists() {
            return Err(StorageError::DatasetNotFound(format!(
                "Data file missing for {}",
                name
            )));
        }

        let file = fs::File::open(&data_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let record_batch_reader = builder
            .with_batch_size(2048)
            .build()
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let mut rows = Vec::new();
        let schema = Arc::new(metadata.schema.clone());

        for batch in record_batch_reader {
            let batch = batch?;
            let batch_rows = record_batch_to_rows(&batch, &schema)?;
            rows.extend(batch_rows);
        }

        let mut dataset = Dataset::new(
            crate::core::dataset_legacy::DatasetId(0),
            schema,
            Some(name.to_string()),
        );
        dataset.rows = rows;
        dataset.metadata = metadata;

        Ok(dataset)
    }

    fn save_reference_dataset(
        &self,
        dataset: &crate::core::dataset::Dataset,
    ) -> Result<(), StorageError> {
        self.ensure_directories(Some(&dataset.name))?;
        let path = format!(
            "{}/datasets/{}/{}.ref.json",
            self.base_path, dataset.name, dataset.name
        );
        let json = serde_json::to_string_pretty(dataset)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load_reference_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::Dataset, StorageError> {
        let path = format!("{}/datasets/{}.ref.json", self.base_path, name);
        if !Path::new(&path).exists() {
            return Err(StorageError::DatasetNotFound(name.to_string()));
        }
        let json = fs::read_to_string(path)?;
        let dataset =
            serde_json::from_str(&json).map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok(dataset)
    }

    fn dataset_exists(&self, name: &str) -> bool {
        Path::new(&self.dataset_data_path(name)).exists()
    }

    fn delete_dataset(&self, name: &str) -> Result<(), StorageError> {
        let dir = self.dataset_dir(name);
        if Path::new(&dir).exists() {
            fs::remove_dir_all(&dir)?;
        }

        let meta_path = self.legacy_metadata_path(name);
        if Path::new(&meta_path).exists() {
            fs::remove_file(&meta_path)?;
        }
        Ok(())
    }

    fn list_datasets(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories(None)?;
        let datasets_dir = format!("{}/datasets", self.base_path);
        let mut datasets = Vec::new();
        for entry in fs::read_dir(&datasets_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    // Check if it's a valid package by looking for data.parquet or manifest.json
                    let data_path = path.join("data.parquet");
                    let manifest_path = path.join("manifest.json");
                    if data_path.exists() || manifest_path.exists() {
                        datasets.push(name.to_string());
                    }
                }
            }
        }
        Ok(datasets)
    }

    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError> {
        self.ensure_directories(None)?;
        let tensor_path = self.tensor_path(name);
        let tensor_json = serde_json::to_string_pretty(tensor)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&tensor_path, tensor_json)?;
        Ok(())
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError> {
        let tensor_path = self.tensor_path(name);
        if !Path::new(&tensor_path).exists() {
            return Err(StorageError::TensorNotFound(name.to_string()));
        }
        let tensor_json = fs::read_to_string(&tensor_path)?;
        let tensor: Tensor = serde_json::from_str(&tensor_json)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok(tensor)
    }

    fn tensor_exists(&self, name: &str) -> bool {
        Path::new(&self.tensor_path(name)).exists()
    }

    fn delete_tensor(&self, name: &str) -> Result<(), StorageError> {
        let tensor_path = self.tensor_path(name);
        if Path::new(&tensor_path).exists() {
            fs::remove_file(&tensor_path)?;
        }
        Ok(())
    }

    fn list_tensors(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories(None)?;
        let tensors_dir = format!("{}/tensors", self.base_path);
        let mut tensors = Vec::new();
        for entry in fs::read_dir(&tensors_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    tensors.push(name.to_string());
                }
            }
        }
        Ok(tensors)
    }
}

/// CSV-based storage implementation
pub struct CsvStorage {
    _path: String,
}

impl CsvStorage {
    pub fn new(path: impl Into<String>) -> Self {
        Self { _path: path.into() }
    }

    pub fn export_dataset(&self, dataset: &Dataset, path: &str) -> Result<(), StorageError> {
        let batch = dataset_to_record_batch(dataset)?;
        let file = fs::File::create(path)?;
        let mut writer = arrow::csv::Writer::new(file);
        writer.write(&batch)?;
        Ok(())
    }

    pub fn import_dataset(&self, path: &str) -> Result<Dataset, StorageError> {
        let file = fs::File::open(path)?;
        // Infer schema from CSV using Format
        let format = csv::reader::Format::default().with_header(true);
        let (arrow_schema, _) = format
            .infer_schema(file, Some(100))
            .map_err(StorageError::Arrow)?;

        let arrow_schema_arc: Arc<ArrowSchema> = Arc::new(arrow_schema.clone());

        let builder = csv::ReaderBuilder::new(arrow_schema_arc).with_header(true);

        let file = fs::File::open(path)?;
        let csv_reader = builder.build(file)?;

        // Use filename as dataset name
        let name = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_data")
            .to_string();

        let mut record_batches: Vec<RecordBatch> = Vec::new();
        for batch in csv_reader {
            record_batches.push(batch?);
        }

        if record_batches.is_empty() {
            return Err(StorageError::Serialization("CSV file is empty".to_string()));
        }

        // Combine arrow schema to LINAL schema
        let fields: Vec<crate::core::tuple::Field> = arrow_schema
            .fields()
            .iter()
            .map(|f: &Arc<ArrowField>| {
                let vt = match f.data_type() {
                    DataType::Int64 => ValueType::Int,
                    DataType::Float32 | DataType::Float64 => ValueType::Float,
                    DataType::Boolean => ValueType::Bool,
                    _ => ValueType::String,
                };
                crate::core::tuple::Field::new(f.name(), vt)
            })
            .collect();

        let schema = Arc::new(crate::core::tuple::Schema::new(fields));
        let mut rows = Vec::new();

        for batch in record_batches {
            rows.extend(record_batch_to_rows(&batch, &schema)?);
        }

        let mut dataset = Dataset::new(
            crate::core::dataset_legacy::DatasetId(0),
            schema,
            Some(name),
        );
        dataset.rows = rows;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_storage_basic() {
        let temp_dir = "/tmp/linal_test_storage";
        let storage = ParquetStorage::new(temp_dir);
        let _ = fs::remove_dir_all(temp_dir);
        storage.ensure_directories(None).unwrap();
        assert!(Path::new(&format!("{}/datasets", temp_dir)).exists());
        assert!(Path::new(&format!("{}/tensors", temp_dir)).exists());
        let _ = fs::remove_dir_all(temp_dir);
    }
}
