//! Loading a dataset straight from in-memory Arrow data (CASMI_WORKLOADS_PLAN.md,
//! P1): the path the embedded bindings use for NumPy arrays and Arrow
//! tables, with no file and no DSL parsing in between.
//!
//! Accepted Arrow column types are the ones this engine's own Parquet
//! packages use: `Int64`/`Int32` -> `Int`, `Float32` -> `Float`, `Float64` ->
//! `Float64`, `Utf8`/`LargeUtf8` -> `String`, `Boolean` -> `Bool`,
//! `FixedSizeList<Float32>` -> `Vector(d)`,
//! `FixedSizeList<FixedSizeList<Float32>>` -> `Matrix(r, c)`, and
//! `FixedSizeBinary(w)` -> `BitVector` (MSB-first packed bits, `8 * w` of
//! them unless the field's `linal.logical_value_type` metadata says
//! `BitVector:N`). Anything else
//! (including `FixedSizeList<Float64>`) is an error naming the column: values
//! are never converted to a different precision behind the caller's back.
//! NaN and infinite floats are rejected too, with the column and row.

use super::TensorDb;
use crate::core::storage::{arrow_schema_to_tuple_schema, record_batch_to_rows};
use crate::engine::error::EngineError;
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// First non-finite value in a float array (or its nested list values), as
/// (row, value). `row` is the top-level row the value belongs to.
fn first_non_finite(array: &ArrayRef) -> Option<(usize, f64)> {
    match array.data_type() {
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            (0..a.len())
                .find(|&i| a.is_valid(i) && !a.value(i).is_finite())
                .map(|i| (i, a.value(i) as f64))
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            (0..a.len())
                .find(|&i| a.is_valid(i) && !a.value(i).is_finite())
                .map(|i| (i, a.value(i)))
        }
        DataType::FixedSizeList(_, size) => {
            let list = array.as_any().downcast_ref::<FixedSizeListArray>()?;
            // Child offsets are relative to the list's own offset.
            let child = list
                .values()
                .slice(list.offset() * *size as usize, list.len() * *size as usize);
            first_non_finite(&child).map(|(i, v)| (i / *size as usize, v))
        }
        DataType::Struct(_) => {
            let st = array.as_any().downcast_ref::<arrow::array::StructArray>()?;
            let values = st.column_by_name("values")?;
            first_non_finite(values)
        }
        DataType::List(_) => {
            let list = array.as_any().downcast_ref::<arrow::array::ListArray>()?;
            (0..list.len())
                .filter(|&i| list.is_valid(i))
                .find_map(|i| first_non_finite(&list.value(i)).map(|(_, v)| (i, v)))
        }
        _ => None,
    }
}

fn check_supported(name: &str, data_type: &DataType) -> Result<(), EngineError> {
    let ok = match data_type {
        // SparseVector: needs its dimension in the field metadata, checked
        // in `load_record_batch`.
        DataType::Struct(fields) => {
            fields.find("indices").is_some() && fields.find("values").is_some()
        }
        DataType::Int64
        | DataType::Int32
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Boolean
        | DataType::FixedSizeBinary(_) => true,
        DataType::FixedSizeList(inner, _) => match inner.data_type() {
            DataType::Float32 => true,
            DataType::FixedSizeList(innermost, _) | DataType::List(innermost) => {
                matches!(innermost.data_type(), DataType::Float32)
            }
            _ => false,
        },
        _ => false,
    };
    if ok {
        return Ok(());
    }
    let hint = match data_type {
        DataType::FixedSizeList(inner, _) if matches!(inner.data_type(), DataType::Float64) => {
            " -- vectors are float32 in this engine; cast to float32 explicitly first"
        }
        DataType::List(_) | DataType::LargeList(_) => {
            " -- vectors need a fixed size; use a FixedSizeList (all rows the same length)"
        }
        _ => "",
    };
    Err(EngineError::InvalidOp(format!(
        "column '{}' has unsupported Arrow type {:?}{}",
        name, data_type, hint
    )))
}

impl TensorDb {
    /// Creates dataset `name` in the active database from `batch`, without
    /// writing a file or parsing DSL. `origin` is a short description of
    /// where the data came from (e.g. `"numpy"`, `"arrow"`); it's recorded
    /// in the provenance log together with the new dataset's content hash,
    /// so `EXPLAIN LINEAGE` shows the load. Fails if `name` already exists.
    /// With the write-ahead log enabled, a checkpoint is taken afterwards,
    /// since the load isn't a DSL statement the log could replay.
    ///
    /// Returns the number of rows loaded.
    pub fn load_record_batch(
        &mut self,
        name: &str,
        batch: &RecordBatch,
        origin: &str,
    ) -> Result<usize, EngineError> {
        if name.is_empty() {
            return Err(EngineError::InvalidOp("dataset name is empty".to_string()));
        }
        if self.get_dataset(name).is_ok() {
            return Err(EngineError::InvalidOp(format!(
                "dataset '{}' already exists -- DROP it first or load under another name",
                name
            )));
        }
        if batch.num_columns() == 0 {
            return Err(EngineError::InvalidOp(
                "cannot load a dataset with no columns".to_string(),
            ));
        }

        // LargeUtf8 is decoded as Utf8 by the shared row conversion.
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
        let mut fields = Vec::with_capacity(batch.num_columns());
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            check_supported(field.name(), field.data_type())?;
            if matches!(field.data_type(), DataType::Struct(_))
                && !field
                    .metadata()
                    .get("linal.logical_value_type")
                    .is_some_and(|t| t.starts_with("SparseVector:"))
            {
                return Err(EngineError::InvalidOp(format!(
                    "column '{}' is a struct: a SparseVector column needs field metadata linal.logical_value_type = \"SparseVector:<dim>\" (linaldb.sparse_array builds it)",
                    field.name()
                )));
            }
            if let Some((row, value)) = first_non_finite(column) {
                return Err(EngineError::InvalidOp(format!(
                    "column '{}' row {} is {} -- NaN and infinite values are rejected",
                    field.name(),
                    row,
                    value
                )));
            }
            if matches!(field.data_type(), DataType::LargeUtf8) {
                columns.push(
                    arrow::compute::cast(column, &DataType::Utf8)
                        .map_err(|e| EngineError::InvalidOp(e.to_string()))?,
                );
                fields.push(field.as_ref().clone().with_data_type(DataType::Utf8));
            } else {
                columns.push(column.clone());
                fields.push(field.as_ref().clone());
            }
        }
        let arrow_schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let batch = RecordBatch::try_new(arrow_schema.clone(), columns)
            .map_err(|e| EngineError::InvalidOp(e.to_string()))?;

        let schema = Arc::new(arrow_schema_to_tuple_schema(&arrow_schema));
        let rows = record_batch_to_rows(&batch, &schema)
            .map_err(|e| EngineError::InvalidOp(e.to_string()))?;
        let n = rows.len();

        let instance = self.active_instance_mut();
        let id = instance.dataset_store.gen_id();
        let dataset = crate::core::dataset_legacy::Dataset::with_rows(
            id,
            schema,
            rows,
            Some(name.to_string()),
        )
        .map_err(EngineError::InvalidOp)?;
        let hash = dataset.content_hash();
        instance
            .dataset_store
            .insert(dataset, Some(name.to_string()))
            .map_err(EngineError::from)?;

        let record = crate::core::provenance::ProvenanceRecord::new(
            "LOAD FROM MEMORY",
            crate::core::tensor::ExecutionId::new(),
        )
        .with_param("origin", origin)
        .with_param("rows", n)
        .with_outputs(vec![crate::core::provenance::ProvenanceEntity::dataset(
            name.to_string(),
            hash,
        )]);
        instance.record_provenance(record);

        if self.config.wal.enabled {
            self.checkpoint()?;
        }
        Ok(n)
    }
}
