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
        // SparseVector (indices + values) or a quantized vector (scale +
        // values); either needs its type in the field metadata, checked in
        // `load_record_batch`.
        DataType::Struct(fields) => {
            fields.find("values").is_some()
                && (fields.find("indices").is_some() || fields.find("scale").is_some())
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

/// One `Matrix(2, *)` peak-list column to build from two variable-length
/// list columns (`PeakLoad`).
#[derive(Debug, Clone)]
pub struct PeakColumns {
    /// Name of the new column.
    pub name: String,
    /// Source column with each spectrum's m/z values.
    pub mz: String,
    /// Source column with each spectrum's intensities.
    pub intensity: String,
}

/// Options for `TensorDb::load_record_batch_with_peaks`.
#[derive(Debug, Clone, Default)]
pub struct PeakLoad {
    pub peaks: Vec<PeakColumns>,
    /// Allow `Float64` source lists, rounded to `Float32` (the only
    /// precision peak lists have). Without it, `Float64` lists are an error:
    /// values are never converted behind the caller's back.
    pub cast_f64: bool,
    /// Sort each spectrum's peaks by m/z instead of rejecting unsorted ones.
    pub sort: bool,
}

/// A list column's per-row values as f32, or an error naming the column.
/// `List` and `LargeList` of `Float32` (always) or `Float64` (`cast_f64`).
fn list_rows(
    batch: &RecordBatch,
    column: &str,
    cast_f64: bool,
) -> Result<Vec<Option<Vec<f32>>>, EngineError> {
    use arrow::array::{GenericListArray, OffsetSizeTrait};
    let bad = |msg: String| EngineError::InvalidOp(format!("peaks: column '{}' {}", column, msg));
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| bad("not found".to_string()))?;
    fn rows<O: OffsetSizeTrait>(
        list: &GenericListArray<O>,
        cast_f64: bool,
        bad: &dyn Fn(String) -> EngineError,
    ) -> Result<Vec<Option<Vec<f32>>>, EngineError> {
        let values = list.values();
        let get: Box<dyn Fn(usize) -> f32> = match values.data_type() {
            DataType::Float32 => {
                let v = values.as_any().downcast_ref::<Float32Array>().unwrap().clone();
                Box::new(move |i| v.value(i))
            }
            DataType::Float64 if cast_f64 => {
                let v = values.as_any().downcast_ref::<Float64Array>().unwrap().clone();
                Box::new(move |i| v.value(i) as f32)
            }
            DataType::Float64 => {
                return Err(bad(
                    "holds float64 values; peak lists are float32 -- pass cast='f32' to round them explicitly"
                        .to_string(),
                ))
            }
            other => return Err(bad(format!("holds {:?} values, not floats", other))),
        };
        if values.null_count() > 0 {
            return Err(bad("has a NULL inside a list".to_string()));
        }
        let offsets = list.value_offsets();
        Ok((0..list.len())
            .map(|r| {
                list.is_valid(r).then(|| {
                    (offsets[r].as_usize()..offsets[r + 1].as_usize())
                        .map(&get)
                        .collect()
                })
            })
            .collect())
    }
    match array.data_type() {
        DataType::List(_) => rows(
            array
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .unwrap(),
            cast_f64,
            &bad,
        ),
        DataType::LargeList(_) => rows(
            array
                .as_any()
                .downcast_ref::<arrow::array::LargeListArray>()
                .unwrap(),
            cast_f64,
            &bad,
        ),
        other => Err(bad(format!(
            "is {:?}, not a variable-length list of floats",
            other
        ))),
    }
}

/// `batch` with each `PeakLoad::peaks` entry's two list columns combined
/// into one `Matrix(2, *)` column (`FixedSizeList<List<Float32>, 2>`, as
/// `linaldb.peaks_array()` builds it), placed where the m/z column was.
/// The two source columns are dropped (variable-length lists are not a
/// column type of their own).
/// Per row: both lists NULL gives a NULL spectrum; otherwise both must be
/// present, of equal length, finite, and with ascending m/z (or sorted,
/// with `sort`). Errors name the column and row.
pub fn combine_peak_columns(
    batch: &RecordBatch,
    opts: &PeakLoad,
) -> Result<RecordBatch, EngineError> {
    use arrow::datatypes::Field;
    let mut built: Vec<(String, String, Field, ArrayRef)> = Vec::new();
    let mut consumed: Vec<&str> = Vec::new();
    for spec in &opts.peaks {
        let mz_rows = list_rows(batch, &spec.mz, opts.cast_f64)?;
        let int_rows = list_rows(batch, &spec.intensity, opts.cast_f64)?;
        let mut offsets: Vec<i32> = vec![0];
        let mut values: Vec<f32> = Vec::new();
        let mut valid: Vec<bool> = Vec::with_capacity(mz_rows.len());
        for (r, (mz, int)) in mz_rows.into_iter().zip(int_rows).enumerate() {
            let err = |msg: String| {
                EngineError::InvalidOp(format!("peaks '{}': row {} {}", spec.name, r, msg))
            };
            let (mut mz, mut int) = match (mz, int) {
                (None, None) => {
                    offsets.push(values.len() as i32);
                    offsets.push(values.len() as i32);
                    valid.push(false);
                    continue;
                }
                (Some(m), Some(i)) => (m, i),
                _ => {
                    return Err(err(format!(
                        "has a NULL in only one of '{}' and '{}'",
                        spec.mz, spec.intensity
                    )))
                }
            };
            if mz.len() != int.len() {
                return Err(err(format!(
                    "has {} m/z values and {} intensities",
                    mz.len(),
                    int.len()
                )));
            }
            if let Some(x) = mz.iter().chain(&int).find(|x| !x.is_finite()) {
                return Err(err(format!(
                    "has {} -- NaN and infinite values are rejected",
                    x
                )));
            }
            if let Some(i) = mz.windows(2).position(|w| w[1] < w[0]) {
                if !opts.sort {
                    return Err(err(format!(
                        "has m/z {} after {} -- peaks must be sorted by m/z (or pass sort=True)",
                        mz[i + 1],
                        mz[i]
                    )));
                }
                let mut order: Vec<usize> = (0..mz.len()).collect();
                order.sort_by(|&a, &b| mz[a].total_cmp(&mz[b]));
                (mz, int) = (
                    order.iter().map(|&k| mz[k]).collect(),
                    order.iter().map(|&k| int[k]).collect(),
                );
            }
            for row in [mz, int] {
                values.extend(row);
                offsets.push(i32::try_from(values.len()).map_err(|_| {
                    EngineError::InvalidOp(format!(
                        "peaks '{}': more than {} values in one load -- load in batches",
                        spec.name,
                        i32::MAX
                    ))
                })?);
            }
            valid.push(true);
        }
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let list = arrow::array::ListArray::try_new(
            item,
            arrow::buffer::OffsetBuffer::new(offsets.into()),
            Arc::new(Float32Array::from(values)),
            None,
        )
        .map_err(|e| EngineError::InvalidOp(e.to_string()))?;
        let row_field = Arc::new(Field::new("item", list.data_type().clone(), false));
        let nulls = valid
            .iter()
            .any(|v| !v)
            .then(|| arrow::buffer::NullBuffer::from(valid));
        let outer = FixedSizeListArray::try_new(row_field.clone(), 2, Arc::new(list), nulls)
            .map_err(|e| EngineError::InvalidOp(e.to_string()))?;
        built.push((
            spec.name.clone(),
            spec.mz.clone(),
            Field::new(&spec.name, DataType::FixedSizeList(row_field, 2), true),
            Arc::new(outer),
        ));
        consumed.push(&spec.mz);
        consumed.push(&spec.intensity);
    }
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let name = field.name().as_str();
        if let Some((_, _, f, a)) = built.iter().find(|(_, mz, _, _)| mz == name) {
            fields.push(f.clone());
            columns.push(a.clone());
        }
        if !consumed.contains(&name) {
            fields.push(field.as_ref().clone());
            columns.push(column.clone());
        }
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(f) = fields.iter().find(|f| !seen.insert(f.name().clone())) {
        return Err(EngineError::InvalidOp(format!(
            "peaks: column '{}' would appear twice -- give the peak column another name",
            f.name()
        )));
    }
    RecordBatch::try_new(Arc::new(arrow::datatypes::Schema::new(fields)), columns)
        .map_err(|e| EngineError::InvalidOp(e.to_string()))
}

impl TensorDb {
    /// `load_record_batch` after combining list columns into peak lists
    /// (`combine_peak_columns`). The lineage record names each combined
    /// column's sources and whether float64 values were rounded.
    pub fn load_record_batch_with_peaks(
        &mut self,
        name: &str,
        batch: &RecordBatch,
        origin: &str,
        peaks: &PeakLoad,
    ) -> Result<usize, EngineError> {
        let combined = combine_peak_columns(batch, peaks)?;
        let params: Vec<(String, serde_json::Value)> = vec![
            (
                "peaks".to_string(),
                serde_json::Value::Object(
                    peaks
                        .peaks
                        .iter()
                        .map(|p| {
                            (
                                p.name.clone(),
                                serde_json::json!([p.mz.clone(), p.intensity.clone()]),
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "cast".to_string(),
                (if peaks.cast_f64 { "f32" } else { "none" }).into(),
            ),
            ("sorted".to_string(), peaks.sort.into()),
        ];
        self.load_record_batch_inner(name, &combined, origin, params)
    }

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
        self.load_record_batch_inner(name, batch, origin, Vec::new())
    }

    fn load_record_batch_inner(
        &mut self,
        name: &str,
        batch: &RecordBatch,
        origin: &str,
        params: Vec<(String, serde_json::Value)>,
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
                    .is_some_and(|t| t.starts_with("SparseVector:") || t.starts_with("QVector:"))
            {
                return Err(EngineError::InvalidOp(format!(
                    "column '{}' is a struct: a SparseVector or quantized vector column needs field metadata linal.logical_value_type = \"SparseVector:<dim>\" or \"QVector:<dim>,<F16|I8>\" (linaldb.sparse_array / load_numpy(quantize=...) set it)",
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
        .with_param("rows", n);
        let record = params
            .into_iter()
            .fold(record, |r, (k, v)| r.with_param(k, v))
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
