//! NetCDF connector with real CF-convention semantics -- SCIENTIFIC_ENGINE_EXPANSION_PLAN.md
//! Phase 1. Before this existed, `.nc` files fell through to `Hdf5Connector` (NetCDF4 files
//! are HDF5 files under the hood, so that "worked" in the sense of reading raw bytes), but
//! treated every variable as an opaque array with no attribute interpretation at all --
//! exactly the audit finding this connector closes.
//!
//! **What "real CF semantics" means here, scoped**: for each variable, this connector reads
//! and applies the CF conventions' packing/masking attributes --
//! `scale_factor`/`add_offset` (`unpacked = raw * scale_factor + add_offset`) and
//! `_FillValue`/`missing_value` (mapped to `NaN`, since this codebase's connector columns are
//! non-nullable Arrow arrays throughout -- see `Hdf5Connector` for the same convention) -- and
//! surfaces `units`/`standard_name`/`long_name` as Arrow field metadata (`UNITS_METADATA_KEY`
//! etc.) so a caller can inspect them via `DatasetSchema`/`SHOW SCHEMA` without them being
//! silently dropped.
//!
//! **Deliberately not implemented** (documented rather than silently unsupported):
//! - Integer-packed variables (`int16`/`int32` storage, the classic CF packing case for raw
//!   satellite/reanalysis data). This engine's connectors are float-only throughout (see
//!   `Hdf5Connector`'s own "we only support numeric datasets" note) -- adding integer tensor
//!   support is out of this connector's scope. `scale_factor`/`add_offset`/`_FillValue` are
//!   still fully honored for float-stored variables, which is the common case for
//!   already-processed/derived scientific data (e.g. a reanalysis product re-exported as
//!   float32).
//! - Distinguishing CF *coordinate variables* (dimension-scale datasets) from *data
//!   variables*. This engine's tabular ingestion model (flat same-length columns, exactly
//!   `Hdf5Connector`'s convention) has no separate "coordinate axis" concept -- every variable,
//!   coordinate or data, becomes an ordinary column.
//! - Calendar-aware time decoding (`units: "days since ..."` + `calendar` attribute resolved
//!   to real timestamps). Time variables ingest as plain numeric columns like any other.

use crate::core::connectors::{field_with_shape, resolve_shape_dims, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hdf5::types::{FloatSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group};
use std::path::Path;
use std::sync::Arc;

/// Arrow `Field` metadata keys this connector attaches from CF attributes, mirroring
/// `SHAPE_METADATA_KEY`'s convention in `connectors::mod`.
pub const UNITS_METADATA_KEY: &str = "linal.units";
pub const STANDARD_NAME_METADATA_KEY: &str = "linal.standard_name";
pub const LONG_NAME_METADATA_KEY: &str = "linal.long_name";

pub struct NetCdfConnector;

/// Same accumulator shape as `Hdf5Connector`'s -- see its doc comment.
struct IngestAccumulator<'a> {
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    num_rows: usize,
    warnings: Vec<String>,
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

impl Connector for NetCdfConnector {
    fn name(&self) -> &str {
        "netcdf"
    }

    fn can_handle(&self, path: &str) -> bool {
        matches!(
            Path::new(path).extension().and_then(|s| s.to_str()),
            Some("nc")
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
                    "FIELDS: variable(s) not found in NetCDF file '{path}': {}",
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
                "No variables found in NetCDF file".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(acc.fields));
        let batch = RecordBatch::try_new(schema, acc.columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "netcdf_import".to_string(),
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

impl NetCdfConnector {
    fn visit_group(
        &self,
        group: &Group,
        prefix: &str,
        acc: &mut IngestAccumulator,
    ) -> Result<(), ConnectorError> {
        for member_name in group
            .member_names()
            .map_err(|e| ConnectorError::Other(e.to_string()))?
        {
            let name = if prefix.is_empty() {
                member_name.clone()
            } else {
                format!("{}_{}", prefix, member_name)
            };

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
            return Ok(());
        }

        let shape = ds.shape();

        // Same up-front declared-dtype check as `Hdf5Connector` -- see its doc comment for why
        // relying on a read error to detect f64 doesn't work (HDF5 silently narrows on read).
        let is_declared_f64 = ds
            .dtype()
            .ok()
            .and_then(|dt| dt.to_descriptor().ok())
            .is_some_and(|td| matches!(td, TypeDescriptor::Float(FloatSize::U8)));

        let raw_f64: Vec<f64> = if is_declared_f64 {
            match ds.read_raw::<f64>() {
                Ok(v) => v,
                Err(e) => return self.skip_or_error(name, requested, acc, &e.to_string()),
            }
        } else {
            match ds.read_raw::<f32>() {
                Ok(v) => v.into_iter().map(|x| x as f64).collect(),
                Err(_) => match ds.read_raw::<f64>() {
                    Ok(v) => v,
                    Err(e) => return self.skip_or_error(name, requested, acc, &e.to_string()),
                },
            }
        };

        let len = raw_f64.len();
        if acc.num_rows == 0 {
            acc.num_rows = len;
        } else if len != acc.num_rows {
            let msg = format!(
                "NetCDF variable '{name}': has {} element(s), expected {} \
                 (doesn't match other variables already ingested from this file)",
                len, acc.num_rows
            );
            if requested {
                return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
            }
            acc.warnings.push(format!("Skipped {msg}"));
            return Ok(());
        }

        // CF decoding: scale_factor/add_offset unpacking, _FillValue/missing_value -> NaN.
        // Computed in f64 throughout regardless of storage precision, same "promote for the
        // math, narrow at the end" policy `core::linalg` uses.
        let scale_factor = read_f64_attr(ds, "scale_factor");
        let add_offset = read_f64_attr(ds, "add_offset");
        let fill_value =
            read_f64_attr(ds, "_FillValue").or_else(|| read_f64_attr(ds, "missing_value"));
        let decoded = apply_cf_decoding(raw_f64, scale_factor, add_offset, fill_value);

        let units = read_string_attr(ds, "units");
        let standard_name = read_string_attr(ds, "standard_name");
        let long_name = read_string_attr(ds, "long_name");

        acc.found.insert(name.to_string());

        let mut field = field_with_shape(
            name,
            if is_declared_f64 {
                DataType::Float64
            } else {
                DataType::Float32
            },
            false,
            &shape,
        );
        field = attach_cf_metadata(field, units, standard_name, long_name);
        acc.fields.push(field);

        if is_declared_f64 {
            acc.columns.push(Arc::new(Float64Array::from(decoded)));
        } else {
            let narrowed: Vec<f32> = decoded.into_iter().map(|v| v as f32).collect();
            acc.columns.push(Arc::new(Float32Array::from(narrowed)));
        }

        Ok(())
    }

    /// Same "explicitly-requested field errors loudly, otherwise skip with a warning" policy
    /// `Hdf5Connector::skip_or_error` uses.
    fn skip_or_error(
        &self,
        name: &str,
        requested: bool,
        acc: &mut IngestAccumulator,
        err: &str,
    ) -> Result<(), ConnectorError> {
        let msg = format!(
            "NetCDF variable '{name}': not readable as a numeric (float-convertible) array ({err})"
        );
        if requested {
            return Err(ConnectorError::Parse(format!("FIELDS: {msg}")));
        }
        acc.warnings.push(format!("Skipped {msg}"));
        Ok(())
    }
}

/// `scale_factor`/`add_offset`/`_FillValue`/`missing_value` are declared as true 0-d HDF5
/// scalars in some NetCDF4 files, but real-world reanalysis products (verified against NCEP/
/// NCAR Reanalysis's own `air.mon.mean.nc`/`slp.mon.mean.nc`) commonly store them as 1-element
/// 1-D arrays instead -- `read_scalar` hard-errors on that shape (`ndim mismatch: expected
/// scalar, got 1`), which `.ok()` then silently swallowed here, disabling CF unpacking/masking
/// entirely against real files without so much as a warning. `read_raw` flattens either shape
/// into a `Vec<T>` uniformly, so this covers both.
fn read_f64_attr(ds: &Dataset, name: &str) -> Option<f64> {
    ds.attr(name)
        .ok()?
        .read_raw::<f64>()
        .ok()?
        .into_iter()
        .next()
}

fn read_string_attr(ds: &Dataset, name: &str) -> Option<String> {
    ds.attr(name)
        .ok()?
        .read_scalar::<VarLenUnicode>()
        .ok()
        .map(|s| s.as_str().to_string())
}

/// `unpacked = raw * scale_factor + add_offset` (identity when neither attribute is present);
/// any raw value exactly equal to `fill_value` becomes `NaN` -- checked *before* unpacking,
/// since `_FillValue`/`missing_value` are defined in the packed (raw, on-disk) domain per the
/// CF conventions, not the unpacked one.
fn apply_cf_decoding(
    raw: Vec<f64>,
    scale_factor: Option<f64>,
    add_offset: Option<f64>,
    fill_value: Option<f64>,
) -> Vec<f64> {
    let scale = scale_factor.unwrap_or(1.0);
    let offset = add_offset.unwrap_or(0.0);
    raw.into_iter()
        .map(|v| match fill_value {
            Some(fill) if v == fill => f64::NAN,
            _ => v * scale + offset,
        })
        .collect()
}

fn attach_cf_metadata(
    field: Field,
    units: Option<String>,
    standard_name: Option<String>,
    long_name: Option<String>,
) -> Field {
    if units.is_none() && standard_name.is_none() && long_name.is_none() {
        return field;
    }
    let mut metadata = field.metadata().clone();
    if let Some(u) = units {
        metadata.insert(UNITS_METADATA_KEY.to_string(), u);
    }
    if let Some(s) = standard_name {
        metadata.insert(STANDARD_NAME_METADATA_KEY.to_string(), s);
    }
    if let Some(l) = long_name {
        metadata.insert(LONG_NAME_METADATA_KEY.to_string(), l);
    }
    field.with_metadata(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf_decoding_applies_scale_and_offset() {
        let raw = vec![0.0, 1.0, 2.0];
        let decoded = apply_cf_decoding(raw, Some(0.5), Some(10.0), None);
        assert_eq!(decoded, vec![10.0, 10.5, 11.0]);
    }

    #[test]
    fn cf_decoding_is_identity_with_no_attributes() {
        let raw = vec![1.5, -2.0, 3.25];
        let decoded = apply_cf_decoding(raw.clone(), None, None, None);
        assert_eq!(decoded, raw);
    }

    #[test]
    fn cf_decoding_maps_fill_value_to_nan_before_unpacking() {
        let raw = vec![1.0, -999.0, 3.0];
        let decoded = apply_cf_decoding(raw, Some(2.0), Some(1.0), Some(-999.0));
        assert_eq!(decoded[0], 3.0); // 1.0 * 2.0 + 1.0
        assert!(decoded[1].is_nan());
        assert_eq!(decoded[2], 7.0); // 3.0 * 2.0 + 1.0
    }
}
