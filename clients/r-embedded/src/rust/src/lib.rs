//! extendr binding wrapping `linal::engine::TensorDb` directly -- no HTTP.
//! See `../../../EMBEDDED_CONTRACT.md` for the result-shape contract this
//! crate implements against (mirrors `clients/CONTRACT.md`'s tagged
//! `DslOutput` shapes, but as native `Robj`s instead of JSON).

use extendr_api::prelude::*;
use linal::core::tuple::Tuple;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::cell::RefCell;
use std::path::PathBuf;

/// One `Value` cell -> an `Robj`. `Null` becomes `NA_real_` (any R `NA`
/// variant satisfies `is.na()`, matching how the HTTP client's
/// `unwrap_value()` treats a wire `"Null"` as a generic `NA`).
fn value_to_robj(value: &Value) -> Robj {
    match value {
        Value::Float(f) => Robj::from(*f as f64),
        Value::Float64(f) => Robj::from(*f),
        // Mirrors the HTTP client's `as.integer(inner)` scalar handling
        // (clients/r/R/wire.R) -- values outside i32 range truncate here
        // rather than round-tripping through JSON first, but the same
        // "large Int loses precision" boundary applies either way.
        Value::Int(i) => Robj::from(*i as i32),
        Value::String(s) => Robj::from(s.as_str()),
        Value::Bool(b) => Robj::from(*b),
        Value::Vector(v) => Robj::from(v.iter().map(|x| *x as f64).collect::<Vec<f64>>()),
        Value::Matrix(m) => {
            let rows: Vec<Robj> = m
                .iter()
                .map(|row| Robj::from(row.iter().map(|x| *x as f64).collect::<Vec<f64>>()))
                .collect();
            List::from_values(rows).into_robj()
        }
        Value::Null => Robj::from(f64::na()),
    }
}

/// `columns: <name> -> list(per-row cell)`, mirroring
/// `table_result_columns()` in `clients/r/R/wire.R` -- the R-side layer
/// (`linal_embedded_query`/`linal_embedded_execute`) collapses each
/// column into an atomic vector or a list-column exactly like the HTTP
/// client's `columns_to_dataframe()` does.
fn table_to_robj(schema_fields: &[String], rows: &[Tuple]) -> extendr_api::Result<Robj> {
    let mut values: Vec<Robj> = Vec::with_capacity(schema_fields.len());
    for (col_idx, _name) in schema_fields.iter().enumerate() {
        let column_cells: Vec<Robj> = rows
            .iter()
            .map(|row| value_to_robj(&row.values[col_idx]))
            .collect();
        values.push(List::from_values(column_cells).into_robj());
    }
    let list = List::from_names_and_values(schema_fields.iter().map(|s| s.as_str()), values)?;
    Ok(list.into_robj())
}

fn output_to_robj(output: DslOutput) -> extendr_api::Result<Robj> {
    match output {
        DslOutput::None => Ok(().into_robj()),
        DslOutput::Message(s) => {
            let list = List::from_names_and_values(["Message"], [Robj::from(s)])?;
            Ok(list.into_robj())
        }
        DslOutput::Table(dataset) => {
            let field_names: Vec<String> = dataset
                .schema
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect();
            let columns = table_to_robj(&field_names, &dataset.rows)?;
            let payload = List::from_names_and_values(["columns"], [columns])?;
            let list = List::from_names_and_values(["Table"], [payload.into_robj()])?;
            Ok(list.into_robj())
        }
        // Zero-copy reference-graph `Dataset` (`core::dataset::Dataset`,
        // `.columns: HashMap<String, ResourceReference>`), not the
        // row-oriented one `Table` wraps -- materializing it needs a
        // `TensorDb`/`DatabaseInstance` reference
        // (`TensorDb::materialize_tensor_dataset`) this free function
        // doesn't have. No verified wire example to shape this against
        // either -- the HTTP R/Python clients punt on `TensorTable` for
        // the same reason (`clients/r/R/wire.R`'s `unwrap_result`). Fail
        // loudly rather than guess.
        DslOutput::TensorTable(..) => Err(Error::Other(
            "TensorTable result is not yet supported by the embedded binding \
             -- SHOW the underlying dataset (materializes to a plain Table) instead"
                .to_string(),
        )),
        DslOutput::Tensor(t) => {
            let shape: Vec<i32> = t.shape.dims.iter().map(|d| *d as i32).collect();
            let data: Vec<f64> = t.data.iter().map(|x| *x as f64).collect();
            let payload = List::from_names_and_values(
                ["shape", "data"],
                [Robj::from(shape), Robj::from(data)],
            )?;
            let list = List::from_names_and_values(["Tensor"], [payload.into_robj()])?;
            Ok(list.into_robj())
        }
        DslOutput::LazyTensor(_) => Err(Error::Other(
            "LazyTensor result is not materialized -- run SHOW <name> first".to_string(),
        )),
    }
}

/// Embedded LINALDB engine -- wraps `TensorDb` directly, in-process, no
/// HTTP server. See `linal_embedded_db()` (R/api.R) for the public
/// constructor; this raw `#[extendr]` type is not meant to be constructed
/// directly from R.
#[extendr]
struct Db {
    inner: RefCell<TensorDb>,
}

#[extendr]
impl Db {
    fn new(data_dir: Nullable<String>) -> Self {
        let mut db = TensorDb::new();
        if let Nullable::NotNull(dir) = data_dir {
            db.config.storage.data_dir = PathBuf::from(dir);
        }
        Db {
            inner: RefCell::new(db),
        }
    }

    fn execute(&self, sql: &str) -> extendr_api::Result<Robj> {
        let mut db = self.inner.borrow_mut();
        match execute_line(&mut db, sql, 1) {
            Ok(output) => output_to_robj(output),
            Err(e) => Err(Error::Other(e.to_string())),
        }
    }

    fn active_db(&self) -> String {
        self.inner.borrow().active_db().to_string()
    }

    fn data_dir(&self) -> String {
        self.inner
            .borrow()
            .config
            .storage
            .data_dir
            .to_string_lossy()
            .to_string()
    }

    /// `{data_dir}/{active_db}/datasets/{name}` -- the on-disk package
    /// directory a `SAVE DATASET <name>` write lands in
    /// (`src/core/storage.rs`), single source of truth for the R-side
    /// `linal_embedded_dataset_*` readers so the path logic isn't
    /// duplicated in Rust and R. Built via a real path join (not string
    /// concatenation) so the returned string uses native separators on
    /// every platform, matching R's own `file.path()` on the caller side.
    fn dataset_dir(&self, name: &str) -> String {
        let db = self.inner.borrow();
        PathBuf::from(&db.config.storage.data_dir)
            .join(db.active_db())
            .join("datasets")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }
}

extendr_module! {
    mod linaldb;
    impl Db;
}
