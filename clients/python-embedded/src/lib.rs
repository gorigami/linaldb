//! PyO3 bridge between LINALDB's synchronous embedded engine
//! (`linal::engine::TensorDb` + `linal::dsl::execute_line`) and Python.
//!
//! Kept deliberately thin: this module only converts `DslOutput`/`Value`
//! into plain Python primitives (`None`/`str`/`dict{columns,rows}`) and
//! maps `DslError` to a Python exception. Ergonomics (pandas/pyarrow
//! conversion, a `Dataset` handle, a friendlier `ExecuteResult` type) live
//! in the pure-Python `python/linaldb_embedded/__init__.py` layer, mirroring
//! how `clients/python/linaldb/wire.py` (raw unwrap) is kept separate from
//! `clients/python/linaldb/client.py` (ergonomics).

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pyo3::create_exception!(_native, LinalError, PyException);

fn value_to_py(py: Python<'_>, value: &Value) -> PyObject {
    match value {
        Value::Float(f) => (*f as f64).into_pyobject(py).unwrap().into_any().unbind(),
        Value::Float64(f) => (*f).into_pyobject(py).unwrap().into_any().unbind(),
        Value::Int(i) => (*i).into_pyobject(py).unwrap().into_any().unbind(),
        Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        Value::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        Value::Vector(v) => {
            let items: Vec<f64> = v.iter().map(|x| *x as f64).collect();
            PyList::new(py, items).unwrap().into_any().unbind()
        }
        Value::Matrix(m) => {
            let rows: Vec<PyObject> = m
                .iter()
                .map(|row| {
                    let items: Vec<f64> = row.iter().map(|x| *x as f64).collect();
                    PyList::new(py, items).unwrap().into_any().unbind()
                })
                .collect();
            PyList::new(py, rows).unwrap().into_any().unbind()
        }
        Value::Null => py.None(),
    }
}

/// Convert a materialized (row-oriented) `Dataset` into the raw
/// `{"columns": [...], "rows": [[...]]}` shape `execute_raw` returns for
/// any table-shaped `DslOutput`.
fn legacy_dataset_to_py(py: Python<'_>, ds: &linal::Dataset) -> PyObject {
    let columns: Vec<String> = ds.schema.fields.iter().map(|f| f.name.clone()).collect();
    let rows: Vec<PyObject> = ds
        .rows
        .iter()
        .map(|tuple| {
            let values: Vec<PyObject> = tuple.values.iter().map(|v| value_to_py(py, v)).collect();
            PyList::new(py, values).unwrap().into_any().unbind()
        })
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("columns", columns).unwrap();
    dict.set_item("rows", rows).unwrap();
    dict.into_any().unbind()
}

/// An embedded LINALDB instance: an in-process `TensorDb`, no server, no
/// network — the same engine the CLI/REPL runs, linked directly into the
/// Python process.
#[pyclass]
struct Db {
    inner: TensorDb,
    next_line_no: usize,
}

#[pymethods]
impl Db {
    /// `data_dir`, when given, overrides where `SAVE DATASET`/persistence
    /// reads and writes (`{data_dir}/{db}/datasets/{name}/...`, same
    /// layout `linal serve`/the CLI use) — defaults to `./data` relative
    /// to the current process's working directory, exactly like the CLI.
    #[new]
    #[pyo3(signature = (data_dir=None))]
    fn new(data_dir: Option<String>) -> Self {
        let mut config = EngineConfig::load();
        if let Some(dir) = data_dir {
            config.storage.data_dir = dir.into();
        }
        Db {
            inner: TensorDb::with_config(config),
            next_line_no: 1,
        }
    }

    /// Run one DSL statement and return the raw wire-shaped result:
    /// `None` (no output), a `str` (`Message`), or a
    /// `dict` with `columns: list[str]` / `rows: list[list[...]]`
    /// (`Table`/`TensorTable`). Raises `LinalError` for a DSL error or
    /// for a bare `Tensor`/`LazyTensor` result (not supported yet here —
    /// `SHOW` it into a table first). This is the raw layer; the
    /// ergonomic `Db.execute()` Python wrapper turns the dict into an
    /// `ExecuteResult`.
    fn execute_raw(&mut self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        let line_no = self.next_line_no;
        self.next_line_no += 1;
        let result = execute_line(&mut self.inner, sql, line_no);
        match result {
            Ok(DslOutput::None) => Ok(py.None()),
            Ok(DslOutput::Message(s)) => Ok(s.into_pyobject(py).unwrap().into_any().unbind()),
            Ok(DslOutput::Table(ds)) => Ok(legacy_dataset_to_py(py, &ds)),
            Ok(DslOutput::TensorTable(ds, _tensor_cols)) => {
                // Zero-copy reference-graph dataset -- materialize it into
                // the row-oriented legacy shape first (same conversion
                // `SELECT`/`SHOW` use internally, `TensorDb::materialize_tensor_dataset`)
                // so callers get one consistent Table shape regardless of
                // which of the two dataset implementations produced it
                // (see CLAUDE.md's "Dual dataset model").
                let materialized = self
                    .inner
                    .materialize_tensor_dataset(&ds.name)
                    .map_err(|e| LinalError::new_err(e.to_string()))?;
                Ok(legacy_dataset_to_py(py, &materialized))
            }
            Ok(DslOutput::Tensor(t)) => {
                let shape: Vec<usize> = t.shape.dims.clone();
                let data: Vec<f64> = t.data.iter().map(|x| *x as f64).collect();
                let dict = PyDict::new(py);
                dict.set_item("shape", shape)?;
                dict.set_item("data", data)?;
                dict.set_item("strides", t.strides.clone())?;
                dict.set_item("offset", t.offset)?;
                Ok(dict.into_any().unbind())
            }
            Ok(DslOutput::LazyTensor(_)) => Err(LinalError::new_err(
                "execute() doesn't support a bare LazyTensor (unevaluated) result — \
                 materialize it first (e.g. `SHOW <name>`) to get a real Tensor result",
            )),
            Err(e) => Err(LinalError::new_err(e.to_string())),
        }
    }

    /// The database currently active on this instance (`USE <db>` changes
    /// it) — needed to compute `dataset_dir()` correctly.
    fn active_db(&self) -> String {
        self.inner.active_db().to_string()
    }

    /// The resolved data directory this instance persists to/recovers
    /// from (`./data` by default).
    fn data_dir(&self) -> String {
        self.inner
            .config
            .storage
            .data_dir
            .to_string_lossy()
            .into_owned()
    }

    /// The on-disk package directory for a saved dataset —
    /// `{data_dir}/{active_db}/datasets/{name}/` — containing
    /// `data.parquet`/`schema.json`/`stats.json`/`manifest.json`, the
    /// same layout `/delivery` serves over HTTP (`src/server/dataset_server.rs`).
    /// The Python `Dataset` class reads these files directly.
    fn dataset_dir(&self, name: &str) -> String {
        format!("{}/{}/datasets/{}", self.data_dir(), self.active_db(), name)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Db>()?;
    m.add("LinalError", m.py().get_type::<LinalError>())?;
    Ok(())
}
