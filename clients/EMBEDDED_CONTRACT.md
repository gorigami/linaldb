# LINALDB Embedded Bindings Contract (Tier B)

This is the contract `clients/python-embedded/` and `clients/r-embedded/`
implement against — the in-process counterpart to
[`CONTRACT.md`](CONTRACT.md)'s HTTP+JSON contract. There is **no server,
no HTTP, no JSON** here: both bindings link `linal::engine::TensorDb` and
`linal::dsl::execute_line` directly into the host process (Python/R) via
a native extension (PyO3 / extendr) and convert `DslOutput`/`Value`
straight from Rust structs into language-native values. If a binding's
actual behavior disagrees with this document, that's a bug in the
binding, the doc, or both — fix the disagreement, don't just pick one
side.

Everything here was verified directly against each binding's own crate
(`clients/python-embedded/src/lib.rs`, `clients/r-embedded/src/rust/src/lib.rs`)
as of engine v0.1.76, and by running each binding's real showcase example
against the real UCI handwritten-digits data.

## 1. `Db.execute(sql)` / `Ndb$execute(sql)` — one DSL statement

Both bindings call `linal::dsl::execute_line(&mut TensorDb, sql, line_no)`
directly and convert the resulting `DslOutput` (or the `DslError` on
failure) into a native value. **The two languages use different raw
shapes at the FFI boundary** — each binding's own ergonomic wrapper layer
(`linaldb_embedded/__init__.py`'s `Db.execute`, R's
`linal_embedded_execute()`) is what a caller should actually use; the raw
shapes below exist to document what that wrapper is built on:

| `DslOutput` variant | Python raw (`Db.execute_raw`, `src/lib.rs`) | R raw (`Db$execute`, `src/rust/src/lib.rs`) | Ergonomic result |
|---|---|---|---|
| `None` | `None` | R `NULL` | `None` / `NULL` |
| `Message(s)` | `str` | `list(Message = <chr>)` | `str` / `character` scalar |
| `Table(dataset)` | `dict(columns=[...], rows=[[...]])` | `list(Table = list(columns = list(name = list(cell, ...), ...)))` | `ExecuteResult` (Python: `.columns`/`.rows`) / `linal_table_result` (R: an S3 object, same shape `clients/r`'s HTTP client uses) |
| `TensorTable(dataset, _)` | materialized into the same `Table` shape via `TensorDb::materialize_tensor_dataset` (Python only — the Rust side has a live `&mut TensorDb` to call it on) | **not supported** — raises `"TensorTable result is not yet supported by the embedded binding"` (the R crate only holds a bare `RefCell<TensorDb>` behind a free conversion function with no dataset-name context to materialize against; `SHOW` the dataset first to get a plain `Table` instead) | same as `Table`, or an error telling you to `SHOW` first |
| `Tensor(t)` | `dict(shape=[...], data=[...], strides=[...], offset=<int>)` | `list(Tensor = list(shape = <int vec>, data = <dbl vec>))` | `TensorResult` (Python: `.to_numpy()`) / raw list (R: reshape yourself, `strides`/`offset` not currently exposed on the R side) |
| `LazyTensor(_)` | raises `LinalError` | raises `"LazyTensor result is not materialized -- run SHOW <name> first"` | error in both — materialize with `SHOW <name>` first |
| `Err(DslError)` | raises `linaldb_embedded.LinalError(str(e))` | raises an R error condition with `e.to_string()` | native exception/condition in both, carrying the engine's own error text verbatim (same rule as `CONTRACT.md` §4) |

Per-cell `Value` conversion (both languages convert every `Value`
directly, no JSON round-trip):

| `Value` | Python | R |
|---|---|---|
| `Float(f32)` / `Float64(f64)` | `float` (both promote to Python `float`) | `double` (both promote to R `double`) |
| `Int(i64)` | `int` | `integer` — **truncates to i32** (same precision boundary `clients/r`'s HTTP client already has via `as.integer()`) |
| `String` | `str` | `character` |
| `Bool` | `bool` | `logical` |
| `Vector(Vec<f32>)` | `list[float]` | numeric vector |
| `Matrix(Vec<Vec<f32>>)` | `list[list[float]]` | list of numeric vectors (row-major) |
| `Null` | `None` | `NA_real_` (satisfies `is.na()`, same rule `CONTRACT.md` §3 documents for the HTTP client's `Value::Null`) |

## 2. Dataset export — no `/delivery`, direct filesystem reads

There is no export endpoint to call: a saved dataset's package
(`data.parquet`/`schema.json`/`stats.json`/`manifest.json`) already lives
on disk at `{data_dir}/{active_db}/datasets/{name}/`
(`src/core/storage.rs`) — the exact same layout `/delivery/*` serves over
HTTP (`src/server/dataset_server.rs`), just reachable directly since the
engine and the binding share a filesystem. Both bindings expose
`dataset_dir(name)` (`{data_dir}/{active_db}/datasets/{name}`, computed
once in Rust as the single source of truth) and read those files with
the host language's own Parquet/JSON libraries (`pyarrow`, R's `arrow` +
`jsonlite`) — the Vector/Matrix dual-encoding rule in `CONTRACT.md` §2
(native `FixedSizeList` vs. legacy JSON-string fallback) still applies
identically here, since it's a property of how the engine wrote the
Parquet file, not of how it's fetched.

There is no ad-hoc-query equivalent to `/delivery` (only *saved* datasets
have an on-disk package) — an ad-hoc `SELECT`'s `Table` result only
exists as the in-memory rows returned by `execute()`. Python's
`ExecuteResult.to_pandas()` is a convenience that builds a DataFrame
directly from those rows (no HTTP client equivalent, since it needs no
Parquet round-trip); the R binding doesn't currently offer the same
convenience for `linal_embedded_execute()`'s result.

## 3. Error semantics

- Every `DslError` surfaces as a native exception (`linaldb_embedded.LinalError`
  in Python) or R error condition, carrying the engine's error string
  verbatim — same rule as `CONTRACT.md` §4, just without an HTTP status
  code or `status: "error"` envelope in between.
- There is no network layer, so there is no connection-failure or
  timeout case to handle — a call either returns or raises/errors
  synchronously in the same process.

## 4. What this contract deliberately does not cover yet

- `/jobs`/`/schedule` equivalents — background execution only exists
  through the HTTP server today; the embedded bindings run everything
  synchronously in the calling thread.
- `TensorTable` support in the R binding (see the table above) — the
  Python binding materializes it via a live `TensorDb` reference; the R
  crate's conversion function doesn't have one. `SHOW` the dataset first
  in DSL to get a plain `Table` instead.
- Multi-threaded/concurrent use of a single `Db`/`Ndb` from multiple
  threads — both bindings assume single-threaded use per instance (the
  Python `Db` holds a plain owned `TensorDb`; the R `Db` wraps one in a
  `RefCell`, which panics on a concurrent borrow). Create one instance
  per thread if you need concurrency, mirroring how the HTTP server
  itself uses `Arc<RwLock<TensorDb>>` only because it's genuinely
  multi-tenant.
