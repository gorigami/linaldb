# Changelog

All notable changes to the `linaldb` R binding will be
documented here. See the parent repository's `CHANGELOG.md` for the
engine's own changelog.

## [0.1.0] - 2026-09-06 (unreleased, not yet published to CRAN)

Initial embedded/native binding, built alongside `clients/python-embedded`
as the in-process counterpart to the HTTP `clients/r` client:

- `linal_embedded_db()` opens a `TensorDb` directly in-process via an
  `extendr` binding (`src/rust/`, crate `linalr`) — no `linal serve`.
- `linal_embedded_execute()` / `linal_embedded_query()`, mirroring
  `clients/r`'s `linal_execute()`/`linal_query()` shape and error
  semantics (`linal_error` condition) exactly, just without HTTP/JSON in
  the middle.
- `linal_embedded_dataset()` / `linal_embedded_dataset_read()` /
  `linal_embedded_dataset_schema()` / `_stats()` / `_manifest()` /
  `_to_arrow()` — read a saved dataset's package
  (`data.parquet`/`schema.json`/...) straight off disk at
  `Db$dataset_dir(name)`, no `/delivery` export step needed since
  embedded mode already has filesystem access to the same layout
  `SAVE DATASET` writes.
- `linal_embedded_active_db()` / `linal_embedded_data_dir()`.
- `TensorTable`/`LazyTensor` results raise a clear "not yet supported"
  `linal_error` rather than guessing a shape — same scope boundary
  `clients/r`'s HTTP client already draws for `TensorTable`.
- Real end-to-end example:
  `examples/digit_classification_embedded.R` (no server, in-process
  replay of the same real UCI handwritten-digits workflow
  `clients/r/examples/digit_classification.R` uses).
- 8 passing `testthat` tests (`tests/testthat/test-embedded.R`).

Standalone Cargo crate (`src/rust/`, not a workspace member of the
repo-root `linal` crate) with a `path` dependency on it — building this
package compiles the full `linal` dependency graph including a
from-source HDF5 vendor build; see `README.md`'s Development section.
