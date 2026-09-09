# Changelog

All notable changes to the `linaldb` Python native bindings will
be documented here. See the parent repository's `CHANGELOG.md` for the
engine's own changelog.

## [0.1.0] - 2026-09-06 (unreleased, not yet published to PyPI)

Initial release: a PyO3 extension (`src/lib.rs`) embedding the engine's
synchronous `TensorDb`/`execute_line` directly in the Python process, no
server required — alongside, not replacing, the HTTP-based
`clients/python` package.

- `Db(data_dir=None)` / `Db.execute()` / `Db.query()`, returning an
  `ExecuteResult` (table) or `TensorResult` (bare tensor) mirroring the
  HTTP client's `Client.execute()`/`Client.query()` shapes.
- `Db.dataset(name)` / `Dataset.to_arrow()` / `Dataset.to_pandas()` /
  `.schema()` / `.stats()` / `.manifest()`, reading a saved dataset's
  on-disk package directly (`{data_dir}/{db}/datasets/{name}/...`) — the
  same layout `/delivery` serves over HTTP, just via a local file read
  instead of a request.
- `LinalError` for both DSL errors and dataset-file-not-found cases.
- Verified against CPython 3.14 with
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (pyo3 0.23's max officially
  supported CPython version is 3.13 at the time of writing).
- Real end-to-end example:
  `examples/digit_classification_embedded.py` and the Jupyter notebook
  `examples/digit_classification_embedded.ipynb`, both ported from
  `clients/python/examples/digit_classification.py` with the
  server/HTTP layer removed entirely.
