# Changelog

All notable changes to the `linaldb` Python native bindings will
be documented here. See the parent repository's `CHANGELOG.md` for the
engine's own changelog.

## [0.1.3] - 2026-09-12

Bumped for the engine features below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- New `EXPLAIN LINEAGE <name> [AS JSON]` DSL command — real, persisted derivation
  history for a tensor or dataset, surviving a restart. `SHOW LINEAGE <name>` keeps
  working as an alias, and now also resolves dataset names (previously tensor-only).
- Real classical linear algebra: `TRACE`, `DETERMINANT`, `RANK`, `INVERSE`, `SOLVE`,
  `EIGENVALUES`, `CHOLESKY`, `PCA`, and the decompositions `QR`/`LU`/`EIGEN`/`SVD` via
  a new multi-output `LET a, b[, c] = <expr>` binding syntax.

See the parent repository's `CHANGELOG.md` (`[0.1.80]`) for full detail.

## [0.1.2] - 2026-09-12

Bumped for the engine fix below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `CORRELATE a WITH b` now computes true Pearson correlation instead of a raw,
  unnormalized dot product it was silently wired to.
- `SUM`/`MEAN`/`STDEV` now return a true scalar (rank-0) instead of a
  rank-1 `Vector(1)` — previously, combining one of these with a longer
  vector via `+`/`-`/`*`/`/` (e.g. `v - MEAN(v)`, the standard way to center
  a vector) silently corrupted every element past the first instead of
  broadcasting correctly.

See the parent repository's `CHANGELOG.md` (`[0.1.79]`) for full detail —
both were found via a real end-to-end notebook (`linal-hub`, a local
project, not yet published anywhere) doing gene-expression marker
selection and classification entirely in `linaldb` DSL.

## [0.1.1] - 2026-09-10

Bumped for the engine fix below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `HAVING` on an aliased aggregate (`HAVING AVG(score) > 0.5` alongside
  `AVG(score) AS avg_score`) no longer silently matches zero rows.
- `HAVING SUM(...)` — previously broken independent of aliasing — now
  resolves correctly.
- `CAST(<float literal> AS DOUBLE)` no longer silently loses precision.
- `INSERT INTO t VALUES (...)` with more positional values than the target
  has columns (or a named `INSERT` naming an unknown column) now errors
  instead of silently truncating/dropping data.

See the parent repository's `CHANGELOG.md` for full detail — all four were
found via a real pytest suite written against this exact published PyPI
package (`linal-hub`, a local project, not yet published anywhere).

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
