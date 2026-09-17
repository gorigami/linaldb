# Changelog

All notable changes to the `linaldb` Python native bindings will
be documented here. See the parent repository's `CHANGELOG.md` for the
engine's own changelog.

## [0.1.10] - 2026-09-16

Picks up the root engine's `v0.1.83` (linked directly in via the `linal` path
dependency — no wire contract or Python-side API changes here). Two real
correctness fixes, both found via a deep audit of `docs/DSL_REFERENCE.md`
against a real build of the engine, and both now flow straight through to
`Db.execute()`:

- `WHERE <bool_col> = 1` and a bare `WHERE <bool_col>` (no explicit `=
  true`) used to silently match zero rows instead of comparing correctly —
  now fixed. Affects every `SELECT`/pipeline/`TRANSFORM` query run through
  `execute()`.
- In-place `TRANSFORM <source> SELECT ...` (no `INTO`) used to corrupt the
  dataset's schema when the projection changed the column set, breaking
  every later read. Now fixed.

See the root repository's `CHANGELOG.md` (`[0.1.83]`) for the full
root-cause writeups.

## [0.1.9] - 2026-09-16

Documentation-only release, no functional changes:

- Added a `## Status` section to `README.md` (none existed before)
  summarizing the real bug-fix history across all 9 releases.
- Added an `## About` section crediting Gorigami and Nicolás Balaguera.
- Standardized the `authors` contact email to `develop@gorigami.xyz`
  (matching `LICENSE`/root `README.md`/`SECURITY.md`/the R client
  packages, which all already used this address — the Python packages
  were the outlier at `gorigamidev@gmail.com`) and added Nicolás
  Balaguera as a named author.
- Added `numpy` to the `dev` extra (already an implicit dependency of the
  README's own Usage example and `examples/digit_classification_embedded.py`
  via pandas, but not declared explicitly — `clients/python`'s `dev`
  extra already listed it).

## [0.1.8] - 2026-09-15

Bumped for the engine fixes below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `LET y = x` / `DERIVE b FROM a` (the RHS a plain existing tensor/dataset-var
  name, not a real expression) silently failed to bind the new name, and even
  misreported which variable it had defined. Found while building a real
  manufacturing-quality-control notebook in `linal-hub`. `LET`/`BIND` now
  create a true zero-copy alias in this case (also accepting a
  `dataset()`-constructed variable, closing a related latent gap in `BIND`
  itself); `LAZY LET`/`DERIVE` with a bare identifier are now clear errors
  instead of silent wrong successes.
- `EXPORT <dataset> TO "*.csv"` crashed whenever the dataset had a populated
  `Vector`/`Matrix` column (Arrow's CSV writer can't serialize the native
  `FixedSizeList` encoding Parquet prefers for that common case). CSV export
  now always uses the existing JSON-string fallback encoding for those
  columns; `SAVE DATASET`/Parquet is unaffected.

See the parent repository's `CHANGELOG.md` (`[Unreleased]`, two entries) for
full detail.

## [0.1.7] - 2026-09-14

Bumped for the engine fix below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `SEARCH ... LIMIT k` without `INTO` always materialized results into a
  `search_results` dataset and returned only a status message, contradicting
  the documented default of returning the top-k rows inline. Found while
  building a real MovieLens collaborative-filtering notebook in `linal-hub`
  (the first workload to exercise `CREATE VECTOR INDEX`'s IVF clustering and
  a real similarity `JOIN` at scale). `SEARCH ... INTO <target>` is
  unchanged.

See the parent repository's `CHANGELOG.md` (`[Unreleased]`) for full detail.

## [0.1.6] - 2026-09-14

Bumped for the engine fixes below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `SimdBackend` rejected a legitimate scalar/shape broadcast (e.g. `matrix * 0.5`)
  with a bare `"Shape mismatch"` error for any tensor at or above the 1024-element
  SIMD threshold, even though the identical operation worked fine below it.
- An un-aliased qualified `SELECT` column (`SELECT t.col FROM t`, no `AS`) was
  labeled `__cmp_0` in the output instead of its real name — the underlying data
  was always correct, only the column label was wrong.
- A qualified column (`t.col`) failed to parse at all in `GROUP BY`, plain
  `ORDER BY`, window `PARTITION BY`/`ORDER BY`, and `LAG`/`LEAD`'s column
  argument, even though the identical qualified column already worked in
  `SELECT`/`WHERE`.

See the parent repository's `CHANGELOG.md` (`[Unreleased]`, three entries) for
full detail.

## [0.1.5] - 2026-09-13

Bumped for the engine fixes below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- A `JOIN`'s `SELECT`/`WHERE`/aggregate expressions could silently return the
  *wrong* table's value for a qualified `table.col` reference when both sides
  of the join shared a bare column name — no error, just wrong data.
- The 12 classical-linear-algebra keywords (`RANK`, `TRACE`, `DETERMINANT`, ...)
  are now usable as ordinary identifiers (column names, `AS` aliases, bare
  references) anywhere the grammar expects one, not just as the operator they
  otherwise start.
- `FROM` is now optional for a literal/computed-only `SELECT`
  (e.g. `SELECT L2_NORM([3.0, 4.0]) AS five`, no dataset needed).

See the parent repository's `CHANGELOG.md` (`[0.1.82]`) for full detail.

## [0.1.4] - 2026-09-13

Bumped for the engine fix below (this crate has no code of its own beyond the
PyO3 bindings, so it just picks up the new `linal` engine behavior):

- `EXPLAIN LINEAGE` could misattribute ancestry across a zero-copy `TRANSPOSE`
  — a transposed matrix and its untransposed source could hash identically,
  since `TRANSPOSE` shares the same underlying buffer as its input in this
  engine's storage model.

See the parent repository's `CHANGELOG.md` (`[0.1.81]`) for full detail.

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

## [0.1.0] - 2026-09-09

Initial release: a PyO3 extension (`src/lib.rs`) embedding the engine's
synchronous `TensorDb`/`execute_line` directly in the Python process, no
server required — alongside, not replacing, the HTTP-based
`clients/python` package. Built 2026-09-06; published to PyPI as
[`linaldb`](https://pypi.org/project/linaldb/0.1.0/) on 2026-09-09
alongside the naming rename from `linaldb-embedded`, with wheels for
macOS (aarch64), Linux (manylinux x86_64), and Windows (x86_64), all
`cp39-abi3`.

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
