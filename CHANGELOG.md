# Changelog

All notable changes to LINAL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.1.56] - 2026-07-17

### Fixed — HDF5/Numpy/Zarr connectors flattened all N-D arrays to 1D, discarding shape

Found via the same real-HDF5-file testing that surfaced v0.1.55's round-trip
bug: `USE DATASET FROM` on a real 4x3 HDF5 matrix produced a `Vector[12]`
resource instead of `Matrix[4,3]`. Root cause: `record_batch_to_tensors`
always built a flat `Shape::new(vec![num_rows])`, and none of the three
scientific connectors attached the original array shape anywhere it could
read it from — each flattened via `.iter().cloned().collect()` and handed
back a plain 1D Arrow array.

Fixed by adding an Arrow `Field` metadata side-channel
(`core::connectors::field_with_shape` / `read_shape_metadata`, keyed
`"linal.shape"`) that connectors populate from their own shape APIs
(`hdf5::Dataset::shape()`, `ndarray::ArrayD::shape()`, zarrs'
`Array::shape()`) whenever a column is genuinely multi-dimensional.
`record_batch_to_tensors` now reads it back (falling back to the flat
assumption if missing or inconsistent with the actual data length — a
defensive guard against stale/corrupt metadata) to build the real `Shape`.
This activates two previously-dead code paths that already correctly
handled rank-2 tensors (`persistence.rs`'s `use_dataset_core` and
`engine/db.rs`'s `materialize_tensor_dataset`), which had been unreachable
since every connector-sourced shape was always rank-1.

**Behavior change to note**: a genuine rank>2 array (e.g. a 3D HDF5
dataset) now correctly gets a real rank-3 `Shape` internally, but
`USE DATASET FROM` then fails loudly with "Cannot materialize tensor with
rank > 2" instead of silently returning flattened, mislabeled data as
before. This is an intentional improvement (loud failure over silent
wrong data); N-D materialization itself remains out of scope.

New `tests/scientific_shape_preservation_test.rs`, using an inline
non-square 4x3 fixture (the checked-in `test_data.h5`/`test_data.npy`
fixtures are a constant 2x2 and can't catch ordering bugs): HDF5 and Numpy
2D shape preservation at the tensor level, an end-to-end `USE DATASET FROM`
check that a 4x3 HDF5 file materializes as 4 rows of `Vector(3)`, and a
rank-3 test documenting that the shape mechanism itself works while
materialization correctly still fails loudly.

## [0.1.55] - 2026-07-17

### Fixed — `IMPORT DATASET FROM` produced a package `LOAD DATASET` could never load back

Found while testing the engine against a real downloaded HDF5 file:
`import_dataset_core`, the shared handler
behind `IMPORT DATASET FROM <path> AS <name>` for every connector (HDF5,
Numpy, Zarr, CSV-via-`IMPORT`), wrote the tensor-first dataset package via
`save_dataset_package` but never wrote the legacy `.meta.json` sidecar file
that `LOAD DATASET`'s existence check and schema reconstruction hard-require.
The command reported success and the data was correctly on disk, but the
dataset was invisible to `SHOW ALL DATASETS` and `LOAD DATASET` failed with
"Dataset not found" — silently breaking the documented "import once, load
later" workflow for every connector-sourced import.

Fixed by adding `ParquetStorage::save_legacy_metadata` /
`save_legacy_metadata_for_batch` (builds a minimal legacy schema from the
Arrow `RecordBatch` via a new `arrow_schema_to_tuple_schema` mapping) and
calling it from `import_dataset_core` right after `save_dataset_package`.
`StorageEngine::save_dataset`'s existing inline sidecar-writing code was
refactored to share the same `save_legacy_metadata` method (no behavior
change there — it still uses its own fully-populated metadata with live
per-column stats).

New tests: `tests/ingestion_test.rs::test_import_dataset_from_csv` now
asserts `LOAD DATASET` succeeds after `IMPORT DATASET FROM` (previously the
test stopped right at the gap); new
`tests/persistence_test.rs::test_import_dataset_hdf5_round_trip` exercises
the fix directly against the HDF5 connector and storage layer.

Note: the scientific connectors (HDF5/Numpy/Zarr) still flatten all
imported arrays to a 1D shape regardless of their original dimensionality —
that is a separate, known limitation, tracked for a follow-up fix.

## [0.1.54] - 2026-07-17

### Added — end-to-end showcase example: single-cell PBMC reference-based cell typing

New `examples/pbmc_cell_typing.lnl` replicates a real single-cell RNA-seq
analysis technique — reference-based "label transfer" / nearest-centroid
classification, as used in tools like Seurat and SingleR — against
synthetic PBMC marker-gene data (not downloaded; deterministically
generated, fixed seed), with ground truth retained so the classifier's
accuracy can be validated directly.

Demonstrates nearly the full DSL feature surface in one coherent workflow:
`Vector`-typed columns, a named pipeline for QC filtering, a vector index +
index-accelerated similarity `JOIN` for classification, `ROW_NUMBER`/`RANK`
window functions, CTEs and `FROM`-subqueries, an equi-`JOIN` against a
small metadata table (the asymmetric-size case the v0.1.51 smaller-side
hash join targets), `AVG_VEC` per-type expression centroids, `CASE`-based
accuracy scoring, `MATMUL`/`TRANSPOSE` for a cell-type similarity matrix,
and dataset persistence. A documented, simplified synthetic effect (a
1.35x CD14 boost on sepsis-condition monocytes) is injected at data
generation time so the differential-analysis step has a genuine,
recoverable signal — not a claim about real measured magnitudes.

Building this surfaced and fixed two real engine bugs, released
separately: `CASE WHEN <comparison>` always taking the `ELSE` branch
(v0.1.52) and `GROUP BY` silently dropping/misnaming aggregate columns
when the `SELECT` list also had a `Computed` item (v0.1.53).

New `tests/examples_cli_smoke_test.rs::test_example_pbmc_cell_typing_runs_clean`
and a deeper `tests/examples_integration.rs::test_pbmc_cell_typing_integration`
asserting on dataset row counts (240 raw cells, 237 after QC, 6 per-type
profiles) and the vector index. Full suite passes, 0 failures.

---

## [0.1.53] - 2026-07-17

### Fixed — GROUP BY silently dropped/misnamed aggregate columns when a Computed item was also in the SELECT list

Found while building the PBMC cell-typing showcase example: `SELECT
small.sid AS sid, small.cond AS cond, COUNT(*) AS n, AVG(big.val) AS
avg_val FROM big JOIN small ON ... GROUP BY sid, cond` returned a 4-column
result where the `n`/`avg_val` columns were silently replaced by duplicated
`sid`/`cond` values — the actual aggregate results never appeared.

Root cause: any qualified column with an alias (`t.col AS col`) — not just
a genuine computed expression — parses as `SelectExpr::Computed`, which
makes `execute_select`'s window/computed post-processing path run even
when a `GROUP BY` is present (that path isn't exclusive to ungrouped
queries, despite a comment added in v0.1.45/Track G3 assuming it was). That
path's `agg_idx` counter (introduced in v0.1.45 to look up an aggregate's
real output name instead of a hardcoded placeholder) assumed
`base_schema.fields` were *only* the aggregate outputs, one per Aggregate
`SelectExpr` in order — true only without a `GROUP BY`. With one,
`LogicalPlan::Aggregate::schema()` puts the group-key fields first, so
`agg_idx` pointed at the wrong fields entirely.

Fixed by starting `agg_idx` at `s.group_by.len()` instead of `0`, so it
correctly skips past the group-key fields before indexing into the
aggregate portion of the schema.

New `tests/group_by_with_computed_column_test.rs`: the exact
qualified-alias-plus-JOIN-plus-GROUP-BY repro, a genuinely computed
expression (`UPPER(...)`) alongside GROUP BY + an aggregate (confirming
the bug wasn't specific to column qualifiers), and a guard test for the
plain GROUP BY + aggregate case. Full suite passes, 0 failures.

---

## [0.1.52] - 2026-07-17

### Fixed — CASE WHEN (and any comparison in a computed column or aggregate) silently always took the ELSE branch

Found while building a real-world showcase example (single-cell reference
classification): `SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END)` — a
standard accuracy-metric pattern — always returned `0`, even when some rows
genuinely matched.

Root cause: `evaluate_expression` (`src/query/physical.rs`), the expression
evaluator used for CASE WHEN conditions, computed `SELECT` columns, and
aggregate inner expressions, had no handling for comparison operators (`=`,
`!=`, `>`, `<`, `>=`, `<=`) in its `Expr::BinaryExpr` arm — only arithmetic
(`+`, `-`, `*`, `/`). Any comparison silently fell through to `Value::Null`,
and since `Value::Null` is never `Value::Bool(true)`, every `CASE WHEN
<comparison>` — including the exact form `DSL_REFERENCE.md`'s own example
uses (`CASE WHEN score > 90 THEN ...`) — always evaluated to the `ELSE`
branch, with no error raised. `WHERE` clauses were never affected: they
route through a separate, already-correct implementation
(`query::planner::evaluate_expr`).

This had **zero test coverage** anywhere in the suite. Fixed by evaluating
comparison operators generically via `Value::compare()` before falling
into the arithmetic type-pair match, mirroring the WHERE-clause evaluator.
New `tests/case_when_comparison_test.rs`: all 6 comparison operators, in a
plain computed column, a bare aggregate, and a `GROUP BY` aggregate, across
Int/String types, plus the exact documented `DSL_REFERENCE.md` grade
example as a direct regression test. Full suite passes, 0 failures.

---

## [0.1.51] - 2026-07-17

### Changed — equi-join now hashes the smaller side; renamed `NestedLoopJoinExec` → `HashJoinExec`

`NestedLoopJoinExec` was already hash-based (not a true nested loop) but
always hashed a *fixed* side per `JoinType` (right for Inner/Left/Full, left
for Right), regardless of which side was actually smaller — `tiny JOIN huge`
ended up hashing `huge` instead of `tiny`, the wrong choice for performance.

- Renamed to `HashJoinExec` and generalized to always build the hash table
  on whichever materialized side has fewer rows (ties build right, matching
  the previous default for Inner/Left/Full), fully decoupled from which side
  outer-join NULL-preservation applies to (a separate, `JoinType`-driven
  decision).
- Hash keys are now `Value` directly — `Value` already implements `Hash`/
  `Eq` with proper float-bits comparison (`src/core/value.rs`) — instead of
  a `format!("{:?}", value)` allocation per row on both sides; the new code
  only clones a key on the smaller (build) side.
- Output rows/columns are unchanged (left-then-right column order
  preserved, NULL-padding semantics identical); internal row ordering may
  differ when the smaller side changes between queries.
- Corrects `docs/ARCHITECTURE.md`'s "Join Execution" section, which
  (incorrectly) said equi-joins were "always a nested-loop scan, not a hash
  join" — also fixed an incidental `JoinKind`/`JoinType` mix-up in the same
  paragraph.

New unit tests for the build-side decision (`src/query/physical.rs`) and
DSL-level regression tests for asymmetric row-count joins in both size
directions (`tests/hash_join_test.rs`). Full suite passes, 0 failures.

---

## [0.1.50] - 2026-07-17

### Fixed — parser errors are no longer discarded; SECURITY.md contact email unified

Fixes the two items flagged, but not fixed, at the end of the round-2
consistency audit (v0.1.45-v0.1.49).

- **Parser errors were silently discarded.** `execute_line_with_context`
  (`src/dsl/mod.rs`) used `if let Ok(stmt) = parser::parse(line)`, which
  threw away the `Err` case entirely — any line that failed to parse
  reported a generic `Unknown command: <raw line>` regardless of what
  actually went wrong, even though the parser had already produced a
  structured `ParseError { offset, msg }` with the real expectation detail.
  Fixed by matching on the `Result` instead: a real parse failure now
  surfaces the parser's own message and byte offset, e.g. `GET * FROM
  users` now reports `expected a statement keyword, found identifier
  `GET` (at byte 0)` instead of `Unknown command: GET * FROM users`.
  `ParseError::into_dsl_error` (previously dead code — never called
  anywhere) now also preserves the byte offset via `Display` instead of
  dropping it, and is the single conversion path used here.
- **Found while fixing the above**: pure `--`-comment lines errored out as
  `Unknown command` when reached directly through `execute_line`/
  `execute_line_with_context` (REPL, server `/execute`) — the blank/comment
  fallback only checked `#`/`//`, not `--`, even though the lexer treats
  all three as comment styles and `execute_script`'s own line-by-line
  pre-filter already checked all three. Only affected direct callers;
  `run`/script execution was never affected since `execute_script` filters
  comment lines before they reach this function at all. Fixed by checking
  `--` too.
- **`SECURITY.md` gave two different contact emails** (`security@gorigami.xyz`
  in one section, `dev@gorigami.xyz` in another) for reporting a
  vulnerability. Unified on `develop@gorigami.xyz`, matching the address
  README.md already uses for commercial licensing inquiries.

Updated `docs/ERROR_REFERENCE.md`'s Parse Error section and example to match
the new, correct behavior. New regression coverage in
`tests/parser_error_surfacing_test.rs`. Full suite passes, 0 failures.

---

## [0.1.49] - 2026-07-16

### Documented — README/CONTRIBUTING cross-consistency, closes the round-2 audit (Track K)

Closes out Track K of `CONSISTENCY_PLAN.md` — the last track of the round-2
follow-up audit (v0.1.45-v0.1.49). No code changes.

- Fixed CONTRIBUTING.md's repo URL: it used `gorigami/linal.git` (wrong,
  including a stale `cd linal` directory name) while README.md and the
  actual git remote both say `gorigami/linaldb.git`.
- Fixed CONTRIBUTING.md's stale "Example tests" description — since v0.1.42,
  `examples/` holds only `.lnl` scripts; the Rust fixture generators moved
  to `tools/fixtures/` as `[[example]]` entries in `Cargo.toml`.
- Added `docs/DATASET_ARCHITECTURE.md` to README's Documentation Hub —
  it was substantively rewritten back in v0.1.44 but never linked.
- Linked `SECURITY.md` from CONTRIBUTING.md's Getting Help section — it
  existed at the repo root but was linked from nowhere.
- Refreshed CONTRIBUTING.md's Project Structure tree: added `tools/`,
  `benches/`, `scripts/`, `data/`, `.github/`, `SECURITY.md`, none of
  which were reflected after the last two reorganizing PRs.

**Flagged, not fixed**: `SECURITY.md` itself gives two different contact
emails for reporting a vulnerability in two different sections of the same
file — an internal inconsistency worth a dedicated look, out of scope for
a cross-doc-consistency pass since it requires knowing which address is
actually monitored.

Full suite passes, 0 failures (no code touched by this release).

**This closes the round-2 consistency audit** (Tracks G-K, v0.1.45-v0.1.49)
— `CONSISTENCY_PLAN.md` is deleted in this same release; see git history for
the full checklist if needed later.

---

## [0.1.48] - 2026-07-16

### Documented — DATASET_ARCHITECTURE.md and ERROR_REFERENCE.md gaps (Track J)

Closes out Track J of `CONSISTENCY_PLAN.md` (round 2). No code changes.
PR #33 (v0.1.44) rewrote both docs, but the follow-up audit found real
inaccuracies survived that rewrite.

**`docs/DATASET_ARCHITECTURE.md`**:
- `ResourceReference` now documents both variants (`Tensor { id }` and the
  previously-omitted `Column { dataset, column }`).
- Fixed the `graph.rs`/`DatasetGraph` attribution: it's actually used by
  `ATTACH` and `AUDIT DATASET`, not `BIND` (plain aliasing) or `DERIVE`
  (unrelated tensor-expression evaluation) as previously claimed.
- Fixed the `schema_evolution.rs` attribution: there is no `LIST VERSIONS`
  command; the real command is `SHOW DATASET VERSIONS <name>`.
- Fixed the `lineage.rs` attribution: `SHOW LINEAGE <name>` actually walks
  an unrelated tensor-computation `LineageNode` type in `engine/db.rs` —
  `core::dataset::lineage` is populated only by the scientific-ingestion
  connectors, tracking data-*import* provenance, not `SHOW LINEAGE`'s
  tensor-derivation DAG.

**`docs/ERROR_REFERENCE.md`**:
- Added the 2 `EngineError` variants missing from the table: `Store` and
  `DatasetError` (the latter is user-reachable — e.g. loading a dataset
  under a name already in use).
- Fixed the sample Parse and Engine error messages to match actual runtime
  `Display` output — the parser's structured `ParseError` (with byte
  offset) is silently discarded at its only call site and replaced with a
  generic "Unknown command" message; noted this explicitly instead of
  documenting the discarded, richer error as if it surfaced.
- Rewrote the "Storage Errors" section entirely: it documented the wrong
  type (`StoreError`, the in-memory *tensor* store error) instead of the
  actual persistence error type (`StorageError`), and included a variant
  (`UnsupportedFormat`) that doesn't exist anywhere in source.

**Flagged, not fixed**: the discarded structured parser error (with byte
offset and expectation detail) is a real DX regression, not just a doc
gap — worth a dedicated future PR to stop throwing it away.

Full suite passes, 0 failures (no code touched by this release).

---

## [0.1.47] - 2026-07-16

### Documented — ARCHITECTURE.md gaps found by the follow-up documentation audit (Track I)

Closes out Track I of `CONSISTENCY_PLAN.md` (round 2). No code changes.

- Fixed the storage-layout description: dataset persistence is a per-dataset
  **directory package** (`datasets/{name}/data.parquet` plus sibling
  `schema.json`/`stats.json`/`lineage.json`/`manifest.json`), not a single
  flat `.parquet` file as previously described.
- Added the sixth executor file, `executor/pipeline.rs`, to the executor-split
  description (previously said "five files").
- Refreshed every stale hardcoded count (parser tests, `Expr`/`Statement`/
  `CallExpr` variants, lexer tokens) by re-reading the source directly rather
  than trusting the previous numbers — several were already wrong even at
  the time they were written. Added a `grep`/source pointer next to each
  count so it can be re-verified instead of silently drifting again.
- Added two new "Query Processing" subsections that were previously entirely
  undocumented despite being real, shipped features: **Join Execution**
  (`JoinKind`, `NestedLoopJoinExec`/`SimilarityJoinExec`, the v0.1.40
  qualified-column/table-alias fixes) and **Window Function Execution**
  (`ROW_NUMBER`/`RANK`/`LAG`/`LEAD`/aggregate-as-window, the v0.1.37
  nullable-schema fix, the v0.1.45 vector-aggregate fix).
- Added `context.rs` (`ExecutionContext`) to the Engine Module listing,
  `persistence.rs` to the DSL Module listing, and `dataset_server.rs`
  (the `/delivery` HTTP routes) to the Server Module listing — all three
  were real, substantial modules with zero mention.
- Extended the `Expr` documentation (both `dsl::ast::Expr` and
  `query::logical::Expr`) to include `Case`/`Coalesce`/`Nullif`/`Cast` and
  the four newer `VectorFnKind` variants (`Matmul`/`Transpose`/`MatShape`/
  `Flatten`) — previously undocumented despite being fully implemented.
- Corrected two minor-but-wrong performance claims: `CpuBackend`'s SIMD
  dispatch checks element count only (contiguity is checked one layer
  down, inside `SimdBackend`), and the `<=16`-element `SmallVec` fast path
  isn't actually zero-heap-allocation overall (the final `.to_vec()` still
  allocates once).

Full suite passes, 0 failures (no code touched by this release).

---

## [0.1.46] - 2026-07-16

### Documented — DSL_REFERENCE.md gaps found by the follow-up doc audit (Track H)

Closes out Track H of `CONSISTENCY_PLAN.md` (round 2) — documentation debt
in `docs/DSL_REFERENCE.md` found by the same audit that produced v0.1.45's
Track G bug fixes. No code changes.

- Corrected the `UNION`/`UNION ALL` claim: chained 3-way+ unions actually
  work (verified live), not just a single union.
- Documented the real default column name for aggregate-as-window functions
  (`sum(expr)_over`, not `sum`).
- Bumped the stale `"version": "0.1.34"` in the example pipeline JSON and
  noted the field is informational only, never read back on load.
- Added a new "DATASET ... FROM (Materialized View)" subsection — this
  form was entirely undocumented; only the `COLUMNS (...)` form existed.
- Documented the WHERE/FILTER predicate vocabulary: `IN (...)`,
  `BETWEEN ... AND ...`, `IS NULL`/`IS NOT NULL`, `DISTINCT`, and
  `LIMIT ... OFFSET ...` — all previously undocumented.
- Added a new "Subqueries in FROM" subsection for `FROM (SELECT ...) AS
  alias`.
- Added `MAT_SHAPE`, `MATMUL`, `TRANSPOSE` to the Vector Scalar Functions
  table (they work in `SELECT` today, alongside the pre-existing
  standalone tensor-DSL keyword forms).
- Documented all `EXPLAIN` target forms (`DATASET`, `SEARCH`, the optional
  `PLAN` keyword) — the doc previously implied `EXPLAIN` only covered
  `SELECT`.
- Documented `LIST DATASET PACKAGES` (alias of `LIST DATASETS`).
- Documented that `#` and `//` are valid line-comment markers alongside
  `--`.
- Documented that the `CSV` keyword in `EXPORT [CSV] name TO "path"` is
  optional.
- H1/H2 (bare-aggregate alias and `SUM_VEC`/`AVG_VEC OVER` doc examples)
  turned out to already be correct as of v0.1.45's Track G fixes — verified
  live, no change needed.

Full suite passes, 0 failures (no code touched by this release).

---

## [0.1.45] - 2026-07-16

### Fixed — silent correctness bugs found by a follow-up documentation audit (Track G)

A second doc-vs-engine audit (this time against `docs/DSL_REFERENCE.md`,
`ARCHITECTURE.md`, `DATASET_ARCHITECTURE.md`, and `ERROR_REFERENCE.md`
individually, not just the top-level docs v0.1.44 covered) surfaced two live
engine bugs while verifying doc examples by execution, plus one CLI bug.
Tracked as Track G in `CONSISTENCY_PLAN.md` (round 2).

- **`SUM_VEC(col) OVER (...)` / `AVG_VEC(col) OVER (...)` silently returned
  `0.0`** instead of an element-wise running aggregate on vector columns.
  Root cause: the parser collapses `SumVec`/`AvgVec` into the generic
  `WindowFunc::Sum`/`Avg` (same as plain `SUM`/`AVG`, matching how the
  non-windowed accumulator already treats them), but the window executor
  (`apply_window_func`, `src/dsl/executor/query.rs`) only ever accumulated
  `Int`/`Float`, defaulting `Value::Vector`/`Value::Matrix` to `0.0`. Added
  `window_running_sum`, a vector/matrix-aware running accumulator mirroring
  `AggregateExec`'s grouped SUM/AVG logic (`src/query/physical.rs`), that
  errors on dimension/shape mismatch instead of silently zeroing. Also fixed
  the window column's output-type inference, which had the same blind spot
  (defaulted anything non-scalar to `Int`).
- **Bare (non-windowed) aggregates silently ignored `AS alias`** — `SELECT
  SUM(price) AS total FROM t` always named the output column `SUM(price)`,
  never `total`. The parser explicitly discarded the alias
  (`src/dsl/parser/dataset.rs`). `SelectExpr::Aggregate` and
  `query::logical::Expr::AggregateExpr` both gained an `alias: Option<String>`
  field, threaded through to the schema-naming logic
  (`query::logical::LogicalPlan::schema`), so the alias is now honored end
  to end.
- **Found while fixing the alias bug**: a `SELECT` mixing a bare aggregate
  with a window function in the same list (no `GROUP BY`, e.g. `SELECT
  SUM(price) AS total, ROW_NUMBER() OVER (...) AS rn FROM t`) could silently
  drop the aggregate column from the final projection — the post-processing
  step looked up the column by a hardcoded `"agg"` placeholder instead of
  its real output name. Fixed by looking up the actual name from the
  aggregate physical plan's schema (which also naturally picks up the alias
  fix above).
- **`linal --version` reported a hardcoded, stale `0.1.9`** (`src/main.rs`)
  while `Cargo.toml` was at `0.1.44` — 35 releases out of date. Now sourced
  from `env!("CARGO_PKG_VERSION")`.

New regression coverage in `tests/silent_correctness_test.rs` (Track G
section) and `tests/cli_hardening_test.rs`. Full suite passes, 0 failures.

Documentation debt found by the same audit (missing/stale coverage in
`DSL_REFERENCE.md`, `ARCHITECTURE.md`, `DATASET_ARCHITECTURE.md`,
`ERROR_REFERENCE.md`, `README.md`/`CONTRIBUTING.md`) is tracked separately
as Tracks H-K in `CONSISTENCY_PLAN.md` — no doc-only changes in this release.

---

## [0.1.44] - 2026-07-16

### Cleaned up — documentation audit against the mission statement

Held every doc against the mission statement ("a high-performance,
in-memory analytical engine... SQL-inspired DSL that treats vectors,
matrices, and multi-dimensional tensors as first-class citizens") and the
current codebase.

- **Deleted** `DOCUMENTATION_ALIGNMENT_REPORT.md` (root) — a one-time
  audit snapshot pinned to "Codebase Version: v0.1.14", ~28 versions and
  a full architecture migration stale; its own recommendations were
  already carried out (the `docs/archive/` convention it proposed exists
  and holds exactly the files it named).
- **Deleted** `docs/SCIENTIFIC_DATASET_INGESTION_PLAN.md` — titled
  "COMPLETED", every phase checked off, fully redundant with the living
  `docs/DSL_REFERENCE.md` scientific-ingestion section.
- **Deleted** `docs/EXAMPLES.md` — linked to `examples/end_to_end.lnl`
  (deleted in v0.1.41), used non-portable absolute `file:///` paths, and
  fully duplicated the freshly-written `examples/README.md`.
- **Archived** `docs/BENCHMARKS.md` → `docs/archive/` (gitignored, local
  only — same convention as the rest of `docs/archive/`) — dated
  2026-01-02 against "v0.1.10 (unreleased)", presenting ~6 months of
  stale numbers as current was actively misleading.
- **Rewrote** `docs/DATASET_ARCHITECTURE.md` — fixed a wrong file path
  (`dataset/dataset.rs` doesn't exist; it's `dataset/mod.rs`), added the
  `metadata` field the struct actually has, and documented all 9
  previously-unmentioned submodules (`reference.rs`, `registry.rs`,
  `graph.rs`, `schema.rs`, `schema_evolution.rs`, `lineage.rs`,
  `manifest.rs`, `stats.rs`, `metadata.rs`).
- **Fixed real bugs in `README.md`**: the Quick Start used the wrong
  binary name (`linaldb` instead of `linal`) in all 4 CLI invocations —
  every command as written failed with "command not found" — and linked
  to `examples/end_to_end.lnl`, which no longer exists. Also updated the
  Documentation Hub links to drop the two removed/archived docs and point
  at `examples/README.md` instead.
- **Fixed dangling links**: `docs/ARCHITECTURE.md` and `CONTRIBUTING.md`
  both referenced `Tasks_implementations.md` at its pre-archive path.
- **Fixed a cosmetic mismatch** in `docs/ERROR_REFERENCE.md`: the sample
  error message used Rust's `Debug` format instead of the real `Display`
  output engine errors actually print.

Full suite passes, 0 failures.

---

## [0.1.43] - 2026-07-16

### Fixed — CI: smoke tests invoked the wrong binary profile

- `tests/examples_cli_smoke_test.rs` and `tests/cli_hardening_test.rs`
  hardcoded `target/debug/linal` as the binary path. CI runs
  `cargo test --release`, so every test in `examples_cli_smoke_test.rs`
  failed on the open PR ("No such file or directory") — passed locally
  only because a debug build happened to already exist. Fixed both to use
  `env!("CARGO_BIN_EXE_linal")`, which Cargo resolves to the correct path
  for whichever profile actually built the test binary. Verified by
  running the exact CI invocation locally (`cargo test --release -j 1`
  with the same `--skip` flags).

---

## [0.1.42] - 2026-07-16

### Cleaned up — follow-up to the examples/tests audit

- **`examples/`**: moved the two fixture-generating Rust binaries
  (`gen_test_data.rs`, `gen_zarr_data.rs`) out of `examples/` into
  `tools/fixtures/`, registered as explicit `[[example]]` entries in
  `Cargo.toml` so `cargo run --example gen_test_data` (used by CI) keeps
  working unchanged. `examples/` now contains only `.lnl` scripts (plus
  `README.md` and the `data/` fixture folder) — no stray `.rs` files.
  Deleted the now-empty `gen_zarr_data_minimal.rs` reference from the
  README table.
- **`tests/`**: removed `dataset_validation_test.rs`, a true duplicate of
  `dataset_integrity_test.rs::test_row_count_validation_error_message`
  (same scenario, different variable names). Folded 3 more single-test
  files into their natural siblings with no loss of coverage:
  `dataset_zero_copy_test.rs` → `tensor_arc_sharing_test.rs` (both assert
  the Arc zero-copy invariant, just at different layers — dataset-column
  vs. raw `Tensor`); `engine_scenarios.rs` →
  `engine_matrix_ops.rs::test_engine_binary_and_unary_scenario` (both
  bypass the DSL and exercise `ExecutionContext` directly); `symbol_resolution.rs` →
  `dsl_dataset_complete.rs::test_dataset_derived_column_and_persistence_round_trip`
  (both are DSL-level dataset workflow tests). 68 → 64 test files. Full
  suite still passes, 0 failures.

---

## [0.1.41] - 2026-07-16

### Cleaned up — examples/ and tests/ audit

- **`examples/`**: fixed 3 example scripts that referenced dead engine
  state (`benchmark.lnl`'s `DROP DATABASE` on a DB that didn't exist yet,
  `export_import_csv.lnl`'s missing CSV fixture and legacy-`IMPORT CSV`
  path-resolution mismatch, `persistence_demo.lnl`'s `LOAD DATASET` path);
  deleted 3 that used syntax the parser no longer accepts at all
  (`end_to_end.lnl`, `test_persistence.lnl`, `verify_dataset_export.lnl`);
  deleted a dead scratch script (`gen_zarr_data_minimal.rs`, a 6-line
  no-op); added `pipelines_and_search.lnl` to cover window functions,
  `CAST`-to-tensor, index-accelerated similarity `JOIN`, and pipelines —
  previously undemonstrated. Every example now has a smoke-test assertion
  (`tests/examples_cli_smoke_test.rs`, new) that it runs clean, in
  addition to the deeper correctness assertions already in
  `tests/examples_integration.rs`. See `examples/README.md` (new) for the
  convention going forward.
- **`tests/`**: merged 2 pairs of near-duplicate single-test files into
  their larger siblings (`dataset_dsl_test.rs` +
  `comprehensive_tensor_dataset_test.rs` → `tensor_dataset_dsl_test.rs`;
  `lazy_dsl_test.rs` → folded into `lazy_evaluation_test.rs`, renamed
  `lazy_expression_test.rs`); renamed 5 more files whose names collided
  across unrelated features — most notably "indexing", which meant both
  tensor element slicing (`tensor_indexing_server_test.rs`,
  `tensor_expression_indexing_server_test.rs`) and the DB index feature
  (`dataset_index_feature_test.rs`) — and "zero copy"
  (`tensor_arc_sharing_test.rs` vs. `dataset_zero_copy_test.rs`, formerly
  distinguished only by a plural `s`). No test coverage was removed; full
  suite still passes (71 binaries, 0 failures).

---

## [0.1.40] - 2026-07-16

### Fixed — Track F of CONSISTENCY_PLAN.md: qualified columns, table aliasing, FLATTEN in SELECT

Closes out the consistency/correctness audit started 2026-07-12 (Tracks
A–F, 31 items). This release fixes two bugs found while verifying Track
D's JOIN documentation — one of them was silently corrupting far more than
just JOIN output.

- **Any unaliased computed SELECT expression was silently dropped from
  the output.** Not specific to qualified columns — `SELECT id, price * 2
  FROM t` (no `AS`) used to return only `id`. Root cause:
  `apply_window_and_computed_exprs` names an unaliased computed column
  `__cmp_{idx}` when appending it to each row, but the final SELECT-order
  projection step looked it up under the unrelated literal string
  `"expr"` — the lookup failed and the column was silently filtered out.
  Fixed by making both sites use the same naming scheme.
- **`table.col` always evaluated to `NULL`.** It parses to
  `Expr::Field { base, field }`, which the SQL row evaluator had no case
  for and fell through to a generic `Null` default. Fixed by resolving it
  to the bare column name, matching how the JOIN `ON` clause already
  treats `table.col` (qualifier accepted for readability, only the column
  name is used to resolve the value).
- **`FROM table alias` / `JOIN table alias ON ...` didn't parse at all.**
  Added optional `[AS] <alias>` parsing after a dataset name in both
  clauses. The alias itself isn't tracked for disambiguation (per the
  point above) — documented as a limitation.
- **`FLATTEN(col)` inside `SELECT` silently evaluated to `NULL`.** Same
  underlying shape as the `table.col` bug: `NORMALIZE`/`MATMUL`/`TRANSPOSE`
  have a dual-branch parser (`NAME(x)` → SQL-context function,
  bare `NAME x` → tensor-DSL keyword) that `FLATTEN` never got, so
  `FLATTEN(v)` fell into the tensor-DSL path, which the SQL evaluator
  doesn't handle. Added the same dual-branch parsing, wired through type
  inference and the physical evaluator. `RESHAPE` inside `SELECT` is left
  as a parse error and documented rather than fixed the same way —
  `CAST(expr AS VECTOR(n)/MATRIX(r,c))` (v0.1.39) already covers
  arbitrary-shape reshaping inside a query.

The `DSL_REFERENCE.md` JOIN example that shipped broken in Track B
(`SELECT o.id, u.name FROM orders o JOIN users u ON ...`) now works
verbatim and has been restored as the primary example.

New tests: `tests/qualified_column_test.rs` (7 tests), plus 3 more added
to `tests/correctness_integration.rs` for `FLATTEN`.

### Removed

`CONSISTENCY_PLAN.md` — the working document tracking this audit. All 6
tracks (A: silent correctness bugs, B: documentation debt, C: test
coverage, D: architecture/design debt, E: window function bug, F: this
release) are complete; deleting it per its own stated completion process.

---

## [0.1.39] - 2026-07-15

### Added — Track D of CONSISTENCY_PLAN.md (architecture/design debt)

All 7 items resolved in one pass — some were fixed, some were bugs found
during investigation (not just naming/design questions), some were
documented as intentional, and one was removed as dead code.

- **D1 — `CAST` to `VECTOR(n)`/`MATRIX(r, c)`**: reshape/flatten a
  Vector/Matrix column inline in a query (row-major, dimension mismatch
  returns `NULL`). A deep-dive found `RESHAPE`/`FLATTEN` — thought to make
  this redundant — don't actually work inside `SELECT` at all (see "Found
  while implementing this release" below), so this fills a real gap.
- **D2 — index-accelerated similarity `JOIN`**: `JOIN <ds> ON
  COSINE_SIM(a.col, b.col) > threshold`. New `SimilarityJoinExec` uses a
  `Vector` index on the right dataset's column when one exists, falling
  back to brute-force O(n·m) otherwise — same index-or-fallback pattern as
  the existing `CosineFilterExec`. Supports INNER/LEFT/RIGHT/FULL.
- **D4 — `AVG(vector_col)` bug fix**: `SUM(vector_col)` already worked
  without the `_VEC` suffix (the executor already merges `Sum`/`SumVec`),
  but `AVG`'s schema-inference hardcoded `Float` regardless of input type,
  so `AVG(vector_col)` errored with a type mismatch even though the
  executor could compute it. Fixed to infer the result type like
  `SUM`/`MIN`/`MAX` do, while still correctly returning `Float` (not `Int`)
  for scalar input.
- **D5 — pipeline persistence path consistency**: relative `SAVE`/`LOAD
  PIPELINE ... TO/FROM` paths now resolve against `<data_dir>/<db>/`,
  matching `TENSOR`/`DATASET` (previously CWD-relative).
- **D7 — `LIST PIPELINES`**: alias for `SHOW PIPELINES`, for naming parity
  with `LIST TENSORS`/`LIST DATASETS`.
- **D3, D6**: no functional change. D3 (the `FilterExec`/`CosineFilterExec`
  split) was confirmed to be a deliberate, correctly-implemented optimizer
  rewrite-rule pattern, not an inconsistency — documented in
  `ARCHITECTURE.md` and a source doc comment. D6
  (`save_all_pipelines`/`load_all_pipelines`) was dead code with no DSL
  command reaching it, and no other object kind has a bulk "SAVE ALL"
  command either — removed rather than wired up.

### Found while implementing this release (not fixed here)

- `RESHAPE(...)` inside a `SELECT` list doesn't parse at all, and
  `FLATTEN(col)` parses but always evaluates to `NULL` — both are only
  wired for the standalone tensor-DSL context, not the SQL `SELECT`
  computed-expression path. `CAST(... AS VECTOR/MATRIX)` above now covers
  the same use case, so this is lower priority, but the silent `NULL` is
  a footgun.
- Table-qualified columns (`table.col`) are only understood inside a
  `JOIN ... ON` clause — the `SELECT` column list doesn't resolve them at
  all (`SELECT a.id FROM a JOIN b ...` silently returns an empty
  schema/row), and `FROM table alias` aliasing doesn't parse. A Track B
  doc example shipped with exactly this bug, undetected because it wasn't
  individually re-verified — now fixed in `DSL_REFERENCE.md`.

Both tracked as Track F (F1, F2) in `CONSISTENCY_PLAN.md`.

---

## [0.1.38] - 2026-07-14

### Testing — Close out Track C of CONSISTENCY_PLAN.md (test coverage gaps)

No functional changes. Closes the remaining Track C items:

- **C2**: `tests/pipeline_vector_engine_test.rs` (5 tests) — pipelines and
  the vector engine were previously tested in total isolation. Covers
  `COSINE_SIM` in a pipeline `WHERE` step (with and without a vector index
  present), `COSINE_SIM`/`MATMUL` as computed `SELECT` columns inside a
  pipeline, and a chained filter-then-`NORMALIZE` pipeline. No bugs found.
- **C3**: `tests/v0128_features_test.rs` (8 tests) — v0.1.28 features
  (subqueries, `IN`/`BETWEEN`, `LIMIT ... OFFSET`, multi-column `ORDER BY`)
  had parser-level unit tests but no end-to-end integration coverage.
  RIGHT/FULL JOIN was already covered elsewhere.
- **C4**: `tests/server_pipeline_search_test.rs` (3 tests) — first
  server-level test coverage for `PIPELINE` and `SEARCH` over HTTP,
  including confirming `APPLY` on a dropped pipeline correctly errors and
  `SEARCH` without a prebuilt vector index correctly errors through the
  HTTP path.
- **C5**: verified (no bug) that `is_read_only()` correctly excludes
  pipeline mutations from the read-only fast path, so the server takes a
  write lock for them; added a regression test.

With this, Tracks A, B, C, and E of `CONSISTENCY_PLAN.md` are fully closed.
Track D (7 items, architecture debt needing design decisions) remains.

---

## [0.1.37] - 2026-07-14

### Fixed — Window functions silently corrupted when combined with differing OVER specs (Track E / E1)

Combining multiple window functions with *different* `OVER (...)` specs in
one `SELECT` — especially mixing `LAG`/`LEAD` with a differently-specced
ranking or aggregate window function — could silently produce wrong values
or an outright schema error, depending on ordering. Found while writing
v0.1.36's window function documentation.

**Root cause**: `apply_window_func` (`src/dsl/executor/query.rs`) built the
new window-result column's `Field` without `.nullable()` — unlike the
sibling `SelectExpr::Computed` code path, which does mark its appended
column nullable. `LAG`/`LEAD` routinely produce `Value::Null` for boundary
rows (the first/last `offset` rows in the window). `Tuple::new`'s schema
validation rejected those rows against the non-nullable field, and the
code silently fell back to the pre-window row via `.unwrap_or(row)` —
leaving the `Vec<Tuple>` with **inconsistent per-row schemas** (some rows
had the new column, some didn't), which cascaded into wrong values or
schema errors in whatever window function ran next.

**Fix**: mark the appended column nullable (matching the `Computed` path),
and replace the silent `.unwrap_or(row)` fallback with a propagated
`DslError` — if `Tuple::new` still fails for any other reason, the query
now errors instead of silently returning a corrupted row.

Adds `tests/window_functions_test.rs` (8 tests) — regression coverage for
this bug plus general window function coverage (`ROW_NUMBER`, `RANK`,
`DENSE_RANK` with ties, `LAG`/`LEAD`, windowed `SUM`), which had almost no
dedicated tests before this.

---

## [0.1.36] - 2026-07-14

### Documentation — Track B of CONSISTENCY_PLAN.md

`docs/DSL_REFERENCE.md` was missing a large share of what's actually
implemented — a 2026-07-12 audit found `INSERT`/`UPDATE`/`DELETE`, `JOIN`,
CTEs/`UNION`, window functions, `CASE`/`COALESCE`/`NULLIF`/`CAST`, string
functions, `SEARCH`, `TRANSFORM`, `CREATE INDEX`, and `SET DATASET METADATA`
entirely undocumented despite being fully implemented and reachable. This
release fills those gaps, adds a new `## 7. Vector Search & Indexing`
section (the doc's numbering previously jumped straight from `## 6` to
`## 8`), and corrects two stale/inaccurate spots:

- `docs/ARCHITECTURE.md`'s "Recovery" section described a metadata-scan and
  lazy-load-on-access mechanism that doesn't exist — startup only registers
  empty `DatabaseInstance` stubs per directory found in `data_dir`. Also
  added a "Pipelines (JSON, v0.1.34)" persistence subsection, which was
  missing entirely.
- A stale source comment in `src/dsl/mod.rs` referenced a "legacy chain"
  fallback for unrecognised DSL syntax that no longer exists (removed in an
  earlier typed-parser migration) — every statement now goes through the
  typed parser; only blank/comment lines are tolerated on parse failure.
- `src/dsl/ast.rs`'s `Statement::Save`/`Statement::Load` doc comments
  implied the `TENSOR`/`DATASET`/`PIPELINE` kind keyword was optional with
  a default; it's actually required.

Every new doc example was executed against the DSL directly (not just
hand-written) to confirm it actually parses and produces the claimed
output before being committed.

### Found while writing this release (not fixed here)

Verifying the window-functions examples surfaced a real engine bug:
combining multiple window functions with *different* `OVER (...)` specs in
one `SELECT` — especially mixing `LAG`/`LEAD` with a differently-specced
ranking or aggregate window function — can silently produce wrong values
or an outright schema error, depending on ordering. Same-spec combinations
work correctly. Tracked as Track E / E1 in `CONSISTENCY_PLAN.md` with
concrete repro queries; the new documentation only uses verified-safe
example combinations and calls this out as a known limitation.

---

## [0.1.35] - 2026-07-12

### Fixed — Silent correctness bugs (Track A of CONSISTENCY_PLAN.md)

A 2026-07-12 audit of the engine/DSL/persistence layers found several
places where invalid or unsupported operations were silently accepted
instead of erroring, producing wrong results with no diagnostic.

- **`ORDER BY` on Vector/Matrix columns** now errors instead of silently
  leaving the sort a no-op (`Value::compare` has no defined ordering for
  these types). Fixed at both the top-level `SELECT ... ORDER BY` path
  (`SortExec`) and inside windowed `ORDER BY` (e.g.
  `ROW_NUMBER() OVER (ORDER BY ...)`).
- **`SUM`/`AVG` (incl. `SUM_VEC`/`AVG_VEC`) on dimension-mismatched
  vectors or matrices** now error instead of silently dropping the
  mismatched row from the aggregate — previously this could silently
  under-count a `SUM`/corrupt an `AVG`'s denominator with no warning.
- **`DELIVER <dataset>`** no longer returns a hardcoded success message
  regardless of input. It now validates the dataset exists, and reports
  whether it has actually been persisted (i.e. has a delivery manifest
  the `/delivery` HTTP routes can serve) — pointing the user at
  `SAVE DATASET` if not.
- **A `SELECT` with an aggregate function and no `GROUP BY`** (e.g.
  `SELECT SUM(price) FROM t`, `SELECT COUNT(*) FROM t`) previously
  silently returned the raw, unaggregated table instead of computing the
  aggregate — the plan-building code path for ungrouped `SELECT`
  statements never checked for aggregate expressions in the select list.
  This was found while testing the fixes above, not in the original
  audit; it's likely the highest-severity fix in this release since it
  affects a very common query shape.

### Internal

- `ParquetStorage::manifest_exists` — new public helper backing the
  `DELIVER` fix, mirroring the existing `metadata_exists`.
- 13 new regression tests in `tests/silent_correctness_test.rs`.

---

## [0.1.34] - 2026-07-10

### Added — Pipeline persistence (`SAVE`/`LOAD PIPELINE`)

**DSL surface**
- `SAVE PIPELINE <name> [TO '<path>']` — serializes a named pipeline to JSON; defaults to `<data_dir>/<db>/pipelines/<name>.json`
- `LOAD PIPELINE <name> [FROM '<path>']` — restores a pipeline from its JSON file by re-parsing the stored DSL source

**JSON format** — human-readable and editable:
```json
{ "name": "...", "source": "DEFINE PIPELINE ... AS ...", "version": "0.1.34" }
```

**Internals**
- `StoredPipeline { steps, source }` — replaces bare `Vec<PipelineStep>` in `TensorDb.pipelines`; keeps the original DSL text alongside the parsed steps
- `PersistKind::Pipeline` variant added to the `PersistKind` enum
- `save_all_pipelines` / `load_all_pipelines` — public helpers for bulk pipeline persistence (building block for future `SAVE DATABASE` / `LOAD DATABASE`)
- Source line attached in `execute_line_with_context` at parse/execute boundary so that `DefinePipelineStmt.source` is always populated before storage

---

## [0.1.33] - 2026-07-10

### Added — Named, reusable, composable pipelines

**DSL surface**
- `DEFINE PIPELINE <name> AS step [THEN step ...]` — defines a named pipeline of sequential data steps
- `APPLY PIPELINE <name> ON <source> [INTO <target>]` — executes a pipeline on a dataset; writes result to `target`, or updates `source` in-place when `INTO` is omitted
- `SHOW PIPELINES` — lists all defined pipelines with step counts
- `DESCRIBE PIPELINE <name>` — prints a human-readable summary of each pipeline step
- `DROP PIPELINE <name>` — removes a pipeline from the session

**Supported pipeline steps** (combinable with `THEN`):
- `SELECT col [AS alias], ...` — column projection
- `WHERE expr` / `FILTER expr` — row filtering
- `ORDER BY col [ASC|DESC] [, ...]` — sorting
- `LIMIT n` — row cap
- `NORMALIZE col` — L2-normalize a vector column in-place (implemented as direct mutation, not routed through SELECT)

**Internals**
- 4 new lexer tokens: `PIPELINE`, `PIPELINES`, `APPLY`, `DESCRIBE`
- `PipelineStep` enum and `DefinePipelineStmt` / `ApplyPipelineStmt` AST structs in `dsl/ast.rs`
- `TensorDb.pipelines: HashMap<String, Vec<PipelineStep>>` — session-scoped pipeline registry (cleared on `RESET`)
- New `src/dsl/executor/pipeline.rs` module with all pipeline execution logic
- `ShowTarget::Pipelines` variant routed through `execute_show`

**Tests** — `tests/pipeline_test.rs` (10 integration tests)

---

## [0.1.32] - 2026-07-10

### Added — Vector engine: index-aware similarity search, matrix SQL expressions, and TRANSFORM

**Feature 1 — Index-aware COSINE_SIM (`COSINE_SIM(col, vec) > threshold`)**
- The query planner now detects `COSINE_SIM(col, vec) > threshold` / `>= threshold` predicates in WHERE clauses
- When a vector index exists on the filtered column, the planner replaces the full table scan with a new `CosineFilterExec` physical node that calls `index.search(query, k=total_rows)` and filters candidates by score, avoiding reading irrelevant rows
- Falls back to brute-force expression evaluation when no vector index is present — behavior is identical, only performance differs
- Supports both `>` (strict) and `>=` (inclusive) threshold comparisons

**Feature 2 — Matrix SQL expressions**
- `[[r1_v1, r1_v2], [r2_v1, r2_v2]]` matrix literals now valid anywhere an expression is expected
- `MATMUL(a, b)` — matrix × matrix or matrix × vector multiplication in SELECT/WHERE
- `TRANSPOSE(col)` — transposes a matrix column; rows and columns are swapped
- `MAT_SHAPE(col)` — returns the shape as a `"RxC"` string (e.g. `"2x3"`)
- All four additions are wired through parser → AST (`Expr::MatLiteral`, `VectorFnKind::{Matmul,Transpose,MatShape}`) → logical plan → physical evaluation
- `MATMUL` and `TRANSPOSE` tokens disambiguated: followed by `(` → SQL-function call; otherwise → existing tensor-algebra prefix syntax

**Feature 3 — TRANSFORM statement**
- New top-level DSL statement: `TRANSFORM <source> SELECT <cols> [WHERE <predicate>] [INTO <target>]`
- Applies a SELECT with optional filter over an existing dataset
- `INTO <target>` writes the result to a new (or existing) dataset; omitting `INTO` replaces the source in-place
- Implemented by wrapping into a `SelectStmt` and delegating to `execute_select`, then reinserting the result

**Tests**
- `tests/v0132_features_test.rs`: 11 integration tests covering all three features (matrix literals, MATMUL, TRANSPOSE, MAT_SHAPE, TRANSFORM with/without INTO, TRANSFORM SELECT *, COSINE_SIM with and without a vector index)

---

## [0.1.31] - 2026-07-10

### Added — Tensor-SQL bridge: vectors as first-class SQL citizens

**Inline vector literals in SQL expressions**
- `[v1, v2, ...]` syntax now valid anywhere an expression is expected: `SELECT [1.0, 0.0] AS query FROM docs`
- `Expr::VecLiteral(Vec<f64>)` added to both the DSL AST and the logical expression layer
- Parser: `Token::LBracket` in `parse_expr_atom` routes to the new `parse_vec_literal()` helper

**Vector scalar functions usable in SELECT, WHERE, ORDER BY**
- `NORMALIZE(col)` — normalizes a column vector to unit length; disambiguated from tensor-algebra `NORMALIZE` by presence of `(`
- `L2_NORM(col)` — returns the Euclidean norm as a `Float`
- `COSINE_SIM(a, b)` — cosine similarity between two vector expressions; works with inline literals
- `DOT(a, b)` — dot product returning `Float`
- `VEC_ADD(a, b)` — element-wise vector addition returning a `Vector`
- `VEC_SCALE(col, factor)` — scalar multiplication returning a `Vector`
- All six functions fully integrated through parser → AST → logical plan → physical evaluation

**Vector aggregate functions for GROUP BY centroid queries**
- `AVG_VEC(col)` — element-wise average across group rows; produces per-group centroid vectors
- `SUM_VEC(col)` — element-wise sum across group rows
- Both functions update `AggregateExec` init, accumulation, and finalization paths
- `AS alias` now consumed (and respected) for aggregate functions in SELECT lists

**Schema compatibility**
- `Vector(0)` in schema fields now accepted as a wildcard matching any vector dimension, enabling aggregate output rows whose dimension is only known at runtime

**Tests**
- `tests/vector_expressions_test.rs`: 11 integration tests covering inline literals, all 6 scalar functions, WHERE filtering with vector functions, and GROUP BY with AVG_VEC / SUM_VEC

---

## [0.1.30] - 2026-07-09

### Fixed — Correctness bug-fix pass with 12 integration tests

**Bug 1 — Computed columns: type inference now uses actual runtime value type**
- `infer_expr_result_type` added `Expr::Ref` → `Float` and `Expr::Infix` with recursive left/right inference
- `apply_window_and_computed_exprs`: field type now derived from `val.value_type()` instead of static inference, preventing `Tuple::new` rejections on Int/Bool/String results
- Projection `extended_schema` now derived from the first row's schema (actual types) instead of static inference

**Bug 2 — NULLABLE columns ignored in `DATASET ... COLUMNS (...)`**
- `Field::new` in `CreateDataset` executor now calls `.nullable()` when `ColumnDef.nullable` is true
- `INSERT INTO ... VALUES (NULL)` now correctly accepted for columns declared `INT NULLABLE`, `STRING NULLABLE`, etc.

**Bug 3 — `TENSOR(n)` parenthesis syntax rejected**
- `parse_col_type` (both `Token::Tensor` and `"TENSOR"` string paths) now accepts `TENSOR(n)` and `TENSOR(r, c)` with parentheses in addition to existing `TENSOR[n]` bracket form

**Bug 4 — COALESCE result type**
- `apply_window_and_computed_exprs` correctly handles `COALESCE` whose args are reference columns: uses actual computed `Value::Int`/`Value::Float`/etc. rather than a static `Float` fallback

### Tests — 12 new integration tests in `tests/correctness_integration.rs`
- `test_multiple_computed_columns` — multiple `expr AS alias` in one SELECT
- `test_window_no_partition_by` — `ROW_NUMBER() OVER (ORDER BY ...)` with no partition
- `test_union_deduplicates` — `UNION` produces distinct rows
- `test_union_all_keeps_duplicates` — `UNION ALL` retains duplicates
- `test_cast_to_bool` — `CAST(0 AS BOOL)` → false, `CAST(1 AS BOOL)` → true
- `test_substr_two_arg` — `SUBSTR(str, 2)` uses 1-based indexing
- `test_coalesce_three_args` — `COALESCE(NULL, NULL, 42)` → 42 with nullable INT columns
- `test_right_join_correctness` — right-only rows have NULL left-side columns
- `test_full_outer_join_correctness` — unmatched rows on both sides carry NULLs
- `test_cte_cleanup_after_query` — CTE temp dataset removed after query completes
- `test_tensor_column_preserves_dimensions` — `TENSOR(128)` → `Vector(128)`
- `test_tensor_2d_column_maps_to_matrix` — `TENSOR(4, 8)` → `Matrix(4, 8)`

---

## [0.1.29] - 2026-07-09

### Added — Window Functions, CTEs, UNION, CASE WHEN, DISTINCT, COALESCE/NULLIF, String Functions, CAST

**CTEs (`WITH name AS (SELECT ...) SELECT * FROM name`):**
- `SelectStmt.ctes: Vec<(String, SelectStmt)>` stores CTE definitions before the main query
- New `Token::With` dispatches to `parse_cte_select` in `parser/dataset.rs`
- Executed in `execute_select`: each CTE is materialized as a temp dataset before the main query runs

**Window functions (`ROW_NUMBER/RANK/DENSE_RANK/LAG/LEAD/SUM/AVG/COUNT/MIN/MAX OVER (PARTITION BY … ORDER BY …)`):**
- `WindowFunc` and `WindowSpec` AST types added
- `SelectExpr::Window { func, spec, alias }` for window columns
- Post-processed after base plan execution in `execute_select` → `apply_window_func`
- Supports partitioned and unpartitioned windows with optional ORDER BY

**UNION / UNION ALL:**
- `SetOpKind { Union, UnionAll }` enum; `SelectStmt.union: Option<(SetOpKind, Box<SelectStmt>)>`
- Parsed via `Token::Union` after the main SELECT body; `UNION ALL` checks for `Token::All`
- Executed by running both queries, concatenating rows, deduplicating for plain UNION

**CASE WHEN (`CASE WHEN cond THEN val … ELSE default END`):**
- `Expr::Case { operand, branches, else_expr }` in AST and logical layer
- Parsed by new `parse_case_expr` in `parser/expr.rs` triggered by `Token::Case`
- Evaluated in `physical::evaluate_expression`; also works inside WHERE/HAVING

**SELECT DISTINCT:**
- `SelectStmt.distinct: bool`; parsed as `Token::Distinct` immediately after `SELECT`
- Maps to `LogicalPlan::Distinct → DistinctExec` (row-level deduplication via HashSet)

**COALESCE / NULLIF:**
- `Expr::Coalesce(Vec<Expr>)` and `Expr::Nullif(Box<Expr>, Box<Expr>)` in AST and logical layer
- Parsed as identifier calls in `parse_expr_atom` (IFNULL treated as NULLIF alias)
- `COALESCE` returns first non-null arg; `NULLIF` returns null when both args equal

**String functions (UPPER, LOWER, LENGTH, TRIM, CONCAT, SUBSTR):**
- `ScalarFnKind` enum; `Expr::ScalarFn { func, args }` in AST and logical layer
- Parsed in `parse_expr_atom` from identifier names with `(` lookahead
- Evaluated in `physical::evaluate_expression`

**CAST (`CAST(expr AS INT|FLOAT|TEXT|BOOL)`):**
- `CastTarget` enum; `Expr::Cast { expr, to }` in AST and logical layer
- Parsed in `parse_expr_atom` from `CAST` identifier with `(` lookahead; type parsed with `AS` keyword
- Evaluated with type coercion in `physical::evaluate_expression`

**`SelectExpr::Computed { expr, alias }`:**
- New variant for arbitrary expressions in SELECT column list (CASE, ScalarFn, CAST, etc.)
- Post-processed after base plan execution via `apply_window_and_computed_exprs`
- Column alias via `AS name` syntax

### Changed
- `parse_select_expr` extended to detect and route window/computed expressions
- `dsl_expr_to_logical_expr` maps all new AST expression variants to logical layer
- All `SelectExpr` match arms updated project-wide for exhaustiveness
- `parse_agg_call` removed (functionality inlined into `parse_select_expr`)
- 23 new parser tests covering all 8 feature areas

---

## [0.1.28] - 2026-07-08

### Added — Subqueries, LIMIT/OFFSET, Multi-col ORDER BY, IN/BETWEEN, RIGHT/FULL JOIN, compound HAVING

**Subqueries (`SELECT * FROM (SELECT ...) AS alias`):**
- `DatasetSource` enum in AST replaces `SelectStmt.dataset: String` — variants `Named(String)` and `Subquery { query, alias }`
- Parsed in `parser/dataset.rs`: `FROM (SELECT ...) AS name` routes into `DatasetSource::Subquery`
- Executed by running the inner query recursively via `execute_select`, registering the result as a temp dataset under the alias, then scanning it in the outer query

**LIMIT + OFFSET (`LIMIT n OFFSET m`):**
- `Token::Offset` added to lexer
- `SelectStmt.offset: Option<usize>` and `DatasetFromClause.offset: Option<usize>` added to AST
- Parsed as `LIMIT n OFFSET m` or standalone `OFFSET m` after LIMIT
- `LogicalPlan::Limit` gains `offset: usize`; `LimitExec.execute` uses `.skip(offset).take(n)`

**Multi-column ORDER BY (`ORDER BY a ASC, b DESC`):**
- `OrderByClause.columns: Vec<(String, bool)>` replaces single `column/ascending` fields
- Parsed with a comma loop supporting any number of sort keys, each with optional `ASC`/`DESC`
- `LogicalPlan::Sort.columns: Vec<(String, bool)>`; `SortExec` pre-resolves column indices once before the sort closure for efficiency

**IN / BETWEEN predicates:**
- `Token::In`, `Token::Between` added to lexer
- `Expr::In { expr, list }` and `Expr::Between { expr, low, high }` added to DSL AST, logical plan, and physical evaluators
- Parsed as postfix operators in the Pratt loop; BETWEEN uses `parse_pratt(4)` to stop before AND, then eats AND explicitly
- Evaluated in `evaluate_expression` (physical) and `eval_value` (planner)

**RIGHT JOIN and FULL OUTER JOIN:**
- `Token::Right`, `Token::Full`, `Token::Outer` added to lexer
- `JoinKind::Right` and `JoinKind::Full` added to AST and `JoinType` in logical plan
- `parse_join_clause` handles `RIGHT [OUTER] JOIN` and `FULL [OUTER] JOIN`
- `NestedLoopJoinExec` completely rewritten: RIGHT uses left-keyed hash map with right-row probe; FULL tracks matched right indices via `HashSet<usize>` and emits unmatched right rows with NULL left values

**Compound HAVING / FILTER via Pratt parser:**
- `DatasetFromClause.filter` and `having` changed from `Option<DatasetFilter>` to `Option<Expr>`
- `parse_dataset_filter` and `parse_cmp_op` removed; FILTER/HAVING now use `parse_expr()` directly
- `DatasetFilter` struct and `CmpOp` enum removed from AST (dead code)
- `Expr::Bool(bool)` added to DSL AST to fix regression where `FILTER active = true` parsed `true` as a column reference — handled in `parse_expr_atom` before the general ident path

**Tests:**
- 11 new parser tests: `select_limit_offset`, `select_multi_col_order_by`, `select_where_in`, `select_where_between`, `select_right_join`, `select_right_outer_join`, `select_full_outer_join`, `select_full_join`, `select_subquery`, `select_order_by_single_no_direction`, `select_between_compound`
- Updated existing tests for `DatasetSource::Named`, `Expr::Infix` (was `DatasetFilter`), multi-col `ord.columns[0]` field access

---

## [0.1.27] - 2026-07-08

### Added — JOIN, UPDATE, DELETE, IS NULL / IS NOT NULL

**IS NULL / IS NOT NULL (`src/dsl/lexer.rs`, `ast.rs`, `parser/expr.rs`, `query/logical.rs`, `query/planner.rs`):**
- `Token::Is` added to lexer
- `Expr::IsNull(Box<Expr>)` and `Expr::IsNotNull(Box<Expr>)` added to DSL AST
- Parsed as postfix operators in the Pratt loop: `col IS NULL`, `col IS NOT NULL`
- Mapped through `dsl_expr_to_logical_expr` → `LogicalExpr::IsNull/IsNotNull`
- Evaluated in physical filter via `evaluate_expr` and `evaluate_predicate`

**UPDATE (`UPDATE <ds> SET col = expr [, ...] [WHERE ...]`):**
- `Token::Update` in lexer; `UpdateStmt` in AST
- `parse_update` in `parser/dataset.rs`
- `execute_update` in `executor/query.rs` — row-level expression evaluation, in-place mutation, returns updated count

**DELETE (`DELETE FROM <ds> [WHERE ...]`):**
- `Token::Delete` in lexer; `DeleteStmt` in AST
- `parse_delete` in `parser/dataset.rs`
- `execute_delete` in `executor/query.rs` — `rows.retain`, returns deleted count; `DELETE FROM ds` without WHERE clears all rows

**JOIN (`SELECT ... FROM ds1 [INNER|LEFT] JOIN ds2 ON col = col`):**
- `Token::Join`, `Token::On`, `Token::Inner`, `Token::Left` in lexer
- `JoinClause`, `JoinKind` structs added to AST; `SelectStmt.joins: Vec<JoinClause>` field added
- `parse_join_clause` and `parse_join_col_ref` in `parser/dataset.rs` — supports qualified (`t.col`) and unqualified column refs
- `LogicalPlan::Join` and `JoinType` (Inner/Left) in `query/logical.rs` — schema merging prefixes colliding right-table columns with `r_`
- `NestedLoopJoinExec` in `query/physical.rs` — hash-map on right side, nested loop probe; LEFT JOIN emits NULLs for unmatched left rows
- Planner creates `NestedLoopJoinExec` from `LogicalPlan::Join`
- `evaluate_predicate` public entry point added to `query/planner.rs` for reuse in UPDATE/DELETE

**Bug fix:** `SEARCH ... ON col` and `CREATE INDEX ON` previously matched `ON` as `Token::Ident("ON")` — updated to `Token::On`.

10 new parser unit tests.

---

## [0.1.26] - 2026-07-08

### Added — compound WHERE clause support (AND / OR / NOT)

**Lexer (`src/dsl/lexer.rs`):**
- `Token::And` (`AND`) and `Token::Or` (`OR`) added

**AST (`src/dsl/ast.rs`):**
- `Expr::And(Box<Expr>, Box<Expr>)` — logical AND
- `Expr::Or(Box<Expr>, Box<Expr>)` — logical OR
- `Expr::Not(Box<Expr>)` — logical NOT

**Parser (`src/dsl/parser/expr.rs`):**
- Pratt operator table updated with correct precedence: OR (1) < AND (3) < comparison (5) < +/- (7) < \*// (9)
- `NOT` handled as unary prefix in `parse_expr_atom`

**Query layer:**
- `dsl_expr_to_logical_expr` in `executor/query.rs` maps `And`/`Or`/`Not` to logical plan nodes
- `Not` variant added to `query::logical::Expr`
- `evaluate_expr` in `query/planner.rs` handles `Not`

**Effect:** `SELECT * FROM ds WHERE age > 30 AND city = "NYC"` now works correctly. Previously, `AND ...` was silently ignored (parsed as trailing tokens). Compound predicates with `OR` and `NOT` also work.

5 new parser unit tests added.

---

## [0.1.25] - 2026-07-08

### Changed — executor and parser split into sub-module directories

**`src/dsl/executor/` (was `executor.rs`, 2014 lines):**
- `mod.rs` — `execute_statement` dispatch, `to_engine_kind`, `col_type_to_value_type`; Search and InsertInto arms remain inline
- `eval.rs` — `eval_let`, `eval_expr_to_name`, `eval_call`, `apply_index`, `fresh_temp`, `expr_to_string`, `infix_to_binary_op`
- `show.rs` — `execute_show` (all ShowTarget variants), `format_lineage_tree`
- `explain.rs` — `execute_explain`; reuses shared helpers from `query.rs` via `use super::query::...`
- `query.rs` — `execute_select`, `execute_create_dataset_from`, `execute_add_computed_column`, plus shared logical-plan helpers (`agg_func_to_logical`, `dataset_filter_to_logical`, `dsl_expr_to_logical_expr`)

**`src/dsl/parser/` (was `parser.rs`, 2581 lines):**
- `mod.rs` — `Parser` struct, all cursor/consuming primitives, `parse_statement` dispatch, small statement parsers, full test suite (58 tests)
- `dataset.rs` — `parse_create_dataset`, `parse_dataset_from_clause`, `parse_select`, `parse_alter`, `parse_insert_into`, `parse_search`, and related helpers
- `expr.rs` — `parse_expr`, `parse_pratt` (Pratt parser), `parse_expr_atom`, `parse_call_expr`, `parse_simple_expr`
- `introspection.rs` — `parse_show`, `parse_explain`, `parse_audit`, `parse_deliver`
- `persistence.rs` — `parse_save`, `parse_load`, `parse_list`, `parse_import`, `parse_export`, `parse_use`

No behavioral changes; all 108+ tests pass.

---

## [0.1.24] - 2026-07-07

### Changed — DELIVER ported to typed pipeline; delivery_dsl.rs deleted

**`execute_line_shared` fully typed:**
- Added `Statement::Deliver` arm — routes through the parsed AST instead of the string fallback
- Removed the `if line.starts_with("DELIVER ")` string stub; `execute_line_shared` now contains zero string-based dispatch
- Collapsed the `if let Ok(stmt) + match + string fallback` pattern into a single `match crate::dsl::parser::parse(line)` expression

**`is_read_only` ported to typed parser:**
- `mod.rs::is_read_only(line)` now calls `parser::parse(line).map(|s| s.is_read_only()).unwrap_or(false)`
- `Statement::is_read_only()` in `ast.rs` extended to include `Statement::Deliver`
- The previous four `starts_with` string checks are gone

**Dead module deleted:**
- `src/dsl/delivery_dsl.rs` deleted — `DeliveryProjection` struct was never referenced outside the file
- `pub mod delivery_dsl` removed from `mod.rs`

---

## [0.1.23] - 2026-07-07

### Changed — `handlers/` directory eliminated; zero fallback campaign complete

**`src/dsl/handlers/` fully deleted:**
- Removed `handlers/dataset.rs`, `handlers/search.rs`, `handlers/metadata.rs`, `handlers/persistence.rs`, `handlers/mod.rs`
- All live logic relocated; all dead string-based wrappers discarded

**`.add_column()` method-call syntax ported to typed parser:**
- Added single-quoted string support to `Token::Str` (second `#[regex]` on the Logos lexer)
- Added `peek_at(offset)` lookahead helper to `Parser`
- Added `parse_method_call` in `parse_statement`: `dataset.add_column("col", tensor)` → `Statement::Attach`
- Removed `.add_column(` fallback branch from `execute_line_with_context` — typed parser now handles 100% of inputs

**New `src/dsl/persistence.rs`:**
- Moved core persistence functions (`save_dataset_core`, `load_dataset_core`, `export_csv_core`, `import_csv_typed`, et al.) from `handlers/persistence.rs` into a top-level `dsl/persistence` module
- Dead string-based wrappers (`handle_save`, `handle_load`, `handle_import`, `handle_export`, etc.) not ported
- `get_connector_registry` made `pub` to support the scientific connectors test suite

**`executor.rs` updated:**
- `Statement::SetMetadata` arm inlined (was delegating to `handlers::metadata::set_metadata_typed`)
- All persistence calls changed from `handlers::persistence::*` to `crate::dsl::persistence::*`
- `use crate::dsl::handlers` removed; `use crate::dsl::persistence` and `use crate::core::storage::ParquetStorage` added

**`mod.rs` updated:**
- `pub mod handlers` replaced with `pub mod persistence`
- `execute_line_shared` `LIST` arm updated to call `persistence::list_typed` directly

---

## [0.1.22] - 2026-07-07

### Changed — final legacy handler cleanup; explain.rs deleted

**Dead fallbacks removed from `execute_line_with_context`:**
- Removed `handle_save`, `handle_load`, `handle_set_metadata`, `handle_import` (DATASET form), `handle_export` string-based fallback branches — all were unreachable because the typed parser already handled every form they covered
- Only remaining legacy fallback: `.add_column(` method-call syntax

**EXPLAIN fully ported — `explain.rs` deleted:**
- Added `ExplainTarget::DatasetQuery { name, from: DatasetFromClause }` to `ast.rs`; handles `EXPLAIN [PLAN] DATASET name FROM source [FILTER …] [SELECT …] [GROUP BY …] [ORDER BY …] [LIMIT …]`
- Extracted `parse_dataset_from_clause(source)` helper from `parse_create_dataset`; reused in both `parse_create_dataset` and `parse_explain`
- `execute_explain` signature changed from `&mut TensorDb` to `&TensorDb` (function is read-only); now `pub` so `execute_line_shared` can call it directly
- `ExplainTarget::DatasetQuery` arm in `execute_explain` builds the full logical plan (Scan → Filter → Aggregate/Project → Sort → Limit) without any string reconstruction
- `handlers/explain.rs` deleted; `handlers/mod.rs` entry removed

**`execute_line_shared` (server read-only path) ported to typed parser:**
- Now tries `parser::parse(line)` first; dispatches `Statement::Explain` → `execute_explain`, `Statement::Audit` → inline logic, `Statement::List` → `list_typed`
- Removed calls to `handle_explain` and `handle_list_datasets` from this path
- DELIVER remains as a stub in the minimal fallback

**`IMPORT CSV FROM path [AS name]` ported:**
- Added `Statement::ImportCsv(ImportCsvStmt)` to AST
- `parse_import` detects `IMPORT CSV` vs `IMPORT DATASET` and emits the appropriate statement
- `import_csv_typed` added to `persistence.rs`; executor dispatches `Statement::ImportCsv` to it

**`EXPORT [CSV] name TO path` ported:**
- `parse_export` now accepts an optional `CSV` keyword (all exports produce CSV; the keyword is redundant but accepted for backward compat)

**`SET DATASET name [METADATA] key = value` fixed:**
- `parse_set` now consumes the optional `Token::Metadata` keyword before reading the key; previously `METADATA` as a keyword token caused `eat_ident()` to fail

---

## [0.1.21] - 2026-07-06

### Changed — dataset.rs and search.rs ported; parser extended

**Legacy handlers dataset.rs and search.rs fully ported:**
- Removed all string-based handler functions (`handle_dataset`, `handle_dataset_creation`, `handle_dataset_query`, `handle_select`, `handle_insert`, `handle_deliver`, `handle_materialize`, `handle_add_column`, `handle_search`) — all replaced by typed parser + executor paths
- `handlers/dataset.rs` retains only `handle_add_tensor_column` (for `.add_column()` method fallback) and `build_select_query_plan` / `build_dataset_query_plan` (still used by `explain.rs`)
- `handlers/search.rs` retains only `build_search_query_plan` / `build_search_plan_internal` (used by `explain.rs`)
- Fallback chain in `mod.rs` trimmed: removed SELECT, DATASET, INSERT INTO, SEARCH, MATERIALIZE, ALTER branches

**Parser extensions:**
- **`INSERT INTO VALUES`**: `parse_insert_value` now handles vector literals `[v1, v2]` and matrix literals `[[r0c0, r0c1], ...]` → new `InsertValue::Vector` / `InsertValue::Matrix` variants; also handles boolean literals `true` / `false` → `InsertValue::Bool`
- **SEARCH**: `parse_search` rewrote to handle all three syntaxes: `SEARCH ds ON col QUERY [...] LIMIT k [INTO target]`, `SEARCH target FROM source QUERY [...] ON col K=k`, `SEARCH source WHERE col ~= [...] LIMIT k`
- **Comparison operators in expressions**: `InfixOp` extended with `Eq`, `NotEq`, `Gt`, `Lt`, `GtEq`, `LtEq`; Pratt parser precedence table updated; `dsl_expr_to_logical_expr` and `expr_to_string` handle all six
- **Integer literals**: `Expr::Int(i64)` variant added to preserve Int vs Float distinction through the expression pipeline; integer literals in computed expressions now produce `Value::Int` results
- **Aggregate functions in HAVING**: `COUNT(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)` in WHERE/HAVING clauses are parsed as `Expr::Ref("COUNT(col)")`, matching the post-aggregation column names the physical plan emits
- **`DATASET name ADD COLUMN col = expr [LAZY]`**: new computed-column form; `AlterOp::AddComputedColumn { name, expr, lazy }` added to AST; executor evaluates expression per row and calls `alter_dataset_add_computed_column`
- **`DATASET name COLUMNS`**: parentheses around column list now optional (both `COLUMNS (a: INT)` and `COLUMNS a: INT` accepted)
- **`parse_agg_call`**: now parses a full `Expr` inside aggregate calls, enabling `SUM(price * qty)` and other computed aggregate expressions
- **Column definition**: `nullable` defaults to `false` when no explicit `?` / `NULLABLE` marker — columns without a marker now use type-appropriate zero defaults instead of `Null`

**New AST nodes:**
- `InsertValue::Bool(bool)`, `InsertValue::Vector(Vec<f64>)`, `InsertValue::Matrix(Vec<Vec<f64>>)`
- `FilterValue::Bool(bool)`
- `AlterOp::AddComputedColumn { name: String, expr: Box<Expr>, lazy: bool }`
- `SelectExpr::Aggregate { expr: Box<Expr> }` — field renamed from `column: String` to `expr: Box<Expr>`
- `Expr::Int(i64)`
- `InfixOp::Eq / NotEq / Gt / Lt / GtEq / LtEq`

**New lexer tokens:**
- `Token::ApproxEq` (`~=`) — for vector similarity WHERE clauses
- `Token::Question` (`?`) — for nullable column syntax

---

## [0.1.20] - 2026-07-06

### Changed — Explain typed, five dead handlers deleted, parser hardened

**Explain — final round-trip eliminated:**
- Added `ExplainTarget` enum (`Dataset(String)`, `Search(SearchStmt)`, `Select(SelectStmt)`) to `ast.rs`
- Rewrote `parse_explain()` to branch on `DATASET`/`SEARCH`/`SELECT` keywords (or a bare ident treated as a dataset scan); optional `PLAN` keyword consumed
- Added `execute_explain(db, ExplainTarget, line_no)` private function in `executor.rs` that builds `LogicalPlan` directly from the typed AST and runs it through the planner — zero string reconstruction
- Legacy `EXPLAIN DATASET name FROM source FILTER …` syntax continues to fall through to `handle_explain` for backward compat

**Parser hardening:**
- `CREATE DATABASE [IF NOT EXISTS] name` — `if_not_exists: bool` added to `CreateDatabaseStmt`; executor skips silently when flag is set and DB already exists
- `DROP DATABASE [IF EXISTS] name` — `if_exists: bool` added to `DropDatabaseStmt`; executor skips silently when flag is set and DB is absent
- `CREATE VECTOR INDEX` — fixed broken match arm (`Token::Ident("VECTOR")` → `Token::Vector`); `IndexKindAst::Vector` added; executor dispatches to `db.create_vector_index()`
- `RESET SESSION` — parser now consumes the optional `SESSION` ident so this form goes through the typed path
- `AUDIT DATASET name` — parser consumes the optional `Token::Dataset` keyword so the full form goes through the typed path
- `USE DATASET FROM "path" AS name` and `IMPORT DATASET FROM "path" AS name` — both parsers now handle the optional `AS <ident>` clause

**Dead handler files deleted (5):**
- `src/dsl/handlers/audit.rs` — superseded by executor `Audit` arm
- `src/dsl/handlers/session.rs` — superseded by typed RESET parser
- `src/dsl/handlers/introspection.rs` — superseded by `execute_show()` in executor
- `src/dsl/handlers/index.rs` — superseded by typed CREATE VECTOR INDEX parser
- `src/dsl/handlers/instance.rs` — superseded by typed CREATE/DROP DATABASE and USE parsers

**Fallback chain pruned in `mod.rs`:** removed branches for SHOW, DELIVER, AUDIT, CREATE, USE, DROP, RESET — all now handled exclusively by the typed executor path.

**`execute_line_shared` updated:** AUDIT logic inlined (no more `handle_audit` call), SHOW error stub removed, DELIVER inlined.

---

## [0.1.19] - 2026-07-02

### Changed — All remaining AST → string → re-parse round-trips eliminated

Every `Statement` variant in `executor.rs` now dispatches directly to typed engine APIs. No string reconstruction anywhere in the typed dispatch path.

**Ported from string round-trip to direct typed dispatch:**

- `Show` — new `execute_show(db, ShowTarget, line_no)` in `executor.rs` matches on the `ShowTarget` enum directly; removes `show_to_string()`
- `Deliver` — inlined: one `DslOutput::Message` with no handler call
- `SetMetadata` — new `set_metadata_typed(db, dataset, key, value, line_no)` in `handlers/metadata.rs`; removes format string
- `Save` — new `save_typed(db, PersistKind, name, path, line_no)` in `handlers/persistence.rs`
- `Load` — new `load_typed(db, PersistKind, name, path, line_no)` in `handlers/persistence.rs`
- `List` — new `list_typed(db, &ListTarget, line_no)` in `handlers/persistence.rs`
- `Import` — new `import_typed(db, ephemeral, path, name, line_no)` in `handlers/persistence.rs`
- `Export` — new `export_typed(db, name, path, line_no)` in `handlers/persistence.rs`

**Dead code removed:** `show_to_string`, `deliver_to_string`, `save_to_string`, `load_to_string`, `list_to_string` helpers deleted from `executor.rs`.

**`handlers/persistence.rs` restructured:** Private `handle_*_dataset` / `handle_*_tensor` bodies extracted into `*_core` private functions with typed signatures. String-based `handle_*` wrappers (used by the legacy fallback chain in `mod.rs`) now parse and delegate to the cores — no logic duplication.

`Explain` is intentionally unchanged — the typed parser's `parse_explain()` only captures a single ident while `handle_explain` requires the full `EXPLAIN DATASET|SEARCH|SELECT …` string; fixing it needs an `ExplainStmt` redesign.

---

## [0.1.18] - 2026-07-02

### Added / Changed — DATASET FROM and SEARCH inline vector fully typed

**DATASET target FROM source — direct typed dispatch (no more string round-trip):**

`DATASET target FROM source [FILTER …] [SELECT …] [GROUP BY …] [HAVING …] [ORDER BY … [DESC]] [LIMIT n]` is now fully parsed by the typed parser and dispatched directly from `executor.rs` — no string reconstruction, no re-parse through `handle_dataset()`.

Changes:
- `CreateDatasetStmt.from` changed from `Option<String>` to `Option<DatasetFromClause>`
- New AST types in `ast.rs`: `DatasetFromClause`, `DatasetFilter`, `CmpOp` (`Eq`, `NotEq`, `Gt`, `GtEq`, `Lt`, `LtEq`), `FilterValue` (`Int`, `Float`, `Str`)
- New lexer tokens: `Gt`, `Lt`, `GtEq`, `LtEq`, `NotEq` — comparisons now lexed as dedicated tokens
- Parser: `parse_create_dataset()` extended to parse all optional clauses (`FILTER`/`WHERE`, `SELECT`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`) using new helpers `parse_dataset_filter()`, `parse_cmp_op()`, `parse_filter_value()`
- Executor: `CreateDataset` FROM arm now builds the `LogicalPlan` (`Scan` → `Filter` → `Aggregate`/`Project` → `Having` → `Sort` → `Limit`) directly, then executes and persists the result dataset
- `build_dataset_query_plan` in `handlers/dataset.rs` is preserved — still used by `explain.rs`

**SEARCH inline vector literal support:**

`SEARCH <dataset> ON <column> QUERY [v1, v2, …] LIMIT <k> [INTO <target>]` now accepted by the typed parser alongside the existing named-tensor form (`QUERY tensor_name`).

Changes:
- New `SearchQuery` enum in `ast.rs`: `TensorRef(String)` | `Inline(Vec<f64>)`
- `SearchStmt.query_tensor: String` renamed to `query: SearchQuery`
- Parser: `parse_search()` detects `[` after `QUERY` and calls `parse_f64_list()` for inline vectors
- Executor: `Search` arm materialises the inline vector as an anonymous `Tensor` before building `LogicalPlan::VectorSearch`
- Legacy `SEARCH target FROM source QUERY vector ON column K=k` and `SEARCH source WHERE col ~= vector LIMIT k` syntaxes unchanged — still handled via the legacy fallback chain

### Tests

Added 6 new parser unit tests: `search_inline_vector`, `search_inline_vector_into`, `dataset_from_source`, `dataset_from_with_filter`, `dataset_from_full_clauses`.

---

## [0.1.17] - 2026-07-02

### Added — Typed SEARCH and SELECT GROUP BY in the Executor

**SEARCH — new typed syntax with direct engine dispatch:**

The `SEARCH` statement is redesigned with a complete typed AST and dispatches directly to the query engine with no string reconstruction. Old string-based syntax (`SEARCH target FROM source QUERY vector ON column K=k`) still works via the legacy fallback chain.

New syntax parsed by the typed parser:
```
SEARCH <dataset> ON <column> QUERY <tensor_name> LIMIT <k> [INTO <target>]
```

- Updated `SearchStmt` fields: `dataset`, `column`, `query_tensor` (renamed to `query: SearchQuery` in v0.1.18), `top_k`, `target`
- Parser: `parse_search()` rewritten; `parse_select_expr()` and `parse_agg_call()` added
- Executor: builds `LogicalPlan::VectorSearch` directly from typed AST; result stored in `target` dataset (default: `"search_results"`)
- Removed dead helper `search_to_string`

**SELECT GROUP BY — direct aggregate plan construction:**

The GROUP BY fallback to `select_to_string` + `handle_select` is eliminated. Aggregate queries now build `LogicalPlan::Aggregate` directly from the typed AST.

- Added `SelectExpr` enum (`Column(String)` | `Aggregate { func: AggFuncAst, column: String }`) to `ast.rs`
- Added `AggFuncAst` enum: `Sum`, `Avg`, `Count`, `Min`, `Max`
- `SelectColumns::Named` now carries `Vec<SelectExpr>` instead of `Vec<String>`
- Parser: SELECT column list now calls `parse_select_expr()` which recognises `SUM(col)`, `AVG(col)`, `COUNT(*)`, `MIN(col)`, `MAX(col)` in addition to plain column names
- Executor: `agg_func_to_logical()` helper maps `AggFuncAst` → `query::logical::AggregateFunction`
- Removed dead helper `select_to_string`

### Tests

Added 5 new parser unit tests: `search_statement`, `search_with_into`, `select_aggregate_columns`, `select_count_star`, `select_mixed_plain_and_agg`.

---

## [0.1.16] - 2026-07-01

### Fixed — Executor Dispatch Bugs (All Remaining String Round-Trip Bugs)

Eliminated five latent format-string bugs and ported five more statement handlers to direct engine API calls in `src/dsl/executor.rs`:

- **`Audit`** — was passing `"AUDIT {target}"` but `handle_audit` expected `"AUDIT DATASET {target}"`; always errored at runtime. Now calls `db.verify_tensor_dataset()` directly.
- **`Reset`** — was passing `"RESET"` but `handle_session` stripped prefix and checked `== "SESSION"`; always errored. Now calls `db.reset_session()` directly.
- **`SetMetadata`** — format string was `"SET DATASET {ds} {key} = "{val}""` but handler expected `"SET DATASET {ds} METADATA {key} = "{val}""` (missing `METADATA` keyword); always errored.
- **`Export`** — format string was `"EXPORT {name} TO "{path}""` but handler checked `rest.starts_with("CSV ")`; always errored. Fixed to `"EXPORT CSV {name} TO "{path}""`.
- **`Import`** — the `name` field (`AS <alias>`) from `ImportStmt` was silently dropped from the format string; imports with an explicit alias were always ignored.

### Changed — More Direct Engine Dispatch

- **`CreateDatabase`** — calls `db.create_database()` directly (was delegating to `handlers::instance`)
- **`DropDatabase`** — calls `db.drop_database()` directly
- **`UseDatabase`** — calls `db.use_database()` directly
- **`CreateIndex`** — calls `db.create_index()` directly based on `IndexKindAst`; removes the always-`HASH` format-string assumption

### Changed — Direct Executor Dispatch for Dataset Operations

Eliminated four string-reconstruction round-trips in `src/dsl/executor.rs`. These statements now build engine objects directly from the typed AST instead of serialising back to a string and re-parsing:

- **`CreateDataset`** — builds `Schema` from `Vec<ColumnDef>` via `col_type_to_value_type`, calls `db.create_dataset()` directly; the `FROM source` variant still falls through to the legacy handler
- **`AlterDataset`** — calls `db.alter_dataset_add_column()` directly from `AlterOp::AddColumn(ColumnDef)`
- **`InsertInto`** — resolves named column values against the schema, builds `Tuple`, calls `db.insert_row()` directly; fixes a latent format-mismatch bug where `insert_to_string` generated a non-`VALUES` format that `handle_insert` could not re-parse
- **`Select`** — builds `LogicalPlan` directly from `SelectStmt` fields, bypassing `build_select_query_plan`; GROUP BY queries fall through to the legacy handler since `SelectColumns` only carries strings
- **`Materialize`** — calls `db.materialize_lazy_columns()` directly

Removed now-dead helpers: `create_dataset_to_string`, `alter_dataset_to_string`, `col_type_to_string`, `insert_to_string`.

Added helpers: `col_type_to_value_type` (ColType → ValueType), `dsl_expr_to_logical_expr` (ast::Expr → query::logical::Expr for WHERE/HAVING clauses).

### Added — Query Engine Quick Wins

- **`src/core/storage.rs`** — `compute_stats` now fills `min`, `max`, and `mean` for `Float32`, `Float64`, `Int64`, and `Int32` columns (previously always `None`); non-numeric columns remain `None`
- **`src/query/logical.rs`** — added `Expr::And` and `Expr::Or` variants so compound predicates (`col > 5 AND col < 10`) can be represented in the logical plan
- **`src/query/planner.rs`** — `evaluate_expr` now handles `>=`, `<=`, `AND`, and `OR`; previously `>=`/`<=` silently returned `false` and compound predicates were unsupported
- **`src/query/physical.rs`** — `evaluate_expression` handles `Expr::And`/`Expr::Or`, returning `Value::Bool`

### Removed — Legacy Handler Cleanup

Deleted three handler files that were fully superseded by `src/dsl/executor.rs` (landed in v0.1.15). The typed executor now owns all dispatch for these statement types; no functionality was removed.

- **Deleted `src/dsl/handlers/operations.rs`** (739 lines) — string-dispatch LET handler with sub-parsers for indexing, dot notation, infix ops, and all math operations
- **Deleted `src/dsl/handlers/semantics.rs`** (107 lines) — BIND / ATTACH / DERIVE handlers; `handle_derive` internally delegated to `operations::handle_let`
- **Deleted `src/dsl/handlers/tensor.rs`** (197 lines) — DEFINE / VECTOR / MATRIX handlers

### Changed

- **`src/dsl/handlers/mod.rs`** — removed `pub mod` and `pub use` declarations for all three deleted modules
- **`src/dsl/mod.rs`** — removed `handle_define` / `handle_let` imports; removed DEFINE, VECTOR, MATRIX, LET/LAZY LET, BIND, ATTACH, and DERIVE branches from the legacy fallback chain in `execute_line_with_context`

### Fixed

- **`tests/dataset_integrity_test.rs`** — changed three `dataset('name')` calls (single-quoted) to `dataset("name")` (double-quoted) so the Logos lexer can tokenize them and the typed parser handles them correctly

---

## [0.1.15] - 2026-07-01

### Added — Typed DSL Parser

Replaced the `if/else if starts_with()` string-dispatch chain in `execute_line_with_context` with a proper compiler-grade pipeline. The old chain remains as a fallback for any input the new parser doesn't handle.

- **`src/dsl/lexer.rs`** — Logos 0.14 DFA tokenizer
  - 80+ tokens: all keywords, operators, punctuation, float/int/string/identifier literals
  - Skips whitespace and all three comment styles (`--`, `#`, `//`)
  - Keyword tokens always take priority over the `Ident` regex (DFA property)
  - 10 unit tests

- **`src/dsl/ast.rs`** — Fully-typed AST
  - `Statement` enum with 27 variants covering every DSL command
  - `Statement::is_read_only()` for gating shared-reference execution paths
  - `Expr` sub-language: `Ref`, `Scalar`, `StringLit`, `Infix`, `Call`, `Index`, `Field`, `DatasetRef`
  - `CallExpr`: 18 named-prefix operations (binary, unary, n-ary)
  - All type nodes (`ColType`, `TensorKindAst`, `InfixOp`, `IndexSpec`) decoupled from engine internals

- **`src/dsl/parser.rs`** — Recursive-descent + Pratt parser
  - One function per statement type; dispatches on first token
  - Pratt parser for the expression sub-language with correct operator precedence
  - Postfix `.field` and `[...]` subscript handled before infix
  - `ParseError { offset, msg }` with `into_dsl_error(line)` for integration
  - 42 unit tests

- **`src/dsl/executor.rs`** — Typed dispatch layer
  - `execute_statement()` — single `match` on `Statement`, routes each variant directly to the engine API
  - `eval_expr_to_name()` — recursive `Expr` → engine call evaluator (replaces per-handler sub-parsing)
  - `eval_call()` — maps all 18 `CallExpr` variants to `eval_binary`/`eval_unary`/`eval_matmul`/etc. with lazy/eager branching
  - String reconstruction helpers (`show_to_string`, `select_to_string`, etc.) for complex statements still delegated to legacy handlers
  - `expr_to_string()` for round-tripping `Expr` nodes back to DSL text when needed
  - `fresh_temp()` — atomic counter for intermediate tensor names

- **`Cargo.toml`**: added `logos = "0.14"` dependency

### Fixed

- **`SHOW SCHEMA` for tensor-first datasets** (`src/dsl/handlers/introspection.rs`)
  - Previously only checked the legacy `DatasetStore`; now falls through to `DatasetRegistry` when no match is found
  - Tensor-first output includes a `Role` column (`Feature`, `Target`, `Weight`, `Guid`, `Generic`) from `ColumnSchema`
  - Returns `DatasetNotFound` instead of an opaque error when neither store contains the name

### Documentation

- **`docs/DSL_REFERENCE.md`** — documented previously undocumented features:
  - `DEFINE x AS STRICT TENSOR [dims] VALUES [...]` and `TensorKind::Strict` semantics
  - `LET ds = dataset("name")` tensor-first dataset constructor
  - `<ds>.add_column(<col>, <tensor>)` zero-copy column attachment syntax
  - `LAZY LET` / `LET LAZY` as interchangeable aliases
  - `SAVE TENSOR`, `LOAD TENSOR`, `LIST TENSORS`, `LIST DATASET VERSIONS`
  - Full `SHOW` command family: `SHOW ALL`, `SHOW ALL DATASETS`, `SHOW SHAPE`, `SHOW DATASET METADATA`, `SHOW DATASET VERSIONS`, `SHOW "<string>"`
  - Complete server endpoint tables for jobs, scheduler, databases, and delivery routes
  - Per-request DB isolation guarantee (`X-Linal-Database` header restores previous context after each request)
- **`docs/ARCHITECTURE.md`** — corrected stale/inaccurate claims:
  - Backend dispatch diagram: Rayon is embedded in `engine/kernels.rs` kernel functions (≥50k element threshold), not a named third `CpuBackend` tier
  - SIMD threshold corrected to `1024` elements (was undocumented)
  - SIMD op coverage: listed the 6 ops with SIMD implementations (`add`, `sub`, `multiply`, `matmul`, `dot`, `distance`) and the 11 with scalar fallback + TODOs
  - Added `TensorKind` (`Normal`, `Strict`, `Lazy`) to the Type System section
  - Corrected `dataset/dataset.rs` path to `dataset/mod.rs`
  - Expanded server module docs with full API surface and scheduler endpoints
- **`docs/ERROR_REFERENCE.md`** — marked `ConstraintViolation` and `ReferenceError` as `*(Reserved)*`; both variants exist in `error.rs` but are not currently emitted — errors surface as `InvalidOp` instead

## [0.1.14] - 2026-01-08

### Added - Scientific Dataset Ingestion

- **Connector Architecture**
  - Implemented `Connector` trait for pluggable data format support
  - `ConnectorRegistry` for automatic format detection and connector management
  - Automatic format detection based on file extension
- **Scientific Format Support**
  - `HDF5Connector` for HDF5 files (.h5, .h5ad) with recursive group traversal
  - `NumpyConnector` for Numpy files (.npy single arrays, .npz archives)
  - `ZarrConnector` for Zarr stores with group and array support
  - `CsvConnector` refactored to use new connector architecture
- **DSL Integration**
  - `USE DATASET FROM "path"` - Load external data as ephemeral tensors and dataset view
  - `IMPORT DATASET FROM "path" AS name` - Persist external data to Parquet with full metadata
  - Format auto-detection for CSV, HDF5, Numpy, and Zarr files
- **Dataset Lineage Tracking**
  - `DatasetLineage` with hierarchical `LineageNode` structure
  - Provenance tracking for all ingested datasets
  - Lineage metadata saved alongside dataset packages

### Changed

- Updated `docs/DSL_REFERENCE.md` with scientific ingestion commands
- Updated `docs/ARCHITECTURE.md` with connector architecture section
- Updated `README.md` to highlight scientific data ingestion capabilities

### Fixed

- CI/CD pipeline stability with HDF5 dependency management
- Test data generation for scientific connectors in GitHub Actions
- Memory optimization for resource-constrained CI runners

## [0.1.13] - 2026-01-07

### Added - Phase 16: Dataset Delivery & Server Exposure

- **Standardized Dataset Packaging**
  - Implemented folder-based storage for datasets: `datasets/{name}/`.
  - Automated generation of `data.parquet` (authoritative data layer).
  - Automated generation of `manifest.json` (delivery contract & entrypoint).
  - Automated generation of `schema.json` (logical + physical schema mapping).
  - Automated generation of `stats.json` (columnar statistics & sparsity).
  - Automated generation of `lineage.json` (DAG-based derivation history).
- **Read-Only Dataset Server**
  - New modular HTTP server for sub-resource delivery.
  - Integration into main Axum server at `/delivery` prefix.
  - Endpoints for metadata introspection and component retrieval.
- **Delivery DSL Foundations**
  - Support for `DELIVER` and `SELECT` commands in the delivery context.
  - Read-only projection engine for serving customized views.

### Changed

- Updated `ParquetStorage` to use directory-based discovery.
- Refactored `tests/persistence_test.rs` to validate new package structure.

## [0.1.12] - 2026-01-06

### Added - Phase 12: Advanced Tensor & Analytical Capabilities

- **N-Dimensional Tensor Support**
  - Generalized core kernels (add, multiply, scale, flatten) to support arbitrary rank (Rank > 2).
  - Optimized incremental offset traversal for high-dimensional tensor math.
  - Integration tests for Rank-3 and Rank-4 tensors.
- **Lazy Evaluation Engine**
  - Computation Graph abstraction (`Expression` and `LazyTensor`).
  - `LAZY LET` command for deferred compute definitions.
  - Transparent materialization via `SHOW` command (automatic graph evaluation).
  - Support for mutable `SHOW` in server context for on-demand evaluation.

### Added - Phase 14: Statistical Aggregations

- **Numerical Aggregation Primitives**
  - `SUM`: Optimized reduction across all dimensions.
  - `MEAN`: Arithmetic average calculation for tensors of any rank.
  - `STDEV`: Population standard deviation implementation.
- **Improved Analytical DSL**
  - New keywords: `SUM`, `MEAN`, `STDEV`, `NORMALIZE`, `SCALE`, `RESHAPE`, `FLATTEN`, `STACK`.
  - Statistical transformation keywords: `CORRELATE`, `SIMILARITY`, `DISTANCE`.
  - Full support for indexing syntax (`v[0:10]`, `m[0, *]`).

### Improved

- Centralized shape validation and error reporting.
- Enhanced `DslOutput` with metadata for lazy tensors.

## [0.1.11] - 2026-01-05

### Added - Phase 3: Server Concurrency & Async Jobs

- **High-Concurrency Analytical Reads**
  - Refactored global state from `Mutex` to `RwLock` for parallel execution.
  - Implemented `execute_line_shared` for safe concurrent analytical query dispatch.
  - Optimized resource locking to allow multiple readers without blocking write-intensive operations.
- **Asynchronous Job System**
  - Integrated `JobManager` for background execution of long-running DSL commands.
  - New REST API endpoints: `POST /jobs`, `GET /jobs`, `GET /jobs/:id`, `GET /jobs/:id/result`.
  - Support for job cancellation via `DELETE /jobs/:id`.
- **Operational Polish**
  - Implemented multi-platform Graceful Shutdown (SIGINT/SIGTERM handling) for the Axum server.
  - Enhanced CLI with server management subcommands: `linal server status`.

### Changed

- Updated all integration tests to support the new `RwLock`-based architecture.
- Improved server responsiveness during heavy analytical workloads.

## [0.1.10] - 2026-01-02

### Performance Improvements - Phases 7-11

**Phase 7: Zero-Overhead Push**

- Eliminated metadata syscall overhead (`Utc::now()` bypass for intermediates)
- Uninitialized allocation to avoid zero-filling
- Kernel specialization for same-shape operations
- **Result**: ~10% improvement on small operations

**Phase 8: Zero-Copy Views**

- Metadata-only reshape (O(1) operation)
- Metadata-only transpose (stride manipulation)
- Metadata-only slice (view over same Arc)
- **Result**: Zero allocation for view operations

**Phase 9: Parallel & SIMD Execution**

- Rayon parallelization for large tensors (threshold: 50k elements)
- SIMD kernels (add, sub, mul, matmul with tiling)
- Dataset batching (1024-row chunks)
- **Result**: 2.5x speedup on 100k-element vectors

**Phase 10: Resource Governance**

- Arena-backed tensor allocation via `ExecutionContext`
- Memory limit enforcement (default 100MB per context)
- `ResourceError` for limit violations
- **Result**: Production-ready resource controls

**Phase 11: Allocation Optimization**

- Tensor pooling for common sizes (128-8192 elements)
- Size threshold optimization (256 elements)
- Stack allocation for tiny tensors (≤16 elements via SmallVec)
- **Result**: 3-18% improvement, zero regression

### Added

- `ExecutionContext::with_memory_limit(bytes)` for configurable memory limits
- `ExecutionContext::acquire_vec()` / `release_vec()` for tensor pooling
- `TensorPool` with automatic size matching
- `ResourceError::MemoryLimitExceeded` error type
- `SmallVec` dependency for stack-based tiny tensor allocation

### Changed

- `ComputeBackend::alloc_output()` now uses three-tier strategy:
  - Stack allocation for ≤16 elements (zero heap allocation)
  - Direct allocation for 17-255 elements (avoid pool overhead)
  - Pool reuse for ≥256 elements (reduce allocation cost)
- Backend dispatch optimized with SIMD thresholds
- Dataset operations support batched execution

### Documentation

- Added `docs/DATASET_ARCHITECTURE.md` explaining dataset_legacy vs dataset
- Updated `docs/PERFORMANCE_ROADMAP_V2.md` with Phase 7-11 completion status

## [0.1.9] - 2025-12-29

### Added

- **Phase 6: Usability Hardening (Managed Service)**
  - **Managed Instances**: Integrated database lifecycle API (`/databases`) and CLI (`linal db`) for persistent instance management.
  - **Server Multitenancy**: Support for `X-Linal-Database` header to isolate execution contexts within a single server.
  - **Background Scheduler**: In-memory task scheduler for periodic DSL execution and automated analytical pipelines.
  - **Remote Execution Mode**: CLI `query` command now supports `--url` to act as a client for remote LINAL servers.
  - **Context-Aware REPL**: Shell prompt now displays active database and supports `.use <db>` meta-command.

- **Phase 5: Internal Consistency & Validation**
  - **Lineage Introspection**: `SHOW LINEAGE <tensor>` provides a recursive tree view of data provenance.
  - **Deep Resource Auditing**: `AUDIT DATASET <name>` verifies integrity of zero-copy reference graphs.
  - **Diagnostic Exports**: Added `LineageNode` to public engine API for external tool integration.
  - **Enhanced Displays**: Improved tensor formatting in DSL output including source op and creation time.

## [0.1.8] - 2025-12-29

### Added

- **Phase 3: Execution Context & Lineage**
  - **Persistent Lineage**: Tensors now track their full derivation history (execution ID, operation, inputs).
  - **ExecutionContext**: Introduced a thread-safe context to propagate unique execution IDs across operations.
  - **Metadata Preservation**: Ensured lineage and extra metadata survive save/load cycles via disk storage.
  - **Transitive Provenance**: Support for tracking lineage through complex calculation chains.

- **Phase 4: DSL Semantic Expansion**
  - **Declarative Keywords**: Added `BIND`, `ATTACH`, and `DERIVE` for explicit resource management.
  - **Zero-Copy Aliasing**: `BIND` allows multiple names to point to the same internal resource without copying.
  - **Dataset Linking**: `ATTACH` provides a way to link independent tensors as virtual dataset columns.
  - **Explicit Derivation**: `DERIVE` emphasizes the creation of new artifacts from existing ones while maintaining full lineage.
  - **DSL Retrocompatibility**: Ensured all existing commands (`LET`, `DEFINE`, etc.) work seamlessly alongside new semantics.

## [0.1.7] - 2025-12-28

### Added

- **Phase 2: Dataset as Reference Graph**
  - **Formal Reference System**: Datasets now serve as semantic views using `ResourceReference`.
  - **DatasetGraph**: Recursive resolver supporting transitive links (View of a View) and cycle detection.
  - **Semantic Roles**: Introduced `ColumnRole` (Feature, Target, Weight, Guid) for rich metadata.
  - **Reference Persistence**: Support for saving lightweight Dataset views as JSON metadata.
  - **Hybrid Storage**: Maintained Parquet materialization by default for portable data sharing.
  - **Zero-Copy Chain**: Guaranteed shared memory access across arbitrary reference depths.
  - **Verification Suite**: Dedicated tests and examples for Graph resolution and Zero-copy guarantees.

## [0.1.6] - 2025-12-27

### Added

- **Tensor-First Datasets (Zero-Copy Views)**
  - Support for creating datasets directly from named tensors via `LET ds = dataset("name")`.
  - Dot notation support for dataset columns in DSL expressions (`ds.column`).
  - Metadata-only column addition (`ds.add_column("name", var)`) for O(1) complexity.
  - Reverse integration: Operation results can be added back to datasets without data copies.
  - On-demand materialization: Automatic conversion of tensor views to Parquet during `SAVE DATASET`.

- **Robustness & Integrity**
  - Strict row-count validation for all dataset columns.
  - On-demand integrity audits for dangling tensor references.
  - Health warnings in `SHOW` command for missing data dependencies.

- **Specialized Benchmarking**
  - New `benches/dataset_ops.rs` suite for tracking metadata and resolution overhead.
  - Performance report available in `docs/TENSOR_FIRST_PERFORMANCE.md`.

### Improved

- **Tensor Kernel Performance**
  - Implemented fast-path optimization for identical-shape tensor operations.
  - Recovered performance regressions, resulting in 10-15% speedups for core vector/matrix math.
- **Maintenance**
  - Updated `SECURITY.md` contact information to `dev@gorigami.xyz`.
  - Updated `README.md` copyright to 2025.

## [0.1.5] - Phase 12: Public Readiness

### Added

- **Architectural Documentation**
  - Comprehensive architecture document (`docs/ARCHITECTURE.md`)
  - System architecture overview
  - Component descriptions
  - Execution flow documentation
  - Design principles

- **End-to-End Examples**
  - Complete workflow example (`examples/end_to_end.lnl`)
  - Demonstrates full LINAL capabilities
  - ML/AI use case scenarios

- **Benchmark Suite**
  - Performance benchmark script (`examples/benchmark.lnl`)
  - In-memory vs persisted workload comparison
  - Index performance testing
  - Vector operation benchmarks

- **Contribution Guidelines**
  - `CONTRIBUTING.md` with development workflow
  - Coding standards and best practices
  - Testing guidelines
  - Pull request process

- **Security Documentation**
  - `SECURITY.md` with security policy
  - Vulnerability reporting process
  - Security considerations and best practices
  - Known limitations and recommendations

### Changed

- Updated README with links to new documentation
- Enhanced documentation structure
- Project ready for public release

## [0.1.4] - Phase 11: CLI & Server Hardening

### Added

- **Professional REPL (LINAL Shell)**
  - Integrated `rustyline` for persistent command history
  - Multi-line input support with balanced parentheses logic
  - Colored output for improved readability and error reporting
  - Basic auto-completion via rustyline

- **Administrative CLI Commands**
  - `linal init`: Automated setup for `./data` directory and `linal.toml` configuration file
  - `linal load <file> <dataset>`: Direct Parquet file ingestion via CLI
  - `linal serve`: Shorthand alias for starting the HTTP server

- **Server Robustness & API Documentation**
  - Query timeouts: Long-running queries automatically cancel after 30 seconds
  - Request validation: Size limits (16KB max) and non-empty checks for all incoming commands
  - OpenAPI / Swagger UI: Built-in interactive API documentation available at `/swagger-ui`

### Changed

- Improved REPL user experience with better error messages and visual feedback
- Server now validates all requests before processing

## [0.1.3] - Phase 10: Engine Lifecycle & Instance Management

### Added

- **Multi-Database Engine**
  - Named database instances with isolated DatasetStores
  - `CREATE DATABASE` and `DROP DATABASE` commands
  - `USE database` command for context switching
  - `SHOW DATABASES` command

- **Engine Configuration**
  - `linal.toml` configuration file support
  - Customizable storage paths and default database settings
  - Startup/shutdown hooks with graceful recovery from disk

- **Robust Metadata System (Phase 10.5)**
  - `chrono` dependency for ISO-8601 timestamps
  - Enhanced `DatasetMetadata` with versioning, `updated_at`, and `extra` fields
  - `SET DATASET METADATA` DSL command
  - Automatic timestamp tracking (created_at, updated_at)

- **CLI Parity & Multi-line Support (Phase 10.6)**
  - Refactored script runner for multi-line command support
  - `ALTER DATASET` routing in DSL
  - Fixed `GROUP BY` type inference for grouping columns
  - Comprehensive smoke test suite

## [0.1.2] - Phase 8.5 & 9: Interface Standardization & Persistence

### Added

- **Interface Standardization (Phase 8.5)**
  - Server API refactor: Accept raw DSL text via `text/plain` content type
  - JSON backward compatibility with deprecation warnings
  - TOON format as default output
  - CLI `--format` flag for REPL and Run commands (display/toon)
  - Response format selection: `?format=toon` (default) or `?format=json` query parameter

- **Persistence Layer (Phase 9)**
  - StorageEngine trait abstraction
  - Parquet-based storage for datasets
  - JSON format for tensor storage
  - `SAVE DATASET` and `SAVE TENSOR` commands
  - `LOAD DATASET` (Parquet -> Dataset conversion) and `LOAD TENSOR` commands
  - `LIST DATASETS` and `LIST TENSORS` commands
  - Full persistence test suite

- **AVG Aggregation**
  - Full implementation with proper sum/count tracking
  - Supports Int, Float, Vector, and Matrix types
  - Automatic type conversion (Int → Float for precision)
  - Works with GROUP BY and computed expressions

- **Computed Columns**
  - Materialized columns (evaluated immediately)
  - Lazy columns (evaluated on access)
  - `MATERIALIZE` command to convert lazy to materialized
  - Automatic lazy evaluation in queries

### Changed

- Server now defaults to TOON format output
- CLI output format can be controlled via `--format` flag

## [0.1.1] - Phase 8: Aggregations & GROUP BY

### Added

- **GROUP BY Execution**
  - Full GROUP BY support with multiple grouping columns
  - Aggregation functions:
    - `SUM` - Element-wise summation for vectors and matrices
    - `AVG` - Average with proper sum/count tracking
    - `COUNT` - Count rows or elements
    - `MIN` / `MAX` - Minimum and maximum values
  - Aggregations over:
    - Scalars (Int, Float)
    - Vectors (element-wise)
    - Matrices (axis-based)
  - `HAVING` clause support
  - Aggregations over computed columns

## [0.1.0] - Phase 7: Query Planning & Optimization

### Added

- **Query Planning System**
  - Logical query plan representation
  - Physical execution plan
  - Index-aware execution
  - Basic query optimizer:
    - Index selection
    - Predicate pushdown
  - `EXPLAIN` / `EXPLAIN PLAN` DSL command

## [0.0.9] - Phase 6: Indexing & Access Paths

### Added

- **Index System**
  - `Index` trait definition
  - `HashIndex` implementation for exact match lookups
  - `VectorIndex` implementation for similarity search:
    - Cosine similarity
    - Euclidean distance
  - `CREATE INDEX` DSL command
  - `CREATE VECTOR INDEX` DSL command
  - `SHOW INDEXES` command
  - Automatic index maintenance on INSERT

## [0.0.8] - Phase 5.5: Feature Catch-up

### Added

- **STACK Operation**
  - Tensor stacking operation

- **Schema Introspection**
  - `SHOW SCHEMA <dataset>` command
  - Enhanced `SHOW` command for all types

- **ADD COLUMN Enhancements**
  - Computed columns with expressions (`ADD COLUMN x = a + b`)
  - Materialized evaluation (immediate computation)
  - Lazy evaluation (`ADD COLUMN x = expr LAZY`)
  - Automatic lazy evaluation in queries
  - `MATERIALIZE` command

- **Indexing Syntax**
  - Tensor indexing: `m[0, *]`, `m[:, 1]`
  - Tuple access: `row.field`, `dataset.column`

- **Expression Improvements**
  - Better typing and error messages
  - Extended SHOW to cover scalars, vectors, matrices, tensors, tuples, and datasets

## [0.0.7] - Phase 5: TOON Integration & Server Refactor

### Added

- **TOON Format Support**
  - `toon-format` dependency
  - Serialize implementation for core types (Tensor, Dataset, DslOutput)
  - Server returns TOON format by default
  - Automated tests for TOON header and body

### Changed

- Server output format changed to TOON
- Project structure cleanup (moved docs, deleted temp files)

## [0.0.6] - Phase 4: Server & CLI

### Added

- **CLI Implementation**
  - Subcommands: `repl`, `run`, `server`
  - Structured output via `DslOutput`

- **REST API**
  - `POST /execute` endpoint
  - Dependencies: `clap`, `tokio`, `axum`, `serde`

## [0.0.5] - Restructuring (Architectural Overhaul)

### Changed

- **Modular Architecture**
  - Restructured `src/` into modular components:
    - `core/` - tensor, value, tuple, dataset, store
    - `engine/` - db, operations, error
    - `dsl/` - parser, error, handlers
    - `utils/` - parsing
  - Cleaned up `lib.rs` exports for unified API
  - Deleted legacy files

## [0.0.4] - Phase 3: DSL Dataset Operations

### Added

- **Dataset DSL Commands**
  - `DATASET` command for dataset creation
  - `INSERT INTO` command for row insertion
  - `SELECT` / `FILTER` / `ORDER BY` / `LIMIT` commands for querying

## [0.0.3] - Phase 2: Engine Integration

### Added

- DatasetStore integration into TensorDb
- `create_dataset` and `insert_row` methods
- EngineError to DatasetStoreError mapping

## [0.0.2] - Phase 1: Dataset Store

### Added

- **DatasetStore Implementation**
  - Name-based and ID-based access
  - Insert, get, remove operations
  - Duplicate name validation
  - Comprehensive unit tests (4 tests passing)

## [0.0.1] - Phase 0: Preparation

### Added

- Fixed Cargo.toml edition (2024 → 2021)
- `ADD COLUMN` for datasets (with DEFAULT values and nullable support)
- `GROUP BY` with aggregations (SUM, AVG, COUNT, MIN, MAX)
- Matrix operations (MATMUL, TRANSPOSE, RESHAPE, FLATTEN)
- Indexing syntax (m[0, *], tuple.field, dataset.column)
- `SHOW` command for all types (tensors, datasets, schemas, indexes)
- `SHOW SHAPE` introspection
- `SHOW SCHEMA` introspection

---

## Project Identity (Phase 13)

### Naming Decisions

- **Project Name**: **LINAL** (derived from *Linear Algebra*)
- **Engine**: LINAL Engine
- **CLI Binary**: `linal`
- **DSL Name**: LINAL Script
- **File Extension**: `.lnl` for LINAL scripts

### Scope

LINAL is positioned as:

- An **in-memory analytical engine**
- Focused on linear algebra (vectors, matrices, tensors) and structured datasets
- SQL-inspired querying combined with algebraic operations
- Designed for Machine Learning, AI research, Statistical analysis, and Scientific computing

---

[0.1.15]: https://github.com/gorigami/linal/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/gorigami/linal/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/gorigami/linal/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/gorigami/linal/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/gorigami/linal/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/gorigami/linal/compare/v0.1.9...v0.1.10
[0.1.8]: https://github.com/gorigami/linal/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/gorigami/linal/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/gorigami/linal/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/gorigami/linal/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/gorigami/linal/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/gorigami/linal/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/gorigami/linal/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/gorigami/linal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gorigami/linal/compare/v0.0.9...v0.1.0
[0.0.9]: https://github.com/gorigami/linal/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/gorigami/linal/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/gorigami/linal/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/gorigami/linal/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/gorigami/linal/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/gorigami/linal/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/gorigami/linal/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/gorigami/linal/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/gorigami/linal/releases/tag/v0.0.1
