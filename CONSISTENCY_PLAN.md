# LinalDB Consistency & Correctness Plan (Round 2)

**Status:** in progress · **Created:** 2026-07-16 · **Baseline:** v0.1.44 (main @ f42d260)

This is a working document, not permanent documentation. Delete it once every
item below is checked off (see "Completion" at the bottom).

## Why this exists

PR #33 (v0.1.44) did a first documentation-alignment pass against the mission
statement ("vectors, matrices, tensors as first-class citizens"), but focused
on the high-level docs (README, ARCHITECTURE structure, DATASET_ARCHITECTURE,
ERROR_REFERENCE sample errors, dead links/files). A follow-up 4-track audit
(DSL_REFERENCE.md vs. parser/executor, ARCHITECTURE.md vs. full module
structure, DATASET_ARCHITECTURE.md + ERROR_REFERENCE.md vs. source,
README/CONTRIBUTING/CHANGELOG cross-consistency) requested 2026-07-16 found
real gaps — including two live engine bugs, not just doc drift. This doc
tracks fixing them, one PR at a time. Track lettering continues from the
first cycle (Tracks A-F, completed through v0.1.44).

## Process for every PR against this plan

- [ ] Implement the change
- [ ] Add/update tests covering it
- [ ] Update relevant docs (`DSL_REFERENCE.md` / `ARCHITECTURE.md` /
      `DATASET_ARCHITECTURE.md` / `ERROR_REFERENCE.md` / `README.md` /
      `CONTRIBUTING.md` / `CHANGELOG.md`) to match the change
- [ ] `cargo test` passes clean (full suite, not just the touched crate)
- [ ] `cargo clean` (or otherwise check `target/` size) before ending the
      session — this repo's build artifacts have hit 50GB+ before and eaten
      all free disk
- [ ] Check off the corresponding item(s) below in the same PR

---

## Track G — Silent correctness bugs (highest priority)

These actively mislead users: no error, no warning, wrong/dropped results.

- [x] **G1.** `SUM_VEC(col) OVER (...)` / `AVG_VEC(col) OVER (...)` silently
      return `0.0` instead of an element-wise running aggregate on vector
      columns. Confirmed by live execution against a `Vector(2)` column.
      Root cause: the parser (`src/dsl/parser/dataset.rs:802-808`) collapses
      `AvgVec`/`SumVec` into the generic `WindowFunc::Sum`/`Avg`, and the
      executor (`src/dsl/executor/query.rs:817-840`) only extracts
      `Value::Int`/`Value::Float`, defaulting `Value::Vector` to `0.0`. Add
      explicit vector-aware accumulation, or error clearly if unsupported —
      silent zeroing is the worst option.
      **Fixed in v0.1.45** — added `window_running_sum`, a vector/matrix-aware
      running accumulator mirroring `AggregateExec`'s grouped SUM/AVG
      (`src/query/physical.rs`), errors on dimension/shape mismatch instead
      of zeroing. Also fixed the window column's output-type inference
      (same Vector/Matrix blind spot). Tests: `tests/silent_correctness_test.rs`.
- [x] **G2.** Plain aggregate functions (`SUM`, `AVG`, `COUNT`, `MIN`, `MAX`,
      `SUM_VEC`, `AVG_VEC`) with an `AS alias` in a non-windowed `SELECT`
      silently ignore the alias — the output column is always named after
      the function call (e.g. `AVGVEC(embedding)`), never the requested
      alias. Root cause: `src/dsl/parser/dataset.rs:815-819` explicitly
      discards the alias for `SelectExpr::Aggregate`. Either honor the
      alias (preferred — matches SQL expectations and every other
      expression type in `SELECT`) or make the parser reject/warn on
      `AS` for bare aggregates instead of silently swallowing it.
      **Fixed in v0.1.45** — `SelectExpr::Aggregate` and
      `query::logical::Expr::AggregateExpr` both gained an
      `alias: Option<String>` field, threaded through to the schema-naming
      logic (`LogicalPlan::schema`). Tests: `tests/silent_correctness_test.rs`.
- [x] **G3.** Column-drop when a `SELECT` mixes a non-windowed aggregate with
      a window function: `src/dsl/executor/query.rs:474-494` maps any
      `SelectExpr::Aggregate` to the placeholder name `"agg"` when building
      `ordered_cols`, but the actual produced column is named `FUNC(col)`
      (per G2) — a name mismatch that silently drops the aggregate column
      from the final projection.
      **Confirmed and fixed in v0.1.45** — `SELECT SUM(price) AS total,
      ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t` did reproduce (no
      `GROUP BY`, so `AggregateExec`'s schema fields correspond 1:1, in
      order, to the `SelectExpr::Aggregate` entries — used that to look up
      the real output name instead of the placeholder, which also
      naturally picks up the G2 alias fix). Tests:
      `tests/silent_correctness_test.rs`.
- [x] **G4.** `linal --version` reports a hardcoded, stale `0.1.9`
      (`src/main.rs:14`, `#[command(version = "0.1.9")]`) while `Cargo.toml`
      is at `0.1.44` — 35 versions out of date. Replace the literal with
      `env!("CARGO_PKG_VERSION")` so it can never drift again.
      **Fixed in v0.1.45.** Test: `tests/cli_hardening_test.rs`.

---

## Track H — DSL_REFERENCE.md doc debt

`docs/DSL_REFERENCE.md` was last substantially updated in the Track B pass
(v0.1.36) but the engine has since gained/changed features (v0.1.37 window
fix, v0.1.39 architecture debt, v0.1.40 JOIN projection/FLATTEN) not fully
reflected. Note: G1/G2 above must be fixed (or explicitly documented as
known limitations) before writing examples that rely on them.

- [ ] **H1.** Document that bare (non-windowed) aggregates ignore `AS`
      aliases (until/unless G2 is fixed, in which case just verify the
      existing doc examples are accurate instead).
- [ ] **H2.** Fix/verify the `SUM_VEC`/`AVG_VEC ... OVER (...)` window
      example (line 313) once G1 is fixed — currently documents broken
      behavior as working.
- [ ] **H3.** Correct the claim that only a single `UNION`/`UNION ALL` is
      supported (line 281) — chained 3-way+ unions actually work (verified
      live); update wording.
- [ ] **H4.** Document the actual default alias format for aggregate-as-
      window functions (`sum(expr)_over`, not `sum`) — line 315 is wrong.
- [ ] **H5.** Bump the stale `"version": "0.1.34"` in the example pipeline
      JSON (line 449) to track the current release, or note it's
      illustrative and version-independent.
- [ ] **H6.** Document the materialized-view `DATASET <name> FROM <source>
      [FILTER|WHERE ...] [SELECT ...] [GROUP BY ...] [HAVING ...]
      [ORDER BY ...] [LIMIT ...] [OFFSET ...]` form — currently only the
      `COLUMNS (...)` form is documented (doc §"DATASET", lines 80-91).
- [ ] **H7.** Document `IN (...)`, `BETWEEN ... AND ...`, `IS NULL`/
      `IS NOT NULL`, `DISTINCT`, and standalone/chained `OFFSET` in the
      WHERE/FILTER predicate vocabulary — all real, all currently
      undocumented (zero doc mentions).
- [ ] **H8.** Document `FROM (SELECT ...) AS alias` subquery support in the
      FROM clause.
- [ ] **H9.** Add `MAT_SHAPE(v)`, `MATMUL(a, b)`, `TRANSPOSE(a)` to the
      "Vector Scalar Functions" table (lines 159-166) — all work as SQL
      SELECT-context functions today, only `NORMALIZE`/`L2_NORM`/
      `COSINE_SIM`/`DOT`/`VEC_ADD`/`VEC_SCALE` are currently listed.
- [ ] **H10.** Document `EXPLAIN DATASET <name> [FROM <clause>]`,
      `EXPLAIN SEARCH ...`, and the optional `EXPLAIN PLAN` prefix — line
      553 currently implies `EXPLAIN` only covers `SELECT`.
- [ ] **H11.** Document `LIST DATASET PACKAGES` in the Persistence &
      Ingestion section (lines 372-399), alongside `LIST DATASETS`/
      `LIST TENSORS`/`LIST DATASET VERSIONS`.
- [ ] **H12.** Document that `#` and `//` are also valid line-comment
      markers, not just `--` (all examples currently only use `--`).
- [ ] **H13.** Note that the `CSV` keyword in `EXPORT [CSV] <name> TO
      <path>` is optional (line 381 only shows the `CSV` form).

---

## Track I — ARCHITECTURE.md doc debt

- [ ] **I1.** Fix the storage-layout description (line 441): it's not a
      flat `.parquet` file — `SAVE`/`IMPORT DATASET` write a per-dataset
      **directory package** (`{base}/datasets/{name}/data.parquet` plus
      sibling `schema.json`/`stats.json`/`lineage.json`/`manifest.json`,
      per `src/core/storage.rs:109-110,200`).
- [ ] **I2.** Add `pipeline.rs` to the executor-split description (lines
      259-271) — it's a sixth file (`execute_define_pipeline`/
      `execute_apply_pipeline`), not five.
- [ ] **I3.** Refresh stale counts: parser test count (58 -> 107), `Expr`
      variant count (10 documented -> 24 actual, missing `Int, Bool, And,
      Or, Not, IsNull, IsNotNull, In, Between, Case, Coalesce, Nullif,
      ScalarFn, Cast, MatLiteral`), `VectorFnKind` (6 -> 10, missing
      `Matmul, Transpose, MatShape, Flatten`), `Statement` variant count
      ("27+" -> 35), lexer token count ("80+" -> 130). Consider using
      "N+" language deliberately or a note that exact counts drift and
      point to the source as ground truth, rather than hardcoding numbers
      that go stale every few releases.
- [ ] **I4.** Add a JOIN section: `JoinClause`/`JoinKind` (INNER/LEFT/
      RIGHT/FULL), `SimilarityJoinExec`, and the v0.1.40 qualified-column
      (`table.col`) / `FROM table alias` fixes. Currently zero mentions
      despite `SimilarityJoinExec`'s own source comment pointing back at
      ARCHITECTURE.md's "Index-Aware Execution" section as its pattern
      reference.
- [ ] **I5.** Add a window functions section: `WindowFunc` (RowNumber,
      Rank, DenseRank, Lag, Lead) — currently undocumented despite having
      its own dedicated correctness-bug release (v0.1.37).
- [ ] **I6.** Add `src/dsl/persistence.rs` (758 lines — all SAVE/LOAD/LIST/
      IMPORT/EXPORT logic) to the DSL Module component listing (§3),
      currently missing.
- [ ] **I7.** Add `src/server/dataset_server.rs` (the `/delivery/*` route
      mount) to the Server Module section (§5), currently only covers
      jobs/scheduler.
- [ ] **I8.** Cite `src/engine/context.rs` (`ExecutionContext`) by file path
      in the Engine Module component listing (§2) — currently described
      conceptually in the Performance Optimizations appendix but not linked
      to its actual module.
- [ ] **I9.** Add CASE/COALESCE/NULLIF/CAST and CTE/subquery/UNION to the
      Expr-surface documentation — fully implemented (CHANGELOG v0.1.28/
      v0.1.36) but absent from ARCHITECTURE.md, consistent with the I3
      `Expr` enum gap.
- [ ] **I10 (minor).** Correct the SIMD dispatch description (lines
      692-696): `CpuBackend::use_simd` only checks length (`>=1024`); the
      contiguity check happens inside `SimdBackend`'s individual op
      methods, not at the `CpuBackend` dispatch point as currently implied.
- [ ] **I11 (minor).** Correct the "zero heap allocation" claim for the
      `<=16` element SmallVec path (line 667) — `.to_vec()` heap-allocates
      to satisfy the `Vec<f32>` return type.

---

## Track J — DATASET_ARCHITECTURE.md + ERROR_REFERENCE.md doc debt

PR #33 rewrote `DATASET_ARCHITECTURE.md` and touched `ERROR_REFERENCE.md`,
but the follow-up audit found both still have real inaccuracies.

- [ ] **J1.** `ResourceReference` doc (line 56) omits the `Column {
      dataset, column }` variant — only documents `Tensor { id }`.
- [ ] **J2.** Fix the `graph.rs` / `DatasetGraph` attribution (line 58):
      it's actually only used by `ATTACH` and `AUDIT`, not `BIND` (plain
      aliasing, no graph involved) or `DERIVE` (pure tensor-expression
      eval, unrelated module).
- [ ] **J3.** Fix `schema_evolution.rs` attribution (line 60): there is no
      `LIST VERSIONS` command — the real command is `SHOW DATASET VERSIONS
      <name>`.
- [ ] **J4.** Fix `lineage.rs` attribution (line 61): `SHOW LINEAGE` uses
      an unrelated `LineageNode` type in `src/engine/db.rs`, not
      `core::dataset::lineage`. The latter is actually consumed only by
      import connectors (csv/hdf5/numpy/zarr) and `core/storage.rs` to
      track data-import provenance.
- [ ] **J5.** Add the 2 missing `EngineError` variants to ERROR_REFERENCE.md
      (lines 13-18): `Store(StoreError)` and `DatasetError
      (DatasetStoreError)` — the latter is confirmed user-reachable (e.g.
      loading a dataset whose name already exists).
- [ ] **J6.** Fix the sample Parse error (lines 30-32): the parser's
      structured `ParseError` (with byte offset) is actually discarded at
      the only call site (`src/dsl/mod.rs:222`) and replaced with a generic
      `[line N] Parse error: Unknown command: <raw line>` — no offset, no
      specific "expected X found Y" detail ever surfaces. Either fix the
      doc to match reality, or (better, and worth considering as a Track G
      item) stop discarding the rich parser error in the first place.
- [ ] **J7.** Fix the "unrecognized command returns a ParseError directly"
      claim (line 38) to match the actual generic-`DslError::Parse`
      behavior described in J6.
- [ ] **J8.** Fix the sample Engine error (lines 44-46): actual `Display`
      format is `[line N] Engine error: <msg>`, not `"Engine error at line
      N:"` — same class of bug PR #33 was meant to fix elsewhere, missed
      here.
- [ ] **J9.** Rewrite the "Storage Errors" section (lines 50-56) entirely:
      it documents the wrong type (`StoreError`, the *tensor* store error —
      `ShapeMismatch`/`TensorNotFound`/`InvalidTensor`) instead of the
      actual persistence error type `StorageError`
      (`src/core/storage.rs:24-42`: `Io`, `Serialization`, `Parquet`,
      `Arrow`, `DatasetNotFound`, `TensorNotFound`). The doc's
      `UnsupportedFormat` variant doesn't exist anywhere in source.

---

## Track K — README / CONTRIBUTING / CHANGELOG cross-consistency

- [ ] **K1.** Reconcile the repo clone URL: README.md's Quick Start uses
      `github.com/gorigami/linaldb.git`, CONTRIBUTING.md's fork/upstream
      instructions use `github.com/gorigami/linal.git` — pick the actual
      URL and fix whichever is wrong.
- [ ] **K2.** Fix CONTRIBUTING.md's stale "Example tests: In `examples/`
      directory (run with `cargo test --examples`)" (line 218) — since
      PR #32 (v0.1.42), the Rust fixture generators moved to `tools/
      fixtures/` with explicit `[[example]]` entries in `Cargo.toml`;
      `examples/` now holds only `.lnl` scripts.
- [ ] **K3.** Add `docs/DATASET_ARCHITECTURE.md` to README's Documentation
      Hub (lines 139-142) — omitted even though PR #33 substantively
      rewrote it in the same pass that touched this section.
- [ ] **K4.** Link `SECURITY.md` from README.md or CONTRIBUTING.md's
      "Getting Help" section (currently orphaned — exists at repo root,
      linked from nowhere).
- [ ] **K5.** Refresh CONTRIBUTING.md's Project Structure tree (lines
      324-343): missing `tools/`, `benches/`, `scripts/`, `data/`,
      `.github/`, `SECURITY.md` — stale relative to the last two PRs that
      reorganized exactly this area.

---

## Completion

- [ ] All tracks (G, H, I, J, K) fully checked off
- [ ] Final PR deletes this file (`CONSISTENCY_PLAN.md`)
