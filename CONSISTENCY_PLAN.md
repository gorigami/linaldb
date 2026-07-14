# LinalDB Consistency & Correctness Plan

**Status:** in progress · **Created:** 2026-07-12 · **Baseline:** v0.1.34 (main @ 8f0d2ef)

This is a working document, not permanent documentation. Delete it once every
item below is checked off (see "Completion" at the bottom).

## Why this exists

Mission statement: *"a high-performance, in-memory analytical engine built to
bridge the gap between relational data engineering and scientific computing,
providing a SQL-inspired DSL that treats vectors, matrices, and
multi-dimensional tensors as first-class citizens."*

A 4-angle audit (DSL docs-vs-impl, tensor/vector first-class-citizen check,
storage/persistence consistency, server/test-coverage parity) across the
codebase at v0.1.34 turned up real gaps in all four areas. This doc tracks
fixing them, one PR at a time.

## Process for every PR against this plan

- [ ] Implement the change
- [ ] Add/update tests covering it
- [ ] Update relevant docs (`DSL_REFERENCE.md` / `ARCHITECTURE.md` /
      `CHANGELOG.md`) to match the change
- [ ] `cargo test` passes clean (full suite, not just the touched crate)
- [ ] `cargo clean` (or otherwise check `target/` size) before ending the
      session — this repo's build artifacts have hit 50GB+ before and eaten
      all free disk
- [ ] Check off the corresponding item(s) below in the same PR

---

## Track A — Silent correctness bugs (highest priority)

These actively mislead users: no error, no warning, wrong/dropped results.

- [x] **A1.** `ORDER BY` on a raw vector/matrix column silently becomes a
      no-op sort instead of erroring. `Value::compare()` returns `None` for
      `(Vector, _)`/`(Matrix, _)` (`src/core/value.rs:187`); `SortExec`
      swallows that via `.unwrap_or(Ordering::Equal)`
      (`src/query/physical.rs:253`). Should return a clear DSL error instead.
      **Fixed in v0.1.35** — also fixed the same bug in windowed `ORDER BY`
      (`apply_window_func`, `src/dsl/executor/query.rs`), found while
      implementing this fix. Tests: `tests/silent_correctness_test.rs`.
- [x] **A2.** `SUM`/`AVG` on vectors with mismatched dimensions silently
      drops the row instead of erroring (`src/query/physical.rs:430`,
      `465-480`). Silent data loss — should error.
      **Fixed in v0.1.35.** Tests: `tests/silent_correctness_test.rs`.
- [x] **A3.** `DELIVER <name>` parses and "succeeds" but does nothing real —
      hardcoded stub message ("Phase 1 Read-Only View"), no projection
      created (`src/dsl/executor/mod.rs:335-338`). Either implement real
      semantics or make it clearly error/warn as not-yet-implemented.
      **Fixed in v0.1.35** — now validates the dataset exists and reports
      whether it's actually been persisted (has a delivery manifest), since
      that's what `/delivery` HTTP routes actually serve; real "delivery
      packaging" already happens on `SAVE DATASET` (see `SCIENTIFIC_DATASET_
      INGESTION_PLAN.md`), so DELIVER's job is to report state honestly, not
      create anything new. Tests: `tests/silent_correctness_test.rs`.
- [x] **A4. (found during A1-A3 testing, not in original audit)** A `SELECT`
      with an aggregate function and **no `GROUP BY`** (e.g.
      `SELECT SUM(price) FROM t`) silently returned the raw, unaggregated
      table instead of computing the aggregate. Root cause:
      `src/dsl/executor/query.rs`'s ungrouped-`SELECT` plan-building branch
      never checked whether the select list contained an aggregate before
      falling through to plain-column projection, which silently discarded
      aggregate expressions. Likely the highest-severity fix in this
      release — affects one of the most common query shapes in SQL.
      **Fixed in v0.1.35.** Tests: `tests/silent_correctness_test.rs`.

## Track B — Documentation debt

`docs/DSL_REFERENCE.md` and `docs/ARCHITECTURE.md` are missing large swaths
of what's actually implemented. Low-risk, high-value, no code changes.

- [x] **B1.** Document `INSERT INTO` / `UPDATE ... SET ... WHERE` /
      `DELETE FROM ... WHERE`. **Done in v0.1.36** (§4).
- [x] **B2.** Document `JOIN`/`LEFT JOIN`/`RIGHT JOIN`/`FULL JOIN`. **Done in
      v0.1.36** (§4) — also documents that `table.col` in `ON` only uses the
      column part (no true qualified-name resolution).
- [x] **B3.** Document `WITH ... AS (...)` CTEs and `UNION`/`UNION ALL`.
      **Done in v0.1.36** (§4).
- [x] **B4.** Document window functions: `ROW_NUMBER() OVER (...)`, `RANK`,
      `DENSE_RANK`, `LAG`/`LEAD`, windowed aggregates. **Done in v0.1.36**
      (§4) — found and worked around a real bug while writing this (see
      Track E / E1 below); examples only use verified-safe combinations.
- [x] **B5.** Document `CASE WHEN`, `COALESCE`, `NULLIF`, `CAST`, and scalar
      string functions (`UPPER`/`LOWER`/`LENGTH`/`TRIM`/`CONCAT`/`SUBSTR`).
      **Done in v0.1.36** (§4).
- [x] **B6.** Document `SEARCH` (vector similarity search) — currently has
      **zero** doc coverage despite being a headline feature. Cover both
      `SEARCH <ds> ON <col> QUERY [...] TOP k` and the legacy
      `SEARCH target FROM source QUERY [...] ON col K=k` form. **Done in
      v0.1.36** (new §7) — covers all 3 syntax forms (modern, `WHERE ~=`
      shorthand, legacy) and notes all 3 require a prebuilt `CREATE VECTOR
      INDEX`.
- [x] **B7.** Document `TRANSFORM <source> SELECT ... [WHERE ...] [INTO <target>]`.
      **Done in v0.1.36** (new §7) — corrected an assumption while writing
      this: without `INTO`, `TRANSFORM` overwrites the **source** dataset in
      place, it does not return results inline like `SELECT`.
- [x] **B8.** Document `CREATE INDEX ON ds(col)` / `CREATE VECTOR INDEX`.
      **Done in v0.1.36** (new §7).
- [x] **B9.** Document `SET DATASET <name> [METADATA] <key> = <value>` syntax.
      **Done in v0.1.36** (§4, Schema Evolution).
- [x] **B10.** Fix `SAVE`/`LOAD` doc inconsistency — the kind keyword
      (`TENSOR`/`DATASET`/`PIPELINE`) is required, not optional; the
      `ast.rs` doc-comment implying a default is wrong. **Fixed in v0.1.36.**
- [x] **B11.** Update `ARCHITECTURE.md`: add a section on pipeline
      persistence (v0.1.34); fix the "Recovery" section, which describes
      metadata/lazy-loading on startup that doesn't actually exist
      (`recover_databases` just creates empty `DatabaseInstance` stubs).
      **Done in v0.1.36.**
- [x] **B12.** Remove stale references (in docs or code comments) to a
      string-matching "legacy fallback chain" in `dsl/mod.rs` — it no longer
      exists; `execute_line_with_context` is typed-parser-only now.
      **Fixed in v0.1.36** (`src/dsl/mod.rs` comment corrected).

## Track C — Test coverage gaps

- [ ] **C1.** Add window function tests: `PARTITION BY`, `RANK`,
      `DENSE_RANK`, `LAG`/`LEAD`. Currently exactly one test exists
      (`test_window_no_partition_by`), despite full support being claimed
      shipped in v0.1.29.
- [ ] **C2.** Add pipeline × vector-engine integration tests — no test
      currently chains `COSINE_SIM`/`MATMUL`/index-aware search inside an
      `APPLY PIPELINE` step; v0.1.31-33 and v0.1.33-34 features are tested
      in total isolation from each other.
- [ ] **C3.** Add a dedicated test file for v0.1.28 features (subqueries,
      `RIGHT`/`FULL JOIN`, `IN`/`BETWEEN`, multi-column `ORDER BY`) — only
      incidental coverage exists today, higher silent-regression risk.
- [ ] **C4.** Add server-level test(s) exercising `PIPELINE` and `SEARCH`
      through `/execute` — no server test currently references either.
- [ ] **C5.** Verify `is_read_only` correctly classifies
      `APPLY PIPELINE`/`SAVE PIPELINE`/`LOAD PIPELINE` as write operations
      (not yet confirmed either way — check it doesn't run under a read lock).

## Track D — Architecture/design debt

These need a design decision before implementation, not just a bug fix.

- [ ] **D1.** `CAST` has zero `Vector`/`Matrix`/`Tensor` support — entirely
      scalar-only (`CastTarget` enum, `src/dsl/ast.rs:680-685`). Decide:
      add tensor-aware casting, or explicitly document it as out of scope.
- [ ] **D2.** `JOIN` only supports scalar equality — no similarity/ANN join,
      and it's an undocumented limitation. Decide: document as a known
      limitation, or scope a similarity-join feature.
- [ ] **D3.** `WHERE`-clause vector filtering has two parallel execution
      paths for the same conceptual operation — generic `FilterExec` (with
      `COSINE_SIM` as a predicate function) vs. index-driven
      `CosineFilterExec` (`src/query/physical.rs:597-655`). Consolidate or
      document why both exist.
- [ ] **D4.** Aggregate surface is inconsistent: `SUM`/`AVG` need
      vector-suffixed variants (`SumVec`/`AvgVec`) while `MIN`/`MAX` got
      unified/polymorphic naming. Decide whether to unify `SUM`/`AVG` syntax
      or document the asymmetry as intentional.
- [ ] **D5.** Pipeline persistence bypasses the shared
      `resolve_persistence_path` helper that `TENSOR`/`DATASET` use, and
      hand-rolls its own `pipeline_dir()` path logic
      (`src/dsl/persistence.rs:547-568` vs. `16-30`). Route through the
      shared helper, or document the intentional divergence.
- [ ] **D6.** `save_all_pipelines`/`load_all_pipelines` are dead code — no
      DSL command (e.g. `SAVE ALL PIPELINES`) invokes them, and no
      startup/shutdown lifecycle calls them either (note: this matches
      tensors/datasets, which also have no auto load/save wiring — so this
      may be "add the DSL command" rather than "fix a regression"). Either
      wire them to something reachable or remove them.
- [ ] **D7.** Add `list_pipelines_core`/`LIST PIPELINES` for parity with
      `LIST TENSORS`/`LIST DATASETS`, if genuinely missing.

---

## Track E — Window function combination bug (found while writing Track B docs)

- [ ] **E1.** Combining multiple window functions with *different* `OVER (...)`
      specs in one `SELECT` — especially mixing `LAG`/`LEAD` with a
      differently-specced ranking or aggregate window function — silently
      produces wrong values or an outright schema-mismatch error, depending
      on ordering. Reproduced directly against `main` @ v0.1.36 baseline:
      - `SELECT id, LAG(price) OVER (ORDER BY id) AS prev, SUM(price) OVER (PARTITION BY category ORDER BY id) AS running FROM items`
        gives the *same* `running` value for every row (not a cumulative
        sum) — LAG comes first, corrupts the following aggregate window.
      - `SELECT id, LAG(price) OVER (ORDER BY id) AS prev, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM items`
        errors outright: `Parse { msg: "Row 1 has incompatible schema" }`.
      - `SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn, LAG(price) OVER (ORDER BY id) AS prev FROM items`
        (reversed order) succeeds but silently **drops the `prev` column
        from the output entirely** — 2 columns come back instead of 3.
      - Same-spec combinations (identical `PARTITION BY`/`ORDER BY` across
        all window functions in the `SELECT`) work correctly — this is
        specifically about *differing* specs interacting, most reliably
        triggered by `LAG`/`LEAD`.
      - Likely root cause: `apply_window_and_computed_exprs`
        (`src/dsl/executor/query.rs`) calls `apply_window_func` once per
        `SelectExpr::Window` entry, threading `rows` through sequentially;
        something in how a later call's schema/column-count assumptions
        interact with an earlier call's appended column (or how `LAG`/`LEAD`
        specifically build their result column vs. how ranking/aggregate
        window functions build theirs) is inconsistent. Not yet root-caused
        beyond this — needs a debugger/print-driven trace through
        `apply_window_func` for the exact indices/schema at the point of
        divergence.
      - `docs/DSL_REFERENCE.md`'s Window Functions section was written to
        only use verified-safe example combinations and calls this out as a
        known limitation with a pointer to this entry — update that note if
        this gets fixed.

---

## Completion

- [ ] All tracks (A, B, C, D, E) fully checked off
- [ ] Final PR deletes this file (`CONSISTENCY_PLAN.md`)
