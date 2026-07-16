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

- [x] **C1.** Add window function tests: `PARTITION BY`, `RANK`,
      `DENSE_RANK`, `LAG`/`LEAD`. Currently exactly one test exists
      (`test_window_no_partition_by`), despite full support being claimed
      shipped in v0.1.29.
      **Done in v0.1.37** as a side effect of fixing Track E / E1 —
      `tests/window_functions_test.rs` covers `ROW_NUMBER`, `RANK`,
      `DENSE_RANK` (including ties), `LAG`/`LEAD`, and windowed `SUM`.
- [x] **C2.** Add pipeline × vector-engine integration tests — no test
      currently chains `COSINE_SIM`/`MATMUL`/index-aware search inside an
      `APPLY PIPELINE` step; v0.1.31-33 and v0.1.33-34 features are tested
      in total isolation from each other.
      **Done in v0.1.38** — `tests/pipeline_vector_engine_test.rs` (5 tests):
      `COSINE_SIM` in a pipeline `WHERE` step (with and without a vector
      index present), `COSINE_SIM`/`MATMUL` as computed `SELECT` columns,
      and a chained `WHERE COSINE_SIM(...) ... THEN NORMALIZE` pipeline. No
      bugs found — all combinations produced correct results.
- [x] **C3.** Add a dedicated test file for v0.1.28 features (subqueries,
      `RIGHT`/`FULL JOIN`, `IN`/`BETWEEN`, multi-column `ORDER BY`) — only
      incidental coverage exists today, higher silent-regression risk.
      **Done in v0.1.38** — `tests/v0128_features_test.rs` (8 tests).
      RIGHT/FULL JOIN row correctness was already covered by
      `correctness_integration.rs`, so this file covers what wasn't:
      subqueries (incl. nested), `IN`, `BETWEEN` (incl. compound `AND`),
      `LIMIT ... OFFSET`, multi-column `ORDER BY`, and the
      `FILTER x = true` boolean-literal regression guard.
- [x] **C4.** Add server-level test(s) exercising `PIPELINE` and `SEARCH`
      through `/execute` — no server test currently references either.
      **Done in v0.1.38** — `tests/server_pipeline_search_test.rs` (3
      tests): full `DEFINE`/`SHOW`/`APPLY`/`DROP PIPELINE` lifecycle over
      HTTP (incl. confirming `APPLY` on a dropped pipeline now errors
      rather than silently succeeding), `CREATE VECTOR INDEX` + `SEARCH`
      over HTTP, and `SEARCH` without a prebuilt index correctly erroring
      through the HTTP path too.
- [x] **C5.** Verify `is_read_only` correctly classifies
      `APPLY PIPELINE`/`SAVE PIPELINE`/`LOAD PIPELINE` as write operations
      (not yet confirmed either way — check it doesn't run under a read lock).
      **Verified correct in v0.1.38, no bug** — `Statement::is_read_only()`
      (`src/dsl/ast.rs`) only lists `Explain`/`Audit`/`List`/`Deliver` as
      read-only; pipeline mutations fall through to `false` by default and
      correctly take the server's write lock
      (`src/server/mod.rs` picks `db_arc.read()` vs `db_arc.write()` based
      on this). Added `is_read_only_pipeline_mutations_require_write_lock`
      unit test in `src/dsl/parser/mod.rs` to lock this in.

## Track D — Architecture/design debt

These need a design decision before implementation, not just a bug fix.

- [x] **D1.** `CAST` has zero `Vector`/`Matrix`/`Tensor` support — entirely
      scalar-only (`CastTarget` enum, `src/dsl/ast.rs:680-685`). Decide:
      add tensor-aware casting, or explicitly document it as out of scope.
      **Implemented in v0.1.39.** Deep-dive found the "redundant with
      RESHAPE/FLATTEN" assumption was wrong: `RESHAPE` can't be used inside
      a `SELECT` at all (hard parse error), and `FLATTEN` parses in
      `SELECT` but silently returns `NULL` (tracked as Track F / F2) — so
      there was no working way to reshape a Vector/Matrix *column* inline
      in a query. Added `CAST(expr AS VECTOR(n))` /
      `CAST(expr AS MATRIX(r, c))`, reshaping/flattening row-major;
      dimension mismatch returns `NULL` (consistent with other invalid
      `CAST` combinations) rather than erroring. Tests in
      `tests/correctness_integration.rs`.
- [x] **D2.** `JOIN` only supports scalar equality — no similarity/ANN join,
      and it's an undocumented limitation. Decide: document as a known
      limitation, or scope a similarity-join feature.
      **Implemented in v0.1.39**, index-accelerated per the project's
      efficiency-first philosophy: `JOIN <ds> ON COSINE_SIM(a.col, b.col) >
      threshold`. New `SimilarityJoinExec` (`src/query/physical.rs`) uses a
      `Vector` index on the right dataset's join column when one exists
      (`Index::search`, same pattern as `CosineFilterExec`), falling back
      to brute-force O(n·m) otherwise; both paths verified to produce
      identical results. Supports INNER/LEFT/RIGHT/FULL. Tests in
      `tests/similarity_join_test.rs`.
- [x] **D3.** `WHERE`-clause vector filtering has two parallel execution
      paths for the same conceptual operation — generic `FilterExec` (with
      `COSINE_SIM` as a predicate function) vs. index-driven
      `CosineFilterExec` (`src/query/physical.rs:597-655`). Consolidate or
      document why both exist.
      **Documented in v0.1.39, no code change** — re-reading confirmed this
      is a deliberate optimizer rewrite-rule pattern (substitute an
      index-accelerated executor only when a matching index exists,
      otherwise the generic path still evaluates the predicate correctly
      on its own), the same pattern `IndexScanExec` uses for hash-indexed
      equality. Documented in `ARCHITECTURE.md`'s "Index-Aware Execution"
      and a doc comment on `Planner::try_optimize_filter`.
- [x] **D4.** Aggregate surface is inconsistent: `SUM`/`AVG` need
      vector-suffixed variants (`SumVec`/`AvgVec`) while `MIN`/`MAX` got
      unified/polymorphic naming. Decide whether to unify `SUM`/`AVG` syntax
      or document the asymmetry as intentional.
      **Turned out to be a real bug, fixed in v0.1.39** — not just a naming
      question. The executor already merges `Sum`/`SumVec` (and
      `Avg`/`AvgVec`) into the same accumulator logic, and `SUM(vector_col)`
      already worked without the `_VEC` suffix. But `AVG`'s schema-inference
      (`query/logical.rs`) hardcoded `Float` regardless of input type
      (unlike `SUM`/`MIN`/`MAX`, which correctly infer from the column), so
      `AVG(vector_col)` errored with a type mismatch even though the
      executor could compute it fine. Fixed by inferring `AVG`'s result
      type the same way, with a Vector/Matrix-aware branch that still
      returns `Float` for scalar input (not `Int`, matching the actual
      runtime averaging behavior). Tests in `tests/vector_expressions_test.rs`.
- [x] **D5.** Pipeline persistence bypasses the shared
      `resolve_persistence_path` helper that `TENSOR`/`DATASET` use, and
      hand-rolls its own `pipeline_dir()` path logic
      (`src/dsl/persistence.rs:547-568` vs. `16-30`). Route through the
      shared helper, or document the intentional divergence.
      **Fixed in v0.1.39.** Extracted a shared `instance_base_dir` helper
      (dedupes the `data_dir`/instance-name composition), and routed
      pipeline's explicit `TO`/`FROM` paths through `resolve_persistence_path`
      so relative paths now resolve against `<data_dir>/<db>/`, consistent
      with `TENSOR`/`DATASET` (previously CWD-relative — no documented
      example relied on that behavior, so this was safe to change). Test
      in `tests/pipeline_persistence_test.rs`.
- [x] **D6.** `save_all_pipelines`/`load_all_pipelines` are dead code — no
      DSL command (e.g. `SAVE ALL PIPELINES`) invokes them, and no
      startup/shutdown lifecycle calls them either (note: this matches
      tensors/datasets, which also have no auto load/save wiring — so this
      may be "add the DSL command" rather than "fix a regression"). Either
      wire them to something reachable or remove them.
      **Removed in v0.1.39** — no object kind (tensor, dataset, or
      pipeline) has a bulk "SAVE ALL" DSL command, so adding one only for
      pipelines would itself be inconsistent. Removed the two functions and
      their test.
- [x] **D7.** Add `list_pipelines_core`/`LIST PIPELINES` for parity with
      `LIST TENSORS`/`LIST DATASETS`, if genuinely missing.
      **Added in v0.1.39** as a thin alias for the existing
      `SHOW PIPELINES` (`ListTarget::Pipelines` dispatches to the same
      `execute_show_pipelines`). Test in `tests/pipeline_test.rs`.

---

## Track E — Window function combination bug (found while writing Track B docs)

- [x] **E1.** Combining multiple window functions with *different* `OVER (...)`
      specs in one `SELECT` — especially mixing `LAG`/`LEAD` with a
      differently-specced ranking or aggregate window function — silently
      produced wrong values or an outright schema-mismatch error, depending
      on ordering.
      **Fixed in v0.1.37.** Root cause: `apply_window_func`
      (`src/dsl/executor/query.rs`) built the new window-result column's
      `Field` without `.nullable()` (unlike the sibling `SelectExpr::Computed`
      path, which does). `LAG`/`LEAD` produce `Value::Null` for boundary
      rows, so `Tuple::new`'s schema validation rejected those rows against
      the non-nullable field, and the code silently fell back to the
      pre-window row via `.unwrap_or(row)` — leaving the `Vec<Tuple>` with
      inconsistent per-row schemas, which cascaded into wrong values or
      errors in whatever window function ran next. Fix: mark the column
      nullable, and replace the silent `.unwrap_or(row)` with a propagated
      `DslError` for defense in depth. `docs/DSL_REFERENCE.md`'s Window
      Functions section restored the full combined example (previously
      hedged with a "known limitation" note, now removed). Tests:
      `tests/window_functions_test.rs` (8 tests, also substantially covers
      Track C / C1's window function coverage gap).

---

## Track F — Found while implementing Track D (v0.1.39)

- [ ] **F1.** Table-qualified columns (`table.col`) are only understood
      inside a `JOIN ... ON` clause. The `SELECT` column list does not
      resolve them at all — `SELECT a.id, b.name FROM a JOIN b ON ...`
      returns an empty schema/row (not an error, just silently wrong/empty
      output). Worse: `FROM orders o JOIN users u ON o.user_id = u.id`
      (table aliasing) fails to parse entirely
      (`"Unknown command: SELECT o.id, ..."`). This affects even a plain
      equi-join, not just the new similarity join — reproduced directly:
      `SELECT a.id, b.id FROM a JOIN b ON a.id = b.id` →
      `Schema { fields: [], field_indices: {} }`. A Track B doc example
      (`SELECT o.id, u.name FROM orders o JOIN users u ON ...`) shipped
      with this exact bug, undetected because that example wasn't
      individually re-verified after the surrounding JOIN section was
      drafted — fixed in v0.1.39 by rewriting the example to avoid
      aliasing/qualified projection and adding a "Known limitation" note.
      Not root-caused or fixed here — worth a dedicated session, since
      `table.col` projection is a pretty basic SQL expectation.
- [ ] **F2.** `FLATTEN(col)` inside a `SELECT` list parses successfully but
      always evaluates to `NULL` (schema inferred as `Float`, should be
      `Vector`/scalar depending on input). `RESHAPE(...)` inside `SELECT`
      doesn't even parse (`"Unknown command"`). Both are only wired for the
      standalone tensor-DSL context (`LET x = RESHAPE t TO [dims]`), not
      the `parse_select_expr` computed-expression path (unlike `NORMALIZE`,
      `MATMUL`, `TRANSPOSE`, which do work in `SELECT`). `CAST(... AS
      VECTOR(n)/MATRIX(r,c))` (Track D / D1) now covers the same use case
      via a different syntax, so this is lower priority, but the silent
      `NULL` for `FLATTEN` in particular is a footgun worth fixing or at
      minimum erroring on instead of silently returning `NULL`.

---

## Completion

- [ ] All tracks (A, B, C, D, E, F) fully checked off
- [ ] Final PR deletes this file (`CONSISTENCY_PLAN.md`)
