# Changelog

All notable changes to LINAL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
