# LINAL Architecture

This document provides a comprehensive overview of the LINAL engine architecture, its components, and design decisions.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Execution Flow](#execution-flow)
5. [Storage Layer](#storage-layer)
6. [Query Processing](#query-processing)
7. [Type System](#type-system)
8. [Lineage & Provenance](#lineage--provenance)
9. [Consistency & Auditing](#consistency--auditing)
10. [Design Principles](#design-principles)

---

## Overview

LINAL is an in-memory analytical engine designed for linear algebra operations, structured data analysis, and machine learning workloads. It combines:

- **Tensor computation** (vectors, matrices, higher-dimensional tensors)
- **Structured datasets** (SQL-like tables with heterogeneous types)
- **Query optimization** (index-aware execution, predicate pushdown)
- **Persistence** (Parquet for datasets, JSON for tensors)

The engine is built in Rust with a modular architecture that separates concerns into distinct layers.

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                      Application Layer                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI    │  │  Server  │  │   REPL   │  │  Scripts │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │     DSL Layer (Compiler Pipeline)   │
        │  ┌───────────┐  ┌────────────────┐  │
        │  │  Lexer    │→ │ Parser (RD+    │  │
        │  │  (Logos)  │  │  Pratt)        │  │
        │  └───────────┘  └───────┬────────┘  │
        │  ┌────────────────────────────────┐ │
        │  │  Executor (typed AST → engine) │ │
        │  │  (zero string round-trips)     │ │
        │  └────────────────────────────────┘ │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │       Query Planning & Execution    │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Logical    │→ │   Physical   │ │
        │  │    Plan      │  │    Plan      │ │
        │  └──────────────┘  └──────────────┘ │
        │  ┌──────────────────────────────┐   │
        │  │      Query Optimizer         │   │
        │  │  - Index Selection           │   │
        │  │  - Predicate Pushdown        │   │
        │  └──────────────────────────────┘   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │         Engine Layer (TensorDb)     │
        │  ┌──────────────────────────────┐   │
        │  │   Database Instance Mgmt     │   │
        │  │   - Multi-database support   │   │
        │  │   - Context switching        │   │
        │  └──────────────────────────────┘   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │          Storage Layer              │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Tensor     │  │   Dataset    │ │
        │  │   Store      │  │   Store      │ │
        │  └──────────────┘  └──────────────┘ │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Hash       │  │   Vector     │ │
        │  │   Index      │  │   Index      │ │
        │  └──────────────┘  └──────────────┘ │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │        Persistence Layer            │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Parquet    │  │     JSON     │ │
        │  │  (Datasets)  │  │   (Tensors)  │ │
        │  └──────────────┘  └──────────────┘ │
        │  ┌──────────────┐                   │
        │  │     CSV      │                   │
        │  │  (I/O Opts)  │                   │
        │  └──────────────┘                   │
        └─────────────────────────────────────┘
```

---

## Core Components

### 1. Core Module (`src/core/`)

The core module contains fundamental data structures and abstractions:

#### `tensor.rs`

- **Tensor**: Multi-dimensional array with shape `[d1, d2, ..., dn]` and f32 data
- **TensorId**: Unique identifier for tensors.
- **ExecutionId**: Unique identifier for an execution/query session.
- **Lineage**: Information about how a tensor was derived (operation, inputs, execution ID).
- **Shape**: Dimension specification supporting scalars, vectors, matrices, and higher-order tensors.

#### `value.rs`

- **Value**: Enum representing all possible data types:
  - `Int`, `Float`, `String`, `Bool`
  - `Vector(usize)`, `Matrix(usize, usize)`, `Tensor(Shape)`
- **ValueType**: Type information for schema definitions

#### `tuple.rs`

- **Tuple**: Row representation with named fields
- **Schema**: Column definitions with types and constraints
- **Field**: Individual column specification

#### `dataset/` (Reference Graph)

- **Dataset**: A semantic view over existing tensors or other dataset columns. It does not own data directly but stores a map of `ResourceReference`s.
- **ResourceReference**: An enum representing a link to either a specific `TensorId` or a `(dataset, column)` pair in another dataset.
- **DatasetGraph**: A component responsible for resolving references. It supports **Transitive Resolution** (e.g., resolving a view of a view) and implements **Circular Dependency Detection** to prevent infinite resolution loops.
- **ColumnRole**: Metadata categorizing columns by their semantic purpose (e.g., `Feature`, `Target`, `Weight`, `Guid`).
- **Zero-Copy Guarantees**: Adding a column is an O(1) metadata operation. Underlying tensor data is shared via atomic reference counting (`Arc`), ensuring no data duplication.
- **Materialization**: While datasets are views in-memory, they can be materialized into physical rows and persisted via standard Parquet for portability.

#### `dataset/metadata.rs`

- **DatasetMetadata**: The central structure for dataset lifecycle management.
  - **Versioning**: Monotonically increasing version number for every `SAVE` operation.
  - **Identity**: Content-based hashing for integrity verification.
  - **Provenance**: `DatasetOrigin` tracking (Created, Imported, Derived, etc.).
  - **Evolution**: `SchemaHistory` recording every schema change with migration context.
  - **Timestamps**: `created_at` and `updated_at` (SystemTime) with microsecond precision.
  - **Custom Tags**: User-defined key-value pairs (`SET DATASET METADATA`).

#### `store/`

- **InMemoryTensorStore**: In-memory storage for tensors.
- **DatasetStore**: In-memory storage for legacy datasets.

#### `index/`

- **HashIndex**: Exact match lookups (equality predicates)
- **VectorIndex**: Similarity search (cosine, Euclidean distance)

#### `storage.rs`

- **StorageEngine**: Trait for persistence abstraction
- **ParquetStorage**: Parquet-based dataset persistence
- **JsonStorage**: JSON-based tensor persistence
- **CsvStorage**: CSV-based import/export with schema inference (Legacy)

#### `connectors/` (Scientific Ingestion)

- **Connector**: Trait for format-specific ingestion (translation only).
- **ConnectorRegistry**: Global registry for format handlers.
- **CsvConnector**: High-performance Arrow-based CSV ingestion.

### 2. Engine Module (`src/engine/`)

The engine module orchestrates execution:

#### `db.rs`

- **TensorDb**: Main database engine managing multiple database instances
- **DatabaseInstance**: Isolated database with its own stores
- Features:
  - Multi-database support with context switching
  - Automatic recovery from disk on startup
  - Configuration via `linal.toml`
  - **Session Management**: Explicit `RESET SESSION` capability to clear in-memory state

#### `operations.rs`

- **BinaryOp**: Binary operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, etc.)
- **UnaryOp**: Unary operations (TRANSPOSE, FLATTEN, RESHAPE, etc.)
- **TensorKind**: Classification of tensor types

#### `kernels.rs`

- Low-level computational kernels:
  - Element-wise operations
  - Matrix multiplication (MATMUL)
  - Vector operations (dot product, cosine similarity, L2 distance)
  - Broadcasting and relaxed mode operations

#### `context.rs`

- **`ExecutionContext`**: per-execution arena allocator and resource tracker — bump allocation for scratch buffers (`alloc_temp`/`alloc_slice`/`alloc_vec`), a pooled `acquire_vec`/`release_vec` for reusable `Vec<f32>` scratch space, optional memory-limit enforcement (`check_allocation`, `with_memory_limit`), and tensor/dataset ID tracking for the current execution. See "Performance Optimizations" below for the arena/pool sizing details.
- **`ResourceError`**: raised when `check_allocation` would exceed a configured memory limit.

#### `error.rs`

- **EngineError**: Unified error type for engine operations

### 3. DSL Module (`src/dsl/`)

The DSL module implements a full compiler-grade pipeline from source text to engine calls.

#### `lexer.rs` — DFA Tokenizer (Logos 0.14)

- 130+ tokens covering all keywords, operators, punctuation, and literals (exact count drifts every release — `grep -cE '#\[(token|regex)' src/dsl/lexer.rs` for the current figure)
- Skips whitespace and `--` / `#` / `//` comment styles automatically
- Keyword tokens always win over the `Ident` regex (DFA property ensures no ambiguity)
- `tokenize(source) -> Result<Vec<(Token, Span)>, usize>` — returns error offset on unknown character

#### `ast.rs` — Typed AST

- `Statement` — 36 variants as of v0.1.47, one per top-level command (exact count drifts every release — see the enum in `src/dsl/ast.rs` for the current, authoritative list)
- `Statement::is_read_only()` — gates the shared-reference `execute_line_shared` path
- `Expr` — expression sub-language, 25 variants as of v0.1.47 (see `src/dsl/ast.rs` for the current list): scalar/literal (`Ref`, `Int`, `Scalar`, `StringLit`, `Bool`, `VecLiteral`, `MatLiteral`), boolean/predicate (`Infix`, `And`, `Or`, `Not`, `IsNull`, `IsNotNull`, `In`, `Between`), structural (`Call`, `Index`, `Field`, `DatasetRef`), and SQL-surface (`Case`, `Coalesce`, `Nullif`, `ScalarFn`, `Cast`, `VectorFn`)
  - `Expr::VecLiteral` / `Expr::MatLiteral` — inline vector/matrix constants (`[v1, v2, ...]` / `[[r1c1, r1c2], ...]`) usable anywhere in a SQL expression
  - `Expr::Case` / `Expr::Coalesce` / `Expr::Nullif` / `Expr::Cast` — `CASE WHEN`, `COALESCE`, `NULLIF`/`IFNULL`, `CAST(expr AS type)` (including `CAST(... AS VECTOR(n)/MATRIX(r,c))` for in-query reshaping) — see DSL_REFERENCE.md §4
  - `Expr::VectorFn` — one of ten SQL-style vector/matrix functions (see `VectorFnKind`) usable in SELECT, WHERE, ORDER BY
- `VectorFnKind` — `Normalize`, `L2Norm`, `CosineSim`, `Dot`, `VecAdd`, `VecScale` (v0.1.31), plus `Matmul`, `Transpose`, `MatShape`, `Flatten` (v0.1.40, for in-`SELECT` matrix operations — distinct from the standalone `CallExpr` keyword forms below)
- `CallExpr` — 17 named-prefix operations: binary (`ADD`, `MATMUL`, `CORRELATE`, …), unary (`NORMALIZE`, `RESHAPE`, …), n-ary (`STACK`)
- `AggFuncAst` — aggregate functions: `Sum`, `Avg`, `Count`, `Min`, `Max`, `AvgVec`, `SumVec` (v0.1.31)
  - `AvgVec` / `SumVec` — element-wise vector aggregates for GROUP BY centroid queries
- All types (`ColType`, `TensorKindAst`, `InfixOp`, `CmpOp`, `FilterValue`) are decoupled from engine internals; the executor maps them
- `DatasetFromClause` — typed clause bag for `DATASET … FROM source [FILTER …] [SELECT …] [GROUP BY …] [HAVING …] [ORDER BY …] [LIMIT …]`
- `DatasetFilter { column, op: CmpOp, value: FilterValue }` — typed predicate for FILTER/HAVING in dataset queries
- `SearchQuery` — enum: `TensorRef(String)` (named tensor) | `Inline(Vec<f64>)` (inline vector literal)
- `ExplainTarget` — enum: `Dataset(String)` | `Search(SearchStmt)` | `Select(SelectStmt)`; carries the full typed sub-statement into `execute_explain()` so no string reconstruction is needed
- `CreateDatabaseStmt.if_not_exists` / `DropDatabaseStmt.if_exists` — boolean flags enabling idempotent DDL (`CREATE DATABASE IF NOT EXISTS`, `DROP DATABASE IF EXISTS`)
- `IndexKindAst::Vector` — variant added for `CREATE VECTOR INDEX`; previously absent, causing all vector index creation to fall through to the legacy handler

#### `parser/` — Recursive-Descent + Pratt Parser (sub-module directory, v0.1.25)

Split from a single 2581-line `parser.rs` into five focused files. All files share the same `impl Parser` block pattern — Rust allows multiple `impl` blocks per type across files within a module.

- **`parser/mod.rs`** — `Parser` struct, all cursor/consuming primitives (`peek`, `eat`, `eat_ident`, etc.), `parse_statement` dispatch, small statement parsers (`parse_define_tensor`, `parse_let`, `parse_create`, `parse_drop`, etc.), full test suite (107 tests as of v0.1.47 — grew steadily since the v0.1.25 split; `grep -c '#\[test\]' src/dsl/parser/mod.rs` for the current count)
- **`parser/dataset.rs`** — `parse_create_dataset`, `parse_dataset_from_clause`, `parse_select`, `parse_alter`, `parse_insert_into`, `parse_search`, `parse_materialize`, and related helpers (`parse_cmp_op`, `parse_filter_value`, `parse_agg_call`, `parse_select_expr`)
  - `parse_select_expr` detects `AVG_VEC` / `SUM_VEC` identifiers and routes to `SelectExpr::Aggregate { func: AvgVec/SumVec, ... }`; consumes optional `AS alias` (v0.1.31)
- **`parser/expr.rs`** — `parse_expr`, `parse_pratt` (Pratt precedence climber), `parse_expr_atom`, `parse_call_expr`, `parse_simple_expr`, `can_start_simple_expr`
  - `parse_vec_literal()` — parses `[n, n, ...]` into `Expr::VecLiteral`; triggered by `Token::LBracket` at atom position (v0.1.31)
  - `Token::Normalize` in atom position: if next token is `(` → SQL-style `Expr::VectorFn { Normalize, ... }`, otherwise → tensor-algebra `CallExpr::Normalize` (v0.1.31)
  - Ident `L2_NORM` / `COSINE_SIM` / `DOT` / `VEC_ADD` / `VEC_SCALE` followed by `(` → `Expr::VectorFn` (v0.1.31)
- **`parser/introspection.rs`** — `parse_show`, `parse_explain`, `parse_audit`, `parse_deliver`
- **`parser/persistence.rs`** — `parse_save`, `parse_load`, `parse_list`, `parse_import`, `parse_export`, `parse_use`

Public API (unchanged): `parse(source) -> Result<Statement, ParseError>` — entry point in `mod.rs`.

Key properties:
- Pratt parser for the expression sub-language with correct infix precedence (`*`/`/` > `+`/`-` > comparisons) and postfix `.field` / `[...]` binding
- `Token::LBracket` in atom position always routes to `parse_vec_literal()` (SQL vector literal); in postfix position routes to `parse_index_specs()` (tensor indexing)
- `ParseError { offset: usize, msg: String }` with `into_dsl_error(line)` for integration
- All statement forms from v0.1.20–v0.1.21 (hardened `IF NOT EXISTS`, SEARCH, computed columns, `Expr::Int`, aggregate expressions) preserved unchanged

#### `executor/` — Typed Dispatch Layer (sub-module directory, v0.1.25)

Split from a single 2014-line `executor.rs` into six focused files, each with a single responsibility.

- **`executor/mod.rs`** — `execute_statement` (single `match` on `Statement`, all `Statement` variants; zero string round-trips); `to_engine_kind`, `col_type_to_value_type` (small helpers); Search and InsertInto arms remain inline
- **`executor/eval.rs`** — `eval_let` (entry point for Let/Derive arms); `eval_expr_to_name` (recursive `Expr` → engine call, generates temp names via atomic counter); `eval_call` (maps all 18 `CallExpr` variants to engine ops); `apply_index` (subscript operations); `fresh_temp`; `infix_to_binary_op`; `expr_to_string` (debug/tracing only)
- **`executor/show.rs`** — `execute_show` (all `ShowTarget` variants; calls engine APIs directly); `format_lineage_tree` (private)
- **`executor/explain.rs`** — `execute_explain`; builds `LogicalPlan` directly from typed `ExplainTarget` (Dataset/Search/Select) through the `Planner`; reuses shared helpers from `query.rs` via `use super::query::...` to avoid duplication
- **`executor/query.rs`** — `execute_select`, `execute_create_dataset_from`, `execute_add_computed_column`; shared logical-plan helpers (`agg_func_to_logical`, `dataset_filter_to_logical`, `dsl_expr_to_logical_expr`); `eval_row_expr` (pure per-row evaluator for computed columns, walks `Expr::Infix` with column lookup)
  - `dsl_expr_to_logical_expr` maps `Expr::VecLiteral` → `LogicalExpr::Literal(Value::Vector(...))` and `Expr::VectorFn` → `LogicalExpr::VectorFn` (v0.1.31)
  - `agg_func_to_logical` maps `AggFuncAst::AvgVec` / `SumVec` → `AggregateFunction::AvgVec` / `SumVec` (v0.1.31)
  - `infer_expr_result_type` extended for `VecLiteral` (→ `Vector(n)`) and `VectorFn` (→ `Vector(0)` for shape-returning fns, `Float` for scalar-returning fns) (v0.1.31)
- **`executor/pipeline.rs`** — `execute_define_pipeline`, `execute_apply_pipeline`, `execute_drop_pipeline`, `execute_describe_pipeline`; named/reusable pipeline lifecycle (`DEFINE PIPELINE`, `APPLY PIPELINE`, `DROP PIPELINE`, `DESCRIBE PIPELINE`, v0.1.33; persistence added v0.1.34, see §6)

#### `mod.rs` — Dispatch Entry Point

- **`execute_line_with_context()`**: calls `parser::parse`, then dispatches through `execute_statement`; all `Statement` variants are handled in the typed path — no string fallback remains
- **`execute_line()`**: convenience wrapper (no context)
- **`execute_script()`**: multi-line runner with paren-balance tracking
- **`DslOutput`**: structured output enum (`None`, `Message`, `Table`, `TensorTable`, `Tensor`, `LazyTensor`)

#### `persistence.rs` — SAVE/LOAD/LIST/IMPORT/EXPORT

All persistence-and-ingestion statement handling lives here (758 lines) —
`save_typed`, `load_typed`, `list_typed`, `import_typed`, `export_typed`,
`import_csv_typed` are the top-level entry points `execute_statement`
dispatches to for `Statement::Save/Load/List/Import/Export/ImportCsv`, each
covering all its object kinds (Tensor/Dataset/Pipeline for Save/Load,
Datasets/Tensors/DatasetVersions/DatasetPackages for List). Also owns the
shared `resolve_persistence_path`/`instance_base_dir` helpers that
`TENSOR`/`DATASET`/`PIPELINE` all route relative paths through.

#### ~~`handlers/`~~ — Deleted (v0.1.23)

The `handlers/` directory was fully eliminated in v0.1.23. All logic was either ported to the typed executor or discarded (string-based wrappers with no live callers). The typed executor (`executor/`) is now the sole dispatch layer with zero string round-trips across all `Statement` variants.

#### `error.rs`

- **DslError**: DSL-specific error types (`Parse { line, msg }`, `Engine { line, source }`)

### 4. Query Module (`src/query/`)

The query module implements query planning and optimization:

#### `logical.rs`

- **LogicalPlan**: High-level query representation — `Scan`, `Filter`, `Project`, `Aggregate`, `Join`, `Union`, `Distinct`, `Sort`, `Limit`, `VectorSearch`
- **`Expr`** (this module's own expression type, mirroring `dsl::ast::Expr` but engine-facing): `Column`, `Literal`, `BinaryExpr`, `And`, `Or`, `Not`, `IsNull`, `IsNotNull`, `In`, `Between`, `AggregateExpr { func, expr, alias }`, `Case`, `Coalesce`, `Nullif`, `ScalarFn`, `Cast`, `VecLiteral`, `VectorFn`, `MatLiteral`
  - `Expr::VecLiteral(Vec<f64>)` / `Expr::MatLiteral(Vec<Vec<f64>>)` — compile-time vector/matrix constants (v0.1.31 / v0.1.36)
  - `Expr::VectorFn { func: VectorFnKind, args: Vec<Expr> }` — vector/matrix scalar functions
  - `Expr::AggregateExpr.alias` — carries the `AS alias` from `SelectExpr::Aggregate` through to schema naming (v0.1.45; previously discarded, see CHANGELOG)
  - `VectorFnKind`: `Normalize`, `L2Norm`, `CosineSim`, `Dot`, `VecAdd`, `VecScale` (v0.1.31), `Matmul`, `Transpose`, `MatShape`, `Flatten` (v0.1.40)
- **`AggregateFunction`** extended: `AvgVec`, `SumVec` for element-wise vector aggregation
- `infer_expr_type_full`: `VecLiteral` → `Vector(n)`, vector scalar fns → `Vector(0)` or `Float`; `AvgVec`/`SumVec` → `Vector(0)` (wildcard, actual dim known at runtime)

#### `physical.rs`

- **PhysicalPlan**: Executable query plan
- **Executor**: Executes physical plans with index-aware execution
- **`evaluate_expression`** extended in v0.1.31:
  - `VecLiteral` → constructs `Value::Vector` from the literal's `f64` values cast to `f32`
  - `VectorFn` → evaluates all 6 functions: NORMALIZE (unit-length), L2_NORM (Euclidean norm → Float), COSINE_SIM (dot/(‖a‖·‖b‖) → Float), DOT (dot product → Float), VEC_ADD (element-wise add → Vector), VEC_SCALE (scalar multiply → Vector)
- **`AggregateExec`** extended in v0.1.31:
  - `SumVec` shares the `Sum` accumulation path (element-wise addition on `Value::Vector`)
  - `AvgVec` shares the `Avg` accumulation path (element-wise sum + count, finalized as element-wise divide)

#### `planner.rs`

- **QueryPlanner**: Converts logical plans to physical plans
- **Optimizer**: Applies optimizations:
  - Index selection
  - Predicate pushdown
  - Projection pruning

### 5. Server Module (`src/server/`)

HTTP server implementation built with **Axum**:

- **High-Concurrency Model**: Uses `Arc<RwLock<TensorDb>>` to allow multiple parallel read operations (analytical queries) while maintaining exclusive access for state-modifying commands.
- **Asynchronous Job System** (`jobs.rs`):
  - `POST /jobs`: Submit long-running queries for background execution.
  - `GET /jobs`: List all jobs.
  - `GET /jobs/:id`: Poll for status (Pending, Running, Completed, Failed).
  - `GET /jobs/:id/result`: Retrieve structured `DslOutput`.
  - `DELETE /jobs/:id`: Cancel a **Pending** job. Running or Completed jobs return `400 Bad Request`.
- **Background Scheduler** (`scheduler.rs`): Cron-like execution of DSL commands registered at runtime.
  - `POST /schedule`: Register a named task (`name`, `command`, `interval_secs`, optional `target_db`).
  - `GET /schedule`: List active scheduled tasks.
  - `DELETE /schedule/:id`: Remove a scheduled task.
- **Dataset Delivery** (`dataset_server.rs`): a separate `DatasetServer` router, mounted at `/delivery` in `server/mod.rs`, that serves a saved dataset package's files read-only over HTTP — `GET /delivery/datasets/:name/manifest.json`, `.../schema.json`, `.../stats.json`, `.../data.parquet` — the same four files `storage.rs`'s `save_dataset_package` writes to disk (see Storage Layer). This is what `DELIVER <dataset>` (DSL_REFERENCE.md §9) checks the readiness of.
- **Database Management API**: `GET /databases`, `POST /databases/:name`, `DELETE /databases/:name`.
- **Multi-tenant Isolation**: Isolated database contexts via `X-Linal-Database` header. After each request the server restores the previously active database, so concurrent requests with different headers cannot affect each other's context.
- **Graceful Shutdown**: Native support for `SIGINT` and `SIGTERM` to ensure in-flight requests complete before termination.
- **OpenAPI/Swagger documentation**: Interactive API explorer at `/swagger-ui`. Note: only `/execute` and `/health` are currently included in the generated schema; all other routes are functional but undocumented in the spec.

### 6. Utils Module (`src/utils/`)

Utility functions:

- **parsing.rs**: String parsing helpers

---

## Execution Flow

### 1. Command Parsing

```text
DSL source
  → lexer::tokenize()       — DFA, produces Vec<(Token, Span)>
  → parser::parse()         — recursive-descent, produces Statement AST
  → executor::execute_statement()  — typed match, calls engine API directly
```

All `Statement` variants are handled in the typed path — `execute_line_with_context` contains no string fallback.

Example: `SELECT * FROM users WHERE id > 10`

- Lexer produces `[Select, Star, From, Ident("users"), Where, Ident("id"), ...]`
- Parser builds `Statement::Select(SelectStmt { dataset: "users", columns: All, filter: Some(Expr::Infix {...}), ... })`
- Executor builds `LogicalPlan::Scan → Filter → Project` directly from the typed `SelectStmt` AST and executes it through the `Planner` (no string round-trip)

### 2. Query Planning (for SELECT queries)

```rs
SELECT Query → Logical Plan → Physical Plan → Execution
```

1. **Logical Plan**: High-level representation

   ```rs
   Project(columns: [*])
     └─ Filter(predicate: id > 10)
         └─ Scan(table: users)
   ```

2. **Optimization**: Apply optimizations
   - Check for indexes on `id`
   - Push predicate to index scan if available

3. **Physical Plan**: Executable plan

   ```rs
   IndexScan(index: id_idx, predicate: > 10)
     └─ Project(columns: [*])
   ```

4. **Execution**: Execute physical plan
   - Use index for fast lookup
   - Apply projection
   - Return results

### 3. Expression Evaluation

Expressions are evaluated recursively:

- **Literals**: Direct value
- **Variables**: Lookup in tensor/dataset store
- **Binary Operations**: Evaluate operands, apply operation
- **Indexing**: `tensor[i, j]`, `tuple.field`, `dataset.column`

### 4. Aggregation

GROUP BY queries:

1. Group rows by grouping columns
2. Apply aggregation functions (SUM, AVG, COUNT, MIN, MAX, AVG_VEC, SUM_VEC)
3. Support element-wise aggregation for vectors/matrices
4. Apply HAVING clause filter

**Vector aggregates** (v0.1.31): `AVG_VEC` and `SUM_VEC` compute element-wise statistics across all vectors in a group, enabling centroid computation directly in SQL:

```sql
SELECT category, AVG_VEC(embedding) AS centroid
FROM docs GROUP BY category
```

---

## Storage Layer

### In-Memory Storage

- **TensorStore**: HashMap-based storage keyed by TensorId
- **DatasetStore**: HashMap-based storage with name and ID indexes
- **Indices**: Maintained automatically on INSERT

### Persistence

#### Datasets (Parquet)

- **Data**: Columnar Apache Parquet format for high-performance retrieval.
- **Metadata**: Stored in JSON format with two distinct naming conventions:
  - `.metadata.json`: The standard format for all new datasets, containing rich metadata, versioning, and schema history.
  - `.meta.json`: Legacy format maintained for backward compatibility.
- **Path Resolution**: Each dataset is a self-contained **directory package** under `data_dir / db_name / [optional_subpath] / datasets / [name]/`, not a single flat file: `data.parquet` (the columnar data) alongside sibling `schema.json`, `stats.json`, `lineage.json`, and `manifest.json`. The flat `[name].meta.json` / `[name].metadata.json` files described above live next to this directory, not inside it.

#### Tensors (JSON)

- Full tensor serialization.
- Shape and data preserved.
- Suitable for weights and model parameters.

#### Pipelines (JSON, v0.1.34)

- **Format**: Human-readable JSON (`{name, source, version}`) — the pipeline's original DSL source text is stored alongside its parsed steps, not just the parsed AST.
- **Round-trip**: `SAVE PIPELINE` writes `source`; `LOAD PIPELINE` re-parses it via the same typed parser used for live DSL execution, reconstructing the steps rather than deserializing them directly. This makes the file hand-editable — any valid `DEFINE PIPELINE` DSL can replace the `source` field.
- **Path convention**: Defaults to `<data_dir>/<db>/pipelines/<name>.json`. An explicit `TO`/`FROM` path is resolved the same way as tensors/datasets — absolute paths are used as-is, relative paths resolve against `<data_dir>/<db>/`.

#### Tensor-First Datasets (In-Memory)

- **Zero-Copy Architecture**: Datasets reference tensors in the `TensorStore` by ID. Adding a column is an O(1) metadata operation.
- **Math Integration**: Columns are exposed as standard LINAL symbols via dot notation. `LET x = ds.vec * 2.0` resolves `ds.vec` to its underlying `TensorId` and executes normally.
- **Reverse Integration**: Results of any tensor operation can be added back to a dataset as a new column, maintaining the zero-copy chain.
- **Persistence**: While primarily in-memory views, they can be persisted to Parquet using the `SAVE DATASET` command, which triggers on-demand materialization.

#### Scientific Dataset Ingestion

LINAL implements a connector-based architecture for high-performance scientific data (HDF5, Numpy, Zarr, CSV, etc.):

1. **Connector Isolation**: Connectors are responsible ONLY for translating external formats into Arrow `RecordBatch`es.
2. **Ephemeral Context (USE)**: `USE DATASET FROM` loads data directly into memory as tensors and registers a temporary dataset view. No persistence on disk.
3. **Persistent Normalization (IMPORT)**: `IMPORT DATASET FROM` translates the source, normalizes it into a LINAL Dataset Package (Parquet + Metadata), and persists it for future use.
4. **Reproducibility**: Source path and format are tracked in `DatasetOrigin` metadata.
5. **Format Support**:
   - **HDF5**: Recursive group traversal and flattening.
   - **Numpy**: Direct ingestion of `.npy` and multi-array `.npz`.
   - **Zarr**: Full support for V3 stores and hierarchical data.

- **Row Count Validation**: The engine strictly enforces that all columns within a tensor-first dataset have a consistent "row count" (dimension 0 of the tensor). This prevents malformed data from entering analytical pipelines.
- **Dangling Reference Detection**: Since datasets reference tensors by `TensorId`, the engine performs on-demand audits. The `SHOW` command generates **Health Warnings** if a dataset column points to a tensor that has been deleted from the `TensorStore`.
- **Zero-Copy Guarantees**: Metadata-only operations ensure that datasets never duplicate underlying vector data, preserving memory and cache locality.

### Recovery

On engine startup (`TensorDb::recover_databases`), the engine scans `data_dir`
for subdirectories and registers an **empty** `DatabaseInstance` per directory
name found. This only makes existing database names selectable via `USE`; it
does **not** load any dataset, tensor, or pipeline content — there is
currently no metadata scan and no lazy-load-on-first-access mechanism for any
persisted object kind. Datasets, tensors, and pipelines must each be restored
explicitly with `LOAD DATASET` / `LOAD TENSOR` / `LOAD PIPELINE` — there is
no bulk "load everything" command for any object kind.

---

## Query Processing

### Index-Aware Execution

1. **Index Selection**: Planner checks for applicable indexes
   - HashIndex for equality predicates (`WHERE id = 5`)
   - VectorIndex for similarity search (`WHERE embedding ~= [...]`)

2. **Predicate Pushdown**: Filters applied as early as possible
   - Use index to filter before scanning full dataset

3. **Execution**: Physical plan uses index when available
   - IndexScan instead of full table scan
   - Significant performance improvement for filtered queries

**Rewrite-rule pattern** (`query/planner.rs`, `try_optimize_filter`): both index
optimizations follow the same shape — pattern-match a specific predicate form,
and *only* substitute the specialized executor if a matching index actually
exists on the referenced column; otherwise fall through to the generic
executor, which evaluates the same predicate correctly, just without the
index. This is deliberate duplication in service of an optimization, not an
architectural inconsistency:

- `col = literal` + a `Hash` index on `col` → `IndexScanExec`, else generic
  `SeqScanExec` + `FilterExec` (which evaluates `col = literal` itself).
- `COSINE_SIM(col, query_vec) > threshold` + a `Vector` index on `col` →
  `CosineFilterExec`, else generic `FilterExec` (which evaluates
  `COSINE_SIM(...)` itself via the normal expression evaluator).

Adding a new index type/predicate pattern means adding one more branch to
`try_optimize_filter`, not building a parallel execution path — the generic
`FilterExec` path must still handle the predicate correctly on its own for
the no-index case.

### Join Execution

`LogicalPlan::Join` carries a `JoinKind` (`Inner`, `Left`, `Right`, `Full`) and an
optional similarity threshold; the planner (`query/planner.rs`) always marks the
output schema's fields nullable, since LEFT/RIGHT/FULL OUTER joins pad unmatched
rows with `NULL` on one side.

- **Equi-join** (`ON a.col = b.col`): `NestedLoopJoinExec` — no join index exists
  yet, so this is always a nested-loop scan, not a hash join.
- **Similarity join** (`ON COSINE_SIM(a.col, b.col) > threshold`, two `Vector`
  columns): `SimilarityJoinExec` — brute-force by default, or index-accelerated
  if the right dataset's column has a `CREATE VECTOR INDEX`. Follows the same
  optional-index rewrite-rule pattern as `CosineFilterExec` above (see
  `try_optimize_filter`'s source comment for the cross-reference). Only `>` is
  supported as the threshold comparison for now.
- **Table aliasing & qualified columns** (v0.1.40): `FROM orders o JOIN users u
  ON o.user_id = u.id` — a dataset in `FROM`/`JOIN` may be aliased (`AS` is
  optional), and `table.col`/`alias.col` qualifiers are accepted anywhere a
  column is referenced (`ON`, `WHERE`, `SELECT`). The qualifier is **not**
  used to disambiguate, though — only the bare column name resolves — so
  column names must still be unique across the joined datasets. See
  `DSL_REFERENCE.md` §4 for the full caveat and self-join behavior.

### Window Function Execution

Window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, and
aggregate-as-window `SUM`/`AVG`/`COUNT`/`MIN`/`MAX`) are a post-processing step
over the already-executed physical plan's row set
(`apply_window_and_computed_exprs` / `apply_window_func`,
`src/dsl/executor/query.rs`), not a `PhysicalPlan` variant — the executor
partitions rows by `PARTITION BY` key, sorts within each partition by
`ORDER BY`, computes the function's value per row, and appends it as a new
nullable schema field before final projection. Multiple window functions with
*different* `PARTITION BY`/`ORDER BY` specs can be freely combined in one
`SELECT` (fixed in v0.1.37 — see CHANGELOG for the schema-nullability root
cause of the original bug). `SUM`/`AVG`/`SUM_VEC`/`AVG_VEC` as a window
function compute a *running* aggregate within the window (element-wise for
vector columns, fixed in v0.1.45) instead of collapsing to one row.

### Aggregation Execution

1. **Grouping**: Hash-based grouping by grouping columns
2. **Aggregation**: Apply aggregation functions per group
   - Element-wise for vectors/matrices
   - Scalar for numeric types
3. **HAVING**: Filter groups after aggregation

---

## Type System

### Type Hierarchy

```text
Value
├── Scalar
│   ├── Int
│   ├── Float
│   ├── String
│   └── Bool
└── Tensor
    ├── Vector(n)
    ├── Matrix(m, n)
    └── Tensor(Shape)
```

### TensorKind

Every stored tensor carries a `TensorKind` tag:

- **`Normal`**: Default. Participates in all operations without restriction.
- **`Strict`**: Shape-strictness flag. Any binary operation that takes at least one Strict operand produces a Strict result, making strictness contagious through a computation graph. Defined via `DEFINE x AS STRICT TENSOR [dims] VALUES [...]`.
- **`Lazy`**: Set internally by `LAZY LET` / `LET LAZY`. The tensor has no data in the store; instead an `Expression` tree is kept in `lazy_store` and evaluated on first `SHOW`.

### Type Inference

- **Arithmetic**: Int + Float → Float
- **Broadcasting**: Scalar * Vector → Vector
- **Matrix Operations**: Matrix * Matrix → Matrix (if compatible)

### Type Safety

- Compile-time type checking in expressions
- Runtime validation for dataset operations
- Clear error messages for type mismatches
- `Vector(0)` in schema field declarations acts as a dimension wildcard, accepting any `Value::Vector(n)` at runtime. Used by aggregate outputs (`AVG_VEC`, `SUM_VEC`) whose dimension is only known once actual data flows through (v0.1.31)

---

## Lineage & Provenance

LINAL implements a robust **Lineage Tracking** system that ensures every derived tensor carries its computational history.

- **Persistent Provenance**: Lineage metadata (Source Operation, Input Tensor IDs, Execution Context) is serialized alongside the tensor data.
- **Audit Trails**: Users can trace any final result back to its root "ground-truth" tensors using the `SHOW LINEAGE` command.
- **Traceability**: Every execution batch is assigned a unique `ExecutionId`, allowing auditors to group related operations.

---

## Consistency & Auditing

As LINAL moves toward a "Reference Graph" model for data, the engine provides tools to maintain structural integrity.

- **Reference Validation**: The `AUDIT DATASET` command performs a deep scan of the Reference Graph, verifying that all terminal nodes (Tensor IDs) exist in the store.
- **Dangling Reference Detection**: Identifies dataset columns that point to deleted or missing tensors.
- **Self-Healing Diagnostics**: The `SHOW` command for Tensor-First datasets automatically triggers a sanity check, providing visual warnings if the dataset is in an "unhealthy" state.

---

## Design Principles

### 1. Modularity

- Clear separation of concerns
- Minimal coupling between modules
- Easy to extend and test

### 2. Performance

- In-memory first design
- Index-aware query execution
- Efficient tensor operations

### 3. Expressiveness

- Rich type system
- SQL-inspired querying
- Linear algebra operations

### 4. Usability

- Human-friendly DSL
- Multiple access methods (CLI, REPL, Server)
- Comprehensive error messages

### 5. Extensibility

- Trait-based abstractions (StorageEngine, Index)
- Plugin-friendly architecture
- Easy to add new operations

---

## Configuration

Engine configuration via `linal.toml`:

```toml
[storage]
data_dir = "./data"
default_db = "default"
```

- **data_dir**: Root directory for persistence
- **default_db**: Default database name

---

## Error Handling

Unified error types:

- **EngineError**: Engine-level errors
- **DslError**: DSL parsing/execution errors
- **DatasetStoreError**: Dataset operation errors

All errors propagate with context for debugging.

---

## Performance Optimizations (Phases 7-11)

LINAL has undergone comprehensive performance optimization across multiple phases:

### Memory Management

**Three-Tier Allocation Strategy**:

```
Tensor Size → Allocation Strategy:
├─ ≤16 elements: Stack allocation (SmallVec) - avoids heap alloc for the intermediate buffer
├─ 17-255 elements: Direct heap allocation - avoid pool overhead  
└─ ≥256 elements: Tensor pooling - reuse allocations
```

**Tensor Pool**:

- Pools common sizes: 128, 256, 512, 1024, 2048, 4096, 8192 elements
- Max 8 vectors per size
- Automatic size matching (request 100 → get 128 capacity)
- Per-context cleanup

**Arena Allocation**:

- `ExecutionContext` uses `bumpalo::Bump` for ephemeral allocations
- Batch cleanup on context drop
- Memory limits (default 100MB per context)
- `ResourceError` for limit violations

### Execution Model

**Backend Dispatch**:

```
Operation → CpuBackend:
├─ ≥1024 elements → SimdBackend (SIMD optimized, if contiguous)
└─ Otherwise → ScalarBackend (fallback)
```

`CpuBackend::use_simd` dispatches to `SimdBackend` purely on element count (≥1024); the contiguity check happens one layer down, inside each of `SimdBackend`'s individual op methods, which fall back to scalar internally for non-contiguous input rather than at the `CpuBackend` dispatch point. There is no separate Rayon backend tier — Rayon parallelism is embedded directly inside the kernel functions in `engine/kernels.rs`.

**SIMD Kernels** (`src/core/backend/simd.rs`):

- Platform-specific: NEON (ARM), SSE/AVX (x86_64)
- Operations with SIMD implementations: `add`, `sub`, `multiply`, `matmul` (tiled), `dot`, `distance`
- Operations currently using scalar fallback (with TODOs): `divide`, `scale`, `normalize`, `sum`, `mean`, `stdev`, `cosine_similarity`, `transpose`, `flatten`, `reshape`, `stack`
- Automatic dispatch based on tensor contiguity and element count

**Parallel Execution** (via `rayon`, threshold: 50k elements):

Rayon parallelism fires inside individual kernel functions in `engine/kernels.rs`, not as a separate dispatch tier:

- `add`, `sub`, `multiply`: `par_iter()` for contiguous tensors ≥50k elements
- `scalar_mul` (backing `SCALE`): `par_iter()` at ≥50k elements
- `matmul`: `par_chunks_mut` for large tile passes
- Dataset batch operations: `par_chunks` for row processing ≥10k rows
- 2.5x speedup on 100k-element vectors

### Zero-Copy Operations

**Metadata-Only Transformations**:

- `reshape`: O(1) - only updates shape metadata
- `transpose`: O(1) - stride manipulation
- `slice`: O(1) - view over same `Arc<Vec<f32>>`

**Benefits**:

- Zero allocation for view operations
- Shared memory via `Arc`
- Cache-friendly access patterns

### Dataset Operations

**Batching**:

- Process datasets in 1024-row chunks
- Parallel execution for ≥10k rows
- Better cache locality

**Architecture**:

- `dataset_legacy.rs`: Row-based materialized tables (current, active for relational workloads)
- `dataset/mod.rs`: Zero-copy reference graph datasets (active for tensor-first workloads)

### Performance Results

| Optimization | Impact |
|--------------|--------|
| Zero-overhead metadata | ~10% improvement |
| Zero-copy views | Zero allocation for transforms |
| Rayon parallelization | 2.5x on large tensors |
| SIMD kernels | Platform-dependent speedup |
| Tensor pooling | 3-18% improvement |
| Stack allocation | One fewer heap alloc for tiny tensors (see caveat below) |

---

## Technical Appendix: Core Subsystems

### 1. Dual Dataset Model

LINAL supports two distinct dataset implementations to balance flexibility and performance:

- **Legacy Datasets (`dataset_legacy.rs`)**: Traditional row-oriented, fully materialized tables. Ideal for small datasets or cases where diverse scalar types are primary.
- **Tensor-First Datasets (`dataset/`)**: Advanced reference graphs where columns point to `TensorId`s in the `TensorStore`. These are zero-copy views that enable high-performance algebraic workflows and on-demand materialization.

### 2. Compute Backend Dispatch

The `CpuBackend` acts as an intelligent dispatcher for all numerical operations:

- **SIMD Selection**: If the platform supports it (NEON/SSE/AVX) and the tensor is physically contiguous, the `SimdBackend` is prioritized for a 2x-8x speedup.
- **Rayon Parallelization**: For very large tensors (typically ≥50k elements), work is automatically distributed across all available CPU cores.
- **Scalar Fallback**: For complex strided layouts or small tensors where overhead exceeds benefit, a robust `ScalarBackend` ensures correctness.

### 3. Resource Governance & Memory Limits

To prevent system-level instability, LINAL implements per-query resource limits:

- **Arena Allocation**: `ExecutionContext` utilizes a `Bump` arena for ephemeral results, significantly reducing heap fragmentation.
- **Memory Limits**: Default per-context limit is 100MB. Exceeding this triggers a `ResourceError`, terminating the query safely.
- **Tensor Pooling**: Reuses buffers for common tensor sizes to minimize syscall overhead during high-frequency allocation.

### 4. Three-Tier Allocation Strategy

LINAL optimizes memory layout based on tensor dimensionality:

- **Stack (≤16 elements)**: Builds the intermediate result in a `SmallVec<[f32; 16]>`, avoiding a heap allocation for that scratch buffer — the final `.to_vec()` into the function's `Vec<f32>` return type still heap-allocates once, so this isn't zero heap allocation overall, just one fewer than the naive approach.
- **Direct (17-255 elements)**: Standard heap allocation for small, unpredictable sizes.
- **Pooled (≥256 elements)**: Buffer reuse for large analytical payloads.

---

## Semantic Invariants

To ensure a stable foundation, LINAL guarantees the following semantic behaviors:

1. **Tensor Immutability**: Once a tensor is stored in the `TensorStore`, its data buffer is logically immutable. Transformations (Scale, Add, etc.) always produce new tensor IDs.
2. **Identity-Preserving Lineage**: Every tensor carrying an `ExecutionId` maintains an unbroken link to its source operation and inputs.
3. **Reference Integrity**: Tensor-First datasets purely store references. Deleting a tensor from the store will trigger a "Dangling Reference" warning in the dataset, but will not corrupt the dataset schema.
4. **Deterministic Math**: Given the same floating-point precision and backend (SIMD/Scalar), operations are guaranteed to be bit-deterministic.

## Core vs. Extensions

| Component | Classification | Stability |
|-----------|----------------|-----------|
| `Tensor` / `Shape` | **Semantic Core** | Frozen (v1) |
| `ReferenceGraph` (TF Datasets) | **Semantic Core** | Frozen (v1) |
| `DslParser` / `Lexer` / `Executor` | **Semantic Core** | Stable (v0.1.25) |
| `SimdBackend` (NEON/AVX) | **Engine Extension** | Evolving |
| `ParquetPersistence` | **Engine Extension** | Evolving |
| `HttpServer` / `REST API` | **Application Layer** | Flexible |

---

## Future Enhancements

- GPU-backed tensor execution
- Distributed execution
- Columnar execution engine
- Python/WASM integration
- Native ML operators (KNN, clustering, PCA)

---

For more details, see:

- [DSL Reference](DSL_REFERENCE.md)
- [Changelog](../CHANGELOG.md)
