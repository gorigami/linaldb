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
        │  │  Legacy handlers  (fallback)   │ │
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

#### `error.rs`

- **EngineError**: Unified error type for engine operations

### 3. DSL Module (`src/dsl/`)

The DSL module implements a full compiler-grade pipeline from source text to engine calls.

#### `lexer.rs` — DFA Tokenizer (Logos 0.14)

- 80+ tokens covering all keywords, operators, punctuation, and literals
- Skips whitespace and `--` / `#` / `//` comment styles automatically
- Keyword tokens always win over the `Ident` regex (DFA property ensures no ambiguity)
- `tokenize(source) -> Result<Vec<(Token, Span)>, usize>` — returns error offset on unknown character

#### `ast.rs` — Typed AST

- `Statement` — 27 variants, one per top-level command
- `Statement::is_read_only()` — gates the shared-reference `execute_line_shared` path
- `Expr` — expression sub-language: `Ref`, `Scalar`, `StringLit`, `Infix`, `Call`, `Index`, `Field`, `DatasetRef`
- `CallExpr` — 18 named-prefix operations: binary (`ADD`, `MATMUL`, `CORRELATE`, …), unary (`NORMALIZE`, `RESHAPE`, …), n-ary (`STACK`)
- All types (`ColType`, `TensorKindAst`, `InfixOp`, `CmpOp`, `FilterValue`) are decoupled from engine internals; the executor maps them
- `DatasetFromClause` — typed clause bag for `DATASET … FROM source [FILTER …] [SELECT …] [GROUP BY …] [HAVING …] [ORDER BY …] [LIMIT …]`
- `DatasetFilter { column, op: CmpOp, value: FilterValue }` — typed predicate for FILTER/HAVING in dataset queries
- `SearchQuery` — enum: `TensorRef(String)` (named tensor) | `Inline(Vec<f64>)` (inline vector literal)
- `ExplainTarget` — enum: `Dataset(String)` | `Search(SearchStmt)` | `Select(SelectStmt)`; carries the full typed sub-statement into `execute_explain()` so no string reconstruction is needed
- `CreateDatabaseStmt.if_not_exists` / `DropDatabaseStmt.if_exists` — boolean flags enabling idempotent DDL (`CREATE DATABASE IF NOT EXISTS`, `DROP DATABASE IF EXISTS`)
- `IndexKindAst::Vector` — variant added for `CREATE VECTOR INDEX`; previously absent, causing all vector index creation to fall through to the legacy handler

#### `parser.rs` — Recursive-Descent + Pratt Parser

- `parse(source) -> Result<Statement, ParseError>` — main entry point
- One `parse_*` function per statement type; dispatches on first token
- Pratt parser for the expression sub-language with correct infix precedence (`*`/`/` > `+`/`-`) and postfix `.field` / `[...]` binding
- `ParseError { offset: usize, msg: String }` with `into_dsl_error(line)` for integration
- **Hardened forms (v0.1.20)**:
  - `CREATE DATABASE IF NOT EXISTS <name>` — `IF`/`NOT`/`EXISTS` tokens consumed in `parse_create()`
  - `DROP DATABASE IF EXISTS <name>` — same pattern in `parse_drop()`
  - `CREATE VECTOR INDEX` — fixed from broken `Token::Ident("VECTOR")` guard (never fired) to correct `Token::Vector` match arm
  - `RESET SESSION` — optional `SESSION` ident consumed in `parse_reset()` before falling through to `Token::Reset` handling
  - `AUDIT DATASET <name>` — optional `Token::Dataset` consumed in `parse_audit()` so both `AUDIT ds` and `AUDIT DATASET ds` parse cleanly
  - `USE/IMPORT DATASET FROM "path" AS <alias>` — optional `AS <name>` clause added to `parse_use()` and `parse_import()`, enabling named ephemeral and persistent imports
- **Extended forms (v0.1.21)**:
  - `INSERT INTO VALUES` — `parse_insert_value` handles vector `[v1,v2]` / matrix `[[r0c0,…],…]` / boolean `true|false` literals; `InsertRow::Positional` stores them in new `InsertValue::Vector / Matrix / Bool` variants
  - `SEARCH` — all three syntax forms handled: `ON col QUERY [...] LIMIT k [INTO target]`, `target FROM source QUERY [...] ON col K=k`, `source WHERE col ~= [...] LIMIT k`
  - `DATASET name ADD COLUMN col = expr [LAZY]` — computed-column form; parsed into `AlterOp::AddComputedColumn { name, expr, lazy }`; executor evaluates per-row via `eval_row_expr`
  - `DATASET name COLUMNS col: TYPE …` — parentheses around column list now optional
  - `DEFAULT <bool>` / `INSERT … true|false` — `FilterValue::Bool` and `InsertValue::Bool` added; `parse_filter_value` handles `true`/`false` idents
  - **Comparison operators in expressions**: `InfixOp` extended with `Eq / NotEq / Gt / Lt / GtEq / LtEq`; Pratt parser precedence set below arithmetic; `dsl_expr_to_logical_expr` emits `LogicalExpr::BinaryExpr` with correct op strings
  - **Aggregate functions in WHERE/HAVING**: `COUNT(col)`, `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)` in expression context parsed as `Expr::Ref("FUNC(col)")` — matching post-aggregation column names
  - **`Expr::Int(i64)`**: integer literals now distinguished from float literals through the full expression pipeline; eliminates Int→Float promotion in computed expressions
  - **`parse_agg_call`**: now parses a full `Expr` inside aggregate calls, enabling `SUM(price * qty)` and other computed aggregate expressions

#### `executor.rs` — Typed Dispatch Layer

- `execute_statement(db, stmt, line_no, ctx)` — single `match` on `Statement`, routes every variant directly to the engine API; **zero string round-trips** in the typed path as of v0.1.21 (all 27+ variants)
- `execute_show(db, ShowTarget, line_no)` — private function that matches on the `ShowTarget` enum and calls engine APIs directly (no string reconstruction)
- `execute_explain(db, ExplainTarget, line_no)` — private function that builds a `LogicalPlan` directly from the typed `ExplainTarget` (Dataset/Search/Select) and runs it through the `Planner`; produces a formatted logical + physical plan string with zero string reconstruction
- `execute_add_computed_column()` — evaluates a DSL `Expr` per row via `eval_row_expr` to compute derived column values; infers the value type from the first row; calls `alter_dataset_add_computed_column` with `lazy=true/false`
- `eval_row_expr()` — pure per-row evaluator: walks `Expr::Infix` trees with column lookup from a `HashMap<&str, &Value>`, returning `Value`; handles Int/Float promotion correctly for mixed arithmetic
- `eval_expr_to_name()` — recursive `Expr` → engine call evaluator; generates temporary tensor names via an atomic counter for sub-expressions
- `eval_call()` — maps all 18 `CallExpr` variants to `eval_binary` / `eval_unary` / `eval_matmul` / etc., with lazy/eager branching
- `col_type_to_value_type()` — converts `ColType` → `ValueType` without a string round-trip
- `dsl_expr_to_logical_expr()` — converts `ast::Expr` → `query::logical::Expr` for WHERE/HAVING clauses; handles all `InfixOp` variants including comparison operators (`>`, `<`, `=`, etc.); `Expr::Int` → `Value::Int` (preserves integer type)
- `dataset_filter_to_logical()` — converts `DatasetFilter` → `query::logical::Expr` for FILTER/HAVING in `DATASET … FROM` queries
- `expr_to_string()` — converts `Expr` back to DSL text; dead code in the main statement path (used only for tracing/debug)

#### `mod.rs` — Dispatch Entry Point

- **`execute_line_with_context()`**: tries `parser::parse` first; on success dispatches through `execute_statement`; on parse error falls through to the legacy `if/else if starts_with()` chain
- **`execute_line()`**: convenience wrapper (no context)
- **`execute_script()`**: multi-line runner with paren-balance tracking
- **`DslOutput`**: structured output enum (`None`, `Message`, `Table`, `TensorTable`, `Tensor`, `LazyTensor`)

#### `handlers/`

Minimal handler files retained for plan-building helpers used by `explain.rs`, and a single legacy fallback for non-standard syntax. All `Statement` variants dispatch exclusively through the typed executor:

- **dataset.rs**: `build_dataset_query_plan` / `build_select_query_plan` — used by `explain.rs` for legacy multi-clause EXPLAIN forms; `handle_add_tensor_column` — retained for `.add_column("col", var)` method-call syntax (no typed AST equivalent)
- **explain.rs**: Legacy `EXPLAIN` fallback — handles old multi-clause forms like `EXPLAIN DATASET x FROM y FILTER …`; real plan-building logic lives in `build_dataset_query_plan`, `build_select_query_plan`, `build_search_query_plan`; typed EXPLAIN forms use `execute_explain()` in the executor
- **persistence.rs**: `save_typed`, `load_typed`, `list_typed`, `import_typed`, `export_typed` — typed entry points called from the executor
- **search.rs**: `build_search_query_plan` / `build_search_plan_internal` — used by `explain.rs`
- **metadata.rs**: `set_metadata_typed` — typed entry point; `handle_set_metadata` parses the string and delegates

The following handlers were **deleted** (fully superseded by the typed executor):

- ~~**tensor.rs**~~: DEFINE, VECTOR, MATRIX
- ~~**operations.rs**~~: LET, binary/unary operations
- ~~**semantics.rs**~~: BIND, ATTACH, DERIVE
- ~~**audit.rs**~~: AUDIT DATASET (now calls `db.verify_tensor_dataset()` directly)
- ~~**session.rs**~~: RESET (now calls `db.reset_session()` directly)
- ~~**index.rs**~~: CREATE INDEX (now calls `db.create_index()` / `db.create_vector_index()` directly)
- ~~**instance.rs**~~: CREATE DATABASE, USE, DROP DATABASE (now calls engine methods directly)
- ~~**introspection.rs**~~: SHOW (now handled by `executor::execute_show`)
- ~~**dataset.rs `handle_*`**~~: SELECT, INSERT, DATASET queries, MATERIALIZE, ADD COLUMN (all ported to typed path as of v0.1.21)
- ~~**search.rs `handle_search`**~~: all three SEARCH syntax forms now parsed and dispatched by the typed executor

#### `error.rs`

- **DslError**: DSL-specific error types (`Parse { line, msg }`, `Engine { line, source }`)

### 4. Query Module (`src/query/`)

The query module implements query planning and optimization:

#### `logical.rs`

- **LogicalPlan**: High-level query representation
- Operations: Scan, Filter, Project, Aggregate, GroupBy, Limit

#### `physical.rs`

- **PhysicalPlan**: Executable query plan
- **Executor**: Executes physical plans with index-aware execution

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

If `parser::parse()` returns an error (unrecognized syntax), `execute_line_with_context` falls through to the legacy `if/else if starts_with()` chain for backward compatibility.

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
2. Apply aggregation functions (SUM, AVG, COUNT, MIN, MAX)
3. Support element-wise aggregation for vectors/matrices
4. Apply HAVING clause filter

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
- **Path Resolution**: The engine uses a managed directory structure: `data_dir / db_name / [optional_subpath] / datasets / [name].parquet`.

#### Tensors (JSON)

- Full tensor serialization.
- Shape and data preserved.
- Suitable for weights and model parameters.

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

On engine startup:

1. Scan `data_dir` for database directories
2. Load dataset metadata from JSON
3. Load tensor metadata
4. Lazy-load actual data on first access

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
├─ ≤16 elements: Stack allocation (SmallVec) - zero heap allocation
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
├─ Contiguous + ≥1024 elements → SimdBackend (SIMD optimized)
└─ Otherwise → ScalarBackend (fallback)
```

`CpuBackend` dispatches to `SimdBackend` when the tensor is contiguous and has ≥1024 elements; otherwise it falls through to `ScalarBackend`. There is no separate Rayon backend tier — Rayon parallelism is embedded directly inside the kernel functions in `engine/kernels.rs`.

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
| Stack allocation | Zero heap for tiny tensors |

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

- **Stack (≤16 elements)**: Uses `SmallVec` to avoid heap allocation entirely for tiny vectors.
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
| `DslParser` / `Lexer` / `Executor` | **Semantic Core** | Stable (v0.1.21) |
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
- [Tasks & Implementation](Tasks_implementations.md)
- [Changelog](../CHANGELOG.md)
