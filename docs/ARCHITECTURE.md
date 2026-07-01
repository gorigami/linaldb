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
        │          DSL Layer (Parser)         │
        │  ┌──────────────────────────────┐   │
        │  │  Command Parsing & Routing   │   │
        │  │  Expression Evaluation       │   │
        │  │  Error Handling              │   │
        │  └──────────────────────────────┘   │
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

The DSL module handles command parsing and execution:

#### `mod.rs`

- **execute_line()**: Execute a single DSL command
- **execute_script()**: Execute a script file
- **DslOutput**: Structured output format

#### `handlers/`

Command-specific handlers:

- **tensor.rs**: DEFINE, VECTOR, MATRIX, SHOW commands
- **dataset.rs**: DATASET, INSERT INTO, SELECT, FILTER, etc.
- **operations.rs**: LET, binary/unary operations
- **index.rs**: CREATE INDEX, CREATE VECTOR INDEX
- **search.rs**: SEARCH (vector similarity)
- **persistence.rs**: SAVE, LOAD, LIST, IMPORT, EXPORT commands
- **instance.rs**: CREATE DATABASE, USE, DROP DATABASE
- **metadata.rs**: SET DATASET METADATA
- **explain.rs**: EXPLAIN, EXPLAIN PLAN
- **introspection.rs**: SHOW commands (SCHEMA, INDEXES, LINEAGE)
- **semantics.rs**: Explicit resource handlers (BIND, ATTACH, DERIVE)
- **audit.rs**: Consistency checking (AUDIT DATASET)

#### 3.1 `error.rs`

- **DslError**: DSL-specific error types

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

```#rs
DSL Command → Parser → Command AST → Route to Handler
```

Example: `SELECT * FROM users WHERE id > 10`

- Parser identifies `SELECT` command
- Routes to `dataset.rs` handler
- Parses query components (columns, table, filter, etc.)

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
| `DslParser` / `Lexer` | **Semantic Core** | Solidified |
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
