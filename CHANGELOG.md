# Changelog

All notable changes to LINAL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
