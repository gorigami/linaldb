# Dataset Architecture Explanation

## Two Dataset Implementations

### 1. `dataset_legacy.rs` - **Currently Active**

**Location**: `src/core/dataset_legacy.rs`

**Structure**:

```rust
pub struct Dataset {
    pub id: DatasetId,
    pub schema: Arc<Schema>,
    pub rows: Vec<Tuple>,  // ← Stores actual row data
    pub metadata: DatasetMetadata,
    pub indices: HashMap<String, Box<dyn Index>>,
    pub lazy_expressions: HashMap<String, Expr>,
}
```

**Characteristics**:

- **Row-based storage**: Stores actual `Vec<Tuple>` data
- **Used by**: DSL (`src/dsl/mod.rs`), Engine
- **Operations**: filter, map, select, join, etc.
- **Memory**: Copies data for transformations
- **Status**: **Active** - this is what the engine uses

### 2. `dataset/mod.rs` - **Integrated View Layer**

**Location**: `src/core/dataset/mod.rs`

**Structure**:

```rust
pub struct Dataset {
    pub name: String,
    pub schema: DatasetSchema,
    pub columns: HashMap<String, ResourceReference>,  // ← References, not copies
    pub metadata: Option<DatasetMetadata>,
}
```

**Characteristics**:

- **Column-based storage**: References to tensor data
- **Zero-copy**: Uses `ResourceReference` (views over existing data)
- **Memory efficient**: No data duplication
- **Status**: **Integrated Production Layer** - handles `BIND`, `ATTACH`, and `DERIVE`.

**Submodules** (`src/core/dataset/`):

| File | Purpose |
|---|---|
| `reference.rs` | `ResourceReference` — the enum a column resolves to: either `Tensor { id }` (a direct tensor reference) or `Column { dataset, column }` (a reference to another dataset's column, enabling views/virtual datasets); the zero-copy indirection this whole layer exists for |
| `registry.rs` | `DatasetRegistry` — owns the `HashMap<String, Dataset>` for a runtime scope (a `DatabaseInstance` or `ExecutionContext`) |
| `graph.rs` | `DatasetGraph` — resolves references across a `DatasetRegistry`. Actually used by `ATTACH` (linking a tensor into a dataset column) and `AUDIT DATASET` (walking the graph to detect dangling references); **not** used by `BIND` (plain name/entry aliasing, no graph involved — `src/engine/db.rs:bind_resource`) or `DERIVE` (pure tensor-expression evaluation via `eval_let`, unrelated to this module) |
| `schema.rs` | `DatasetSchema`, `ColumnSchema`, `ColumnRole` — column-level typing for the reference-graph model |
| `schema_evolution.rs` | `SchemaVersion`, `Migration` — non-breaking schema versioning, backs `SHOW DATASET VERSIONS <name>` (aliased as `LIST DATASET VERSIONS <name>`; there is no bare `LIST VERSIONS` command) |
| `lineage.rs` | `DatasetLineage`, `LineageNode` — a DAG of *data-import* provenance (dataset name, content hash, operation, parent nodes), populated by the scientific-ingestion connectors (CSV/HDF5/Numpy/Zarr) and `core/storage.rs`. **Not** what `SHOW LINEAGE <name>` displays — that command walks a different, tensor-computation `LineageNode` type defined in `src/engine/db.rs` (tracks how a tensor was derived via `ADD`/`MATMUL`/etc., not dataset import history) |
| `manifest.rs` | `DatasetManifest` — the delivery contract for a portable LINAL dataset package (name, version, hash, entrypoints) |
| `stats.rs` | `DatasetStats`, `ColumnStats` — row counts and per-column summaries |
| `metadata.rs` | `DatasetMetadata`, `DatasetOrigin` — the `metadata` field on `Dataset` above; provenance and free-form metadata set via `SET DATASET METADATA` |

## Why Both Exist? (Hybrid Architecture)

LINALDB uses a hybrid approach to balance performance and flexibility:

1. **Relational/Heavy Path** (`dataset_legacy.rs`):
    - Optimized for **row-level operations** (INSERT/UPDATE).
    - Used for standard SQL `DATASET` creation and `SELECT` query results.
    - Primary format for **Persistence** (Parquet). `Vector`/`Matrix` columns
      write as native Arrow `FixedSizeList` types (v0.1.72,
      `core::storage::dataset_to_record_batch`) when the column has no
      `NULL`s and a uniform width — real numeric columns readable by
      pandas/polars/pyarrow/R `arrow`, not JSON strings. A column with any
      actual `NULL` falls back to the pre-v0.1.72 per-cell JSON-string
      encoding: writing a nullable `FixedSizeList` round-trips fine through
      this engine's own arrow-rs reader but pyarrow rejects it
      (`ArrowInvalid: Expected all lists to be of size=N ...`) — an
      external-reader-verified limitation, not a theoretical one. Read-side
      (`arrow_array_to_values`) transparently supports both encodings, so
      older Parquet packages keep loading. `schema.json` (the
      `/delivery`-exposed package metadata, see Server Module in
      `ARCHITECTURE.md`) correctly reports the *logical* type even for a
      fallback-encoded column (v0.1.73) via a `linal.logical_value_type`
      Arrow field-metadata entry, mirroring the existing
      `core::connectors::SHAPE_METADATA_KEY` pattern — without it,
      `schema.json` would report a fallback column as plain `"String"`,
      since it's otherwise derived purely from the physical (post-fallback)
      Arrow type.
2. **Semantic/Light Path** (`core/dataset/`):
    - Optimized for **Zero-Copy Views** and **Tensor Algebra**.
    - Allows linking independent tensors as virtual columns (`ATTACH`).
    - Tracks complex lineage through the **Reference Graph**.

## Current Status

**Active Hybrid Model**:

- both systems are active and integrated.
- **Engine Bridge**: `materialize_tensor_dataset()` converts a Reference View into a Relational Object (`dataset_legacy::Dataset`) for high-speed row scanning or Parquet export. Until v0.1.60 this was the *only* way to reach that conversion, and nothing ever called it except once, transiently, to build the one-shot table `USE DATASET FROM`/`dataset()`+`.add_column()` print to the console — the result was never stored anywhere queryable. `SHOW ALL DATASETS`, `SELECT ... FROM <name>`, and `MATERIALIZE <name>` all read only from the legacy dataset store, which nothing ever populated, so every one of them reported "Dataset not found" for a Reference View dataset that `SHOW <name>` (a separate, tensor-first-specific health-check display) showed just fine — a real bug in the exact workflow `DSL_REFERENCE.md` §2 documents (`LET ds = dataset(...)` + `.add_column()`). Fixed by new `DatabaseInstance::sync_tensor_dataset_to_legacy()` (`src/engine/db.rs`), which calls `materialize_tensor_dataset()` and additionally stores/refreshes the result in the legacy dataset store under the same name. Called from every mutation site that changes a Reference View's columns (`add_column_to_tensor_dataset`) and from `use_dataset_core`, so the legacy copy never goes stale. See CHANGELOG.md v0.1.60.

**Future Work**:

- Merge the two into a unified `VirtualTable` that can switch between Column-Referencing and Row-Owning modes transparently.
- Standardize all DSL commands to target the unified interface.
