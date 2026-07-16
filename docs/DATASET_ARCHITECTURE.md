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
    - Primary format for **Persistence** (Parquet).
2. **Semantic/Light Path** (`core/dataset/`):
    - Optimized for **Zero-Copy Views** and **Tensor Algebra**.
    - Allows linking independent tensors as virtual columns (`ATTACH`).
    - Tracks complex lineage through the **Reference Graph**.

## Current Status

**Active Hybrid Model**:

- both systems are active and integrated.
- **Engine Bridge**: Use `materialize_tensor_dataset()` to convert a Reference View into a Relational Object for high-speed row scanning or Parquet export.

**Future Work**:

- Merge the two into a unified `VirtualTable` that can switch between Column-Referencing and Row-Owning modes transparently.
- Standardize all DSL commands to target the unified interface.
