# LINAL DSL Reference

**LINAL Script** is a high-performance, SQL-inspired language for tensor algebra and relational analytics. This document serves as the complete technical specification for all keywords, operators, and built-in functions.

---

## 1. Data Types & Literals

LINAL supports both standard relational types and multi-dimensional numeric structures.

### Relational Types

- `Int`: 64-bit signed integer.
- `Float`: 32-bit floating point (standard for tensor values).
- `String`: UTF-8 character sequence.
- `Bool`: `true` or `false`.
- `Null`: Represents a missing value. Use the `?` suffix in `DATASET` definitions for nullable columns (e.g., `score: Float?`).

### Tensor Types

Defined with specific dimensionality:

- `Vector(N)`: A 1D tensor with `N` elements.
- `Matrix(R, C)`: A 2D tensor with `R` rows and `C` columns.
- `Tensor(d1, d2, ...)`: An N-dimensional tensor.

---

## 2. Resource Definition

Create and initialize numeric resources and structured schemas.

### VECTOR / MATRIX

Quick shorthand for defining tensors.

```sql
VECTOR v = [1.0, 2.0, 3.0]
MATRIX m = [[1, 2], [3, 4]]
```

### DEFINE

Explicit tensor definition for higher dimensions. Shape is specified as a bracket-delimited list.

```sql
DEFINE t AS TENSOR [2, 2, 2] VALUES [1, 2, 3, 4, 5, 6, 7, 8]
```

The optional `STRICT` modifier enforces shape-strictness: any binary operation involving a strict tensor propagates the strict flag to its output, preventing accidental shape relaxation.

```sql
DEFINE w AS STRICT TENSOR [3] VALUES [1, 0, 0]
```

### Tensor-First Dataset Constructor

Create a zero-copy tensor-first dataset using the `dataset()` constructor inside a `LET` assignment:

```sql
LET ds = dataset("my_dataset")
```

This registers an empty named dataset in the `DatasetRegistry`. Columns are added later via `.add_column()`.

### Adding Columns to a Tensor-First Dataset

After creating a tensor-first dataset you can attach any in-memory tensor as a column using dot-method syntax:

```sql
VECTOR v_temp = [36.6, 37.1, 36.9]
LET raw = dataset("raw")
raw.add_column(temp, v_temp)
```

Syntax: `<dataset_var>.add_column(<column_name>, <tensor_var>)`

This is an O(1) metadata operation — no data is copied.

### DATASET

Define a persistent relational structure.

```sql
DATASET diagnostics COLUMNS (
    id: Int,
    region: String,
    score: Float?,           -- Nullable column
    features: Vector(128)    -- Embedded tensor
)
```

---

## 3. Numerical DSL (Core Algebra)

LINAL provides two ways to perform math: Functional keywords and Infix operators.

### Functional Keywords

- `ADD a b`: Element-wise addition.
- `SUBTRACT a b`: Element-wise subtraction.
- `MULTIPLY a b`: Element-wise multiplication (Hadamard product).
- `DIVIDE a b`: Element-wise division.
- `MATMUL a b`: Standard matrix multiplication.
- `TRANSPOSE a`: Swap dimensions of a matrix/tensor.
- `RESHAPE a TO [dims]`: Change shape without copying data.
- `FLATTEN a`: Convert multidimensional tensor to a 1D vector.
- `NORMALIZE a`: Scales vector to unit length (L2 norm).
- `SCALE a BY n`: Multiplies all elements by a scalar `n`.
- `STACK t1 t2 ...`: Combines tensors along Axis 0.
- `SUM a`: Sum of all elements in the tensor.
- `MEAN a`: Arithmetic mean of all elements.
- `STDEV a`: Standard deviation of all elements.

### Lazy Evaluation

Prefix a `LET` with `LAZY` (either word order is accepted) to defer computation. The expression is stored as a computation graph and materialized only when `SHOW` is called.

```sql
LAZY LET trend = STDEV sensor_3d   -- deferred
LET LAZY trend = STDEV sensor_3d   -- identical alias
SHOW trend                         -- triggers materialization
```

### Infix Operators

Standard math notation for scalar and tensor variables:

```sql
LET result = (v_a + v_b) / 2.0
LET scaled = m_a * 10
```

### Advanced Operators

- `CORRELATE a WITH b`: Pearson correlation between two vectors.
- `SIMILARITY a WITH b`: Cosine similarity score [-1.0, 1.0].
- `DISTANCE a TO b`: Euclidean distance between points.

---

## 4. Query & Engineering (SQL)

### SELECT

Query datasets with familiar syntax.

```sql
SELECT region, AVG(score) 
FROM diagnostics 
WHERE id > 100 
GROUP BY region 
HAVING AVG(score) > 0.5 
LIMIT 10
```

- **Aggregate Functions**: `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`.
- **Filtering**: `WHERE` or `FILTER` can be used interchangeably.

### Semantic Transforms (Zero-Copy)

- `BIND alias TO resource`: Create a semantic link (alias) to a tensor or dataset.
- `ATTACH tensor TO ds.col`: Link an independent tensor into a dataset column.
- `DERIVE target FROM expr`: Create a new resource with full automated lineage tracking.

### Schema Evolution

- `ALTER DATASET ds ADD COLUMN col: type [DEFAULT val]`
- `ALTER DATASET ds ADD COLUMN col = expression [LAZY]`
- `MATERIALIZE ds`: Physicalize all `LAZY` columns in a dataset.

---

## 5. Persistence & Ingestion

Load and save data across different formats.

- `USE DATASET FROM "path" [AS name]`: Load external data (CSV, HDF5, Numpy, Zarr) into the current session as ephemeral tensors and a dataset view.
  - Automatically detects format from file extension (`.csv`, `.h5`, `.npy`, `.npz`, `.zarr`).
- `IMPORT DATASET FROM "path" [AS name]`: Load and normalize external data into a persistent LINAL Dataset Package.
  - Supports CSV, HDF5, Numpy, and Zarr.
- `IMPORT CSV FROM "path" AS name`: (Legacy) Auto-infer schema and load CSV into a relational dataset.
- `EXPORT CSV name TO "path"`: Save dataset to CSV.
- `SAVE DATASET name [TO "path"]`: Persist to Parquet (includes metadata/lineage).
- `LOAD DATASET name [FROM "path"]`: Restore a persisted dataset.
- `SAVE TENSOR name [TO "path"]`: Persist a tensor to JSON.
- `LOAD TENSOR name [FROM "path"]`: Restore a persisted tensor (preserves lineage).
- `LIST DATASETS [FROM "path"]`: Show available datasets in the current database context.
- `LIST TENSORS [FROM "path"]`: Show available tensors in the current storage path.
- `LIST DATASET VERSIONS <name>`: Show version history and schema evolution log for a persisted dataset.

### Scientific Data Ingestion

LINAL supports direct ingestion of multi-dimensional data:

- **HDF5**: Ingests datasets from groups; flattens them into columns.
- **Numpy**: Supports `.npy` (single vector/matrix) and `.npz` (named collections).
- **Zarr**: Supports V3 Zarr stores with recursive group traversal.

---

## 6. Instance & Session Management

### Database Management

LINAL supports multi-platform isolated instances.

```sql
CREATE DATABASE research
USE research
DROP DATABASE obsolete_db
SHOW DATABASES          -- also: SHOW ALL DATABASES
```

### RESET SESSION

Clears all in-memory registers (Tensors and Datasets) for the current session.

---

## 7. Diagnostics

### Resource Display

- `SHOW <name>`: Display contents of any resource — tensor, legacy dataset, or tensor-first dataset. Automatically materializes lazy tensors before displaying.
- `SHOW ALL` / `SHOW ALL TENSORS`: List all in-memory tensors with shapes and data.
- `SHOW ALL DATASETS`: List all legacy datasets with row/column counts.
- `SHOW DATABASES` / `SHOW ALL DATABASES`: List all database instances.
- `SHOW SCHEMA <dataset>`: Display column names and types for a legacy dataset.
- `SHOW SHAPE <name>`: Display only the shape dimensions of a tensor.
- `SHOW LINEAGE <name>`: Display the recursive computation graph that produced a tensor.
- `SHOW INDEXES [<dataset>]`: List all indexes; optionally filter to a specific dataset.

### Dataset Metadata & Versioning

- `SHOW DATASET METADATA <name>`: Display version, hash, origin, author, tags, and timestamps for a dataset (checks in-memory first, falls back to disk).
- `SHOW DATASET VERSIONS <name>`: Display the full schema evolution history for a persisted dataset.
- `LIST DATASET VERSIONS <name>`: Equivalent to `SHOW DATASET VERSIONS` — returns the same schema history output.

### Utility

- `SHOW "<string>"`: Print a string literal directly. Useful for annotating script output.

```sql
SHOW "--- Begin training phase ---"
```

### Query Planning

- `EXPLAIN <query>`: Show the logical execution plan for a SELECT query.
- `AUDIT DATASET <name>`: Perform a deep health check on the reference graph — detects dangling tensor references.

---

## 8. Server & Job Management

For remote execution and production workloads.

### Background Jobs

| Endpoint | Method | Description |
|---|---|---|
| `/jobs` | `POST` | Submit a DSL command for background execution. Returns `job_id`. |
| `/jobs` | `GET` | List all jobs and their statuses. |
| `/jobs/:id` | `GET` | Poll a specific job — returns `Pending`, `Running`, `Completed`, or `Failed`. |
| `/jobs/:id/result` | `GET` | Retrieve structured `DslOutput` for a completed job. |
| `/jobs/:id` | `DELETE` | Cancel a **Pending** job. Running or finished jobs cannot be cancelled. |

### Scheduler

Submit recurring DSL commands that execute on a fixed interval:

| Endpoint | Method | Description |
|---|---|---|
| `/schedule` | `POST` | Register a named scheduled command (`name`, `command`, `interval_secs`, optional `target_db`). |
| `/schedule` | `GET` | List all active scheduled tasks. |
| `/schedule/:id` | `DELETE` | Remove a scheduled task by ID. |

### Other Server Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Server health check. |
| `/execute` | `POST` | Execute a DSL command synchronously. |
| `/databases` | `GET` | List database instances. |
| `/databases/:name` | `POST` | Create a database instance. |
| `/databases/:name` | `DELETE` | Drop a database instance. |
| `/delivery/...` | `GET` | Read-only dataset delivery endpoints. |

Multi-tenant isolation is provided via the `X-Linal-Database: <db_name>` request header. Each request restores the previous active database after execution, so concurrent requests with different headers do not interfere.

- **Graceful Shutdown**: Server handles `SIGINT`/`SIGTERM` to safely close connections.

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
Licensed under the LinalDB Community License v1.0
