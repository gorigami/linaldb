# LINAL Error Reference

This document provides detailed information about the errors you might encounter while using the LINAL engine and how to resolve them.

---

## 1. Engine Errors (`EngineError`)

Engine errors occur during the internal execution of algebraic or data operations.

| Error | Description | Resolution |
|-------|-------------|------------|
| `NameNotFound` | Referred to a tensor variable that is not in the store. | Verify the variable name or check if the tensor was deleted. |
| `InvalidOp` | Attempted an operation that is mathematically impossible (e.g., MATMUL with incompatible shapes), or a classical linear algebra operator (`INVERSE`/`SOLVE`/`CHOLESKY`/`EIGENVALUES`/`EIGEN`) hit a matrix it can't handle — singular, non-square, or non-symmetric. These never return a silent `NaN`; they always error. Examples: `INVERSE: matrix is singular (not invertible)`, `SOLVE: matrix \`a\` is singular -- no unique solution`, `TRACE requires a square matrix, got 2x3`, `EIGENVALUES: matrix is not symmetric (entries [0][1]=2 vs [1][0]=0 differ) -- only symmetric matrices are supported today`, `CHOLESKY: matrix is not positive-definite`. Also covers `Value count mismatch: expected N, got M` (an `INSERT` with the wrong number of positional values, or naming an unknown column) and `HAVING references unknown column '...' — available: ...` (a `HAVING` clause referencing a column/alias that doesn't exist in the query's output schema). | Verify dimensions (e.g., Matrix A: 2x3, Matrix B: 3x5 for MATMUL). For a singular/non-square/non-symmetric matrix, check the actual values/shape — these are correctness errors in your data or query, not something to work around. For `INSERT`, match the value count/names to the dataset's schema (`SHOW SCHEMA <name>`). For `HAVING`, check the column/alias is actually present in the `SELECT` list. |
| `DatasetNotFound` | Referred to a dataset that does not exist in the active database. | Check your spelling or run `SHOW ALL DATASETS`. |
| `InvalidOp` (aggregates) | `ARG_MAX/ARG_MIN take two arguments`, `ARG_MAX: the \`by\` argument must be a scalar ...`, `... is NaN, which has no ordering`, `RRF: expected a numeric rank ...`, `RRF: k + rank must be positive and finite`. | Pass `ARG_MAX(value_column, by_column)` with a scalar `by`; give `RRF` a numeric rank column and a non-negative `k`. |
| `InvalidOp` (batch `SEARCH`) | `SEARCH QUERIES: query N has dimension D, but '<col>' is Vector(E)`, `tensor '<name>' has shape [...]; a batch of queries must be a 2-D matrix`, `row N of '<ds>.<col>' is ..., not a Vector`, `KEY column '<col>' not found`, `dataset '<ds>' has a column named 'score', which collides with the batch result column`. | Make every query the indexed column's dimension; use a 2-D matrix or a vector column as the query source; rename a colliding column before searching. |
| `InvalidOp` (bit vectors) | `TANIMOTO: BitVector lengths differ (70 vs 64)`, `TANIMOTO expects BitVector arguments, got VECTOR[70]`, `BIT_COUNT expects BitVector arguments, got INT` (all raised before any row is evaluated); on `INSERT`: `column 'fp' is BitVector(70), got 4 bits`, `a bit string may only contain '0' and '1'`, `a vector literal must hold only 0 and 1`. | Compare fingerprints of the same length; write `CAST("..." AS BITVECTOR(n))` with the length so a mismatch is caught up front. |
| `InvalidOp` (spectra) | `SPEC_COSINE: first spectrum: m/z values must be sorted ascending (...)`, `... has a non-finite value at position N`, `tolerance must be a finite, non-negative number`; before execution: `SPEC_COSINE takes 3 to 5 arguments`, `SPEC_COSINE: argument 1 must be a peak list Matrix(2, n), got INT`. A failing `UPDATE`/`DELETE` changes nothing. | Sort each spectrum's peaks by m/z (and drop NaN peaks) before loading; pass the peak-list columns first. |
| `InvalidOp` (sparse vectors) | `SPARSE: sparse vector indices must be increasing`, `... duplicate index N`, `... index N is out of range for dimension D`, `COSINE_SIM: dimensions differ (40 vs 30)`, `DOT expects Vector or SparseVector arguments`, `vector indexes (IVF, HNSW) need a dense Vector column`, `SEARCH: column 's' is a SparseVector ... add PREFILTER`; on load, `column 's' is a struct: a SparseVector or quantized vector column needs field metadata ...`. | Sort and deduplicate indices; search sparse columns with `SEARCH ... PREFILTER`; build Arrow columns with `linaldb.sparse_array`. |
| `InvalidOp` (quantized vectors) | `column 'e': element N (1e6) is outside the F16 range (±65504)`, `column 'e' is Vector(2, F16), got a vector of length 1`, `unknown vector encoding 'F8' (use F16 or I8)`. | Scale values into range or use `I8` (which scales per vector); match the declared length. |
| `InvalidOp` (`APPROX`) | `PREFILTER ... APPROX needs an HNSW index on 'e'`. | `CREATE VECTOR INDEX ON ds(e) USING HNSW`, or drop `APPROX` for the exact search. |
| `InvalidOp` (UPDATE) | `UPDATE 't' row N: column 'id' is Int, but the new value is Float`, `... is not nullable, but the new value is NULL`. Nothing is changed. | Cast the assigned value to the column's type, or make the column nullable. |
| `InvalidOp` (SORTED index, PREFILTER) | `SORTED index supports Int, Float, Float64 and String columns`, `SORTED index: NaN has no ordering`, `PREFILTER: unknown query column 'q.nope'`, `PREFILTER: unknown column 'nope' in dataset 'lib'`. | Index a scalar column; reference query values as `<query dataset>.<column>`. |
| `InvalidOp` (load from memory) | `column '<c>' has unsupported Arrow type ...` (with a hint for float64 vectors and variable-length lists), `column '<c>' row N is NaN -- NaN and infinite values are rejected`, `dataset '<name>' already exists`. Raised by `Db.load_numpy`/`Db.load_arrow` as `LinalError`. | Cast vectors to `float32` and fixed-size lists; clean or drop non-finite values before loading; load under a new name or drop the old dataset. |
| `DatasetError` | Wraps a `DatasetStoreError` from the in-memory dataset store — e.g. `NameAlreadyExists` (creating/loading a dataset under a name already in use), `DatasetNotFound`, `InvalidDataset`. | For `NameAlreadyExists`, drop/rename the existing dataset first, or pick a different name. |
| `Store` | Wraps a `StoreError` from the in-memory *tensor* store: `ShapeMismatch`, `TensorNotFound`, `InvalidTensor`. Distinct from persistence/disk errors — see §3 below. | Check the tensor's shape/existence with `SHOW SHAPE <name>` / `SHOW ALL TENSORS`. |
| `ConstraintViolation` | *(Reserved)* Intended for type/schema constraint violations. Currently not emitted — type mismatches surface as `InvalidOp`. | Check the input types against the `SHOW SCHEMA` output. |
| `ReferenceError` | *(Reserved)* Intended for failures resolving zero-copy reference graph links. Currently not emitted — reference errors surface as `InvalidOp`. | Run `AUDIT DATASET <name>` to check for dangling references. |
| `ExecutionError` | A generic failure in the computational kernel or parallel execution. | Check for resource exhaustion or complex tensor layouts. |

---

## 2. DSL Errors (`DslError`)

DSL errors occur during the parsing or initial routing of your script commands.

### Parse Error

Happens when the command doesn't match LINAL's expected grammar. The engine runs a full Logos lexer + recursive-descent parser first, which produces a structured `ParseError { offset, msg }` with a byte offset and expectation detail — as of v0.1.50, that detail survives all the way to the message you see, instead of being discarded in favor of a generic "Unknown command":

```
[line 1] Parse error: expected a statement keyword, found identifier `GET` (at byte 0)
```

- **Example**: `GET * FROM users` → "expected a statement keyword, found identifier `GET`" (`GET` is not a LINAL keyword)
- **Example**: `DEFINE t AS TENSOR(2,2) VALUES [...]` → "expected `[`, found `(`" (old paren syntax; use brackets: `TENSOR [2, 2]`)
- **Fix**: Refer to [DSL_REFERENCE.md](DSL_REFERENCE.md) for correct syntax and type keywords. The message tells you what token the parser expected and what it actually found, plus the byte offset into the line — use that to locate the problem directly instead of scanning the whole line.

All `Statement` variants are handled in the typed pipeline — there is no legacy string-dispatch fallback. Comment-only lines (`--`, `#`, `//`) and blank lines are the only inputs that fail to parse without becoming an error — they're recognized before the structured error would otherwise surface and treated as a no-op.

### Engine Error (from DSL)

Wraps an `EngineError` with a source line number. Occurs when the grammar is valid but the operation fails at runtime (e.g., shape mismatch in `MATMUL`). Actual `Display` format:

```
[line 5] Engine error: Invalid operation: shape mismatch: [3] vs [4]
```

---

## 3. Storage Errors (`StorageError`)

Errors related to Parquet/JSON persistence or disk access (`src/core/storage.rs`) — distinct from the in-memory tensor `StoreError` covered under `EngineError::Store` in §1. These surface through `SAVE`/`LOAD`/`IMPORT`/`EXPORT`/`LIST` commands (`src/dsl/persistence.rs`), wrapped as a `DslError::Parse` with the `StorageError`'s `Display` text as the message — not as a `DslError::Engine`.

| Error | Description |
|-------|-------------|
| `Io` | Permissions issue or disk full when reading/writing to `./data` (or the configured `data_dir`). |
| `Serialization` | Failed to convert data to/from JSON (schema, stats, lineage, manifest, or legacy metadata files). |
| `Parquet` | Failed to read or write the dataset's `data.parquet` file. |
| `Arrow` | Failed converting between LINAL's row/tuple representation and Arrow's columnar `RecordBatch`. |
| `DatasetNotFound` | Attempted to `LOAD`/read a dataset package that doesn't exist on disk. |
| `TensorNotFound` | Attempted to `LOAD`/read a tensor JSON file that doesn't exist on disk. |

### Write-ahead log errors

These only occur with `[wal] enabled = true` (see DSL_REFERENCE.md §8).

| Message (prefix) | Cause | Fix |
|---|---|---|
| `CHECKPOINT requires the write-ahead log` | `CHECKPOINT` with the WAL disabled. | Set `[wal] enabled = true` in `linal.toml`, or don't checkpoint. |
| `Database '<db>' failed WAL recovery on startup and is unavailable: ...` | Replay at startup failed. The cause follows: typically a `LOAD`/`IMPORT` whose file `has changed` since it was logged, or a replayed statement that now errors. | Restore the original file and restart. Or move `{data_dir}/<db>/wal.log` aside and restart, which loses changes since the last checkpoint. |
| `... corrupt record at line N` | `wal.log` has an unreadable record before its last line. | Same as above. A torn *final* record is repaired automatically. |
| `The statement was applied but could not be written to the WAL` / `has un-logged changes because a WAL write failed` | An append failed (e.g. a full disk) after the statement took effect in memory. | Free space, then run `CHECKPOINT`. It snapshots the current state and resumes logging. |

---

## 4. Common Troubleshooting

### "My command does nothing"

LINAL script requires a NEWLINE or semicolon-equivalent completion. If you are in the REPL and see no output, verify your parentheses are balanced.

### "Dangling Reference" Warning

If `SHOW <dataset>` displays a warning, it means one of your columns points to a `TensorId` that was manually removed from the store.

- **Fix**: Re-attach the data using `ATTACH <tensor> TO <dataset>.<column>`.

### "Backend Fallback"

LINAL automatically falls back to scalar execution if SIMD is not supported or the tensor layout is too complex. This is transparent but might be slower for massive datasets.

---

**LINAL**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
