# LINAL Error Reference

This document provides detailed information about the errors you might encounter while using the LINAL engine and how to resolve them.

---

## 1. Engine Errors (`EngineError`)

Engine errors occur during the internal execution of algebraic or data operations.

| Error | Description | Resolution |
|-------|-------------|------------|
| `NameNotFound` | Referred to a tensor variable that is not in the store. | Verify the variable name or check if the tensor was deleted. |
| `InvalidOp` | Attempted an operation that is mathematically impossible (e.g., MATMUL with incompatible shapes). | Verify dimensions (e.g., Matrix A: 2x3, Matrix B: 3x5 for MATMUL). |
| `DatasetNotFound` | Referred to a dataset that does not exist in the active database. | Check your spelling or run `SHOW ALL DATASETS`. |
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
