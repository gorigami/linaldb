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
| `ConstraintViolation` | *(Reserved)* Intended for type/schema constraint violations. Currently not emitted — type mismatches surface as `InvalidOp`. | Check the input types against the `SHOW SCHEMA` output. |
| `ReferenceError` | *(Reserved)* Intended for failures resolving zero-copy reference graph links. Currently not emitted — reference errors surface as `InvalidOp`. | Run `AUDIT DATASET <name>` to check for dangling references. |
| `ExecutionError` | A generic failure in the computational kernel or parallel execution. | Check for resource exhaustion or complex tensor layouts. |

---

## 2. DSL Errors (`DslError`)

DSL errors occur during the parsing or initial routing of your script commands.

### Parse Error

Happens when the command doesn't match LINAL's expected grammar.

- **Example**: `GET * FROM users` (Should be `SELECT`)
- **Example**: `INSERT INTO ds (id) VALUES ("invalid")` (Type mismatch during parsing)
- **Fix**: Refer to [DSL_REFERENCE.md](DSL_REFERENCE.md) for correct syntax and type keywords.

---

## 3. Storage Errors (`StoreError`)

Errors related to Parquet/JSON persistence or disk access.

- **`SerializationError`**: Failed to convert data to disk format.
- **`IOError`**: Permissions issue or disk full when saving to `./data`.
- **`UnsupportedFormat`**: Attempted to load a file that is not a valid Parquet or LINAL JSON.

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
