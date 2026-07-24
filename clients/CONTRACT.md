# LINALDB Client Wire Contract

This is the contract both `clients/python/` and `clients/r/` implement
against. It exists so the two implementations can be built independently
(checkpoints 1-2 and 3-4 of `PYTHON_R_INTEROP_PLAN.md`) without silently
drifting apart. If either client's actual behavior disagrees with this
document, that's a bug in the client, the doc, or both — fix the
disagreement, don't just pick one side.

Everything here was verified directly against the server implementation
(`src/server/mod.rs`, `src/core/storage.rs`) as of engine v0.1.74, not
assumed from the DSL reference alone.

## 1. `POST /execute` — ad-hoc DSL execution

Request: raw DSL text as the request body (`Content-Type: text/plain`
preferred; a legacy `{"command": "..."}` JSON body is still accepted but
deprecated server-side — clients should always send plain text).

Query params: `?format=json` (recommended for clients — default is
`toon`, a human-oriented text format not meant for programmatic parsing).

Headers: `X-Linal-Database: <name>` to target a non-default database for
that one request only (the server reverts to whatever was active before
once this request finishes, so concurrent requests targeting different
databases via the header can't clobber each other's context).

**A request with no `X-Linal-Database` header operates on — and can
change — the server's session-wide active database.** A plain `USE <db>`
statement sent with no header genuinely persists: every subsequent
headerless request sees the new active database, exactly like the
embedded CLI/REPL (confirmed against a live v0.1.74 server; **this was
a real, severe bug before v0.1.74** — the restore-after-request logic
described above used to run unconditionally, silently undoing a
headerless request's own `USE`, so the whole session-level `USE`
workflow was a no-op over HTTP). A client wanting to pin every request to
one database regardless of ambient session state should pass the header
on every request (as both `clients/python`'s and `clients/r`'s
`database=`/`database` connection parameter already do) rather than rely
on a one-time `USE`.

Response body (`format=json`):

```json
{
  "status": "ok" | "error",
  "result": <DslOutput JSON, present iff status is "ok" and the command produced output>,
  "error": <string, present iff status is "error">
}
```

`result` is present iff the command produced output — e.g. `USE <db>`
returns `{"status":"ok","result":{"Message":"Switched to database
'<db>'"}}`, `format=json` empirically confirmed against a live v0.1.72
server.

`result`, when present, is one of `DslOutput`'s serde-tagged variants:

- `{"Message": "<string>"}` — informational text (`CREATE DATASET`/`USE`
  confirmations, etc.).
- `{"Table": {...}}` / `{"TensorTable": [...]}` — row-oriented query
  results. **Verified exact shape** (`SELECT * FROM probe` against a
  2-column, 2-row dataset with one `NULL` vector cell, v0.1.72):

  ```json
  {"status":"ok","result":{"Table":{
    "id": 0,
    "schema": {
      "fields": [
        {"name":"id","value_type":"Int","nullable":false,"is_lazy":false},
        {"name":"emb","value_type":{"Vector":3},"nullable":true,"is_lazy":false}
      ],
      "field_indices": {"id":0,"emb":1}
    },
    "rows": [
      {"schema": { /* same shape repeated per row */ }, "values": [{"Int":1},{"Vector":[1.0,2.0,3.0]}]},
      {"schema": { /* same shape repeated per row */ }, "values": [{"Int":2},"Null"]}
    ],
    "metadata": {
      "name": "Query Result", "created_at": "...", "updated_at": "...",
      "version": 1, "row_count": 2,
      "column_stats": {"id": {"value_type":"Int","null_count":0,"min":{"Int":1},"max":{"Int":2}}, "emb": {...}},
      "schema": { /* same shape again */ }, "extra": {}
    }
  }}}
  ```

  Two things a client must not get wrong here: **(a)** each row's cells
  are under a `values` key, not the row object itself — `row.values[i]`,
  not `row[i]`; **(b)** the per-column schema is repeated three times
  (top-level, per-row, and inside `metadata`) — always redundant across
  a single response, a client only needs to read it once (top-level is
  simplest) rather than per-row.
- `{"Tensor": {...}}` / `{"LazyTensor": {...}}` — a standalone tensor
  result (not a table), from tensor-DSL statements like `LET spectrum =
  FFT signal`. Structural shape (from `core::tensor::Tensor`'s derived
  `Serialize`, not independently re-verified against a live response —
  do that before building tensor-result support in a client): `{"id":
  ..., "shape": {"dims": [...]}, "data": [<flat f32 array, row-major>],
  "metadata": {...}, "strides": [...], "offset": <int>}`.

A client's `execute()` MUST raise/throw on `status: error`, surfacing the
server's real `error` string — never synthesize a generic "request
failed" message when the server sent a specific one.

## 2. `/delivery/*` — read-only dataset export

Mounted per-dataset at `/delivery/datasets/:name/`:

- `manifest.json` — format versions + entrypoints (`{"formats":
  {"parquet": "data.parquet"}, ...}`).
- `schema.json` — **the authoritative column typing for a client to
  trust**, not something to infer from the Parquet file's physical type.
  Each column: `{"name": ..., "value_type": "Int"|"Float"|"String"|"Bool"
  | {"Vector": <dim>} | {"Matrix": [<rows>, <cols>]}, "shape": {"dims":
  [...]}, "nullable": bool}`.
- `stats.json` — per-column min/max/mean/null_count/sparsity, row count.
- `data.parquet` — the actual data.

### Vector/Matrix column encoding in `data.parquet` (as of v0.1.72)

A `schema.json` column with `value_type: {"Vector": n}` or `{"Matrix":
[r, c]}` is encoded in the Parquet file one of two ways, and **a client
must handle both**:

1. **Native** (the common case: a fully-populated column with no actual
   `NULL` values) — a real Arrow `FixedSizeList<Float32>` (Vector) or
   `FixedSizeList<FixedSizeList<Float32>>` (Matrix) column. Any Arrow/
   Parquet-aware library (`pyarrow`, R's `arrow`) reads this natively as
   numeric list-of-floats / nested list-of-lists — no special handling
   needed beyond what the library already does for those Arrow types.
2. **Legacy JSON-string fallback** (only when the column contains at
   least one actual `NULL`, or — for datasets written before v0.1.72 —
   unconditionally) — an Arrow `Utf8` column where each non-null cell is
   the literal text of the tagged JSON encoding, e.g. `{"Vector":
   [1.0,2.0,3.0]}`, and an actual SQL `NULL` cell is an Arrow-null string
   (not the text `"null"`). A client must detect this case (the
   Parquet/Arrow physical type for that column is `Utf8`/`string`, not
   `FixedSizeList`, even though `schema.json` still reports `Vector`/
   `Matrix`) and parse each non-null cell as JSON, unwrapping the
   `{"Vector": [...]}` / `{"Matrix": [[...]]}` tagging into a plain
   numeric list — a client must never surface the raw tagged-JSON string
   to the end user as if it were the column's real content.

Why this dual encoding exists at all: a nullable `FixedSizeList` Parquet
column round-trips correctly through this engine's own arrow-rs-based
reader but is rejected by `pyarrow` (`ArrowInvalid: Expected all lists to
be of size=N but index M had size=0`) — a cross-library Parquet encoding
disagreement, not a bug in either reader in isolation. See the v0.1.72
`CHANGELOG.md` entry for the full root-cause.

## 3. The tagged `Value` encoding (used throughout `/execute` results)

`core::value::Value` derives plain (externally-tagged) serde
`Serialize`. Every scalar cell in a `Table`/`TensorTable` result is one
of:

| Wire form | Client-native equivalent |
|---|---|
| `{"Float": 1.5}` | float |
| `{"Int": 5}` | int |
| `{"String": "x"}` | string |
| `{"Bool": true}` | bool |
| `{"Vector": [1.0, 2.0, 3.0]}` | list/array of floats |
| `{"Matrix": [[1.0, 0.0], [0.0, 1.0]]}` | nested list / 2D array |
| `"Null"` | null / NA / None (unit variant — **not** `{"Null": ...}`) |

Note the last row: `Value::Null` is a unit enum variant, so serde emits
the bare string `"Null"`, not an object — a client's unwrapper must check
for that string form specifically, not assume every cell is a
single-key object.

## 4. Error semantics

- Network/connection failure (server unreachable): client-native
  exception (e.g. Python `ConnectionError`, R condition), not swallowed.
- `status: "error"` in a 200 response: raise/throw with the server's
  `error` string verbatim.
- HTTP non-2xx (e.g. 400 for an empty/oversized command, see
  `MAX_COMMAND_LENGTH` in `src/server/mod.rs`): treat the same as
  `status: "error"` if the body parses as the standard error shape,
  otherwise surface the raw HTTP status + body.
- Query timeout (server enforces `QUERY_TIMEOUT_SECS = 30` per request):
  arrives as a normal `status: "error"` response, not a connection drop —
  no special client handling needed beyond the standard error path.

## 5. What this contract deliberately does not cover yet

- `/jobs` and `/schedule` (background execution, recurring tasks) —
  real server endpoints (see `docs/ARCHITECTURE.md` §5), but out of scope
  for the Tier A clients this round. A client may add thin wrappers later
  without needing a new major version of this contract.
- Tier B (in-process `pyo3`/`extendr` bindings) — a different, lower-level
  contract (direct Rust struct access / Arrow C Data Interface), not this
  HTTP+Parquet one. See `PYTHON_R_INTEROP_PLAN.md`'s design decisions for
  why that's a separate, later effort.
