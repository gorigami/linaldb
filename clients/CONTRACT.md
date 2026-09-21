# LINALDB Client Wire Contract

This is the contract both `clients/python/` and `clients/r/` implement
against. It exists so the two implementations can be built independently
(checkpoints 1-2 and 3-4 of the now-completed `PYTHON_R_INTEROP_PLAN.md`
effort) without silently drifting apart. If either client's actual
behavior disagrees with this document, that's a bug in the client, the
doc, or both — fix the disagreement, don't just pick one side.

Everything here was verified directly against the server implementation
(`src/server/mod.rs`, `src/core/storage.rs`) as of engine v0.1.74, not
assumed from the DSL reference alone. §2 (`/execute/batch`) and the
`USE`+header rejection in §1 were added and verified against a live
server afterward, alongside `docs/ARCHITECTURE.md`'s matching update.

## 1. `POST /execute` — ad-hoc DSL execution

Request: raw DSL text as the request body (`Content-Type: text/plain`
preferred; a legacy `{"command": "..."}` JSON body is still accepted but
deprecated server-side — clients should always send plain text).

Query params: `?format=json` (recommended for clients — default is
`toon`, a human-oriented text format not meant for programmatic parsing).
`?format=arrow` is a third, additive option (`PERFORMANCE_OPTIMIZATION_PLAN.md`
Phase 3): a binary Arrow IPC stream (`Content-Type:
application/vnd.apache.arrow.stream`), decodable with any standard Arrow
reader (e.g. `pyarrow.ipc.open_stream`), for a `Table` result specifically
— measured ~80-180x faster to produce and ~2.4x smaller on the wire than
JSON for a representative bulk query result (10k rows × a `Vector(128)`
column), which is the workload this option exists for. Only a *successful*
`Table` result actually comes back as Arrow bytes; anything else under
`?format=arrow` (an execution error, or a non-tabular success like a bare
`Message`/`Tensor`) falls back to a `format=json`-shaped JSON body instead
(`Content-Type: application/json`) — a client requesting `arrow` must still
check the response `Content-Type` before attempting to decode it as Arrow.

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

**A `USE <db>` statement combined *with* `X-Linal-Database` on this
endpoint is a client error (`400`), not a silent no-op.** A single
statement has no "rest of the request" for `USE` to persist across, so a
header-bearing `/execute` call whose body is `USE <db>` is rejected
outright — `{"status":"error","error":"USE has no persisting effect
when X-Linal-Database is set on a single-statement request..."}` — rather
than reporting `"Switched to database 'db'"` and reverting it before the
response goes out (which is what it silently did before this contract
version). A client that needs `USE`/`CREATE DATABASE` to control several
subsequent statements should send them all to `POST /execute/batch`
(§2) instead of one `/execute` call per statement.

Response body (`format=json`):

```json
{
  "status": "ok" | "error",
  "result": <DslOutput JSON, present iff status is "ok" and the command produced output>,
  "error": <string, present iff status is "error">
}
```

`result` is present iff the command produced output — e.g. a headerless
`USE <db>` returns `{"status":"ok","result":{"Message":"Switched to
database '<db>'"}}`, `format=json` empirically confirmed against a live
v0.1.72 server. (A header-bearing `USE <db>` request instead returns the
`400` error described above.)

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
  FFT signal`. Shape (from `core::tensor::Tensor`'s derived `Serialize`):
  `{"id": ..., "shape": {"dims": [...]}, "data": [<f32 array>],
  "metadata": {...}, "strides": [...], "offset": <int>}`. **`data` is the
  tensor's raw underlying buffer, not necessarily its logical values in
  row-major order** — a zero-copy op like `TRANSPOSE` swaps `strides`
  (and adjusts `offset`) without copying `data`, so `data` can be longer
  than the logical element count and/or in a different order than `shape`
  implies on its own. `strides`/`offset` are element counts (not bytes),
  row-major convention: the logical element at multi-index `(i, j, ...)`
  is `data[offset + i*strides[0] + j*strides[1] + ...]` — the same model
  `numpy.lib.stride_tricks.as_strided` uses. A client MUST reconstruct
  logical values via `strides`/`offset`, not via a plain reshape of
  `data` — confirmed by a real, shipped bug: `clients/python`'s
  `TensorResult.to_numpy()` originally did a plain reshape and silently
  returned wrong values for any transposed/strided result (fixed in
  `linaldb-server` 0.1.1). Verified against a live server for `shape`/
  `strides`/`offset`/`data` together (a real `TRANSPOSE` over HTTP,
  reconstructed correctly) — see `clients/python/tests/
  test_client_integration.py::test_transpose_over_http_to_numpy_end_to_end`.

A client's `execute()` MUST raise/throw on `status: error`, surfacing the
server's real `error` string — never synthesize a generic "request
failed" message when the server sent a specific one.

## 2. `POST /execute/batch` — batch DSL execution

Request: a whole multi-statement script as the request body
(`Content-Type: text/plain`), in the exact same format `linal run`
accepts from a `.lnl` file — one statement per line, or a statement
spanning multiple lines (it ends once its parentheses balance out, not
at the next line break), with `#`/`--`/`//` comment lines skipped between
statements. Statements execute in order, under one write-lock hold, and
the batch stops at the first error.

Query params: `?format=json` (same as `/execute`).

Headers: `X-Linal-Database: <name>`, same semantics as `/execute` but
scoped to the whole batch instead of one statement — the server restores
the previously active database once, after the batch finishes (or stops
on an error), not after each individual statement. This is why `USE`/
`CREATE DATABASE` inside a batch body works exactly as written and
*does* persist for the rest of that batch: nothing restores the active
database mid-batch, only at the very end. Send a script that needs `USE`
to control several subsequent statements here instead of as separate
`/execute` calls (where the identical combination is now a `400` error —
see §1).

Response body (`format=json`):

```json
{
  "status": "ok" | "error",
  "statements": [
    {
      "statement": "<the exact joined statement text that ran>",
      "status": "ok" | "error",
      "result": <DslOutput JSON, present iff status is "ok" and the statement produced output>,
      "error": <string, present iff status is "error">
    },
    ...
  ],
  "error": <string, present only for a script-level failure before any statement ran -- e.g. unbalanced parentheses, an empty body, or a body over the same MAX_COMMAND_LENGTH limit /execute enforces>
}
```

`status` at the top level is `"ok"` only if every statement in
`statements` succeeded. `statements` contains one entry per statement
that actually ran — if the script stops at the Nth statement's error,
entries `N+1..` are never attempted and never appear in the array at
all (not present-with-an-error-placeholder — simply absent).

A client's batch-execute method MUST surface both levels of failure
distinctly: a top-level `error` (the script itself was rejected, nothing
ran) versus a `statements[i].status == "error"` entry (the script ran
partway, and this is where it stopped) are different situations worth
telling apart, not both collapsed into one generic exception.

## 3. `/delivery/*` — read-only dataset export

Mounted per-dataset at `/delivery/datasets/:name/`:

- `manifest.json` — format versions + entrypoints (`{"formats":
  {"parquet": "data.parquet"}, ...}`).
- `schema.json` — **the authoritative column typing for a client to
  trust**, not something to infer from the Parquet file's physical type.
  Each column: `{"name": ..., "value_type": "Int"|"Float"|"Float64"|
  "String"|"Bool"|"Complex" | {"Vector": <dim>} | {"Matrix": [<rows>,
  <cols>]}, "shape": {"dims": [...]}, "nullable": bool}`.
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

### `Complex` column encoding in `data.parquet` (Phase 3)

A `schema.json` column with `value_type: "Complex"` has **no native Arrow
encoding at all** (unlike Vector/Matrix's `FixedSizeList` above) — it
always uses the legacy JSON-string fallback: an Arrow `Utf8` column where
each non-null cell is the literal text `{"Complex":[re,im]}`. A client
must always take the JSON-parsing path for a `Complex` column, never the
native-list path (there is no "common case" native encoding to fall back
from, unlike Vector/Matrix).

## 4. The tagged `Value` encoding (used throughout `/execute` results)

`core::value::Value` derives plain (externally-tagged) serde
`Serialize`. Every scalar cell in a `Table`/`TensorTable` result is one
of:

| Wire form | Client-native equivalent |
|---|---|
| `{"Float": 1.5}` | float (32-bit) |
| `{"Float64": 1.5}` | float (64-bit / double) |
| `{"Int": 5}` | int |
| `{"String": "x"}` | string |
| `{"Bool": true}` | bool |
| `{"Vector": [1.0, 2.0, 3.0]}` | list/array of floats |
| `{"Matrix": [[1.0, 0.0], [0.0, 1.0]]}` | nested list / 2D array |
| `{"Complex": [3.0, 4.0]}` | `[re, im]` pair — see below |
| `"Null"` | null / NA / None (unit variant — **not** `{"Null": ...}`) |

Note the last row: `Value::Null` is a unit enum variant, so serde emits
the bare string `"Null"`, not an object — a client's unwrapper must check
for that string form specifically, not assume every cell is a
single-key object.

**`Complex` (added Phase 3 of `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md`)**: wraps
`num_complex::Complex64` (`num-complex`'s own `Serialize`, `(re, im).serialize(...)`),
so the payload is a plain 2-element JSON array `[re, im]` — **not** an
object with `re`/`im` keys. `re`/`im` are always full `f64`. A client with
no native complex type should represent this as a `(float, float)` pair or
an equivalent small struct; there is no dedicated complex type in the wire
contract beyond this array shape. `Value::equals()`'s real-equality
semantics apply (`3+4i == 3+4i`), but there is **no ordering** — a
`Complex` cell can never be the sort key for a client-side sort matching
this engine's own (`ORDER BY <complex column>` is a hard server-side
error, not a silently arbitrary order). The corresponding `ValueType` tag
is the bare string `"Complex"` (a unit variant, same convention as
`"Null"` in `schema.json`'s `value_type` field), not `{"Vector": n}`/
`{"Matrix": [r, c]}`'s parameterized object form.

## 5. Error semantics

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

## 6. What this contract deliberately does not cover yet

- `/jobs` and `/schedule` (background execution, recurring tasks) —
  real server endpoints (see `docs/ARCHITECTURE.md` §5), but out of scope
  for the Tier A clients this round. A client may add thin wrappers later
  without needing a new major version of this contract.
- Tier B (in-process `pyo3`/`extendr` bindings) — a different, lower-level
  contract (direct Rust struct access, no HTTP/JSON at all), not this
  HTTP+Parquet one. Now implemented — see
  [`EMBEDDED_CONTRACT.md`](EMBEDDED_CONTRACT.md) for the result shapes
  `clients/python-embedded` and `clients/r-embedded` implement against.
