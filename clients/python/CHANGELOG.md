# Changelog

All notable changes to the `linaldb-server` Python client will be documented
here. See the parent repository's `CHANGELOG.md` for the engine's own
changelog and `PYTHON_R_INTEROP_PLAN.md` (repo root, until all its
checkpoints land) for the full build history.

## [0.1.1] - 2026-09-16

Two issues found via real-world testing in `linal-hub` (a sibling project
that drives the real, installed PyPI packages against real data):

- Fixed silent data corruption in `TensorResult.to_numpy()`: it called
  `.reshape(self.shape)` on the raw wire `data` buffer, completely ignoring
  the `strides`/`offset` fields the payload actually carries. This is wrong
  for any tensor that isn't a fresh, contiguous, zero-offset result —
  `TRANSPOSE` is zero-copy on the engine side (swaps strides, clones the
  same underlying buffer rather than copying, see `src/engine/kernels.rs`),
  so `to_numpy()` on a transposed tensor silently returned an ordinary
  reshape of the pre-transpose buffer instead of the transposed data, with
  no exception raised. `to_numpy()` now reconstructs the array respecting
  `strides`/`offset` (the same model as `numpy.lib.stride_tricks.as_strided`,
  which the engine's own tensors already use — element-counted, row-major),
  returning an owned copy, and raises `LinalError` instead of silently
  reading out of bounds if a payload's `data` is shorter than
  shape/strides/offset would require. This also changes a rank-0 (scalar)
  tensor's `.to_numpy()` from a length-1 1-D array to a true 0-d array,
  which no prior example exercised.
- `Client.execute()` now emits a `UserWarning` when a `Client` created with
  `database=None` (the default) is given a bare `USE <database>` statement.
  With no `database=` set, no `X-Linal-Database` header is sent, so per the
  documented contract (`clients/CONTRACT.md`) the statement mutates the
  server's single shared, process-wide active database rather than
  anything scoped to this client — two `Client`s in this mode running
  concurrently reliably race on each other's active database (confirmed:
  100% collision rate across 80 interleaved iterations in a live
  reproduction). This is not a behavior change — headerless ambient-session
  semantics are intentional and unchanged, matching the embedded CLI/REPL —
  it's a warning so the footgun isn't silent. Pass `database=...` to
  `connect()`/`Client()` for an isolated session, which already sends
  `X-Linal-Database` on every request and was verified to eliminate the
  race.

Requires engine `>= 0.1.74` (unchanged from 0.1.0).

## [0.1.0] - 2026-09-09

Initial client, built across checkpoints 0-5 of `PYTHON_R_INTEROP_PLAN.md`
(2026-07-23); published to PyPI as
[`linaldb-server`](https://pypi.org/project/linaldb-server/0.1.0/) on
2026-09-09 alongside the naming rename from `linaldb`:

- `connect()` / `Client.execute()` / `Client.query()` against `/execute`.
- `Client.dataset()` / `Dataset.to_arrow()` / `Dataset.to_pandas()`
  against `/delivery`, transparently handling both the native
  `FixedSizeList` and legacy JSON-string-fallback Vector/Matrix column
  encodings (see the engine's own `CHANGELOG.md` v0.1.72/v0.1.73).
- `database=` parameter on `connect()`, honored by both `/execute` (via
  the `X-Linal-Database` header) and `/delivery` (fixed after initially
  being silently ignored there — see below).
- Real end-to-end example: `examples/digit_classification.py`.

Requires engine `>= 0.1.74` (the version that fixed `USE <database>`
sent to `/execute` to actually persist, and fixed `/delivery` to honor a
non-default database — both found building this client).
