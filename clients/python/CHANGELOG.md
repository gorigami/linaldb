# Changelog

All notable changes to the `linaldb` Python client will be documented
here. See the parent repository's `CHANGELOG.md` for the engine's own
changelog and `PYTHON_R_INTEROP_PLAN.md` (repo root, until all its
checkpoints land) for the full build history.

## [0.1.0] - 2026-07-23 (unreleased, not yet published to PyPI)

Initial client, built across checkpoints 0-5 of `PYTHON_R_INTEROP_PLAN.md`:

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
