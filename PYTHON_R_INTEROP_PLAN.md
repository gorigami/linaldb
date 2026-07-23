# Python / R Interop — Tracked Plan

Started 2026-07-23. Follow-up to the CLI/server alignment audit (v0.1.71)
and the Vector/Matrix → Arrow `FixedSizeList` Parquet fix (v0.1.72), both
done as prep work once this initiative started.

**Why this exists**: LINALDB's mission is bridging relational data
engineering and scientific computing, but today the *only* way to use the
engine is its own DSL via the CLI or a raw HTTP call — there is no
supported way to reach it from Python or R, the two languages that
audience actually works in day to day. This plan builds that bridge.

**Prior discussion outcome (recorded in full in project memory, summarized
here)**: two architectural tiers were identified —

- **Tier A** (this plan): a thin client per language over the existing
  `POST /execute` (DSL in, JSON/TOON out) and `/delivery/*` (real Parquet
  files) HTTP surface. No compiled extension, ships against the server
  as-is, days not weeks of work.
- **Tier B** (sketched, *not* built in this plan): in-process `pyo3`/
  `extendr` bindings embedding `TensorDb` directly, with Arrow C Data
  Interface zero-copy handoff. Real packaging cost (`maturin`, CRAN,
  cross-compiling this crate's native deps like `hdf5`/`zarrs`). A
  credible later step once Tier A proves the contract out, not scheduled
  here.

User's explicit choices for this round: **both Python and R in parallel**
(one shared HTTP/Parquet contract, two thin wrappers around it), and
prioritize the **`/delivery` saved-dataset Parquet export path** over
ad-hoc `/execute` query-result ergonomics. That prioritization is what
surfaced the v0.1.72 bug (Vector/Matrix columns serializing as JSON
strings in Parquet) — fixed before any client code was written, since
building the client's first workload against a known-broken wire format
would have been the wrong order.

---

## Design decisions (confirmed before implementation)

1. **Location in the repo**: new top-level `clients/` directory, sibling
   to `tools/`, `examples/`, `scripts/` — `clients/python/` and
   `clients/r/`. Both live in this one tracked repo/CHANGELOG, not a
   separate repo, so they version alongside the engine they depend on.
2. **`examples/` stays `.lnl`-only** (see `examples/README.md`'s existing,
   explicit convention) — any Python/R example scripts this plan produces
   live under `clients/python/examples/` and `clients/r/examples/`
   instead, not in the shared `examples/` directory.
3. **Python dependencies**: `requests` (HTTP) + `pyarrow` (required —
   parsing the delivered Parquet is the core value, not optional) +
   `pandas` as an *extra* (`linaldb[pandas]`) for `.to_pandas()`
   convenience, so a polars-only user isn't forced to install pandas.
   Packaged with a standard `pyproject.toml` (no compiled extension —
   pure Python, matches the Tier A no-build-step goal).
4. **R dependencies**: `httr2` (HTTP) + `jsonlite` (JSON) + `arrow`
   (Parquet) as `Imports`. Standard R package layout (`DESCRIPTION`,
   `NAMESPACE`, `R/`, `man/`, `tests/testthat/`).
5. **Client package versioning**: each client gets its own version
   (starting `0.1.0`), independent of the Rust crate's `Cargo.toml`
   version, but each README states the minimum compatible engine version
   (`>= 0.1.72`, for the `FixedSizeList` fix this plan's first real
   workload depends on).
6. **Wire contract, matched to what's actually implemented** (verified
   against `src/server/mod.rs` and `src/core/storage.rs`, not assumed):
   - `/execute` returns `{"status": "ok"|"error", "result": <DslOutput
     JSON>, "error": <string>}`. `DslOutput::Table`/`TensorTable` rows
     serialize through the tagged `Value` enum (`{"Vector": [...]}`,
     `{"Int": 5}`, etc.) — the client's job is unwrapping that tagging
     into native types / a DataFrame column.
   - `/delivery/datasets/:name/{manifest,schema,stats}.json` +
     `data.parquet` — the schema.json's `value_type` field (now correctly
     `{"Vector": n}` / `{"Matrix": [r, c]}` post-v0.1.72, not `"String"`)
     is what a client should trust for column typing, not sniffing the
     Parquet file itself.
   - Nulls inside a Vector/Matrix column mean the *server* fell back to
     the legacy JSON-string Parquet encoding for that whole column (see
     v0.1.72's CHANGELOG entry) — the client must handle both encodings
     when reading `/delivery` Parquet directly (real `FixedSizeList` vs.
     a string column of JSON), not assume one.
7. **Testing strategy**: no CI harness exists yet to auto-spin-up the
   Rust server for client test suites, so integration tests in both
   clients launch `linal serve --port <test-port>` as a subprocess
   fixture (build once via `cargo build --release --bin linal` at the
   start of the client test run, reuse the binary), point the client at
   it, and tear it down after — mirrors how this session manually
   verified v0.1.71/v0.1.72 by hand. Unit tests (response-unwrapping
   logic, error handling) run against fixture JSON/Parquet files, no
   server needed.

---

## Checkpoints

Each checkpoint: implement + test + doc update together, `cargo test`
(or the client language's test runner) green before moving on. One
checkpoint may span more than one commit but should land as a coherent
working state.

- [x] **0. Scaffolding + shared contract doc** — **Done 2026-07-23**
  - `clients/python/` (`pyproject.toml`, `linaldb/__init__.py`, `tests/`,
    `examples/`) and `clients/r/` (`DESCRIPTION`, `NAMESPACE`,
    `R/linaldb-package.R`, `tests/testthat/`, `examples/`) skeletons —
    packaging metadata only, no functional client code yet. Verified the
    Python package actually imports (`import linaldb` → `__version__`);
    R package not yet loadable-checked (no `devtools`/R toolchain
    available in this environment — do that before checkpoint 3 starts).
  - `clients/CONTRACT.md` written and, importantly, **fact-checked against
    a live v0.1.72 server** rather than written from the DSL reference
    alone — an initial draft of the `Table` JSON shape was wrong (assumed
    a flat `rows: [[...]]` array; the real shape nests cells under a
    `values` key per row and repeats the schema three times per
    response). Also confirmed empirically: `Value::Null` serializes as
    the bare string `"Null"`, not `{"Null": ...}` — easy to get wrong
    from the enum definition alone since serde's unit-variant-in-an-
    externally-tagged-enum behavior isn't obvious without checking.
  - Added `.gitignore` entries for both clients (`__pycache__/`,
    `*.egg-info/`, `.pytest_cache/`, R's `.Rhistory`/`.RData`/etc.).
  - No engine changes; `cargo build`/`cargo test --lib` reconfirmed
    unaffected by the new `clients/` directory.

- [ ] **1. Python client core (`/execute`)**
  - `linaldb.connect(url, database=None) -> Client`.
  - `Client.execute(dsl: str) -> ExecuteResult` — unwraps the tagged
    `Value` JSON into native Python types (`Vector` → `list[float]` or
    `numpy.ndarray`, `Matrix` → nested list/`numpy.ndarray`, `Null` →
    `None`); raises `LinalError` with the server's real error message on
    `status: error`, not a generic HTTP exception.
  - `Client.query(dsl: str) -> pandas.DataFrame` convenience wrapper
    (requires the `pandas` extra).
  - Unit tests against fixture JSON (all `Value` variants, both
    `status: ok`/`error`, malformed/timeout responses). Integration test
    against a real `linal serve` subprocess running a small real script.

- [ ] **2. Python client dataset export (`/delivery`)**
  - `Client.dataset(name) -> Dataset` — fetches `manifest.json`/
    `schema.json`/`stats.json`.
  - `Dataset.to_pandas() -> pandas.DataFrame` / `Dataset.to_arrow() ->
    pyarrow.Table` — reads `data.parquet` directly via `pyarrow`,
    trusting `schema.json`'s `value_type` for any column where the
    Parquet physical type is ambiguous between "native FixedSizeList" and
    "legacy JSON-string fallback" (design decision 6's null case).
  - Real end-to-end integration test: start a real server, `SAVE` a
    dataset with a non-null Vector/Matrix column (expect real numeric
    columns back) *and* one with an actual-null Vector column (expect the
    JSON-string fallback, and that `to_pandas()` still unwraps it
    correctly rather than leaking the raw JSON text) — exercises both
    v0.1.72 code paths from the client side, not just the engine side.

- [ ] **3. R client core (`/execute`)**
  - Mirrors checkpoint 1: `linal_connect(url, database = NULL)`,
    `linal_execute(conn, dsl)` unwrapping the same tagged `Value` JSON via
    `jsonlite`, condition-based error handling (an R error condition
    carrying the server's real message, not a bare `httr2` HTTP error).
  - `testthat` unit tests mirroring the Python fixture set; integration
    test against the same `linal serve` subprocess pattern.

- [ ] **4. R client dataset export (`/delivery`)**
  - Mirrors checkpoint 2: `linal_dataset(conn, name)`, reading
    `data.parquet` via the `arrow` package, same schema.json-trusting /
    dual-encoding handling, same non-null + actual-null integration test
    pair.

- [ ] **5. Real end-to-end example, both languages**
  - Per this project's own recurring lesson (a real workflow against real
    data finds bugs isolated unit tests don't — confirmed four separate
    times across the PBMC/HDF5/GW rounds, and again by v0.1.72 itself),
    build one real example per client under `clients/python/examples/`
    and `clients/r/examples/`: start a real `linal serve`, load a
    non-trivial real dataset (reuse an existing real fixture — e.g. the
    HDF5 digit-classification or GW data already checked into
    `examples/data/` — rather than inventing synthetic data), query it
    via `/execute`, export it via `/delivery`, and confirm the numbers
    match between the two paths.
  - Explicitly budget time in this checkpoint for **hidden bugs this
    surfaces** — document them the same way every prior real-example
    round did (what broke, root cause, fix, which layer). This is the
    checkpoint most likely to find something, based on this project's own
    track record.

- [ ] **6. Docs pass**
  - `README.md`: new "Python / R Clients" section (install + minimal
    usage snippet for each), linked from the Documentation Hub.
  - `docs/ARCHITECTURE.md`: new subsection under/near the Server Module
    describing the client bindings as consumers of `/execute` +
    `/delivery`, referencing `clients/CONTRACT.md`.
  - `docs/DATASET_ARCHITECTURE.md`: note that the Parquet persistence
    format is now a real external interop surface (not just an internal
    storage detail), pointing at the v0.1.72 entry already there.
  - `CHANGELOG.md`: per-checkpoint entries already required by the
    process below; this step is a final consistency pass across all of
    them plus the client packages' own README/CHANGELOG files.
  - `clients/python/README.md` / `clients/r/README.md`: standalone
    install/usage docs for anyone landing directly in that directory
    (e.g. from PyPI/CRAN metadata) without having read the main repo
    README first.

- [ ] **7. Wrap-up**
  - Full `cargo test` (engine side) + both clients' test suites green.
  - `cargo clean` (this repo's build artifacts grow large fast — see
    e.g. v0.1.59's "freed 22.1GiB" note in project history; do this after
    the test-heavy checkpoints above, not just once at the very end, so
    `target/` doesn't balloon across the whole plan).
  - Confirm working tree clean, no stray branches.
  - This plan file deleted once every checkpoint above is checked off,
    matching this repo's existing `CONSISTENCY_PLAN.md`/
    `SIGNAL_PROCESSING_PLAN.md` convention.

---

## Process for every checkpoint

- Implement + unit test + doc update together, not
  implementation-then-tests-later.
- Engine-side changes (if any checkpoint needs one): `cargo build`,
  `cargo test`, `cargo clippy --lib --bins` all clean before checking off
  a box. Client-side changes: that language's test runner green
  (`pytest` / `testthat::test_dir()`).
- Any bug found — in the engine, in a client, or in the contract between
  them — gets written down inline in this file under the checkpoint that
  found it (root cause + fix + layer), not silently fixed and forgotten;
  promoted to project memory / CHANGELOG once the checkpoint lands, same
  as every prior tracked-plan round in this repo.
- Run `cargo clean` after any checkpoint that did a lot of engine
  building/testing (not required after pure client-side checkpoints,
  which don't touch `target/`).
- Check off the box here with a `**Done in <version/commit>**` note once
  merged.

## Completion

This file is deleted once every checkpoint is checked off, both clients
have a working real-data example, and the docs pass (checkpoint 6) has
landed.
