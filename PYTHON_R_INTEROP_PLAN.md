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

- [x] **1. Python client core (`/execute`)** — **Done 2026-07-23**
  - `linaldb.connect(url, database=None) -> Client` (`clients/python/linaldb/client.py`).
  - `Client.execute(dsl: str)` — unwraps the tagged `Value` JSON into
    native Python types (`Vector` → `list[float]`, `Matrix` → nested
    list, `Null`/unit-variant → `None`) via new `linaldb/wire.py`
    (`unwrap_value`/`unwrap_result`/`TableResult`/`TensorResult`, kept
    separate from `client.py` so it's unit-testable with no HTTP
    involved); raises `LinalError` with the server's real error message
    on `status: "error"`.
  - `Client.query(dsl: str) -> pandas.DataFrame` convenience wrapper
    (requires the `pandas` extra; raises `LinalError` if the result isn't
    table-shaped).
  - `TensorTable` unwrapping deliberately **not implemented** — no DSL
    command was found in this checkpoint that produces one on the wire to
    verify against (unlike `Table`, which was captured from a real
    response for `clients/CONTRACT.md`). Raises a clear `LinalError`
    rather than guessing the shape; revisit if/when a real example turns
    up (checkpoint 5's real example is a natural place to hit this).
  - `TensorResult.to_numpy()` similarly caveated in its docstring — the
    contiguous/zero-offset reshape assumption was never verified against
    a live `Tensor` response either (contract §1 flags the same gap).
  - **Real tests, not just written-and-hoped**: created a `.venv` under
    `clients/python/`, installed the package with its `dev` extra
    (`pytest`, `pandas`), and ran the full suite against a real `linal
    serve` subprocess (`tests/conftest.py`'s `linal_server` fixture —
    finds the built binary under `target/{debug,release}/linal`, launches
    it on a free port, polls `/health`, tears down via SIGTERM). 16/16
    tests pass: 9 fixture-based unit tests in `test_wire.py` (including
    the exact real payload captured for `CONTRACT.md`) + 7 integration
    tests in `test_client_integration.py` (Message/Table/error results,
    the `X-Linal-Database` header actually isolating two databases,
    `NULL` round-tripping through a Vector column, `query()`'s DataFrame
    output). Confirmed no leaked server subprocess after the run.
  - No engine changes this checkpoint.

- [x] **2. Python client dataset export (`/delivery`)** — **Done 2026-07-23,
  engine v0.1.73**
  - `Client.dataset(name) -> Dataset` (new `linaldb/dataset.py`) —
    `.manifest()`/`.schema()`/`.stats()`, `.to_arrow() -> pyarrow.Table`,
    `.to_pandas()`. Trusts `schema.json`'s `value_type` to detect the
    legacy JSON-string fallback encoding (a column typed Vector/Matrix
    whose actual Parquet column is `Utf8`/`string`) and transparently
    unwraps it via `wire.unwrap_value`, reusing the same tagged-JSON
    parser as `/execute` results.
  - **Two real, previously undiscovered engine bugs found and fixed
    (v0.1.73), both by driving this checkpoint's own integration tests
    against a real server rather than trusting `/delivery` worked because
    it existed and had a test**:
    1. `/delivery/*` 404'd for **every** dataset ever saved through the
       real `SAVE DATASET` path, in every database including `default`.
       `dsl::persistence` always writes to
       `{data_dir}/{database}/datasets/{name}/...`; `dataset_server.rs`'s
       handlers read `{data_dir}/datasets/{name}/...`, missing the
       database segment. The pre-existing `dataset_delivery_test.rs`
       never caught it because it built its fixture directory by hand to
       match whatever the handler currently expected, instead of going
       through the real DSL save path. Fixed: handlers now resolve the
       database segment via the same `X-Linal-Database` header
       `/execute` already honors (default `"default"`). Added
       `test_dataset_delivery_matches_real_save_dataset_path`, which
       drives a real `TensorDb` through actual DSL statements and closes
       the exact gap that hid the original bug.
    2. `schema.json` reported a fallback-encoded Vector/Matrix column as
       plain `"String"` — it's derived from the *physical* Arrow type,
       which is indistinguishable from a real String column once a
       column has fallen back to JSON-string encoding. This directly
       contradicted `clients/CONTRACT.md`'s claim that `schema.json` is
       authoritative for logical column typing — a claim that was true
       for the native-encoding case checkpoint 0 verified, but not yet
       true for the fallback case, which checkpoint 0 didn't happen to
       exercise. Fixed via a new `linal.logical_value_type` Arrow
       field-metadata entry (mirrors `core::connectors::
       SHAPE_METADATA_KEY`'s existing pattern), attached whenever
       `dataset_to_record_batch` falls back, read back by both
       `DatasetSchema::from` and `arrow_schema_to_tuple_schema` via new
       `logical_vector_or_matrix_type`.
  - Real end-to-end integration tests (`test_dataset_integration.py`):
    native no-null Vector column, native Matrix column, the legacy
    fallback Vector column with an actual `NULL` (this is the test that
    caught both bugs above), `to_pandas()`, and `manifest()`/`schema()`/
    `stats()`. 21/21 Python tests pass (16 from checkpoint 1 + 5 new).
  - Also fixed test isolation: the `linal_server` fixture now launches
    the server subprocess with its `cwd` set to a fresh `tmp_path_factory`
    directory instead of `clients/python/` — a real issue hit while
    re-running this checkpoint's tests: the server's disk auto-recovery
    picked up a `data/` directory left over from earlier manual runs in
    that directory, and fixed-name `CREATE DATABASE` calls in the test
    suite started failing with "already exists" on rerun.
  - `cargo clean` run after this checkpoint's build/test cycle (freed
    19.1GiB); full rebuild from a clean `target/` reconfirmed both the
    full Rust suite and the full Python suite green.

- [x] **3. R client core (`/execute`)** — **Done 2026-07-23**
  - Mirrors checkpoint 1: `linal_connect(url, database = NULL)`,
    `linal_execute(conn, dsl)` unwrapping the same tagged `Value` JSON via
    `jsonlite` (new `R/wire.R`), `linal_query(conn, dsl)` returning a
    `data.frame`. Condition-based error handling: a `linal_error`
    S3/condition class (`R/errors.R`) carrying the server's real message,
    raised via `stop()`, catchable via `tryCatch(..., linal_error = ...)`.
  - Environment setup required first: R wasn't installed on this
    machine. Installed via Homebrew (`r` formula) plus `httr2`/
    `jsonlite`/`arrow`/`testthat`/`pkgload`/`roxygen2` from CRAN — a
    real, visible machine-level change, done only after confirming with
    the user rather than assuming it was fine to install a new language
    toolchain unprompted.
  - **Real tests, both the normal dev-mode way and the strict CRAN-style
    way**: `pkgload::load_all()` + `testthat::test_dir()` against a real
    `linal serve` subprocess (`tests/testthat/helper-server.R`, mirrors
    the Python `conftest.py` fixture — one server for the whole test run,
    memoized, torn down via `withr::defer(..., teardown_env())` since
    testthat has no built-in session fixture) — 40/40 tests pass (25 wire
    unit tests + 15 integration tests). Then went further and ran a real
    `R CMD build` + `R CMD check`, which found two genuine issues
    `load_all()`-only testing couldn't have caught:
    1. `DESCRIPTION`'s `Authors@R` had no email — `R CMD INSTALL` refuses
       to install without one. Fixed using the same `develop@gorigami.xyz`
       contact this repo's own `LICENSE`/`README.md` already use.
    2. **A real path-resolution bug**: `find_linal_binary()`'s original
       fixed relative-path climb (`../../../..` from the test file) only
       works when tests run from the real source tree
       (`pkgload::load_all()` + `test_dir()`); `R CMD check` copies the
       package into its own sandboxed `<pkg>.Rcheck/` directory with
       different nesting, so the same fixed climb resolved to the wrong
       directory entirely and the integration tests hard-failed. Fixed
       properly: new `find_repo_root()` walks upward from the test file
       looking for `Cargo.toml` (robust to nesting depth) instead of
       assuming a fixed depth, and `find_linal_binary()` now calls
       `testthat::skip()` rather than `stop()` when the repo root or
       binary genuinely can't be found — correct behavior for a package
       whose integration tests depend on a sibling Rust checkout that
       won't exist on CRAN or in generic CI, not just a check-sandbox
       workaround. Also fixed the parallel gap in the Python client
       (`clients/python/tests/conftest.py`'s `_find_linal_binary()`) for
       consistency: `pytest.skip()` instead of `raise RuntimeError`.
    Also cleared a `NOTE` (`arrow` declared in `Imports` but unused —
    correct for checkpoint 3's scope, since `/delivery` support with real
    `arrow::` calls is checkpoint 4; moved to `Suggests` for now, will
    move back to `Imports` when checkpoint 4 lands) and confirmed the
    remaining `WARNING` (non-standard license string) is the
    already-known, already-documented pre-CRAN-submission item from
    checkpoint 0, not a new issue. Final `R CMD check` result: 40/40
    tests pass inside the real check sandbox, 1 WARNING (the known
    license one), 0 ERRORs, 0 NOTEs.
  - `free_port()` needed a non-obvious implementation: R's
    `socketConnection(port = 0, server = TRUE)` (the direct analogue of
    the Python fixture's bind-to-0-then-read-back-the-port trick) blocks
    indefinitely in `open()` waiting for a client to connect that never
    comes — confirmed by hand (it hung for real, had to be killed).
    `serverSocket()` doesn't block but also doesn't expose the
    OS-assigned port. Landed on: pick a random high port, confirm it's
    free by successfully binding `serverSocket()` to it (closing
    immediately), retry on collision.

- [x] **4. R client dataset export (`/delivery`)** — **Done 2026-07-23**
  - Mirrors checkpoint 2: `linal_dataset(conn, name)` +
    `linal_dataset_schema()`/`_manifest()`/`_stats()`/`linal_dataset_read()`
    (returns a `data.frame`)/`linal_dataset_to_arrow()` (returns a real
    `arrow::Table`, built by re-encoding the already-unwrapped
    `data.frame` via `arrow::arrow_table()` rather than patching the raw
    Arrow Table's columns directly — one source of truth for the
    unwrapping logic instead of two implementations that could drift).
    `arrow` moved back to `DESCRIPTION`'s `Imports` (was `Suggests` as of
    checkpoint 3, correctly, since checkpoint 3 didn't call it yet).
  - Verified the native-vs-fallback column type detection against real
    engine-written Parquet files (not just R's own round-trip
    assumptions) before writing the unwrap logic: a fallback (`NULL`
    present) column reads back with Arrow field type class `"Utf8"`;
    a native column reads back as `"FixedSizeListType"` — confirmed by
    saving both through a real `linal serve` instance and inspecting
    `schema$field(i)$type` directly, the same "verify against a live
    server, not assumption" discipline checkpoint 0's `CONTRACT.md` used.
  - Integration tests (`test-dataset-integration.R`) mirror
    `test_dataset_integration.py` exactly: native no-null Vector column,
    native Matrix column, the legacy-fallback Vector column with an
    actual `NULL`, `linal_dataset_to_arrow()`, and
    `schema()`/`manifest()`/`stats()`. One test needed
    `ignore_attr = TRUE`: `arrow`'s own `as.data.frame()` represents a
    *nested* `FixedSizeList<FixedSizeList<Float>>` (Matrix) column as its
    own `arrow_fixed_size_list`/`vctrs_list_of` S3 class rather than a
    bare R list (a flat `FixedSizeList<Float>`/Vector column simplifies
    to a plain list of numeric vectors, but a nested one doesn't) — a
    real, if narrow, arrow-R-package behavior worth remembering, not a
    client bug.
  - Re-ran the full `R CMD check` cycle from checkpoint 3 (not just
    `load_all()`) with checkpoint 4's additions and found one more real,
    small issue: a roxygen doc comment containing literal
    `` `{"Vector": [...]}` `` text triggered `checkRd`'s "Lost braces;
    missing escapes or markup?" — Rd's markup parser treats bare `{}` as
    its own syntax. Fixed by rewording to avoid literal JSON braces in
    `@export`ed docs (an internal `@noRd` function's doc comment has the
    same pattern but is harmless, since `@noRd` produces no `.Rd` file to
    check).
  - Final `R CMD check`: 53/53 tests pass in the real check sandbox (40
    from checkpoint 3 + 13 new), 1 WARNING (the known non-standard
    license string), 0 ERRORs, 0 NOTEs. Full Python suite (21/21)
    reconfirmed unaffected (no engine changes this checkpoint).

- [x] **5. Real end-to-end example, both languages** — **Done 2026-07-23,
  engine v0.1.74 — the checkpoint's own prediction held: two more real
  bugs found, one of them severe**
  - Built one real example per client — `clients/python/examples/
    digit_classification.py` and `clients/r/examples/
    digit_classification.R` — reusing the real, already-checked-in UCI
    handwritten-digits data from `examples/hdf5_digit_classification.lnl`
    (real downloaded samples, not synthetic) rather than duplicating ~40
    real 64-dimension vectors as literal data in two more files: both
    scripts start a real `linal serve` and **replay that `.lnl` file's
    actual DSL statements through the client itself** (a small
    paren-balance-aware line joiner mirroring `linal run`'s own
    multi-line statement joiner in `src/main.rs`, since most statements
    in that file are one physical line but its `DATASET ... COLUMNS
    (...)` blocks genuinely span several). Both scripts then: run the
    real nearest-centroid classification query via `/execute`, export
    `query_digits`/`reference_centroids` via `/delivery`
    (`to_pandas()`/`linal_dataset_read()`), and **independently
    recompute** cosine-similarity classification from the raw exported
    vectors in plain numpy/base R — comparing every per-row similarity
    value, every predicted label, and the aggregate accuracy against what
    `/execute`'s SQL engine reported. Both scripts: exact match, 25/30
    (83.3%) — the same real, already-known-genuine accuracy figure from
    the original `.lnl` file, now independently reproduced three ways
    (original CLI run, Python from `/delivery` data, R from `/delivery`
    data).
  - **Bug found before the example even ran, by reviewing the code with
    a non-default-database scenario in mind**: both `Dataset` (Python)
    and `linal_dataset()` (R) sent every `/delivery` HTTP request with no
    `X-Linal-Database` header at all, so a client configured for a
    non-default database silently fell back to the server's *default*
    database instead. Neither checkpoint 2 nor checkpoint 4's tests
    caught this — both only ever used the default database. Fixed in
    both clients; added a same-named-dataset-in-two-databases regression
    test to each (22/22 Python, 55/55 R before the next finding below).
  - **A severe, previously-undiscovered engine bug found while getting
    the example to actually run**: `USE <database>` sent to `/execute`
    with no `X-Linal-Database` header reported success but had **no
    lasting effect at all** — `execute_command`'s "restore previous
    database" logic ran unconditionally after every command, silently
    reverting any active-database change a headerless request made,
    including one whose own command was an explicit `USE`. This meant
    the entire session-level `USE` workflow over HTTP had likely been a
    no-op since the multi-tenancy header feature was first added — every
    dataset the replayed `.lnl` script's own `USE
    hdf5_digit_classification` was supposed to scope actually landed in
    `default`. Root-caused with runtime `eprintln!` instrumentation after
    code-reading alone didn't explain the symptom (a minimal pure-engine
    reproduction proved `TensorDb`/`execute_line` had no bug at all,
    narrowing it correctly to the HTTP layer). Fixed in engine v0.1.74:
    the restore now only fires for a request that itself supplied the
    header. New regression test
    `test_server_use_database_persists_without_header` — the existing
    `test_server_multitenancy` never caught this because it always sends
    the header on every request, never exercising "switch once via a
    plain `USE`, then rely on that being remembered."
  - **A latent test-isolation bug the engine fix immediately exposed**:
    both clients' own test suites had a test that calls `USE <db>` and
    never switches back — accidentally "protected" by the very engine bug
    just fixed (headerless `USE` never really persisting meant the leak
    was invisible). Once fixed, every other headerless test in the same
    session started running against the leaked-into database instead of
    `default`, and 6 Python dataset-export tests failed. Fixed both
    suites to restore `USE default` afterward (`try/finally` in Python,
    `withr::defer()` in R) — a real, worth-remembering lesson: fixing a
    bug that was silently providing test isolation can surface real gaps
    in the tests themselves, not just in the code under test.
  - Final state: 22/22 Python tests, 55/55 R tests (including a full
    `R CMD check` pass), full Rust suite (172 lib + all integration
    suites) all green. Both example scripts produce byte-identical
    cross-validated classification numbers between `/execute` and
    `/delivery`.

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
