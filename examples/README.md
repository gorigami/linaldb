# Examples

This directory contains only `.lnl` LINAL scripts — no other file types.
Fixture-generating Rust binaries live in `tools/fixtures/` (registered as
Cargo examples, so `cargo run --example gen_test_data` still works).

Every `.lnl` file in this directory is a runnable LINAL script:

```sh
linal run examples/<name>.lnl
```

## Convention for adding a new example

1. **Isolate your database.** Start with:
   ```
   DROP DATABASE IF EXISTS my_example
   CREATE DATABASE my_example
   USE my_example
   ```
   and end with `USE default`. This makes the script safe to re-run and
   keeps it from colliding with data left behind by other examples or
   tests, since `linal run` operates on the same `./data` directory
   regardless of which script is invoked.

2. **One statement per line.** `linal run` splits the script into
   statements on newlines, tracking only parenthesis balance — it does
   not understand multi-clause statements broken across lines unless
   they're inside an open `(...)`. Keep each `SELECT`/`INSERT`/etc. on a
   single line (a `DATASET ... COLUMNS (...)` block spanning lines is
   fine because the parens stay open).

3. **Reference external files relative to the repo root**, and only via
   `USE DATASET FROM "..."` (scientific ingestion) or `IMPORT DATASET
   FROM "..."` — those resolve paths the way you'd expect (relative to
   the process's current directory). The legacy `IMPORT CSV FROM "..."`
   is the exception: it resolves relative to the active instance's data
   dir (`./data/<instance>`), so reference repo files from there with a
   `../..`-style relative path, or prefer `USE DATASET FROM` instead.

4. **Add a verification test.** Every example needs at least a smoke
   test — add a `test_example_<name>_runs_clean()` entry to
   `tests/examples_cli_smoke_test.rs` (copy an existing one; it's a
   one-liner via the shared `assert_example_runs_clean` helper). If the
   example is demonstrating a feature's *correctness* (not just that it
   parses), prefer a deeper test in `tests/examples_integration.rs`
   that runs the script via `execute_script` and asserts on the
   resulting `TensorDb` state directly (see `test_example_vdb_integration`
   for the pattern).

## What's here

| File | Demonstrates |
|---|---|
| `example.lnl` | Core tensor ops (vectors, matrices, matmul, transpose, flatten) + basic dataset filter/select/order chains |
| `features_demo.lnl` | Matrix/Vector-typed columns, matrix aggregation, HASH + VECTOR indexes |
| `pipelines_and_search.lnl` | Window functions, `CAST`-to-tensor reshaping, index-accelerated similarity `JOIN`, named pipelines (define/apply/save/load) |
| `matrix_operations.lnl`, `test_matrix_math.lnl` | Tensor/matrix arithmetic |
| `test_multiline.lnl` | Multi-line statement parsing (paren-balance continuation) |
| `advanced_analytics.lnl` | Aggregations, computed columns, `GROUP BY`/`HAVING` |
| `benchmark.lnl` | Rough in-memory vs. indexed vs. persisted workload comparison |
| `export_import_csv.lnl` | Legacy `IMPORT CSV`, scientific `USE DATASET FROM`, `EXPORT CSV`, `RESET SESSION` |
| `persistence_demo.lnl` | Dataset metadata, `SAVE`/`LOAD DATASET`, version history |
| `metadata_demo.lnl` | `SET DATASET METADATA` / `SHOW DATASET METADATA` |
| `introspection_demo.lnl` | `SHOW SCHEMA`, `DESCRIBE`, catalog introspection commands |
| `reference_graph.lnl` | `BIND`/`ATTACH`/`DERIVE` zero-copy lineage tracking |
| `tensor_datasets.lnl` | Tensor-backed dataset columns, save/load round trip |
| `managed_service_demo.lnl` | Multi-database / multi-tenant workflow |
| `pbmc_cell_typing.lnl` | End-to-end showcase: replicates a real single-cell RNA-seq reference-based cell-typing workflow (synthetic PBMC marker data) using nearly the full feature surface in one coherent story — vector index + similarity `JOIN` for nearest-centroid classification, `ROW_NUMBER`/`RANK` window functions, CTEs, `AVG_VEC` per-type centroids, `CASE`-based accuracy scoring, equi-`JOIN` against a small metadata table, `MATMUL`/`TRANSPOSE`, and persistence |
| `hdf5_digit_classification.lnl` | End-to-end showcase using genuinely downloaded external data: the real UCI "Optical Recognition of Handwritten Digits" dataset, ingested via the HDF5 connector. Exercises the `IMPORT DATASET FROM`/`LOAD DATASET` round-trip fix and the N-D shape-preservation fix directly, `TRANSPOSE`/`MATMUL` on the real shape-fixed tensor, then nearest-centroid classification (vector index + similarity `JOIN`, `CASE`-based accuracy scoring) against real held-out query digits |
| `gw_transient_analysis.lnl` | End-to-end showcase using real LIGO/Virgo gravitational-wave data (GWOSC, CC BY 4.0): all 11 GWTC-1-confident event parameters + real 4096 Hz strain HDF5 for 4 events x 2 detectors (1,048,576 real samples). `CASE WHEN` physics classification, window functions (`RANK`/`LAG`), Vector-embedded schema + `CREATE VECTOR INDEX`/`SEARCH`, `CORRELATE`/`SIMILARITY`/`DISTANCE` between real detector pairs, N-D `RESHAPE` into per-second segments + `L2_NORM` energy scan, cross-event CTE/`UNION ALL` leaderboard, `AVG_VEC` centroids, persistence round-trip |
| `smoke_test.lnl` | Broad single-pass sanity check across many commands |
| `hardening_test.lnl` | CLI multiline-parsing regression check (used by `tests/cli_hardening_test.rs`) |
| `data/` | Small fixture files (e.g. `sample_data.csv`, `digits_centroids.h5`) referenced by the examples above |

Fixture-generating binaries (not `.lnl`, so they live outside this
directory): `tools/fixtures/gen_test_data.rs` and
`tools/fixtures/gen_zarr_data.rs` generate `.npy`/`.h5`/`.zarr` files for
manually exercising scientific ingestion — run with `cargo run --example
gen_test_data` / `cargo run --example gen_zarr_data`.
`tools/fixtures/gen_digits_data.rs` downloads the real UCI digits dataset
and derives `examples/data/digits_centroids.h5` from it — run with `cargo
run --example gen_digits_data` (requires network access; re-run only if
regenerating the fixture, the derived `.h5` is checked in).
`tools/fixtures/gen_gw_data.rs` downloads real LIGO/Virgo gravitational-wave
data from GWOSC (gwosc.org, CC BY 4.0) and derives `examples/data/gwtc1_events.csv`
+ `examples/data/gw_strain/*.hdf5` from it — run with `cargo run --example
gen_gw_data` (requires network access; re-run only if regenerating the
fixtures, both are checked in).
