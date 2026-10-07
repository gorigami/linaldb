# Scientific retrieval workloads plan (CASMI 2026 case)

Tracks the implementation of the proposal *"LINALDB: proposed changes for scientific workloads
(CASMI 2026 case)"* (2026-10-06), proposals P1–P7, after checking each one against the engine
code. The motivating workload: rank candidate molecular structures for MS/MS spectra, roughly
2.5M library spectra and 275k structures, many queries at once, within a 9-hour offline Kaggle
notebook. LINALDB is the retrieval and traceability layer only, not the model.

Rules for every change: validate against an independent implementation (NumPy, RDKit,
matchms), fail loudly on bad input, keep new behavior opt-in, and update `DSL_REFERENCE.md`,
`ARCHITECTURE.md`, `CHANGELOG.md` and the client contracts in the same PR.

## Work is split into three tiers by engine impact

One PR per tier, each from `main` after the previous one is merged; one release at the end
(engine + `linaldb` on PyPI), then a check in `linal-hub` with public spectra.

### Low impact — merged (#136)

| Item | What | Where |
|---|---|---|
| P1 | In-memory load: `TensorDb::load_record_batch` (Arrow → dataset, NaN/Inf and type checks, lineage, WAL checkpoint); Python `Db.load_numpy()` / `Db.load_arrow()` | `src/engine/db/memory_load.rs`, `clients/python-embedded` |
| P6 | `ARG_MAX(col, by)`, `ARG_MIN(col, by)`, `RRF(rank[, k])` group aggregates | `query/logical.rs`, `query/physical.rs` (`AggregateExec`) |
| P3a | Batch top-k: `SEARCH ... QUERIES <matrix> \| <dataset>.<col> [KEY <col>]` → `(query_id, rank, score, row_id, ...)` | `BatchVectorSearchExec`, `dsl/executor/query.rs::search_plan` |
| P7 (report) | `SHOW MEMORY [<dataset>]`: estimated bytes per dataset, index, tensor | `Index::memory_bytes`, `dsl/executor/show.rs` |

### Medium impact — merged (#137)

| Item | What |
|---|---|
| P3b/c | Pre-filtered top-k with a query-dependent window (mass ± ppm): a range join with per-query top-k, exact scan inside the window, plus a sorted scalar index |
| P4 | `BitVector(N)` type with `TANIMOTO`, `JACCARD`, `HAMMING`, bit count |
| P5 | `SPEC_COSINE` / `SPEC_COSINE_MOD` on peak lists; needs a variable-length matrix column first (`Matrix(2, N)` columns require one fixed `N` today) |
| P7 (copies) | Stop indexes keeping their own copies of every vector (IVF/HNSW hold 1–2 extra copies plus per-vector metadata); binary index snapshots instead of pretty-printed JSON |

### Large impact — done in this tier's PR

| Item | What |
|---|---|
| P2 | `SparseVector(dim)` type, with IVF/HNSW support or a clear error |
| P7 (quantization) | Opt-in `Vector(d, F16)` / `I8` with per-vector scale; schema format stays backward compatible |
| P7 (mmap) | Memory-mapped snapshot loading |
| Filtered HNSW | Top-k over HNSW restricted to a pre-filter |

Measured (P7 copies): HNSW at 200,000 × 128 now builds in 64 s (was 419 s) with recall@10 0.955
(was 0.905) and 3.6 ms per query (was 8.3 ms), holding one copy of the vectors (was two plus
per-vector metadata). Validation: `TANIMOTO` equals RDKit exactly; `SPEC_COSINE(_MOD)` equals
matchms to 1e-12 on 900 pairs; `PREFILTER` equals a brute-force window + exact ranking.

Large tier, measured: filtered HNSW (`PREFILTER ... APPROX`) returned the exact top-10 on a
6,000-row test with half the rows passing (recall 1.000); quantized HNSW recall@10 against the
exact f32 answer is F16 1.000 / I8 0.990 (4,000 × 64); sparse results are bit-identical to dense;
quantized rows save exactly 2 (F16) or 3 (I8) bytes per element. One more pre-existing bug fixed:
`WHERE true` matched nothing.

Bugs found along the way and fixed in the medium tier: UPDATE/DELETE index and zone-map
staleness; arithmetic in WHERE comparisons matching nothing; f32 literals next to DOUBLE;
`schema.json` reporting Complex as String.

## Findings from checking the proposal against the code

- Storage is row-oriented (`Vec<Tuple>`, one heap `Vec<f32>` per vector cell). Every new
  column type touches ~9 files plus Arrow encoding, the bindings and the contracts.
- The proposal's open question on external Parquet: a `FixedSizeList<Float32>` column does
  ingest as `Vector(d)` (confirmed by `tests/memory_load_test.rs`, which compares an
  `IMPORT` + `LOAD` of such a file with an in-memory load). `FixedSizeList<Float64>` and
  variable-length `List` columns are not vectors.
- `SHOW MEMORY` makes the cost of the index copies visible: an HNSW index reports more than
  twice the bytes of the vectors it indexes.
