# Scientific retrieval workloads plan, round 2 (CASMI 2026 case)

Tracks the implementation of the proposal *"LINALDB: proposed changes for scientific workloads,
round 2 (CASMI 2026 case)"* (2026-10-08), proposals P8–P15, after checking each one against the
engine code. Round 1 (P1–P7) is `CASMI_WORKLOADS_PLAN.md`, released as v0.1.92 / `linaldb`
0.1.17 (#139), with hub fixes in v0.1.93 / 0.1.18 (#140).

Rules for every change: validate against an independent implementation (`ms_entropy`,
matchms, NumPy brute force), fail loudly on bad input, keep new behavior opt-in, and update
`DSL_REFERENCE.md`, `ARCHITECTURE.md`, `CHANGELOG.md` and the client contracts in the same PR.

## Two PRs, then one release

1. **PR A — low and low+ impact:** P8, P10, P11, P12, P15.
2. **PR B — medium and large impact:** P9, P13, P14.
3. **One release PR** (engine binaries + `linaldb` on PyPI).
4. **linal-hub:** re-run every notebook on the published wheel, plus a new notebook exercising
   the round-2 features; any bug found gets a fix PR that goes all the way through a release;
   then the hub docs (DSL reference, Use Cases, playground pin).

### PR A — low and low+ impact

| Item | What | Where |
|---|---|---|
| P8 | `SPEC_ENTROPY(a, b, tolerance [, weighted])` | `core/spectral.rs::entropy_similarity` |
| P10 | `SEARCH ... RETURN <columns> \| RETURN NONE` | `dsl/executor/query.rs::search_plan`, `BatchVectorSearchExec::projection` |
| P11 | `Db.load_arrow(..., peaks=..., cast="f32", sort=...)` | `engine/db/memory_load.rs::combine_peak_columns`, binding |
| P12 | `SPEC_CLEAN(peaks, precursor_mz [, floor, max_peaks, above_precursor, power, normalize, min_distance])` | `core/spectral.rs::clean` |
| P15 | `rows` on lineage records, `ASSERT LINEAGE <name> MATCHES '<file>'`, `Db.lineage()` | `engine/db.rs::record_provenance`, `ProvenanceTree::mismatches` |

Measured:

- P8: equal to a 64-bit port of `ms_entropy` to 5.6e-16 and to `ms_entropy` 1.5.3 itself to
  2.9e-7 on 300 pairs (identical, noisy, partly overlapping, disjoint, single-peak). The
  proposal's "1e-9 against `ms_entropy`" is not reachable: `ms_entropy` computes in float32.
- P12: same peak counts as `ms_entropy.clean_spectrum` and a matchms filter chain on 600
  spectra × 6 parameter sets; m/z within 1e-7 relative, intensities within 6e-8. End to end,
  `SPEC_ENTROPY(SPEC_CLEAN(a), SPEC_CLEAN(b))` equals `ms_entropy`'s default (cleaning) call to
  2.9e-7.
- P11: identical to `peaks_array()` (same content hash) on 20,000 spectra; 2.5M spectra / 75M
  peaks (`list<float64>`, cast to float32) load in 11.9 s on an M-series laptop (target: well
  under a minute).

Bug found and fixed along the way: a computed `SELECT` column whose rows differ in width (any
variable-width matrix or vector expression) was typed from its first row; other rows lost the
value and the query panicked.

### PR B — medium and large impact

| Item | What |
|---|---|
| P9 | `SEARCH ... USING <expression>`: top-k by any score (`SPEC_*`, `DOT`, `TANIMOTO`, ...) |
| P13 | `CANDIDATES n RERANK USING <expression>`: two-stage search in one statement |
| P14 | Memory-mapped columnar storage for `Vector`, `Vector(d, F16\|I8)` and `BitVector` columns, read-only first |

## Findings from checking the proposal against the code

- `EXPLAIN LINEAGE <name> AS JSON` already existed; P15 reduces to row counts, the assertion
  and the Python helper. Loads already recorded their origin (`origin` / `path`).
- P9 is larger than the proposal says: `QueryBatch` holds `Vec<f32>` queries only and both the
  searched and the query column must be vectors today; `collect_columns` copies every column
  for any predicate with a function call, which would clone each candidate's peak list.
- No `:=` named arguments exist in the parser: `SPEC_CLEAN` takes positional arguments.
- `keep=` in P11 was dropped: the source list columns can't be stored (variable-length lists
  aren't a column type), so it could only fail.
- The license is "LinalDB Community License v1.0"; publishing the closing release under an
  OSI-approved license is a decision outside the engine work.
- Seen, not fixed here (pre-existing): numeric literals with an exponent (`1e6`) don't parse;
  there is no `DROP DATASET`, although the `load_arrow` "already exists" error says to drop it.
