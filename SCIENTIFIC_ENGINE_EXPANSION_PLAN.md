# Scientific Engine Expansion Plan

**Status**: not started. This file is deleted in the commit that closes the final checkpoint,
per this repo's established tracked-plan-doc convention (`PYTHON_R_INTEROP_PLAN.md`,
`LINEAGE_AND_LINALG_PLAN.md`, `FLOAT64_PLAN.md`).

## Why this plan exists

Prompted by a user question on `linal-hub` ("what's the difference from MLflow?"), which led to
repositioning the public docs away from an "ML/AI research" framing toward "database engine +
persistence + real linear algebra + vector search, for scientific and engineering research
broadly" (shipped to `linal-hub`; matching edits to this repo's own `docs/ARCHITECTURE.md` and
`CLAUDE.md` are prepared but held for a future PR). The natural follow-up: does the engine itself
already back up that broader positioning, or does it need new capabilities?

Three parallel code-exploration passes (linear-algebra core; data ingestion/signal
processing/statistics; vector search/embeddability) confirmed the classical linear algebra
(`TRACE`/`DETERMINANT`/`RANK`/`INVERSE`/`SOLVE`/`QR`/`LU`/`CHOLESKY`/`SVD`/`PCA`, symmetric-only
`EIGEN`) and the 4-connector ingestion architecture are solid and cleanly extensible, but surfaced
real, evidenced gaps. The user's direction: address the real gaps, sequenced for engine
consistency, each phase gated on full regression testing (existing 11-notebook suite) plus a new
notebook in a genuinely uncovered scientific field, with full documentation updates throughout.

## Audit findings (re-verified against real source)

- **Linear algebra** (`src/core/linalg.rs`, built on `nalgebra 0.33.3`): no least-squares/
  pseudo-inverse operator (`SOLVE`/`INVERSE` require square+nonsingular only); no complex-number
  type anywhere (`ValueType`, `src/core/value.rs:99-108`) — this is *why* `EIGENVALUES`/`EIGEN`
  are symmetric-only today (`require_symmetric`, `linalg.rs:177-192`, reasoning documented at
  `linalg.rs:242-249`); no `f64` tensor/vector/matrix storage (`Tensor.data` is
  `Arc<Vec<f32>>`, `f64` is scalar-only via `Value::Float64`). Cheap to close because nalgebra
  already pulls in `num-complex` transitively and ships `Schur`/general `eigenvalues()` plus
  `SVD::pseudo_inverse`/`QR::solve` — none currently called.
- **Data ingestion & signal processing**: exactly 4 connectors (CSV/HDF5/NumPy/Zarr) via a clean
  one-file `Connector` trait (`src/core/connectors/mod.rs:78-102`, registered in one place,
  `src/dsl/persistence.rs:37-43`) — today's HDF5 connector accepts `.nc`/`.h5ad` only as opaque
  generic containers, no real NetCDF/CF or AnnData semantics. Signal processing
  (`src/core/signal.rs`) has FFT/PSD/whiten/bandpass/matched-filter but no windowing function
  before each FFT/PSD chunk — a gap the code's own doc comment already flags
  (`signal.rs:96-104`). SQL aggregates are only `Sum`/`Avg`/`Count`/`Min`/`Max`/`AvgVec`/`SumVec`
  (`src/query/logical.rs:127-138`, confirmed exhaustive) — no `Variance`, no quantiles/median, no
  covariance matrix (only pairwise `CORRELATE`). No graph/network algorithms exist at all.
- **Vector search & embeddability**: IVF clustering, cosine-only distance metric
  (`VectorIndex::cosine_similarity`, `src/core/index/vector.rs:70-88`), no HNSW/LSH. Flagged
  unprompted as the highest-leverage gap for RAG-style workloads: no filtered/hybrid vector
  search — `try_optimize_filter` (`src/query/planner.rs:243-312`) has no `AND` handling, so
  `WHERE COSINE_SIM(...) > t AND category = 'x'` silently loses all index acceleration. The
  index is never persisted (full rebuild + blocking k-means on every `CREATE INDEX` and every
  `LOAD DATASET`, `src/core/dataset_legacy.rs:560-581`). The engine is single-writer throughout
  (`TensorDb`/`DatabaseInstance` are all `&mut self`, `src/engine/db.rs`) — concurrency exists
  only externally, via the HTTP server's `Arc<RwLock<TensorDb>>`.

## Design decisions locked before implementation

1. **`LSTSQ`** is a new, distinct keyword for least-squares/pseudo-inverse solve. `SOLVE` stays
   square-only and continues to error loudly on non-square input — no polymorphic
   shape-dependent behavior.
2. **`SEARCH`'s new hybrid filter clause is a new `FILTER` keyword**, not a repurposed `WHERE` —
   `WHERE` is already claimed by `SEARCH`'s alternate query-vector syntax
   (`SEARCH source WHERE col ~= [...] LIMIT k`, `src/dsl/parser/dataset.rs:1078-1091`); reusing
   it would silently break existing scripts.
3. **`Value::Complex` is scalar-only.** `FFT`'s output stays `Matrix(2,N)` (re/im row
   convention) — a genuine `Tensor<Complex>` type is a separate, larger initiative, the same way
   the engine still has no `f64` tensor storage today (only `f64` scalars).
4. **True multi-writer concurrency is out of scope for this entire plan.** The engine's zero-copy
   tensor reference model, lazy-tensor `Expression` store, and reference-graph dataset model all
   assume a single mutable owner; real multi-writer support would be a foundational
   rearchitecture, not an incremental addition. Revisit separately later, only if real usage
   shows it's actually needed.
5. Unaccelerated `WHERE`/`FILTER` predicate shapes in `SEARCH` (e.g. `OR`, or a conjunct on a
   non-indexed column) fall back to brute-force scan+filter rather than erroring, consistent with
   `try_optimize_filter`'s existing "recognize the shape or don't accelerate, never error"
   philosophy — but `EXPLAIN` must visibly report whether acceleration was actually used, so the
   fallback is never silent to the user even though it's silent to the query.

## Checkpoints

### Phase 1 — Quick wins (lowest risk, ship first)
- [x] `LSTSQ` (least-squares/pseudo-inverse, `src/core/linalg.rs`, via nalgebra's
      `SVD::pseudo_inverse`)
- [x] New reductions/aggregates: `MEDIAN`, `QUANTILE a AT p`, `VARIANCE`, `COVARIANCE a WITH b`,
      `COVARIANCE MATRIX <expr>`; SQL `AggregateFunction` gains `Variance`/`Median`
- [x] FFT/PSD windowing (`WINDOW HANN`/`WINDOW HAMMING` clause, `src/core/signal.rs`)
- [x] NetCDF/CF-convention connector + external Parquet ingestion connector
      (`src/core/connectors/`) — registered NetCDF before `HDF5Connector` in
      `get_connector_registry()` (`src/dsl/persistence.rs`), with a regression test
      proving generic `.h5`/`.h5ad` files still route to `HDF5Connector` (protects notebook 07)
- [x] Wildcard-arm grep for the new `AggregateFunction` variants — caught and fixed two real
      bugs in `query/logical.rs`'s schema-inference `match`es (new variants silently defaulted
      to `ValueType::Int` instead of `Float64`)
- [ ] Full existing 11-notebook regression run — **deferred**: `linal-hub`'s notebooks install
      `linaldb` from PyPI only (never editable/source), and none of this phase's features are
      released yet, so there is nothing on PyPI for a regression run to exercise. Run this after
      the version-bump/release PR that ships this phase, against the real published wheel.
- [ ] New notebook: `12_climate_science_reanalysis.ipynb` — **deferred**, same reason as above
      (needs a real released `linaldb` to install, per `linal-hub`'s own no-editable-install
      convention). Write and run it as a `linal-hub` follow-up once this phase is on PyPI.
- [x] Docs: `CHANGELOG.md`, `docs/DSL_REFERENCE.md`, `docs/ARCHITECTURE.md`, `README.md`
      updated (`docs/ERROR_REFERENCE.md` describes error *types*, not per-keyword messages —
      nothing to add there for this phase)

### Phase 2 — Filtered/hybrid vector search + index persistence (RAG composability)
- [x] `SEARCH` grammar: `FILTER <predicate>` clause on the "modern" syntax only
      (`src/dsl/ast.rs`'s `SearchStmt`, `src/dsl/parser/dataset.rs`) — legacy `SEARCH`
      forms untouched (regression-tested)
- [x] `try_optimize_filter` decomposes a top-level `AND`, routing the `COSINE_SIM(...) >
      threshold` conjunct to the existing index-accelerated path, remaining conjuncts as
      post-filter (fixes plain `SELECT ... WHERE COSINE_SIM(...) > t AND ...` identically, not
      just `SEARCH`). **Design deviation from this plan's original wording**: implemented as
      physical-plan-only composition (`CosineFilterExec` wrapped in `FilterExec`), not a new
      `LogicalPlan::FilteredVectorSearch` variant — matches this planner's existing convention
      for every other index optimization here (`IndexScanExec`/`CosineFilterExec`/
      `PartitionPrunedScanExec` are all physical-only too); whether an index exists is runtime
      state the logical plan shouldn't need to know about. See `query/planner.rs`'s
      `try_optimize_filter` doc comment for the full reasoning.
- [x] `EXPLAIN` reports whether index acceleration was actually used (`CosineFilterExec` visible
      in the printed physical plan)
- [x] Index persistence: serialize `VectorIndex` clusters/`clustered_count` into the dataset
      package (`vector_index_clusters.json`), load on `LOAD DATASET`, invalidate via
      content-hash on the underlying vector column
- [x] Wildcard-arm grep for the new `LogicalPlan` variant — **not applicable**, no new variant
      was added (see the design-deviation note above)
- [ ] Full 11-notebook regression (watch notebooks 07/08/09 — shared `try_optimize_filter` path)
      — **deferred**, same reason as Phase 1: nothing in this phase is released to PyPI yet for
      `linal-hub`'s notebooks (PyPI-only installs) to exercise. Run after this phase's
      version-bump/release PR.
- [ ] New notebook: `13_astronomy_catalog_hybrid_search.ipynb` (ingested via Phase 1's Parquet
      connector) — **deferred**, same reason as above
- [x] Docs: `docs/DSL_REFERENCE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md` updated
      (`README.md` has no existing `SEARCH`/vector-index Core Capabilities section to extend;
      `docs/ERROR_REFERENCE.md` describes error *types*, nothing to add there for this phase)

### Phase 3 — Complex-number foundation
- [ ] `Value::Complex(f64, f64)` / `ValueType::Complex` (scalar-only), promote `num-complex` to a
      direct dependency
- [ ] `EIGENVALUES_GENERAL`/`EIGEN_GENERAL` (`src/core/linalg.rs`, nalgebra's `Schur`/general
      `.eigenvalues()`) — `EIGENVALUES`/`EIGEN` stay symmetric-only, unchanged
- [ ] Complex arithmetic (`+`/`-`/`*`/`/`, magnitude/phase)
- [ ] `clients/CONTRACT.md` + `clients/EMBEDDED_CONTRACT.md` updated for the new wire type;
      `clients/python-embedded`/`clients/r-embedded` compatibility pass
- [ ] **Dedicated, proactive wildcard-arm consistency-audit checkpoint** on `Value`/`ValueType`
      across the whole codebase, before declaring the phase done (this is the exact bug class
      the `f64` scalar rollout only found after shipping — `Field::is_compatible`,
      arithmetic/aggregate evaluators, computed-column per-row typing)
- [ ] Full 11-notebook regression (highest-risk phase — `Value` touches comparison, table
      rendering, all four connectors' type inference, both wire formats)
- [ ] New notebook: `14_control_systems_stability.ipynb`
- [ ] Docs: `docs/DSL_REFERENCE.md`, `docs/ARCHITECTURE.md` (forward-reference to this phase's
      grep discipline), `README.md`, `CHANGELOG.md`, `docs/ERROR_REFERENCE.md`,
      `clients/CONTRACT.md`, `clients/EMBEDDED_CONTRACT.md`

## Backlog (greenfield, scoping notes only — separate future rounds)

- **Sparse linear algebra**: no `nalgebra-sparse`/`sprs` even transitively; needs a new dep, a
  parallel sparse `Tensor` storage type (CSR/COO), new ops, connectors for genuinely sparse data.
- **Graph algorithms**: no `petgraph`; `DatasetGraph` (`core/dataset/graph.rs`) is unrelated
  internal lineage machinery, not reusable. New dep + `src/core/graph.rs` + adjacency-from-dataset
  DSL surface + Dijkstra/PageRank/centrality/connected-components.
- **ODE/optimization/root-finding**: no crate present (`ode_solvers`/`argmin` candidates); needs
  its own crate-selection spike before sizing.
- **Random sampling/statistical distributions**: `rand` already resolved transitively, low
  conflict risk to promote to direct — cheapest backlog item, would also host the deferred
  hypothesis-tests gap.
- **Bounded UDF/plugin registry**: no registration API at all today; needs its own design spike
  (dynamic dispatch vs. WASM boundary vs. embedder-only trait-object registry).
- **Deeper typed Python/R embeddability**: today only `execute_raw(sql string)` despite `lib.rs`
  being a genuine public Rust API. Additive API surface over already-working internals, lower
  engine-risk than the others.

## Verification (per phase, before moving to the next)

1. `cargo fmt -- --check` / `cargo clippy -- -D warnings` clean.
2. Full CI-exact suite (`CARGO_INCREMENTAL=0 RUSTFLAGS="-C codegen-units=1" cargo test --release
   -j 1 -- --skip test_cli_init --skip test_cli_run_multiline --skip test_cli_serve_alias`).
3. The phase's wildcard-arm grep — a real gate, not optional.
4. Full existing `linal-hub` 11-notebook regression (`jupyter nbconvert --execute --inplace` on
   01-11, zero errors, zero unexpected output drift).
5. The phase's new notebook, run against a real dataset.
6. `cargo clean` (and `rm -rf ./data`) after the test run.
7. Docs updated per phase before the PR is opened; `linal-hub`'s docs site is a separate,
   hand-authored, non-auto-synced site — its update is an explicit follow-up, not assumed
   automatic.
8. Normal branch/CI/merge flow (`main` protected, 4 required checks); version bump/release are
   separate, deliberate follow-up PRs, not bundled into feature PRs.
