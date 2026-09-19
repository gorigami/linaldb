# Performance & Operational Hardening Plan

**Status**: not started. This file is deleted in the commit that closes the final checkpoint,
per this repo's established tracked-plan-doc convention (`PYTHON_R_INTEROP_PLAN.md`,
`LINEAGE_AND_LINALG_PLAN.md`, `FLOAT64_PLAN.md`, `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md`).

## Why this plan exists

Prompted by a maintainer-style optimization review (6 proposals covering tensor persistence,
dataset storage, server transport, vector indexing, provenance-log growth, and the dense linear
algebra backend), submitted 2026-09-19. Before accepting any of it, every claim was re-verified
against the current `main` (v0.1.86) source rather than taken at face value — two of the six
premises turned out to be wrong or actively contested by this repo's own documented design
decisions, which materially changes what's worth building. This plan captures the corrected
findings and a sequenced, risk-ranked execution order for what remains.

## Audit findings (re-verified against real source)

1. **Tensor persistence in JSON — FALSE.** `save_tensor_core` (`src/dsl/persistence.rs:196-226`)
   already writes tensors via `ParquetStorage::save_tensor`, the same columnar format datasets
   use. The `.json` writes elsewhere in `persistence.rs` are for named pipelines, an unrelated
   feature. No action needed; dropped from this plan.
2. **`dataset_legacy` dual storage framed as an "incoherence" to eliminate — CONTESTED, not a
   bug.** True that `dataset_legacy::Dataset` (row-oriented, the actual JOIN/physical-plan
   execution substrate) and `dataset/` (zero-copy `TensorId` reference graph) coexist, bridged by
   `TensorDb::materialize_tensor_dataset()` (`src/engine/db.rs:305,661`). But this repo's own
   `CLAUDE.md` states explicitly: *"Both are genuinely active, not one superseding the other"*
   (see `docs/DATASET_ARCHITECTURE.md` for the full rationale), and
   `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md`'s locked design decision #4 already ruled out exactly
   this class of foundational rearchitecture ("a foundational rearchitecture, not an incremental
   addition... revisit only if real usage shows it's actually needed"). Not pursued as originally
   framed — parked in Backlog, gated on profiling evidence.
3. **`linal serve` is JSON/REST only, no binary transport — TRUE.** `src/server/mod.rs` is Axum,
   `Json<...>` responses throughout, no Arrow Flight/gRPC anywhere in `Cargo.toml`. Real gap. But
   there's already a partial precedent for a binary path in this server: `/delivery/*` serves
   read-only Parquet exports of saved datasets — it isn't JSON end-to-end today.
4. **IVF auto-clusters at 64 rows (`MIN_VECTORS_TO_CLUSTER`, `src/core/index/vector.rs:9`), no
   HNSW — TRUE**, and not a new idea: `IndexType::Vector`'s own doc comment already says *"linear
   scan for MVP, HNSW later"* (`src/core/index/mod.rs:11`). This plan just picks up a gap the
   codebase already named for itself.
5. **Provenance log has no pruning/checkpointing — TRUE.** `src/core/provenance.rs` has
   `append`/`save_jsonl`/`load_jsonl`/`resolve_ancestry`, nothing that shrinks the file. Real gap
   for long-running or edge pipelines. **Correction to the original proposal**: it frames this as
   protecting the engine's "100MB memory limit" — that limit
   (`ExecutionContext::with_memory_limit`, `docs/ARCHITECTURE.md:924`) governs per-execution arena
   memory, a subsystem unrelated to the on-disk `provenance.jsonl` file. The pruning need stands
   on its own merits; the stated reason for it doesn't.
6. **No BLAS backend for dense matmul — TRUE, and cheaper to address than proposed.**
   `matmul`/`matmul_with_timestamp` (`src/engine/kernels.rs:980-1030`) is a hand-rolled,
   Rayon-parallelized triple loop. But `nalgebra 0.33` is already a direct dependency
   (`src/core/linalg.rs`), and this repo's build already goes out of its way to avoid system
   library dependencies — HDF5 is vendored and `reqwest` uses `rustls-tls` specifically so the
   release binary has **no runtime dependency on system libhdf5 or OpenSSL** (`CLAUDE.md`,
   "Build" section). Linking OpenBLAS, as the original proposal suggests, would reintroduce
   exactly the kind of system dependency this project has deliberately engineered away. `faer`
   (pure Rust, no C/Fortran toolchain) is the backend that actually fits this project's existing
   build philosophy.

## Design decisions locked before implementation

1. **No work starts on Phase 3 or Phase 4 without a benchmark spike first.** Both are "this is
   probably slow" claims with no measured baseline in this codebase today. Each phase opens with
   a `cargo bench`/`hyperfine` baseline at realistic payload/matrix sizes for this engine's actual
   workloads (in-memory analytical queries over datasets that fit the existing memory-limited
   execution model — not large-scale ML training) before any implementation, and is scoped down
   or closed without shipping if the baseline doesn't show a real bottleneck.
2. **`dataset_legacy`/`dataset` unification is not scheduled.** Stays in Backlog as a
   profiling-gated future item, consistent with the existing
   `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md` precedent of ruling out unevidenced foundational
   rearchitectures.
3. **HNSW is additive, not a replacement.** New `IndexType::Hnsw` variant alongside the existing
   linear-scan/IVF paths, opted into via explicit DSL syntax (`CREATE VECTOR INDEX ... USING
   HNSW`) — consistent with this engine's "explicit decisions over silent heuristics" philosophy.
   `CREATE VECTOR INDEX` with no `USING` clause keeps today's IVF-after-64-rows default unchanged.
4. **Provenance pruning must never break `resolve_ancestry` for anything still live.** `PRUNE
   LINEAGE BEFORE <timestamp>` (new DSL statement) computes reachability from every currently-live
   tensor/dataset root and only removes `provenance.jsonl` entries with no live descendant older
   than the cutoff — never a blind time-window truncation. Errors loudly (never silently no-ops)
   if the requested cutoff would break a live object's ancestry.
5. **Server transport change, if the benchmark warrants it, is additive and opt-in.** No breaking
   change to `clients/CONTRACT.md`'s existing JSON wire shape. A new
   `Accept: application/vnd.apache.arrow.stream` path on `/execute` (or extending `/delivery/*`'s
   existing Parquet-export precedent to more endpoints) sits alongside JSON, not instead of it. A
   full Arrow Flight/gRPC service is out of scope for this plan — worth scoping separately only if
   the additive Arrow-IPC path itself proves insufficient.
6. **Matmul backend, if the benchmark warrants it, is `faer` behind a Cargo feature flag**, not
   OpenBLAS — keeps the "no system library dependency" build property intact. Existing
   correctness-first validation (singularity/shape checks that raise loud errors) stays in front
   of the kernel call regardless of which backend executes the multiply.

## Checkpoints

### Phase 1 — HNSW vector index (additive, self-contained, highest confidence)
- [x] `IndexType::Hnsw` variant (`src/core/index/mod.rs`), new `HnswIndex`
      (`src/core/index/hnsw.rs`, backed by the `instant-distance` crate -- pure Rust, no
      C/system dependency, chosen for that reason), new `CREATE VECTOR INDEX ... USING HNSW`
      DSL syntax (`src/dsl/ast.rs`'s `IndexKindAst::VectorHnsw`, `src/dsl/parser/mod.rs`)
- [x] HNSW build/query path wired into `VectorSearchExec` (top-k `SEARCH`) alongside the
      existing IVF path -- **scope correction from the original wording below**:
      `CosineFilterExec`/`SimilarityJoinExec` (the *exact*-predicate paths) stay
      `IndexType::Vector`-only, not extended to HNSW. An HNSW graph traversal has no cheap
      provable bound on what it skipped the way IVF's per-cluster angular-radius bound does, so
      it can't honor those paths' exactness contract -- `HnswIndex::search_threshold` always
      brute-force scans directly instead. An HNSW-only-indexed column still answers `WHERE
      COSINE_SIM(...) > t` correctly, just unaccelerated (full scan+filter), consistent with
      this planner's existing "recognize the shape or don't accelerate, never error"
      philosophy. Own index-persistence mechanism (`hnsw_index_graphs.json`, content-hash
      invalidated identically to `vector_index_clusters.json`) -- simpler than `VectorIndex`'s
      snapshot since `instant-distance`'s serialized graph is fully self-contained (its values
      *are* row ids), so restoring it doesn't depend on row re-insertion order.
- [x] `EXPLAIN` reports which index type was actually used (HNSW vs IVF vs none) via
      `VectorSearchExec::resolved_index_type`, resolved by the planner at plan time
- [x] Wildcard-arm grep for the new `IndexType`/`IndexKindAst` variants -- both `match`es found
      (`engine/db.rs::list_indices`, `dsl/persistence.rs` LOAD DATASET restore) were already
      exhaustive with no `_ =>` catch-all; `cargo build` itself caught every call site needing
      an arm, confirming no variant is silently swallowed
- [ ] Benchmark: recall/latency of HNSW vs IVF vs linear scan at representative dataset sizes,
      below and above the existing 64-row IVF threshold -- **deferred**, no `benches/` criterion
      bench added this round; the correctness test suite (unit + integration, see below)
      covers recall/exactness at small-to-moderate scale but not a real latency comparison
- [x] Full CI-exact test suite + `cargo clean`/`rm -rf ./data` after
- [x] Docs: `CHANGELOG.md`, `docs/DSL_REFERENCE.md`, `docs/ARCHITECTURE.md` (also corrected a
      stale, unrelated `docs/ARCHITECTURE.md` claim found while editing this same section --
      tensors persist via Parquet, not a `JsonStorage` type that no longer exists in source)

### Phase 2 — Provenance log pruning/checkpointing
- [x] `PRUNE LINEAGE BEFORE <RFC3339 timestamp string>` DSL statement (new `Token::Prune`
      lexer token, `Statement::PruneLineage`/`PruneLineageStmt` kept as raw string text in the
      AST per this file's decoupling convention, parsed into a real `chrono::DateTime<Utc>` by
      the executor with a loud error on malformed input)
- [x] Live-reachability analysis from current tensor/dataset roots before removing any
      `provenance.jsonl` entry (`ProvenanceStore::reachable_indices`/`prune_before`,
      `DatabaseInstance::live_provenance_roots`) — a record reachable from something live is
      kept regardless of age, never a blind time-window truncation
- [x] **Decided**: in-place compaction (full `provenance.jsonl` rewrite via `save_jsonl`), not a
      checkpoint file — simpler, and pruning is inherently infrequent maintenance, not a hot
      path worth optimizing the write cost of
- [x] Regression coverage: 4 new `core::provenance` unit tests with deterministic timestamps
      (removes old+unreachable, keeps reachable-even-if-stale, keeps young-even-if-unreachable,
      a mixed pass doing both in one call) + 3 new end-to-end DSL integration tests in
      `tests/lineage_provenance_test.rs` proving the live-tensor-ancestry-never-breaks guarantee
      through the real `PRUNE LINEAGE` statement (a future cutoff, a past cutoff, a malformed
      timestamp) — mechanical *removal* is covered precisely at the unit level since there's no
      DSL-level way to make something "no longer live" short of `DROP DATABASE`, which would
      make the removal case trivial/uninteresting to test end-to-end
- [x] Full CI-exact test suite + `cargo clean`/`rm -rf ./data` after
- [x] Docs: `CHANGELOG.md`, `docs/DSL_REFERENCE.md`, `docs/ARCHITECTURE.md` (also notes the
      corrected motivation: disk growth, not the unrelated 100MB execution-memory limit the
      original proposal conflated it with)

### Phase 3 — Server transport benchmark spike + (conditional) additive Arrow IPC path
- [x] Baseline (`benches/server_transport.rs`, new criterion bench, registered in `Cargo.toml`):
      measured JSON serialization, **and** the actual production default `toon` encoding
      (`toon_format::encode_default`), against a hand-rolled Arrow IPC encode of the same
      representative payload (10k rows × `Vector(128)` embedding column) — **scope correction
      from the original wording**: the original plan assumed JSON was `/execute`'s default: it
      is not, `toon` is (`json` is a legacy opt-in), so benchmarking only against JSON would
      have answered the wrong question
- [x] **Gate result: opened.** Arrow IPC was ~80-180x faster and ~2.4x smaller on the wire than
      JSON across 100/1,000/10,000-row payloads — a real, measured cost, not a guess. **Unplanned
      finding, surfaced to the user via AskUserQuestion before proceeding rather than decided
      unilaterally**: the real default (`toon`) is itself substantially slower/larger than even
      `json` for the same payload (~426ms/30MB vs `json`'s ~36ms/13MB vs `arrow`'s ~233µs/5.5MB
      at 10k rows) — reported as a measured fact in `CHANGELOG.md`/`docs/ARCHITECTURE.md`, not
      fixed or asserted to be a bug, since `toon`'s design goal is believed to be LLM-facing
      token efficiency rather than wire/CPU efficiency, and deciding whether that tradeoff is
      still acceptable is the maintainer's call, not this session's.
- [x] Implemented: `?format=arrow` on `POST /execute`, additive third option alongside default
      `toon` and opt-in `json`. Only a successful `DslOutput::Table` result is encoded as real
      Arrow IPC bytes (`dataset_to_arrow_ipc_bytes`, reusing `dataset_to_record_batch` -- the
      same conversion `/delivery`'s Parquet export already trusts); anything else under
      `?format=arrow` (an execution error, or a non-tabular success) falls back to a JSON body,
      matching this endpoint's existing error-always-falls-back-to-JSON convention. No change to
      `toon`/`json`'s existing behavior.
- [x] Docs: `clients/CONTRACT.md` §1 (wire contract for `?format=arrow`, including the
      fall-back-to-JSON behavior a client must handle), `docs/ARCHITECTURE.md`, `CHANGELOG.md`
- [x] Tests: 2 new server integration tests (`tests/server_test.rs`) -- a real `Table` result
      decoded by a real `arrow::ipc::reader::StreamReader` (not just "bytes with the right
      header"), and a non-tabular result confirmed to fall back to JSON -- plus the existing 8
      `server_test.rs` tests reconfirmed unaffected

### Phase 4 — Dense matmul backend benchmark spike + (conditional) `faer` integration
- [ ] Baseline: current Rayon-parallelized `matmul` vs. `faer` at matrix sizes representative of
      this engine's actual usage (scientific/engineering datasets fitting the existing
      memory-limited execution model, not ML-training-scale)
- [ ] **Gate**: only proceed if `faer` shows a meaningful win at those realistic sizes, not just
      at synthetic large-N benchmarks
- [ ] If gated open: `faer` behind a Cargo feature flag; existing correctness-first validation
      (singularity/shape checks) unchanged in front of the kernel call
- [ ] If gated closed: close this phase with the benchmark results recorded in `CHANGELOG.md`
- [ ] Docs: `CHANGELOG.md`, `docs/ARCHITECTURE.md` (kernel selection section) if shipped

## Backlog (not scheduled — profiling-gated future items)

- **`dataset_legacy`/`dataset` unification**: only revisit if profiling of a real workload shows
  `materialize_tensor_dataset()` is an actual bottleneck, and even then scope it narrowly first
  (e.g., caching the materialized form across repeated joins on the same dataset) before
  considering the full single-columnar-layout rewrite the original proposal described. Read
  `docs/DATASET_ARCHITECTURE.md` in full before touching this.
- **Full Arrow Flight / gRPC transport**: only worth scoping as its own plan if Phase 3's
  additive Arrow-IPC path ships and still isn't enough for a demonstrated real workload.

## Verification (per phase, before moving to the next)

Mirrors `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md`'s existing convention:
1. `cargo fmt -- --check` / `cargo clippy -- -D warnings` clean.
2. Full CI-exact suite (`CARGO_INCREMENTAL=0 RUSTFLAGS="-C codegen-units=1" cargo test --release
   -j 1 -- --skip test_cli_init --skip test_cli_run_multiline --skip test_cli_serve_alias`).
3. The phase's wildcard-arm grep, where a new enum variant is introduced.
4. A real end-to-end `.lnl` showcase exercising the new capability, not just isolated unit tests —
   per this repo's own recurring lesson (`CLAUDE.md`, "Recurring project lesson") that this finds
   bugs isolated tests consistently miss.
5. `cargo clean` (and `rm -rf ./data`) after the test run.
6. Docs updated per phase before the PR is opened.
7. Normal branch/CI/merge flow (`main` protected, 4 required checks); version bump/release are
   separate, deliberate follow-up PRs.
