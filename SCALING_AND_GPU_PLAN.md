# Scaling & GPU Plan

**Status**: Track A done, Tracks B–C not started. Per this repo's tracked-plan-doc convention,
when the last track closes this file is deleted. The analysis and backlog sections move to
`docs/SCALING_AND_GPU_ROADMAP.md` as a permanent reference.

## Why this plan exists

Came out of a design review on 2026-09-24: what would it take for LINAL to (a) scale
horizontally across nodes and (b) compute on GPU/VRAM, how would that fit with today's engine,
and is it worth it? Every claim below was checked against `main` at v0.1.89. The review found
three steps worth taking now:

1. **Track A**: per-database locking in `linal serve`.
2. **Track B**: a write-ahead log.
3. **Track C**: a measured GPU spike.

Everything else is a backlog item that needs evidence first, the same gate
`PERFORMANCE_OPTIMIZATION_PLAN.md` uses.

## Checkpoints

### Track A — Per-database locking in `linal serve`

The problem:
- The server shared a single `Arc<RwLock<TensorDb>>`.
- `Statement::is_read_only()` only covers `EXPLAIN`/`AUDIT`/`LIST`/`DELIVER`. So every
  `SELECT`, `SHOW` and `SEARCH` took the **write** lock over *all* databases.
- `X-Linal-Database` worked by mutating the one global `active_db` and restoring it afterwards.

Checklist:
- [x] `server::engine::SharedEngine`: one single-database `TensorDb`
      (`TensorDb::from_instance`) per database, each behind its own `RwLock`.
      - Catalog statements (`CREATE`/`DROP`/`USE DATABASE`, `SHOW DATABASES`) are answered by the
        router, with byte-identical messages.
      - Session semantics are modeled explicitly (`Session::{Server, Pinned, Following}`).
- [x] Pipelines stay session-wide across databases. `TensorDb.pipelines` is now a shared
      `PipelineRegistry` (`Arc<RwLock<..>>`).
- [x] `/execute/batch`: consecutive statements on one database run under one write-lock hold,
      and `USE` semantics inside a batch are unchanged.
- [x] Jobs and scheduler ported with **their existing semantics preserved**:
      - The scheduler's `target_db` switch stays permanent (documented design, locked by
        `test_schedule_target_db_switch_is_permanent`).
      - A headerless job's `USE` still doesn't outlive the job.
- [x] Wider read-lock path: `SHOW` (except `SHOW <name>` on a lazy tensor, which materializes
      it) and `DESCRIBE PIPELINE` now run through `execute_line_shared`
      (`dsl::can_execute_shared`).
      - `SELECT`/`SEARCH` stay on the write path: `execute_select` creates datasets.
- [x] `InMemoryTensorStore::get` is O(1): an id index alongside the insertion-ordered `Vec`.
- [x] Tests: `tests/server_per_db_locking_test.rs`, 10 cases.
- [x] Measured with release builds, 300k-row CSV in db `a`, two threads looping a `GROUP BY` on
      `a`, 20 single writes on db `b`:

      | | median | p95 | max |
      |---|---|---|---|
      | v0.1.89 (global lock) | 42.8 ms | 83.2 ms | 121.8 ms |
      | per-database locks | 0.5 ms | 0.5 ms | 0.7 ms |

Flagged, not changed (need a maintainer decision):
- A headerless `/jobs` submission restores the previous active database, so its `USE` is
  silently undone. This is the same class of bug fixed for `/execute` in v0.1.74.
- The scheduler's `target_db` switch leaks into every later headerless request (documented,
  intentional).
- `/delivery` hard-codes `./data` (`server/mod.rs`, `ParquetStorage::new("./data")`) instead of
  `config.storage.data_dir`.

### Track B — Write-ahead log (not started)

The problem: there is no WAL and no fsync anywhere. On restart, `recover_databases` recreates
empty instances and reloads `provenance.jsonl`, so anything not `SAVE`d is lost.

Checklist:
- [ ] Config: `[storage] wal = "off" | "on"` (default `off`, no behavior change) and
      `wal_sync = "always" | "batch"`.
- [ ] `src/core/wal.rs`: `{db_dir}/wal.log`, JSONL `{seq, ts, statement}`.
      - Append only after a statement succeeds, and only if it mutates.
      - `Statement::is_mutating()` is an exhaustive `match` with no `_` arm, so every new
        variant must be classified.
- [ ] Determinism: statements that depend on external files (`IMPORT`, `LOAD`, `USE DATASET`)
      are re-executed on replay. Replay fails loudly if the file is gone. Persist timestamps in
      the record where the executor already accepts `*_with_timestamp`.
- [ ] `CHECKPOINT`: saves every named dataset/tensor (`save_dataset_core`/`save_tensor_core`),
      writes `checkpoint.json` (names + seq), then atomically truncates the WAL
      (`wal.log.new` + `rename`).
- [ ] Recovery on startup (with `wal = on`): load the checkpoint, then replay `seq >
      checkpoint.seq`.
      - A truncated tail record is dropped with a warning.
      - Corruption mid-file is a loud error.
- [ ] Tests: simulated crash, truncated tail, checkpoint + replay, `wal = off` creates nothing,
      missing import file.
- [ ] Docs: `DSL_REFERENCE.md` (`CHECKPOINT`), `ARCHITECTURE.md` (Persistence),
      `ERROR_REFERENCE.md`.

### Track C — GPU spike, `gpu-wgpu` feature (not started)

Goal: measure first, without committing the architecture. `Tensor.data` is untouched, and data
is copied host→device per operation.

Checklist:
- [ ] Optional `wgpu`/`pollster`/`bytemuck` behind `gpu-wgpu`. All are pure Rust (Metal on
      macOS, Vulkan/DX12 elsewhere). The feature is outside the default build and the CI matrix,
      like `faer-matmul`.
- [ ] `src/core/backend/gpu/`: `GpuBackend: ComputeBackend`.
      - Accelerates `matmul`, `dot` and `cosine_similarity` above a size threshold.
      - Everything else, and anything below the threshold, delegates to an inner `CpuBackend`.
      - If no adapter is found, it falls back to CPU with a warning. No panic.
- [ ] `[compute] backend = "cpu" | "gpu"` (default `cpu`). `DatabaseInstance::new` builds the
      backend from config. `SHOW BACKEND` reports it.
- [ ] `benches/gpu_backend.rs`:
      - matmul 256²–4096²: CPU-SIMD vs `faer` vs GPU, transfer included;
      - batched cosine 10k–1M × 384/768.
- [ ] Parity tests CPU↔GPU (relative tolerance 1e-4, including transposed/sliced views). They
      skip cleanly without an adapter.
- [ ] **Decision gate**: record the results here. If GPU wins at realistic sizes, open the
      residency phase (Backlog 1). If not, record that and stop.

## Analysis (verified against v0.1.89)

### How the engine is built today

| Aspect | Today | Where |
|---|---|---|
| Server concurrency | One `Arc<RwLock<TensorDb>>` shared by every request *(fixed by Track A)* | `src/server/mod.rs` |
| Tensor | `Arc<Vec<f32>>` + shape/strides/offset: f32, **host memory only** | `src/core/tensor.rs` |
| SQL execution substrate | Row-oriented `dataset_legacy::Dataset { rows: Vec<Tuple> }`, a boxed `Value` per cell | `src/core/dataset_legacy.rs` |
| Physical plan | `execute(&self, db: &TensorDb) -> Vec<Tuple>`: fully materialized, bound to the local engine | `src/query/physical.rs` |
| Partitions | Zone maps over 1024-row batches, for pruning only (not physical partitions) | `dataset_legacy.rs`, `planner.rs` |
| Compute backend | `ComputeBackend` trait (elementwise, reductions, matmul, dot/cosine), `CpuBackend` hard-wired per instance | `src/core/backend/`, `engine/db.rs` |
| Backend bypass | Classical linalg (`linalg.rs`, ~29 call sites) calls `nalgebra` directly. Slicing/indexing/lazy eval call `kernels::` directly | `src/core/linalg.rs` |
| Lazy graph | `Expression` DAG (Add/Sub/Mul/Div/MatMul/ScalarMul/Normalize/Sum/Mean…) | `src/core/tensor.rs` |
| Durability | Explicit `SAVE`/`LOAD` to Parquet, append-only `provenance.jsonl`. No WAL, no fsync, no replication *(Track B)* | `src/core/storage.rs` |
| Transport | REST (JSON/TOON) + opt-in Arrow IPC. No Flight/gRPC | `src/server/mod.rs` |

**What already helps:**
- `ComputeBackend` is a real injection point for a GPU backend.
- The `Expression` DAG is what kernel fusion needs.
- Content-hash provenance can key a VRAM residency cache and deduplicate transfers between nodes.
- Arrow is already a dependency: it's the natural interchange format between nodes and with GPU
  libraries (DLPack/cuDF).
- Zone maps are the seed of a partition catalog.

**What blocks both GPU and distribution:**
1. Host memory is the only place a tensor can live.
2. Row-at-a-time execution: neither GPUs nor network shuffles pay off on boxed `Value` rows.
3. The physical plan is bound to a local `&TensorDb`.
4. A single global lock *(Track A)*.
5. No durable change log to replicate *(Track B)*.

### Horizontal scaling options

| Option | What | Cost | Verdict |
|---|---|---|---|
| **H0** per-DB locking | Track A | days | **Done** |
| **H1** cluster per database | Route by `X-Linal-Database` (one DB per node), WAL-shipped read replicas, Parquet in object storage via `object_store` (pure Rust) | 1–2 months | Not yet. Needs real multi-tenant demand. Needs Track B |
| **H2** distributed queries | Physical partitions, Exchange operators (shuffle/broadcast/gather), plan signature becomes a `RecordBatch` stream, Arrow Flight between nodes. Top-k vector search distributes well (local top-k per shard, then merge). Distributed SVD/eig (TSQR, randomized SVD) only pays off for very large matrices | quarters | Not now. If ever: evaluate Apache DataFusion (+ Ballista) instead of a custom engine. Requires the columnar `dataset_legacy`↔`dataset` unification, which is parked in `PERFORMANCE_OPTIMIZATION_PLAN.md`'s backlog |

### GPU options

| Option | Platforms | "No system deps" fit | Linear algebra | Verdict |
|---|---|---|---|---|
| **wgpu** (WGSL) | Metal, Vulkan, DX12, Apple Silicon | Excellent (pure Rust) | Hand-written kernels | First step (Track C) |
| **cudarc** (cuBLAS/cuSOLVER, dlopen) | NVIDIA | Good: loaded at runtime, not a link-time dependency | Best: GEMM, SVD/eig/QR, cuVS ANN | Second step, if Track C wins |
| Burn / CubeCL | CUDA, wgpu, ROCm, Metal | Good | General tensors, weak on classical linalg | Alternative if one codebase for many targets matters; large dependency |
| candle | CUDA, Metal | Medium | ML-oriented | Poor fit |

**What to accelerate, by return:**
1. Dense matmul and `Expression` chains.
2. Exact brute-force vector search. A GEMM followed by top-k beats HNSW up to millions of vectors
   on GPU, and it can honor `search_threshold` exactly, which HNSW can't.
3. Decompositions via cuSOLVER. This needs a `LinalgBackend` trait to end `linalg.rs`'s direct
   `nalgebra` calls.
4. Reductions, but only on large columns.
5. FFT.

### Pros, cons, verdict

| Feature | Effort | Pros | Cons | Worth it? |
|---|---|---|---|---|
| Per-DB locking | days | More concurrent users. Removes latent concurrency bugs. No DSL/contract change. Needed by everything else | Adds no memory/compute | **Yes — done** |
| WAL | 2–4 weeks | Real durability. Groundwork for replicas | fsync latency. Statement-vs-delta logging decision | **Yes, next** |
| wgpu spike | 2–4 weeks | 10–50× on large matmul/cosine. Exact *and* fast vector search. Runs on a Mac. Product differentiator | Slower on small data (transfer cost). Numeric tolerances. Hand-written kernels | **Yes, as a measured spike** |
| Full GPU (VRAM residency, CUDA, DLPack, `[cuda]` wheels) | 1–2 months | Peak performance. Zero-copy PyTorch/CuPy interop | Invasive `Tensor.data` change. Provenance hash vs device. GPU CI cost. Two backends to maintain | Conditional on the spike |
| Cluster per DB | 1–2 months | Multi-tenant scale. HA. Stateless nodes | Doesn't help a DB bigger than one node. Eventual consistency. Ops burden | Not yet |
| Distributed queries | quarters | Only path past one node's memory | Columnar rearchitecture. Shuffle may eat the gains. Over-engineering risk | Not now |

## Backlog (evidence-gated, not scheduled)

1. **VRAM residency + fusion** (only if Track C wins):
   - `TensorStorage { Host(Arc<Vec<f32>>), Device(..) }` with lazy migration.
   - A VRAM cache keyed by content hash, with a VRAM budget in `EngineConfig`.
   - GPU evaluation of `Expression`.
   - Explicit `SET DEVICE`, and the device shown in `EXPLAIN`.
   - A device field in provenance records (not part of the hash).
2. **`gpu-cuda`**:
   - `cudarc` + cuBLAS/cuSOLVER.
   - A `LinalgBackend` trait.
   - Optional cuVS ANN indexes.
   - `linaldb[cuda]` wheels and DLPack.
3. **H1 cluster per database** (needs Track B).
4. **H2 distributed queries**: DataFusion/Ballista evaluation first. Requires the columnar
   unification.

## Process for every PR

- `cargo fmt -- --check`, `cargo clippy -- -D warnings`, and the full `cargo test`. For Track C,
  also `cargo test --features gpu-wgpu`.
- Then `cargo clean` and `rm -rf ./data`.
- A CHANGELOG entry under `[Unreleased]`. There is no version bump in feature PRs: a release is
  a separate PR.
- Before/after numbers recorded in this file.
