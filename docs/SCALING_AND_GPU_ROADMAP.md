# Scaling & GPU Roadmap

The permanent record of LINAL's horizontal-scaling and GPU design review (2026-09-24): what was
analyzed, what was built and measured, and what's deliberately deferred and why. For how to
scale a deployment *today*, see `ARCHITECTURE.md` → "Scaling & Deployment". This document
explains the reasoning behind that section.

This started as the tracked plan `SCALING_AND_GPU_PLAN.md` at the repo root. All three tracks closed, so per the
repo convention it moved here as a reference.

## Why this exists

The design review asked three things:
- What would it take for LINAL to scale horizontally across nodes?
- What would it take to compute on GPU/VRAM?
- How would that fit with the engine as it was, and is it worth it?

Every claim was checked against `main` at v0.1.89. Three steps were worth taking immediately:
- **Track A**: per-database locking in `linal serve`.
- **Track B**: a write-ahead log.
- **Track C**: a measured GPU spike.

All three shipped (#127, #128, #129). Everything else is a backlog item that needs evidence
first, the same gate `PERFORMANCE_OPTIMIZATION_PLAN.md` uses.

## Completed work

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
      - `SELECT`/`SEARCH` stayed on the write path at the time, because `execute_select` created
        datasets. That was fixed later: see backlog item 2.
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

### Track B — Write-ahead log

The problem: there was no WAL and no fsync anywhere. On restart, `recover_databases` recreated
empty instances and reloaded `provenance.jsonl`, so anything not `SAVE`d was lost.

Checklist:
- [x] Config: a `[wal]` section with `enabled` (default `false`), `sync = "always" | "never"`
      and `checkpoint_bytes`.
- [x] `src/engine/wal.rs`: `{db_dir}/wal.log`, JSONL `{seq, ts, statement, fingerprints}`.
      - Appended only after success, and only if `Statement::is_mutating()` (exhaustive, no
        `_` arm).
      - Hook: `dsl::execute_logged`, the one funnel for CLI, REPL, scripts, server and bindings.
- [x] External inputs: `LOAD`/`IMPORT` records fingerprint what they loaded. Replay fails loudly
      on a mismatch.
- [x] `CHECKPOINT` + `src/engine/db/snapshot.rs`: a private snapshot (tensor ids, aliases,
      views, lazy tensors, tensor-first datasets, record datasets via the SAVE/LOAD package path,
      pipelines).
      - Swapped in by rename, then the log is truncated.
      - Automatic after every `SAVE` and past `checkpoint_bytes`.
- [x] Recovery on startup: restore the checkpoint, then replay `seq > checkpoint.seq`.
      - Provenance is suppressed during replay.
      - A torn tail is dropped and the file is rewritten.
      - Mid-file corruption is an error.
      - A failed recovery leaves the database refusing every statement (`recovery_error`).
- [x] Tests: `tests/wal_test.rs` (9) + `engine::wal` unit tests (4).
- [x] End-to-end: `linal serve` with the WAL on, writes to two databases (dataset + vector
      index), `kill -9`, restart. Everything served back, including after a `CHECKPOINT` +
      further writes.
- [x] Measured: INSERT median over HTTP was 0.25 ms (off), 0.23 ms (`never`) and 5.0 ms
      (`always`, macOS `F_FULLFSYNC`).
- [x] Docs: `DSL_REFERENCE.md` §8, `ARCHITECTURE.md` (Recovery + Write-ahead log),
      `ERROR_REFERENCE.md`.

Deviations from the original design, and why:
- **Config.** It's a `[wal]` section rather than keys under `[storage]`. Tests and embedders
  build `StorageConfig` literally, and a separate `#[serde(default)]` section keeps every
  existing `linal.toml` parsing unchanged.
- **Sync modes.** They're `always`/`never` rather than `always`/`batch`. A timed batch flush
  needs a background flusher thread per database. `never` gives the same crash-only durability
  at zero cost.
- **Timestamps.** Original timestamps aren't re-injected on replay: a replayed tensor's
  `created_at` is the replay time. Ancestry is unaffected, because provenance is content-hash
  addressed and already durable.
- **Checkpoint mechanism.** Checkpoints are a private snapshot, not "SAVE everything". Going
  through user-visible SAVE would have bumped user dataset versions, lost aliasing, and changed
  tensor ids (which tensor-first datasets reference). The automatic checkpoint after `SAVE` was
  added because replaying a `LOAD` after a later `SAVE` of the same package would have
  double-applied changes. That was found while designing this track, not in the original plan.

### Track C — GPU spike, `gpu-wgpu` feature

Goal: measure first, without committing the architecture. `Tensor.data` is untouched, and every
GPU call uploads its inputs and reads the result back.

Checklist:
- [x] Optional `wgpu` 30 / `pollster` / `bytemuck` behind `gpu-wgpu`. All are pure Rust. The
      feature is outside the default build and the CI matrix, like `faer-matmul`.
- [x] `src/core/backend/gpu/`: `GpuContext` (process-wide device, WGSL tiled GEMM + batched
      cosine) and `GpuBackend: ComputeBackend`.
      - `GpuBackend` sends rank-2 `matmul` ≥ 2M multiply-adds to the GPU. Everything else,
        including the per-pair `dot`/`cosine`, goes to an inner `CpuBackend`.
      - With no adapter or no feature, it warns and falls back to the CPU.
- [x] `[compute] backend = "cpu" | "gpu"` (default `cpu`), applied per database, and
      `SHOW BACKEND`.
- [x] Parity tests (`tests/gpu_backend_test.rs`, 4 cases, relative tolerance 1e-4, including a
      transposed view and the DSL `MATMUL` path). They skip without an adapter. Run on the Apple
      M4 (Metal).
- [x] `benches/gpu_backend.rs`, with results below.
- [x] **Decision gate: closed, not proceeding to VRAM residency on this evidence.**

Results (Apple M4: 10 CPU cores, 8-core GPU, 16 GB unified memory; criterion medians; GPU
includes the per-call transfer):

| matmul (square) | `CpuBackend` (DSL `MATMUL` path) | `faer` | GPU (wgpu) |
|---|---|---|---|
| 256 | 2.62 ms | 0.16 ms | 0.82 ms |
| 512 | 21.6 ms | 0.74 ms | 3.84 ms |
| 1024 | 169 ms | 4.97 ms | 25.6 ms |
| 2048 | 1.34 s | 45.0 ms | 196 ms |
| 4096 | 10.8 s | 344 ms | 1.52 s |

| batch cosine (rows × dim) | CPU (Rayon) | GPU (wgpu) |
|---|---|---|
| 10k × 384 | 0.33 ms | 3.01 ms |
| 100k × 384 | 2.95 ms | 29.5 ms |
| 1M × 384 | 28.6 ms | 471 ms |
| 10k × 768 | 0.66 ms | 5.16 ms |
| 100k × 768 | 6.14 ms | 55.6 ms |
| 1M × 768 | 59.9 ms | 998 ms |

What the numbers say:
- **Matmul.** The GPU beats the engine's current CPU kernel by 3–7×, but `faer` beats the GPU by
  4–5× at every size. A hand-written WGSL GEMM on an integrated 8-core GPU isn't competitive
  with a tuned CPU GEMM.
- **Batched cosine.** The GPU is 10–16× *slower*. The work is memory-bound, and re-uploading the
  whole matrix on every query dominates. On unified memory the CPU scan already runs at ~54 GB/s
  (1.5 GB in 28.6 ms), so VRAM residency could at best recover the transfer, not beat the CPU by
  much.
- **Not settled.** This spike can't say how CUDA/cuBLAS/cuVS would do on a discrete datacenter
  GPU, where compute is ~10–50× higher and a resident index avoids the transfer. That's the only
  path left worth measuring (Backlog 2), and only if a real workload lives on NVIDIA hardware.

**The actual win found by this spike** (flagged here, then fixed; see "Follow-up: DSL `MATMUL` on
`faer`" below):
- The DSL's `MATMUL` (`eval_matmul` → `backend.matmul` → `SimdBackend::matmul_simd`) never
  reaches `faer`.
- The `faer-matmul` feature only swaps `engine::kernels::matmul`. That's reached from
  `ScalarBackend` (matrices under 1024 elements), from non-contiguous views, and from lazy
  `MatMul` expressions.
- `PERFORMANCE_OPTIMIZATION_PLAN.md` Phase 4's benchmark compared `faer` against
  `kernels::matmul`, not against the SIMD path the DSL actually uses.
- Routing `SimdBackend`/`CpuBackend::matmul` through `faer` when the feature is on would make
  DSL `MATMUL` **~34× faster at 1024²** (169 ms → 5 ms), with no GPU.

### Follow-up: DSL `MATMUL` on `faer` (done)

The maintainer approved acting on the finding above:
- `CpuBackend::matmul` now calls `kernels::matmul_with_timestamp` (faer) whenever `faer-matmul`
  is enabled.
- `faer-matmul` became a **default** feature, so CI, the release binaries and the Python/R
  bindings all get it. `--no-default-features` keeps the old SIMD/scalar path.
- `tests/cpu_matmul_backend_test.rs` covers:
  - parity with an f64 reference across tiny, odd and large shapes;
  - transposed views;
  - shape errors;
  - bit-for-bit determinism over repeated multithreaded runs at 512²;
  - the DSL `MATMUL` path.

  All pass with and without the feature.
- faer reads row-major and column-major inputs as views over their existing buffers, and writes
  straight into the output.
- Measured in `benches/matmul_backend.rs` (`cpu_backend`, Apple M4, "before" =
  `--no-default-features`):

  | n × n | before (SIMD) | after (faer) | speedup |
  |---|---|---|---|
  | 50 | 26 µs | 4.3 µs | 6.1× |
  | 200 | 1.28 ms | 0.105 ms | 12.1× |
  | 500 | 20.1 ms | 0.69 ms | 29.1× |
  | 1000 | 157 ms | 4.57 ms | 34.3× |
  | 2048 | 1.31 s | 45.9 ms | 28.6× |

## Analysis (as verified against v0.1.89, before Tracks A–C)

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

Ordered by expected return per effort. **Near-term** (cheap, measured or clearly scoped):

1. ~~Route DSL `MATMUL` through `faer`~~: **done** (see "Follow-up" under Track C).
   `faer-matmul` is now a default feature.
2. ~~Let `SELECT`/`SEARCH` run under a read lock~~: **done**.
   - **How.** CTEs and `FROM` subqueries now live in a per-query scope (`LogicalPlan::Values`)
     instead of the catalog. `execute_select` and `run_search` take `&TensorDb`, so reads being
     read-only is compiler-enforced. `SEARCH ... INTO` still writes.
   - **Bug fixed along the way.** A `FROM (SELECT ...) AS x` alias used to leak as a permanent
     dataset, and re-running the query failed.
   - **Measured** (Apple M4, concurrent `GROUP BY` on one database): 1.4× with 2 clients and
     1.8× with 4 (300k rows); 1.5× with 4 (30k rows). Throughput flattens or dips past ~4.
   - **Next: cheaper scans.** `SeqScanExec` clones every row of the dataset on every query, and
     large queries already parallelize internally with Rayon. Together these cap concurrent-read
     scaling at ~4 queries on a 10-core machine. The fix is to scan by reference (or go
     columnar, which H2 needs anyway), and possibly to bound per-query Rayon parallelism under
     load.
3. **Small server items flagged during Track A**:
   - a headerless `/jobs` `USE` is undone when the job finishes;
   - a scheduled task's `target_db` switch leaks into later headerless requests (documented as
     intentional);
   - `/delivery` hard-codes `./data` instead of `config.storage.data_dir`, which matters when
     several instances share a host.

**Horizontal** (needs real multi-tenant/HA demand):

4. **H1: read replicas per database.**
   - Ship `wal.log` plus `checkpoint/` to another node, which replays them (the Track B machinery,
     already there).
   - Put dataset packages in object storage (`object_store`, pure Rust).
   - Promote a replica on failover.
   - Open questions: replay verifies external-file fingerprints, so inputs must be reachable from
     the replica too; and routing reads vs. writes.
5. **H2: distributed queries.** Cross-instance joins, and one database spread over nodes.
   - Evaluate DataFusion/Ballista first.
   - Requires the columnar `dataset_legacy`↔`dataset` unification.

**GPU** (the Track C spike did *not* justify these on integrated/unified-memory hardware; revisit
only with discrete-GPU evidence):

6. **VRAM residency + fusion**:
   - `TensorStorage { Host(Arc<Vec<f32>>), Device(..) }` with lazy migration.
   - A VRAM cache keyed by content hash, with a budget in `EngineConfig`.
   - GPU evaluation of `Expression`.
   - `SET DEVICE`, and the device shown in `EXPLAIN`.
   - A device field in provenance records (not part of the hash).
7. **`gpu-cuda`**:
   - `cudarc` + cuBLAS/cuSOLVER.
   - A `LinalgBackend` trait.
   - Optional cuVS ANN indexes.
   - `linaldb[cuda]` wheels and DLPack.
