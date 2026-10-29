# Improvements plan (post v0.1.94)

**Status**: PR 1 in progress. Single tracked plan for everything proposed after engine v0.1.94 /
`linaldb` 0.1.19. Each PR below ticks its row here; per this repo's tracked-plan convention, this
file is deleted in the commit that closes the last PR we decide to do.

It replaces these documents. Their content that still matters is folded in below; the
committed ones stay in git history, and the never-committed proposals survive only where the
table says:

| Source | Items | Where it went |
|---|---|---|
| `LINALDB propuesta de cambios ... (caso CASMI 2026).md` (round 1) | P1–P7 | Done, v0.1.92–0.1.93. Deleted |
| `CASMI_WORKLOADS_PROPOSAL_2.md` (round 2) | P8–P15 | Done, v0.1.94. Deleted |
| `CASMI_LOAD_MEMORY_PROPOSAL.md` | P16–P20 | PR 2, PR 4 below. Deleted |
| `CASMI_WORKLOADS_PROPOSAL_4.md` | P21–P23 | PR 5, backlog. Deleted |
| `CASMI_PUBCHEM_STORE_PROPOSAL.md` | G1–G6 | Backlog. Deleted (full design also on branch `oc/P0e` as `docs/PUBCHEM_STORE_DESIGN.md`) |
| `CASMI_RELEASE_WORKFLOW_PROPOSAL.md` | R1–R8, W0–W4, D1–D4 | PR 1, PR 3. Deleted (copy in CASMI repo, `docs/LINALDB_RELEASE_WORKFLOW.md`) |
| CASMI `docs/LINALDB_SUGGESTIONS.md` (2026-10-28) | DROP DATASET, MMAP for scalar-only datasets, slow wide `SPEC_COSINE_MOD` | PR 2, PR 5, backlog |
| `CASMI_WORKLOADS_PLAN.md`, `CASMI_WORKLOADS_PLAN_2.md`, `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md` | — | Closed. Deleted per the tracked-plan convention; code comments that cite them by name resolve through git history |
| `PERFORMANCE_OPTIMIZATION_PLAN.md` | one open item | Backlog. Deleted |
| `clients/CRAN_PUBLISHING_PLAN.md` | R packages to CRAN | Kept as its own plan and now tracked (detailed audit, cited by `ARCHITECTURE.md`); listed in the backlog |

Rules for every PR (unchanged from the CASMI rounds): validate against an independent
implementation, fail loudly, keep new behavior opt-in, update `DSL_REFERENCE.md`,
`ARCHITECTURE.md`, `CHANGELOG.md` and the client contracts in the same PR, record new operations
in lineage. Engine PRs go through a release and a `linal-hub` re-run before the hub docs change.

## Deadline that drives the order

CASMI 2026 requires the winning code under an OSI license. The Kaggle kernel must pin a `linaldb`
wheel that is **already published under the new license before 2026-12-07** (CASMI
`docs/ELIGIBILITY_AUDIT.md`, `docs/PLAN_PRIZE.md`). Everything else in this plan is
nice-to-have for CASMI: every gap has a working workaround on the CASMI side.

## Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| D1 | License: MIT, Apache-2.0, or both | **Decided 2026-10-29: Apache-2.0** (see "License decision" below) |
| D2 | One version for crate, wheels and binaries, or keep two lines | Keep two lines for now, add a consistency check; unify later if it keeps hurting |
| D3 | Add Linux aarch64 / macOS x86_64 targets now or after CASMI | After CASMI |
| D4 | Earlier versions stay under the Community License or are relicensed | Stay as they are; the new license applies from the first release that ships it |

All commits are by the owner's accounts, so no contributor sign-off is needed to relicense.

## License decision (D1)

**Decision (owner, 2026-10-29): relicense LINALDB under Apache License 2.0**, starting with the
first release that ships PR 1. Chosen over MIT and over dual `MIT OR Apache-2.0` for the stronger
protection, while still meeting the CASMI rules.

### What Apache-2.0 gives us

- **Express patent grant** (§3) from every contributor to every user, with **patent retaliation**:
  anyone who sues claiming LINALDB infringes a patent loses their patent license. MIT is silent on
  patents.
- **Trademark reservation** (§6): the license does not grant use of the "LinalDB"/"LINAL" names
  or the gorigami marks. Forks may use the code, not the name. The license only reserves the
  marks; enforcing them needs a registered trademark (owner action, outside this plan).
- **NOTICE propagation** (§4d): redistributors must carry our `NOTICE` (attribution to gorigami)
  and mark files they changed.
- **Contribution terms** (§5): contributions are under Apache-2.0 unless stated otherwise, so no
  separate CLA is needed for outside PRs.
- The license used by comparable engines (Arrow, DataFusion, Qdrant, Chroma), so companies and
  their legal teams already accept it.

### What we give up (accepted)

- **The commercial restriction ends** for new versions: anyone may sell, embed or host LINALDB
  (including as DBaaS) without a commercial license. Revenue options left: support/consulting
  (the DuckDB Labs model), a hosted service, and the name.
- **GPLv2-only projects can't include it** (Apache-2.0 is GPLv3-compatible, not GPLv2). Minor for
  a database engine.
- Not chosen: AGPL-3.0 (keeps SaaS pressure, but enterprises avoid it and it clashes with the
  CASMI winner license); source-available licenses like BSL/SSPL (not OSI, so they fail CASMI and
  have caused forks elsewhere: Redis → Valkey, Elastic → OpenSearch, Terraform → OpenTofu).

### Earlier versions

Releases up to engine v0.1.94 / `linaldb` 0.1.19 (and any release before PR 1) **stay under the
LinalDB Community License v1.0**; we don't relicense history (D4). The README License section
says this explicitly, with the first Apache-2.0 version numbers.

### How it fits CASMI 2026

The rules (Kaggle CASMI 2026 → Rules; re-check the exact wording there while logged in) have
two separate requirements:

1. **External tools and code** used by the solution must be under an OSI-approved license that
   does not restrict commercial use. Apache-2.0 meets this. The Community License does not, which
   is the current blocker (CASMI `docs/ELIGIBILITY_AUDIT.md`).
2. **Winner license** (listed as "Open Source – MIT"): a winning team licenses *its submission
   and the source code used to generate it* under MIT. This applies to the CASMI repo
   (`CASMI-2026`, kernels, training and pipeline code), which is MIT.

Our position: LINALDB is an **external, general-purpose library**. It is public, released
through public CI to PyPI, has its own changelog and users, and existed before the competition.
The kernel installs a **pinned public release** of it, like numpy or RDKit, under its own OSI
license. It is not code written for the submission. The CASMI-specific constants (windows,
weights, thresholds) already live in the CASMI repo, not here (round-2 rule).

**Risk:** the owner of LINALDB is on the team, so the host could read LINALDB as "source code
used to generate the submission" owned by a winner, and ask for MIT. **Fallback:** as sole
copyright holder, the owner can at any time also offer that exact release (or later ones) under
MIT. Adding a second license is always possible; taking one back is not. So Apache-2.0 now
doesn't close that door. If a prize is at stake and the host asks, publish a dual-licensed
release (`Apache-2.0 OR MIT`) of the pinned version, or a written MIT grant for it. No code
changes are needed.

To keep the position clean before 2026-12-07:
- The pinned wheel is a normal public release under Apache-2.0, built by CI (PR 3), with
  `SHA256SUMS` so the audit can show the Kaggle wheel = the public release.
- CASMI `EXTERNAL_RESOURCES.md` lists LINALDB as an external tool: name, version, PyPI URL,
  license Apache-2.0, how to get it.
- No competition data or CASMI-only code enters this repo (already the rule).

### Obligations we take on

- Ship `LICENSE` (full Apache-2.0 text) and `NOTICE` at the repo root, and inside every wheel,
  R package and release archive.
- Ship third-party notices for our dependencies (see PR 1). Apache-2.0 doesn't remove the MIT,
  BSD and HDF5 notices we owe the 577 crates we link statically.
- Keep the README and package metadata consistent; the CI check in PR 1 enforces it.

This section records a business decision, not legal advice. For the patent and trademark points,
a short review by a lawyer is worth it before any commercial deal that relies on them.

## PRs, in order

Verdict: **do** = worth it now; **if evidence** = only after a measurement says so; **no** = dropped.

### PR 1. License change and this plan — do (blocks CASMI)

Engine impact: none. Metadata, docs and one CI job. Same PR also adds this plan and removes the
proposal documents it replaces.

- [x] `LICENSE` → verbatim Apache-2.0 text (sha256 `cfc7749b…bc523d30`, so scanners detect it).
- [x] `NOTICE`: copyright gorigami 2025-2026, which earlier versions stay under the Community
      License, the §6 name reservation, and the statically linked HDF5 copyright.
- [x] `license = "Apache-2.0"` in `Cargo.toml` (was missing), `clients/python-embedded/Cargo.toml`
      and `clients/r-embedded/src/rust/Cargo.toml`.
- [x] Both `pyproject.toml`: PEP 639 `license = "Apache-2.0"` + `license-files = ["LICENSE",
      "NOTICE"]`, proprietary classifier dropped, build backends raised to versions that support
      PEP 639 (`maturin>=1.9`, `hatchling>=1.27`). `license-files` can't point outside the
      package directory: `clients/python` carries committed copies; `clients/python-embedded`
      gets them copied in by `pypi-publish-embedded.yml` (its maturin sdist is rooted at the
      repo and already packs the root files, so a committed copy collides), and the test job
      asserts the wheel has both under `.dist-info/licenses/`.
- [x] Both R `DESCRIPTION`s: `License: Apache License (== 2.0)` (canonical and verified by
      `tools:::analyze_license`); `clients/CRAN_PUBLISHING_PLAN.md` marks its license step done.
- [x] Release archives (`release.yml`) ship `LICENSE` and `NOTICE`.
- [x] No per-file headers (Apache recommends, doesn't require); `CONTRIBUTING.md` says so and
      that contributions come in under Apache-2.0 (§5).
- [x] `README.md` License section: Apache-2.0, which versions stay under the Community License,
      the name, support contact. `docs/DSL_REFERENCE.md` footer and the R READMEs updated.
- [x] CI job `License` (`scripts/check_license.sh`): verbatim LICENSE, package copies identical,
      every manifest says Apache-2.0, no "Community License" outside the history notes (R1).
- [ ] `linal-hub`: footer / license page of the docs site (separate repo, after this merges).
- [ ] CASMI side (not this repo): `EXTERNAL_RESOURCES.md` and `docs/ELIGIBILITY_AUDIT.md` point
      to the first Apache-2.0 release once it is on PyPI.

Moved to PR 3, so every artifact gets them the same way: full third-party notices for binaries
and wheels (`cargo about` or similar; 577 crates, all permissive except `colored`, MPL-2.0,
file-level, and the statically linked HDF5, whose BSD terms ask for its full notice in binary
distributions), and optionally `cargo deny check licenses` in CI. Releases up to 0.1.94 didn't
ship them either, so this is not a regression.

### PR 2. Small fixes — do

Engine impact: low. No answer or content-hash changes.

- [ ] **Database name validation (bug, to confirm with a failing test first).**
      `TensorDb::create_database` / `drop_database` (`src/engine/db.rs`) and
      `EngineRegistry::drop_database` (`src/server/engine.rs`) join the name onto `data_dir` and
      `remove_dir_all` it with no validation. The DSL only accepts identifiers, but
      `POST/DELETE /databases/:name` (axum decodes `%2E%2E`, `%2F`) and `linal db create|drop
      <name>` take any string, so `..` or `../x` could be created and then dropped outside
      `data_dir`. Fix: one `validate_db_name` (identifier characters only) used by every entry
      point; tests for the HTTP and CLI paths.
- [ ] **`DROP DATASET [IF EXISTS] <name>`** (P20, PubChem G5, CASMI G2). Parser + executor onto
      the existing `remove_dataset` (`src/engine/db.rs`); drops its indexes and any `.lcol`
      column files / maps; document what happens to lineage records (they stay, the dataset is
      gone). Python: `load_arrow(..., replace=True)` optional. Fixes the `memory_load.rs` error
      that already tells users to "DROP it first". Test: load, drop, reload, memory released.
- [ ] **`<>` as alias of `!=`** (P20): one lexer token in `src/dsl/lexer.rs`.
- [ ] **Release-note figure** (P20): `CHANGELOG.md` 0.1.94 and `DSL_REFERENCE.md` say "2.5M
      spectra / 75M peaks in ~12 s"; the real CASMI file is 401M peaks, 132 s in shards on 4
      CPUs. Restate with the measured size (re-measure after PR 4).
- [ ] **P16 + P17** from branch `oc/P0e` (`951dba1`, golden hash identical, tests green):
      stream the content hash into SHA-256 instead of building the JSON in memory
      (`Dataset::content_hash`), and move values into rows instead of cloning them
      (`record_batch_to_rows`). Removes ~30–50 of the ~68 B/peak transient load memory; hashes
      unchanged. Bring the code and `scripts/measure_load_memory.py`, not the design doc.
- [ ] Repo hygiene: untrack `.DS_Store` and add it to `.gitignore`.

### PR 3. Release pipeline — do (before the first release under the new license)

Engine impact: none. CI only. From the release-workflow proposal (R2–R5, R7):

- [ ] Third-party notices (from PR 1) in the release archives and wheels.
- [ ] Version-consistency check: crate, embedded `Cargo.toml`, `pyproject`, tag (R2).
- [ ] Release candidate on manual dispatch: build wheels + binaries, publish to **TestPyPI** and a
      **draft** GitHub Release with `SHA256SUMS` and build-provenance attestations (R3, R5).
- [ ] Verify job: install from TestPyPI in clean containers (Linux x86_64 py3.10–3.13, macOS
      arm64, Windows), run smoke + a golden content-hash test; one Kaggle-like job (py3.12,
      no network after install) (R4).
- [ ] Promote on tag: same artifacts (checked by checksum) to PyPI, undraft; fail if
      `CHANGELOG.md` has no entry for the version (R7).
- [ ] Then: **release** (engine + `linaldb`) with PR 1 and PR 2 inside, target ~Nov 23, leaving
      buffer for a fix release before Dec 7. Kaggle mirror (R8) is CASMI-side work.

Out of scope for now: R6 extra targets (D3).

### PR 4. Streaming load and append — do (after the release)

Engine impact: medium. The fix that makes 10⁷–10⁸-row loads bounded; today a 42M-row string-heavy
`load_arrow` was OOM-killed on 32 GB and the 2.5M-spectra library must load in 11 shards.

- [ ] **P18**: hand Arrow to Rust through the C stream interface (`__arrow_c_stream__` →
      `arrow::ffi_stream`) instead of IPC bytes (copies 1–3); process batch by batch, no
      `concat_batches` (copy 4); build peak values from Arrow slices with preallocation (copies
      5–6); remove the `i32` offset cap. Acceptance: same rows and hashes as 0.1.19; peak extra
      memory ≤ 2× dataset size + one batch (today ~8×).
- [ ] **P19**: `load_arrow(..., append=True)` (and `load_numpy`): exact schema match or error
      naming the column; sorted/vector indexes updated; one lineage record per append.
      Acceptance: N appends = one load of the concatenation (rows, hash, search results).
- Verified in the CASMI repo: full library as one dataset on 32 GB; 80M-row tier by 1M-row
  appends under 32 GB.

### PR 5. MMAP packages — do, smaller scope

Engine impact: low–medium.

- [ ] **P21**: `SAVE DATASET ... MMAP` writes mapped columns only to `.lcol`, not also into
      `data.parquet` (594 B/row today for `BitVector:2048`, of which 256 B is the `.lcol`).
      Plain `LOAD` reads them from `.lcol`; 0.1.19 packages keep loading.
- [ ] **MMAP for scalar/string columns** (PubChem G1, CASMI suggestion G3): today
      `SAVE ... MMAP` refuses a dataset without a `BitVector`/`Vector(F16|I8)` column, and plain
      `SAVE` is 4.3× the zstd parquet (10.2 vs 2.39 GB for 116M rows). Start by lifting the
      check for fixed-width scalars; strings need an offsets+bytes column kind (shared design
      with P22).

## Backlog

| Item | Source | Impact | Verdict |
|---|---|---|---|
| P23 `PREFILTER mass WITHIN TOL(q.mass, 10 PPM, 0.01 DA)` | round 4 | low | **do** whenever, cheap; keeps window parameters in lineage |
| P22 mmap `Matrix(2, *)` peak lists (offsets + interleaved `f32`, borrowed views in `SPEC_*`) | round 4 | medium–large | **if evidence**: saves ~5 min and 1.8 GB per CASMI run, not answers. Design with the compact peak-list value |
| G3 persist the sorted index (12 B/row snapshot, hash-guarded, mmap) instead of rebuilding on `LOAD` | PubChem store | medium | **if evidence**: only at 10⁸ rows |
| G2 compressed fingerprint column (sparse set-bit encoding) | PubChem store | medium | **if evidence**: only if the PubChem channel earns ≥ +0.02 in CASMI CV and needs to run in-engine |
| G4 SIMD/blocked `DOT(BitVector, Vector)` kernel | PubChem store | low–medium | **if evidence**: benchmark first; current cost is ms per query |
| G6 fetch payload columns for hits from another (mapped) dataset | PubChem store | medium | **if evidence**; document the `SEARCH ... INTO` + join path meanwhile |
| Wide `SPEC_COSINE_MOD` search ~10× slower on Kaggle's 4 CPUs than on the Mac | CASMI suggestion G4 | ? | **if evidence**: profile; the method adds no CASMI signal |
| HNSW vs IVF vs scan latency/recall benchmark | performance plan, Phase 1 | low | **do** when touching indexes |
| Columnar `String` values (if load is still > 3× at-rest after PR 4) | round 4 | large | **if evidence** |
| Binary (versioned) content hash | load-memory proposal | medium | **no** for now: changes every saved hash |
| R packages to CRAN | `clients/CRAN_PUBLISHING_PLAN.md` | medium | **do** after PR 1 (license was one of its blockers); the `r-embedded` path dependency is the real blocker |
| Linux aarch64, macOS x86_64, musl targets | release workflow R6 | low | after CASMI (D3) |

## Not doing

- `APPROX` together with `USING`: exact search is already milliseconds per query.
- Changes to `SPEC_COSINE_MOD`: behaves as specified; the method didn't help ranking.
- New lineage features: P15 is enough; CASMI adoption is CASMI-side.
