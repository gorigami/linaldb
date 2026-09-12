# Lineage/Provenance Unification + Linear Algebra Operators — Implementation Plan

Tracked plan doc, repo-root convention (see `CONSISTENCY_PLAN.md`/`SIGNAL_PROCESSING_PLAN.md`/
`FLOAT64_PLAN.md`/`PYTHON_R_INTEROP_PLAN.md` precedent — deleted on completion, full history
lives in git log + `CHANGELOG.md` + `project-linaldb` session memory once done).

## Why this plan exists

An audit (2026-09-12) of the current lineage/provenance code found it more fragmented than
`docs/DSL_REFERENCE.md` describes — see "Audit findings" below. Separately, the engine is
missing classical linear algebra (no inverse/determinant/solve/eigendecomposition/SVD/any
decomposition) despite that being half of its own stated identity ("SQL meets Linear Algebra").
**Explicit priority, set by the user**: fix the lineage/provenance foundation *first*, so every
linear algebra operator built after it emits real provenance from day one instead of needing a
second retrofit pass later — the same class of rework the Float64 initiative's "wildcard-masked
sites" lesson already taught this project once.

## Audit findings (grounds every design decision below — re-verify if code has drifted)

1. **Three separate, weakly-connected lineage constructs exist today**:
   - `core::tensor::Lineage` (`src/core/tensor.rs:41-45`) — `{execution_id, operation: String,
     inputs: Vec<TensorId>}`. Real, auto-attached to most tensor ops, but no parameters, no
     output tracking, string-only operation name.
   - `engine::LineageNode` (`src/engine/db.rs:20-25`) — a genuinely recursive tree, but built
     **on demand, in-memory only** (`resolve_lineage_node`, `src/engine/db.rs:1036`), walking
     the *live* session's tensor store. Lost on restart; `SHOW LINEAGE` (`src/dsl/executor/
     show.rs:98-106`) only works within one session.
   - `core::dataset::lineage::DatasetLineage`/`LineageNode` (`src/core/dataset/lineage.rs`) —
     **does** persist (`lineage.json` in each dataset package) but every populating call site
     (`csv_connector.rs:70`, `hdf5_connector.rs:97`, `numpy_connector.rs:220,257`,
     `zarr_connector.rs:100`, `core/storage.rs:1164-1172`) writes exactly **one** node
     (`"import"` / `"SAVE (Legacy)"`) with `parents: vec![]`, always. No `JOIN`/`GROUP BY`/
     computed column/`DATASET ... FROM` ever appends to it.
2. **`dataset_legacy.rs`** — the actual SQL execution substrate (`JOIN`/`SELECT`/physical
   plans run on it, per `CLAUDE.md`) — has **zero** lineage-carrying fields. The type that does
   carry `DatasetLineage` lives under the *other*, zero-copy `core/dataset/` model.
3. **`AUDIT DATASET`** (`src/dsl/executor/mod.rs:158-172`) is a referential-integrity check
   ("do all column→tensor references still resolve"), unrelated to derivation history — worth
   naming explicitly since the name invites confusion with what this plan builds.
4. **Dataset content hashing is currently a placeholder, not a real hash**:
   `core/storage.rs:1160` — `format!("{}:{}", dataset_name, dataset.rows.len())` — name+rowcount,
   not a hash of actual content, easily collides. The tensor side already has the right pattern
   to reuse: `TensorMetadata::compute_hash` (`core/tensor.rs:146`) does a real SHA256 over the
   data.
5. **Identifiers don't survive reload**: `TensorId`/dataset instance identifiers are
   process-local UUIDs regenerated on `LOAD`. Any new persisted lineage needs a stable,
   content-derived cross-restart key (see finding 4's hash, done properly), not raw UUIDs.
6. **Connector dtype handling is inconsistent across connectors**: `numpy_connector.rs:83-87`'s
   own comment notes HDF5 "does implicit numeric conversion" while the Numpy connector
   deliberately errors instead — a real, separate small finding, not part of this plan's core
   scope but worth a follow-up (see "Explicitly out of scope" below).

## Design decisions locked before implementation starts

- **User-facing command**: `EXPLAIN LINEAGE <name>` (not an `AUDIT` extension — keeps
  referential-integrity and provenance concepts separate, matches existing `EXPLAIN`
  terminology for query plans). `SHOW LINEAGE` stays as a working alias for backward
  compatibility, documented as superseded.
- **Provenance standard**: evaluate **OpenLineage** first (Job/Run/Dataset/facet vocabulary —
  maps reasonably onto `DatabaseInstance`/execution/`Dataset`/`Tensor`); **W3C PROV**
  (Entity/Activity/Agent) as fallback if OpenLineage's model doesn't fit cleanly. Goal is a
  schema *compatible enough* that a real OpenLineage/PROV exporter is possible later without a
  major refactor — not full spec compliance on day one.
- **One shared model for tensors and datasets**: a `ProvenanceRecord` whose `inputs`/`outputs`
  are `Vec<ProvenanceEntity>` where `ProvenanceEntity = Tensor(TensorId) | Dataset(DatasetRef)`
  — the unification point is the *event*, not the data model (a `Dataset` and a `Tensor` stay
  structurally different; the record describing "operation X consumed these, produced those"
  is the same type either way).
- **Granularity**: dataset/statement-level provenance events for v1 (one record per `JOIN`,
  `GROUP BY`, computed column, etc.) — **not** per-row. Row-level provenance is a much larger
  scope explicitly deferred (see "Explicitly out of scope").
- **First-class metadata on every record**: operation name, structured `parameters` (not just a
  bare string), `inputs`, `outputs`, `timestamp`, `execution_id`, `engine_version`, a **real**
  content hash (SHA256, reusing the tensor side's existing pattern) — all locked as required
  fields, not optional add-ons.

## Phase 0 outcome — locked design (written 2026-09-12, gate for gates 0.1-0.6)

**0.1/0.2 — OpenLineage mapping (adopted, with extensions; W3C PROV fallback not needed)**

| linaldb concept | OpenLineage concept |
|---|---|
| One DSL statement execution (`ctx.execution_id()`) | `Run` (+ implicit `Job` = the operation name) |
| `Tensor` / `Dataset` (by name, scoped to a `DatabaseInstance`) | `Dataset` (namespace = db name) |
| A `ProvenanceRecord` | One `RunEvent` (`COMPLETE` state only — v1 doesn't model START/FAIL events) |
| `parameters` map | a custom `linal_parameters` run facet |
| `content_hash` on each input/output | a custom `linal_checksum` dataset facet |

Decision: **adopt OpenLineage's Job/Run/Dataset/facet vocabulary as the JSON export shape**
(top-level `job`/`run`/`inputs`/`outputs` keys, real OpenLineage field names where they map
1:1) but do **not** implement full transport (no Marquez client, no facet URNs/schema
registration) — "schema compatible enough for a real exporter later," per the plan's stated
goal, not spec compliance today. The mapping is clean enough that W3C PROV was not needed as
a fallback.

**0.3 — Unified types (`src/core/provenance.rs`, new module)**

```rust
pub struct ProvenanceId(pub Uuid);

pub enum ProvenanceEntity {
    Tensor { id: TensorId, name: Option<String>, content_hash: String },
    Dataset { name: String, content_hash: String },
}

pub struct ProvenanceRecord {
    pub id: ProvenanceId,
    pub operation: String,               // e.g. "JOIN", "SCALE", "SAVE"
    pub parameters: BTreeMap<String, serde_json::Value>,  // structured, not a bare string
    pub inputs: Vec<ProvenanceEntity>,
    pub outputs: Vec<ProvenanceEntity>,
    pub timestamp: DateTime<Utc>,
    pub execution_id: ExecutionId,
    pub engine_version: String,
}
```

The content hash lives **on each `ProvenanceEntity` reference**, not just top-level on the
record — this is what lets ancestry resolve after a restart via a stable, content-derived key
(fixes finding 5) instead of the process-local `TensorId`/dataset-instance UUIDs that get
regenerated on `LOAD`. `compute_content_hash(bytes: &[u8]) -> String` is a real SHA256,
generalized from `TensorMetadata::compute_hash` (`core/tensor.rs:146`); the tensor call site is
refactored to call through it, and it replaces the dataset-side placeholder
(`core/storage.rs:1160`, `format!("{}:{}", name, row_count)`) with a real hash over the
dataset's serialized row data.

**0.4 — Persistence format**

One shared, append-only `data/{db}/provenance.jsonl` per `DatabaseInstance` (JSON Lines: one
`ProvenanceRecord` per line) — shared across *all* tensors and datasets in that DB, not
per-dataset, because tensors don't live inside a dataset package and the unification point is
the event log, not per-entity storage. Cross-dataset/cross-restart parent resolution walks this
log matching an entity's `content_hash` against prior records' `outputs` (fixes finding 5) —
this walk *is* the new `resolve_lineage_node`/`LineageNode`, i.e. `engine::LineageNode` becomes
an in-memory tree view computed by walking `provenance.jsonl`, not a separate source of truth
(per 1.2). Each dataset package's existing `lineage.json` becomes a **derived, read-compat
export**: still written at `SAVE DATASET` time (old readers/tools keep working, finding 1.2's
compat requirement), but generated by projecting the relevant `ProvenanceRecord`s for that
dataset out of the shared log — it is no longer where new code looks things up.

**0.5 — `EXPLAIN LINEAGE` grammar**

```
EXPLAIN LINEAGE <name>            -- text-tree mode (default; supersedes SHOW LINEAGE's tree)
EXPLAIN LINEAGE <name> AS JSON    -- JSON export mode, OpenLineage-shaped per 0.1
```

AST: add `ExplainTarget::Lineage { name: String, json: bool }`
(`src/dsl/ast.rs`, alongside `Dataset`/`DatasetQuery`/`Search`/`Select`). Parser: new arm in
`parse_explain` (`src/dsl/parser/introspection.rs:84`) — `Some(Token::Lineage) => { self.advance();
... }`, mirroring the existing `SHOW LINEAGE` arm at line 41-44 of the same file, plus an
optional trailing `AS JSON` (reusing the `Token::As`/a new `Json` ident check, consistent with
how `PLAN` is an optional bare-ident check at line 86-88). `SHOW LINEAGE <name>` keeps working,
implemented as a thin call into the same resolver, documented as superseded.

**0.6 — Gate closed.** This section is the locked design; Phase 1 implementation begins below.

**Correction surfaced during Phase 0 research, affects Phase 1.4's scope**: `execute_create_dataset_from`
(`dsl/executor/query.rs`, the `DATASET <name> FROM ...` handler) hardcodes `joins: vec![]` when
building its internal `SelectStmt` — `DatasetFromClause` has no `joins` field at all today. Plain
`SELECT ... JOIN ...` (`Statement::Select`) only ever produces an ephemeral `DslOutput::Table`,
never a named/persisted dataset. So **there is currently no DSL path that turns a JOIN's result
into a named, persistable dataset** — Phase 1.4's "thread provenance emission through JOIN" must
either (a) scope JOIN provenance to the ephemeral-table case only (a `ProvenanceRecord` describing
the JOIN gets created but has no durable named output to attach it to beyond `EXPLAIN` on the
in-flight result, if that's even needed for v1), or (b) first add `joins` to `DatasetFromClause` so
`DATASET ... FROM ... JOIN ...` becomes expressible before it can have anything to record lineage
*for*. Decide which at Phase 1.4 implementation time; (a) is the smaller, in-scope option — adding
JOIN support to `DATASET ... FROM` is a separate, larger DSL feature not part of this plan.

## Explicitly out of scope for this plan (tracked separately, not blocking)

- Row-level provenance (per-INSERT/UPDATE/DELETE granularity).
- Native multi-dtype `Tensor`/`Vector`/`Matrix` (u8/i8/i32/etc. computed natively, not just
  ingested) — large, separate initiative, comparable in scope to autodiff; do not fold in here.
- Connector dtype-acceptance breadth/consistency (finding 6) — small, separate follow-up.
- Full OpenLineage/PROV spec compliance (an actual exporter/integration) — only schema
  *compatibility* is in scope now.

---

## Phase 0 — Provenance model research & design

- [x] 0.1 Map OpenLineage's core entities (Job, Run, Dataset, facets) onto linaldb's concepts
      (`DatabaseInstance`/execution → Run; a `Statement` → Job; `Tensor`/`Dataset` → Dataset).
      Produce a short mapping table. Decide: adopt directly, adopt-with-extensions, or fall
      back to W3C PROV.
- [x] 0.2 If OpenLineage doesn't map cleanly, do the same mapping exercise against W3C PROV
      (Entity/Activity/Agent).
- [x] 0.3 Design the unified `ProvenanceRecord`/`ProvenanceEntity` Rust types (new module,
      e.g. `src/core/provenance.rs`) per the locked design decisions above.
- [x] 0.4 Design the persistence format: how a `ProvenanceRecord` serializes, how records link
      **across** separate dataset packages (a dataset derived from another *saved* dataset must
      reference the ancestor's own persisted record via a stable, content-derived ID — see
      audit findings 4-5), and whether this replaces `lineage.json` or lives alongside it.
- [x] 0.5 Design `EXPLAIN LINEAGE`'s grammar and output: text-tree mode (human-readable, like
      today's `SHOW LINEAGE`) plus a JSON export mode for programmatic/compliance consumption.
- [x] 0.6 Write the locked design (0.1-0.5's outcomes) into this plan doc's "Design decisions"
      section above before any implementation code is written — treat this as a real gate, not
      a formality.

## Phase 1 — Core provenance infrastructure

- [x] 1.1 Implement `ProvenanceRecord`/`ProvenanceEntity` (`src/core/provenance.rs`) per the
      Phase 0 design, including a real SHA256 content hash (reusing/generalizing
      `TensorMetadata::compute_hash`'s pattern — fixes audit finding 4's placeholder hash).
- [x] 1.2 Decide and implement the migration path for the three existing constructs:
      `core::tensor::Lineage` reshaped to match the unified record; `engine::LineageNode`
      becomes the in-memory tree-walk *view* over the new store (not a separate source of
      truth); `core::dataset::lineage::DatasetLineage` deprecated in favor of the new format
      (keep a read-compatibility shim for pre-existing `lineage.json` files — don't break
      old saved datasets). — `core::tensor::Lineage` kept as-is (still attached by every
      `eval_*`, still `TensorMetadata`'s in-memory fast path); `engine::LineageNode`/
      `resolve_lineage_node`/`get_lineage_tree` removed outright and replaced by
      `get_tensor_lineage_tree`/`get_dataset_lineage_tree` returning `ProvenanceTree`;
      `core::dataset::lineage::DatasetLineage`/`lineage.json` kept exactly as the planned
      read-compat shim (`DatabaseInstance::legacy_dataset_lineage`).
- [x] 1.3 Give `dataset_legacy.rs` an actual provenance-carrying hook (it currently has none —
      audit finding 2) so the real SQL execution substrate can record events. —
      `Dataset::content_hash()` (real SHA256 over `schema.fields` + row `values`, deliberately
      *not* whole-`Schema`/whole-`Tuple` serialization — see its doc comment for why).
- [x] 1.4 Thread provenance emission through `dsl/executor/query.rs`: ~~`JOIN`,~~ `GROUP BY`,
      computed/window columns, `DATASET ... FROM` — these currently bypass lineage entirely.
      — **JOIN dropped from scope**: Phase 0 research found no DSL path turns a JOIN's result
      into a named, persistable dataset (`DatasetFromClause` has no `joins` field; plain
      `SELECT ... JOIN` only ever produces an ephemeral, unnamed `DslOutput::Table`) — nothing
      to attach a `ProvenanceRecord` to. `DATASET ... FROM` (covers `GROUP BY`/computed/window
      columns, since they're all part of that one statement) and `ALTER DATASET ADD COLUMN`
      both emit real records.
- [x] 1.5 Update all 4 connectors (`csv_connector.rs`, `hdf5_connector.rs`,
      `numpy_connector.rs`, `zarr_connector.rs`) and the `SAVE DATASET` path
      (`core/storage.rs`) to emit real records via the new model, replacing the current
      always-one-stub-node behavior. — connectors' `lineage.json` node now gets a real
      `record_batch_content_hash`; `IMPORT DATASET FROM`/`SAVE DATASET` (`dsl/persistence.rs`)
      each record a real `ProvenanceRecord` into the unified store.
- [x] 1.6 Persistence: write/read the new format, with cross-dataset parent resolution via
      stable content-derived IDs (fixes audit finding 5). — `ProvenanceStore` JSONL at
      `{data_dir}/{db}/provenance.jsonl`, loaded on `DatabaseInstance` construction.
- [x] 1.7 Verify a real save → restart → load → `EXPLAIN LINEAGE` round trip reconstructs
      genuine multi-step ancestry, not a stub — this is the core correctness bar for the whole
      phase. — `tests/lineage_provenance_test.rs`, verified passing in the real CI-exact suite.
      Found and fixed two genuine pre-existing bugs while getting this to actually pass (see
      `CHANGELOG.md`'s `[Unreleased]` entry): `DatasetMetadata::update_stats` never refreshed
      its own cached `schema` field (silent data loss on `ALTER ADD COLUMN` + `SAVE` + reload),
      and `resolve_ancestry` could self-reference or misattribute on a content-hash collision
      (a no-op `FILTER` leaves input/output hashes identical).

## Phase 2 — Tensor-level integration & unification

- [x] 2.1 Rewire tensor-level ops (`ADD`/`SUBTRACT`/`MATMUL`/`CORRELATE`/etc., `engine/db.rs`'s
      `eval_binary`/`eval_unary`) to emit into the **same** unified provenance store as dataset
      ops, not a tensor-only side path. — all 16 `eval_*` methods (every one of the original 17
      `Lineage{}` construction sites; `eval_unary`/`eval_binary` collapse to one call site each
      given their shared post-match attach point) now also call
      `DatabaseInstance::record_tensor_provenance`.
- [~] 2.2 Audit every existing kernel/`BinaryOp`/`UnaryOp` call site and add real parameter
      capture where an op has one (`SCALE a BY n`'s `n`, `RESHAPE a TO [dims]`'s `dims`,
      `WHITEN a WITH b`, `BANDPASS ... FROM ... TO ... WITH RATE ...`, etc.) — mechanical but
      real work across ~30 existing ops. — **partially done**: operation *names* already carry
      real parameters as formatted strings for most ops (`PSD(window=...)`,
      `BANDPASS(...-...Hz @ ...Hz)`, `STACK(axis=...)`, `INDEX[...]`, `SLICE[...]`,
      `FIELD_ACCESS(...)`, `COLUMN_ACCESS(...)` — unchanged from what `core::tensor::Lineage`
      already produced) and are reused verbatim for the new `ProvenanceRecord.operation`. What's
      *not* done: promoting these into the structured `ProvenanceRecord.parameters` map (the
      locked design's actual ask — a bare formatted string, even an informative one, isn't the
      same as a queryable key-value field). Left as explicit follow-up work, tracked here rather
      than silently dropped.
- [x] 2.3 Confirm `EXPLAIN LINEAGE` produces identically-shaped output for a tensor name and a
      dataset name (the unification's actual observable proof). — verified manually and via
      `tests/consistency_test.rs::test_show_lineage` (tensor) +
      `tests/lineage_provenance_test.rs` (dataset): same `ProvenanceTree`/`format_lineage_tree`
      path, same `"{operation} ({name}) [{hash}]"` shape, either way.

## Phase 3 — `EXPLAIN LINEAGE` command

- [x] 3.1 New DSL statement `EXPLAIN LINEAGE <name>` (lexer/parser/AST/executor) per the
      Phase 0.5 grammar design.
- [x] 3.2 Text-tree output mode (human-readable).
- [x] 3.3 JSON export mode for programmatic/compliance consumption.
- [x] 3.4 `SHOW LINEAGE <name>` kept working as a documented alias.
- [x] 3.5 Verify the persisted/cross-session case explicitly: `EXPLAIN LINEAGE` on a freshly
      `LOAD`ed dataset reconstructs real ancestry from disk, not an in-memory-only tree. —
      `tests/lineage_provenance_test.rs` drops the whole `TensorDb` and constructs a fresh one
      over the same data dir before asserting this.

## Phase 4 — Disambiguate `AUDIT DATASET`

- [x] 4.1 Leave `AUDIT DATASET`'s existing referential-integrity behavior untouched (don't
      risk regressing a working feature). — zero code changes to `AUDIT DATASET`'s own logic;
      `tests/consistency_test.rs::test_audit_dataset` still passes unmodified.
- [x] 4.2 Update `docs/DSL_REFERENCE.md`/`docs/ARCHITECTURE.md` to clearly separate the two
      concepts now that `EXPLAIN LINEAGE` exists, so the naming collision from audit finding 3
      doesn't persist for future readers.

## Phase 5 — Testing & verification

- [x] 5.1 Unit tests: `ProvenanceRecord` serialization round-trip, content-hash correctness,
      parent-resolution correctness. — `src/core/provenance.rs`'s own `#[cfg(test)]` module (8
      tests): round-trip, multi-step chain walk, self-reference termination, name-based
      hash-collision disambiguation, unknown-hash-is-root, JSONL round trip through real disk,
      missing-file-is-empty-store.
- [x] 5.2 Integration test: a real multi-step workflow — `IMPORT` CSV → ~~`JOIN` →~~ `GROUP BY`
      → computed column → `SAVE DATASET` → restart → `LOAD DATASET` → `EXPLAIN LINEAGE` —
      asserting the *full real chain* is reconstructed, not a stub. This is the single most
      important test in the whole plan; it's the concrete proof the initiative worked. —
      `tests/lineage_provenance_test.rs::full_provenance_round_trip_survives_a_restart` (`JOIN`
      dropped per 1.4's note above; everything else exactly as specified, plus JSON mode and the
      hash-collision disambiguation check).
- [x] 5.3 Regression tests confirming existing `SHOW LINEAGE`/`AUDIT DATASET`/
      `SHOW DATASET METADATA` behavior isn't broken by the migration (1.2). — full CI-exact
      suite green (see 5.4); `tests/consistency_test.rs::test_show_lineage`'s assertions updated
      for the new entity-agnostic message wording (`"Lineage for 'd'"`, not `"... for tensor
      'd'"` — necessarily different now that the same command resolves datasets too), everything
      else in that test unchanged and still passing.
- [x] 5.4 Full CI-exact suite green (`cargo test --release` per `CLAUDE.md`'s exact command),
      `cargo fmt -- --check`, `cargo clippy -- -D warnings`, smoke test script. — all green,
      zero failures across every test binary (fixtures regenerated first, per `CLAUDE.md`, since
      all 4 connectors changed).

## Phase 6 — Documentation

- [x] 6.1 `docs/DSL_REFERENCE.md` — `EXPLAIN LINEAGE` syntax + worked example.
- [x] 6.2 `docs/ARCHITECTURE.md` — replace the old fragmented lineage description with the
      unified model; document the OpenLineage/PROV mapping decision.
- [x] 6.3 `CHANGELOG.md` entries for every user-visible change in this initiative.
- [x] 6.4 `clients/EMBEDDED_CONTRACT.md` — confirm/update `DslOutput` shape for
      `EXPLAIN LINEAGE`'s JSON mode as seen through the Python/R embedded bindings. — confirmed,
      no update needed: both `EXPLAIN LINEAGE` modes return `DslOutput::Message(String)`,
      already covered by that variant's existing row in the contract table (JSON mode is just a
      JSON-formatted string payload, not a new `DslOutput` variant).

## Phase 7 — Release (lineage/provenance foundation)

- [ ] 7.1 Version bump + tag → GitHub Release (`release.yml`), matching this repo's
      "Cutting a release" process in `CLAUDE.md`.
- [ ] 7.2 `clients/python-embedded` version bump + `linaldb-vX.Y.Z` tag → PyPI publish
      (`pypi-publish-embedded.yml`) — remember these are two independent release pipelines,
      per this session's own earlier finding; both are needed before `pip install linaldb`
      actually has this.
- [ ] 7.3 Cross-platform validation on all three `release.yml` targets (macOS/Linux/Windows) —
      this session's own testing was macOS-only; explicitly confirm the persistence round-trip
      (5.2) and `EXPLAIN LINEAGE` work on Linux and Windows too, not just where it was built.
- [ ] 7.4 Install the freshly published PyPI wheel (not a local dev build) and re-verify 5.2's
      round trip through the real published package.

---

## Phase 8 — Linear algebra operators (only after Phases 0-7 are fully done)

Every operator below emits a real `ProvenanceRecord` (operation name + structured parameters +
inputs + outputs) from the moment it's implemented — no separate "add lineage later" step,
because Phases 0-7 already built the foundation.

- [x] 8.0 Add `nalgebra` dependency (pure Rust — matches the `rustfft`/`realfft` precedent
      already in this codebase over binding to LAPACK/BLAS). Build the
      `Tensor ↔ nalgebra` conversion scaffolding (f32 tensor → f64 nalgebra matrix → compute →
      f64 → f32 back), reused by every operator below. — `nalgebra = "0.33"`;
      `src/core/linalg.rs`'s `tensor_to_matrix`/`matrix_to_tensor_data`.
- [x] 8.1 `TRACE`, `DETERMINANT`, `RANK` — single scalar outputs, no new DSL binding syntax
      needed. Each emits provenance per the rule above.
- [x] 8.2 `INVERSE`, `SOLVE` (`Ax = b`) — loud-error-on-singular/near-singular philosophy
      (detected via the crate's own condition/pivot reporting — never silently return `NaN`).
- [x] 8.3 `EIGENVALUES` for **symmetric** matrices only (real eigenvalues guaranteed, no
      complex-number handling needed yet).
- [x] 8.4 Design and implement multi-output `LET a, b, c = <expr>` binding syntax (lexer/
      parser/AST/executor) — the prerequisite for every decomposition below. —
      `Statement::LetMulti`/`LetMultiStmt` (new AST variant, `Let`/`LetStmt` untouched);
      arity (names.len() vs. the bound op's real output count) checked at execution time in
      `eval_let_multi`, with a clear error either direction (too few/many names, or a
      single-output op like `CHOLESKY` used with multi-output `LET`).
- [x] 8.5 `LU`, `QR`, `CHOLESKY`, full eigendecomposition (eigenvalues + eigenvectors, general
      case), `SVD` — each using 8.4's multi-output binding. — **two deviations from the literal
      text, both documented in code/docs**: `CHOLESKY` stayed single-output (bind with a plain
      `LET l = CHOLESKY a`) since it only ever has one real output — forcing 8.4's syntax on it
      would add ceremony, not value. `EIGEN`'s "general case" stayed **symmetric-only**, same as
      `EIGENVALUES` (8.3) — a truly general eigendecomposition can produce complex
      eigenvalues/eigenvectors, and this engine has no `Value::Complex`; going "general" here
      would have silently contradicted 8.3's own locked constraint. `LU` returns `P`, `L`, `U`
      (not just `L`, `U`) specifically so `P @ a == L @ U` holds for any input needing pivoting.
- [x] 8.6 `PCA`, built on 8.5's `SVD` — the capstone: real dimensionality reduction natively in
      the engine.
- [x] 8.7 Unit tests per operator: hand-computable small matrices (2x2/3x3) with known
      determinant/inverse/eigenvalues; property checks (`A · A⁻¹ ≈ I`, `A · solve(A,b) ≈ b`,
      `A ≈ V·diag(λ)·V⁻¹` for a diagonalizable test case); singular/ill-conditioned matrix
      error-path tests. — 21 unit tests in `src/core/linalg.rs` (numerics in isolation,
      including `P@a≈L@U`, `a≈L@Lᵗ`, `a≈V@diag@Vᵗ`, `a≈U@diag(s)@Vᵗ` property checks) + 8
      DSL-level integration tests in `tests/linalg_operators_test.rs` (real lexer→parser→
      executor→engine pipeline, multi-output `LET` binding, error paths, provenance).
- [x] 8.8 `docs/DSL_REFERENCE.md` additions for every new keyword. — new "Classical Linear
      Algebra" subsection (§3) + multi-output `LET` grammar note.
- [x] 8.9 `CHANGELOG.md` entries. — includes a real regression found and fixed while wiring
      `RANK` as a lexer keyword: it collided with the pre-existing `RANK()` SQL window function,
      which expected `RANK` to lex as a generic identifier (`src/dsl/parser/dataset.rs`).
- [x] 8.10 Full CI-exact suite, fmt, clippy, smoke test — same bar as Phase 5.4. — all green;
      `cargo test --lib` (212 passed) run explicitly after the `RANK` keyword fix to catch any
      other keyword/SQL-identifier collision the new tokens (`TRACE`/`DETERMINANT`/`INVERSE`/
      `SOLVE`/`EIGENVALUES`/`QR`/`LU`/`CHOLESKY`/`EIGEN`/`SVD`/`PCA`/`COMPONENTS`) might cause —
      none found.
- [ ] 8.11 Version bump + tag → GitHub Release + PyPI publish (same two-pipeline process as
      Phase 7), cross-platform validation, install-and-reverify through the real published
      wheel. — **not started**: this is the same class of irreversible, external action Phase 7
      is gated on; needs explicit user go-ahead, same as Phase 7.

---

## Phase 9 — Final showcase notebook (release-validation artifact, built LAST)

Do **not** start this notebook until every box below is checked — it's a validation artifact
built against the fully shipped, published result, not an exploratory prototype built alongside
development.

- [ ] Implementation is complete (Phases 1-4, 8).
- [ ] Unit tests are complete.
- [ ] Integration tests are complete.
- [ ] Regression tests pass.
- [ ] CI pipelines are green.
- [ ] Release artifacts are generated successfully.
- [ ] Cross-platform validation is complete.
- [ ] PyPI packages are published and validated.
- [ ] Documentation is finalized.

Once all of the above are checked:

- [ ] 9.1 Build a `linal-hub` notebook demonstrating end-to-end provenance across the full
      range this initiative covered: import → transformation → join → aggregation → tensor
      operation → linear algebra decomposition, with `EXPLAIN LINEAGE` inspected at each step,
      against the real published PyPI package (not a local dev build) — matching this
      session's own established verification standard for every prior `linal-hub` notebook.
- [ ] 9.2 Execute end-to-end via `nbconvert`, zero errors, before committing.
- [ ] 9.3 Update `project-linaldb` session memory with the outcome (bugs found, if any —
      matching this project's own recurring pattern of real end-to-end examples surfacing
      issues unit tests miss).
