//! Integration test for LINEAGE_AND_LINALG_PLAN.md Phase 5.2 -- "the single
//! most important test in the whole plan": a real multi-step workflow
//! (IMPORT CSV -> DATASET ... FROM with GROUP BY -> ADD COMPUTED COLUMN ->
//! SAVE DATASET -> simulated restart -> LOAD DATASET -> EXPLAIN LINEAGE)
//! asserting the *full real ancestry chain* is reconstructed from disk, not
//! a stub.
//!
//! Deviates from the plan text's literal "... -> JOIN -> GROUP BY -> ..."
//! step: Phase 0 research found `DATASET ... FROM` has no `JOIN` support
//! (`DatasetFromClause` has no `joins` field -- see LINEAGE_AND_LINALG_PLAN.md's
//! "Correction surfaced during Phase 0 research" note), so there is no DSL
//! path that turns a JOIN's result into a *named, persistable* dataset to
//! trace ancestry through. GROUP BY, a computed column, and a full
//! save/restart/load round trip are exercised instead -- the actual
//! persisted-ancestry surface this plan built.

use linal::core::config::{EngineConfig, StorageConfig};
use linal::dsl::{execute_line, DslError, DslOutput};
use linal::engine::TensorDb;
use std::fs;

fn expect_message(result: Result<DslOutput, DslError>, ctx: &str) -> String {
    match result {
        Ok(DslOutput::Message(msg)) => msg,
        Ok(other) => panic!("{ctx}: expected Message output, got {other:?}"),
        Err(e) => panic!("{ctx}: execution failed: {e:?}"),
    }
}

fn expect_ok(result: Result<DslOutput, DslError>, ctx: &str) {
    if let Err(e) = result {
        panic!("{ctx}: execution failed: {e:?}");
    }
}

#[test]
fn full_provenance_round_trip_survives_a_restart() {
    // Isolated dataset names so this doesn't collide with other tests that
    // share the default `./data` dir convention used across this test suite.
    let _ = fs::remove_dir_all("./data/default/datasets/lineage_it_raw");
    let _ = fs::remove_dir_all("./data/default/datasets/lineage_it_agg");
    let _ = fs::remove_file("./data/default/provenance.jsonl");

    let csv_path = std::env::temp_dir().join("lineage_provenance_it.csv");
    fs::write(
        &csv_path,
        "category,amount\nfood,10\nfood,20\ntravel,5\ntravel,15\n",
    )
    .unwrap();
    let csv_path_str = csv_path.to_str().unwrap();

    // 1. IMPORT CSV
    {
        let mut db = TensorDb::new();
        let import_cmd = format!(
            r#"IMPORT DATASET FROM "{}" AS lineage_it_raw"#,
            csv_path_str
        );
        expect_message(execute_line(&mut db, &import_cmd, 1), "IMPORT");

        // 2. LOAD it into the active session (IMPORT only persists to disk).
        expect_message(
            execute_line(&mut db, "LOAD DATASET lineage_it_raw", 2),
            "LOAD raw",
        );

        // 3. DATASET ... FROM ... GROUP BY (the reachable analogue of the
        //    plan's JOIN -> GROUP BY step, per the correction note above).
        //    `DslOutput::None` on success, not a Message.
        expect_ok(
            execute_line(
                &mut db,
                "DATASET lineage_it_agg FROM lineage_it_raw GROUP BY category SELECT category, SUM(amount) AS total",
                3,
            ),
            "DATASET FROM GROUP BY",
        );

        // 4. ADD COMPUTED COLUMN on the aggregated dataset.
        expect_message(
            execute_line(
                &mut db,
                "ALTER DATASET lineage_it_agg ADD COLUMN doubled = total * 2",
                4,
            ),
            "ADD COLUMN",
        );

        // 5. SAVE DATASET.
        let save_msg = expect_message(
            execute_line(&mut db, "SAVE DATASET lineage_it_agg", 5),
            "SAVE",
        );
        assert!(save_msg.contains("lineage_it_agg"));

        // Same-session sanity check before simulating a restart: the chain
        // should already be walkable from the live in-memory store.
        let tree = expect_message(
            execute_line(&mut db, "EXPLAIN LINEAGE lineage_it_agg", 6),
            "EXPLAIN LINEAGE (pre-restart)",
        );
        assert!(tree.contains("ADD COMPUTED COLUMN"));
        assert!(tree.contains("DATASET FROM"));
        assert!(tree.contains("IMPORT csv"));
    } // `db` dropped here -- nothing survives except what's on disk.

    // 6. Simulate a restart: a brand-new `TensorDb` over the same `./data`
    //    dir has no in-memory state at all: `resolve_lineage_node` (the old,
    //    session-only tree walk this plan replaced) could never have
    //    reconstructed anything here. The unified `ProvenanceStore` must.
    let mut db2 = TensorDb::new();
    expect_message(
        execute_line(&mut db2, "LOAD DATASET lineage_it_agg", 1),
        "LOAD agg (post-restart)",
    );
    expect_message(
        execute_line(&mut db2, "LOAD DATASET lineage_it_raw", 2),
        "LOAD raw (post-restart)",
    );

    // 7. EXPLAIN LINEAGE, text mode: the full real chain, not a stub.
    let text_tree = expect_message(
        execute_line(&mut db2, "EXPLAIN LINEAGE lineage_it_agg", 3),
        "EXPLAIN LINEAGE text (post-restart)",
    );
    println!("{text_tree}");
    assert!(
        text_tree.contains("ADD COMPUTED COLUMN"),
        "missing ADD COMPUTED COLUMN step:\n{text_tree}"
    );
    assert!(
        text_tree.contains("DATASET FROM"),
        "missing DATASET FROM (GROUP BY) step:\n{text_tree}"
    );
    assert!(
        text_tree.contains("IMPORT csv"),
        "missing root IMPORT step:\n{text_tree}"
    );
    // Real ancestry, not a stub: the old always-one-node-with-parents-vec!\
    // behavior (audit finding 1) would show exactly one operation, never
    // a nested tree with all three.
    let depth = text_tree.matches("  ").count();
    assert!(
        depth >= 2,
        "expected a nested multi-step tree:\n{text_tree}"
    );

    // 8. EXPLAIN LINEAGE, JSON mode: structurally the same chain.
    let json_tree = expect_message(
        execute_line(&mut db2, "EXPLAIN LINEAGE lineage_it_agg AS JSON", 4),
        "EXPLAIN LINEAGE JSON (post-restart)",
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&json_tree).expect("EXPLAIN LINEAGE AS JSON must be valid JSON");
    assert_eq!(parsed["operation"], "ADD COMPUTED COLUMN");
    assert!(parsed["inputs"][0]["operation"]
        .as_str()
        .unwrap()
        .starts_with("DATASET FROM"));
    assert_eq!(parsed["inputs"][0]["inputs"][0]["operation"], "IMPORT csv");

    // 9. The raw dataset's own lineage must resolve to its own producer
    //    (IMPORT), not get cross-attributed to the aggregate that consumed
    //    it -- the content-hash-collision disambiguation from Phase 0/1.
    let raw_tree = expect_message(
        execute_line(&mut db2, "EXPLAIN LINEAGE lineage_it_raw", 5),
        "EXPLAIN LINEAGE raw (post-restart)",
    );
    assert!(raw_tree.contains("IMPORT csv"));

    // 10. SHOW LINEAGE stays working as a documented alias, same resolver.
    let show_tree = expect_message(
        execute_line(&mut db2, "SHOW LINEAGE lineage_it_agg", 6),
        "SHOW LINEAGE (post-restart)",
    );
    assert!(show_tree.contains("ADD COMPUTED COLUMN"));
    assert!(show_tree.contains("DATASET FROM"));
    assert!(show_tree.contains("IMPORT csv"));

    // AUDIT DATASET staying a distinct, working, referential-integrity
    // check (Phase 4) is covered by this repo's existing
    // `consistency_test.rs::test_audit_dataset` against a tensor-first
    // dataset -- that check's target model is orthogonal to this test's
    // `dataset_legacy`/`LOAD DATASET` scenario, so it isn't duplicated here.

    let _ = fs::remove_file(&csv_path);
    let _ = fs::remove_dir_all("./data/default/datasets/lineage_it_raw");
    let _ = fs::remove_dir_all("./data/default/datasets/lineage_it_agg");
}

/// End-to-end (DSL `PRUNE LINEAGE` -> `DatabaseInstance::prune_lineage_before`
/// -> `ProvenanceStore::prune_before`) check of the one property that
/// matters most: pruning must never break `EXPLAIN LINEAGE` for a tensor
/// that's still live, even when every one of its ancestry records is older
/// than the requested cutoff. A future cutoff (every real record is
/// necessarily "older" than it) exercises exactly that "stale but still
/// needed" path without depending on wall-clock timing tricks.
///
/// Mechanical *removal* of a genuinely unreachable, stale record is
/// covered precisely (deterministic timestamps, no real clock involved) by
/// `core::provenance::tests::prune_before_removes_old_unreachable_records`
/// and its neighbors -- there is no DSL-level way to make a tensor/dataset
/// stop being "live" short of `DROP DATABASE` (which would trivially make
/// pruning moot), so this integration test's job is proving the safety
/// guarantee end-to-end, not the compaction mechanics.
#[test]
fn prune_lineage_never_breaks_a_live_tensors_ancestry() {
    let mut db = TensorDb::new();
    execute_line(&mut db, "VECTOR a = [1.0, 2.0, 3.0]", 1).expect("setup failed");
    execute_line(&mut db, "LET b = a * 2.0", 2).expect("setup failed");

    let before_prune = expect_message(
        execute_line(&mut db, "EXPLAIN LINEAGE b", 3),
        "EXPLAIN LINEAGE before prune",
    );
    assert!(before_prune.contains("SCALE") || before_prune.contains("MULTIPLY"));

    let prune_output = expect_message(
        execute_line(&mut db, r#"PRUNE LINEAGE BEFORE "2099-01-01T00:00:00Z""#, 4),
        "PRUNE LINEAGE",
    );
    assert!(
        prune_output.contains("retained because a live tensor/dataset's lineage still needs them"),
        "expected the report to explain why nothing was actually removed, got: {prune_output}"
    );

    // Ancestry must resolve identically after the prune -- nothing was
    // actually lost, only reported as protected.
    let after_prune = expect_message(
        execute_line(&mut db, "EXPLAIN LINEAGE b", 5),
        "EXPLAIN LINEAGE after prune",
    );
    assert_eq!(before_prune, after_prune);
}

/// A cutoff in the past (every real record is necessarily younger than it)
/// must prune nothing at all -- `PRUNE LINEAGE` is not a blind "clear
/// everything" command, it only ever removes records that are both stale
/// *and* unreachable.
#[test]
fn prune_lineage_with_a_past_cutoff_prunes_nothing() {
    let mut db = TensorDb::new();
    execute_line(&mut db, "VECTOR a = [1.0, 2.0, 3.0]", 1).expect("setup failed");

    let output = expect_message(
        execute_line(&mut db, r#"PRUNE LINEAGE BEFORE "2000-01-01T00:00:00Z""#, 2),
        "PRUNE LINEAGE with a past cutoff",
    );
    assert!(
        output.contains("Pruned 0 of"),
        "expected nothing to be pruned, got: {output}"
    );
}

#[test]
fn prune_lineage_rejects_a_malformed_timestamp() {
    let mut db = TensorDb::new();
    let err = execute_line(&mut db, "PRUNE LINEAGE BEFORE \"not-a-timestamp\"", 1)
        .expect_err("a malformed timestamp must be a loud parse error, not silently accepted");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("RFC3339"),
        "expected a clear explanation of the expected format, got: {msg}"
    );
}

/// Regression test for a real bug found via the Python-embedded binding: on
/// a brand-new `TensorDb` whose data directory has never been created on
/// disk (no `SAVE DATASET`, no other operation that happened to trigger
/// `record_provenance`'s own `create_dir_all`), `PRUNE LINEAGE` used to fail
/// with `"No such file or directory"` instead of succeeding -- it tried to
/// `save_jsonl` straight into a directory that doesn't exist yet. Uses a
/// unique temp `data_dir` (not the shared `./data/default` other tests in
/// this file/process write to) so this reliably starts from a genuinely
/// nonexistent directory without risking a race with any other test.
#[test]
fn prune_lineage_succeeds_on_a_database_with_no_data_dir_yet() {
    let data_dir =
        std::env::temp_dir().join(format!("linal_prune_no_datadir_{}", uuid::Uuid::new_v4()));
    let _ = fs::remove_dir_all(&data_dir);
    assert!(
        !data_dir.exists(),
        "test precondition: data_dir must not exist yet"
    );

    let mut db = TensorDb::with_config(EngineConfig {
        storage: StorageConfig {
            data_dir: data_dir.clone(),
            default_db: "default".to_string(),
        },
    });

    let output = execute_line(&mut db, r#"PRUNE LINEAGE BEFORE "2099-01-01T00:00:00Z""#, 1)
        .expect("PRUNE LINEAGE must succeed even when the data dir doesn't exist yet");
    let DslOutput::Message(msg) = output else {
        panic!("expected a Message output")
    };
    assert!(msg.contains("Pruned 0 of 0"), "got: {msg}");

    let _ = fs::remove_dir_all(&data_dir);
}

/// End-to-end regression test for a real bug found via `linal-hub`
/// (`05_lineage_and_linear_algebra.ipynb`, building on engine v0.1.87's
/// `PRUNE LINEAGE`): two different tensor operations that happen to produce
/// byte-identical output content (the same deterministic `SCALE` run twice
/// on the same input, under different `LET` bindings) used to be
/// indistinguishable to `resolve_ancestry`'s content-hash-based producer
/// lookup, since tensor *outputs* (unlike dataset outputs) were never
/// recorded with a name -- "most recent record with this hash wins" could
/// then misattribute `keeper`'s ancestry to the unrelated `orphan` record,
/// and (worse) `PRUNE LINEAGE` could delete `keeper`'s own true original
/// record while keeping `orphan`'s look-alike. Drives this through the real
/// DSL/engine path (`execute_line`), not a hand-built `ProvenanceStore`, so
/// it actually exercises `record_tensor_provenance`'s output-naming fix.
#[test]
fn prune_lineage_does_not_misattribute_a_live_tensor_on_a_real_content_hash_collision() {
    let data_dir =
        std::env::temp_dir().join(format!("linal_prune_collision_it_{}", uuid::Uuid::new_v4()));
    let _ = fs::remove_dir_all(&data_dir);

    let mut db = TensorDb::with_config(EngineConfig {
        storage: StorageConfig {
            data_dir: data_dir.clone(),
            default_db: "default".to_string(),
        },
    });

    execute_line(&mut db, "VECTOR a = [1.0, 2.0, 3.0]", 1).expect("setup failed");
    execute_line(&mut db, "LET keeper = SCALE a BY 2.0", 2).expect("setup failed");
    // A different operation that happens to produce byte-identical output
    // content to "keeper" -- the real-world trigger.
    execute_line(&mut db, "LET orphan = SCALE a BY 2.0", 3).expect("setup failed");
    // Rebind "orphan" -- its previous (content-colliding) record is now
    // genuinely unreachable from anything live.
    execute_line(&mut db, "LET orphan = SCALE a BY 3.0", 4).expect("setup failed");

    // Find keeper's own TRUE record directly -- the ground truth the test
    // checks against, identified in a way that's independent of the fix
    // under test (not by output name, which is exactly what the fix adds):
    // the chronologically *first* record whose output content hash matches
    // keeper's real current tensor content (keeper's own LET ran before
    // orphan's identical-content LET). Critically, `EXPLAIN LINEAGE`'s
    // *text* output can't distinguish a correct vs. misattributed
    // resolution here: both candidate records share the exact same
    // operation name and content hash, so the printed string looks
    // identical either way. Only checking which underlying `ProvenanceId`
    // actually survives the prune catches this.
    let keeper_hash = db.get("keeper").expect("keeper must exist").data_hash();
    let keeper_true_id = db
        .active_instance()
        .provenance
        .records()
        .iter()
        .find(|r| r.outputs[0].content_hash() == keeper_hash)
        .expect("a record producing keeper's real content must exist")
        .id;

    let report = expect_message(
        execute_line(&mut db, r#"PRUNE LINEAGE BEFORE "2099-01-01T00:00:00Z""#, 5),
        "PRUNE LINEAGE",
    );
    assert!(
        report.contains("Pruned 1 of"),
        "expected exactly the orphaned first 'orphan' record to be pruned, got: {report}"
    );

    let surviving_ids: Vec<_> = db
        .active_instance()
        .provenance
        .records()
        .iter()
        .map(|r| r.id)
        .collect();
    assert!(
        surviving_ids.contains(&keeper_true_id),
        "keeper's own TRUE record must survive the prune -- before the fix, this could be the \
         one actually deleted (misattributed to the unrelated, content-colliding 'orphan' \
         record instead)"
    );

    // `EXPLAIN LINEAGE keeper` must still resolve correctly post-prune too.
    let keeper_lineage_after = expect_message(
        execute_line(&mut db, "EXPLAIN LINEAGE keeper", 6),
        "EXPLAIN LINEAGE keeper after prune",
    );
    assert!(keeper_lineage_after.contains("SCALE"));

    let _ = fs::remove_dir_all(&data_dir);
}

/// Same regression as the `SCALE`/`eval_unary` test above, but for the
/// separate multi-output `LET a, b = QR ...` path (`eval_linalg_multi`),
/// which builds its `ProvenanceRecord` inline rather than going through
/// `record_tensor_provenance` -- a distinct call site with the exact same
/// bug (output entities recorded with no name), fixed alongside it.
#[test]
fn prune_lineage_does_not_misattribute_a_multi_output_binding_on_a_real_content_hash_collision() {
    let data_dir = std::env::temp_dir().join(format!(
        "linal_prune_collision_multi_it_{}",
        uuid::Uuid::new_v4()
    ));
    let _ = fs::remove_dir_all(&data_dir);

    let mut db = TensorDb::with_config(EngineConfig {
        storage: StorageConfig {
            data_dir: data_dir.clone(),
            default_db: "default".to_string(),
        },
    });

    execute_line(&mut db, "MATRIX m = [[2.0, 1.0], [1.0, 3.0]]", 1).expect("setup failed");
    execute_line(&mut db, "MATRIX m2 = [[1.0, 2.0], [3.0, 4.0]]", 2).expect("setup failed");
    execute_line(&mut db, "LET keeper_q, keeper_r = QR m", 3).expect("setup failed");
    // Same input, same deterministic op -- byte-identical output content to
    // keeper_q/keeper_r under different names.
    execute_line(&mut db, "LET orphan_q, orphan_r = QR m", 4).expect("setup failed");
    // Rebind with a different input -- orphan_q/orphan_r's previous
    // (content-colliding) record is now genuinely unreachable from
    // anything live.
    execute_line(&mut db, "LET orphan_q, orphan_r = QR m2", 5).expect("setup failed");

    let keeper_q_hash = db.get("keeper_q").expect("keeper_q must exist").data_hash();
    let keeper_q_true_id = db
        .active_instance()
        .provenance
        .records()
        .iter()
        .find(|r| r.outputs.iter().any(|o| o.content_hash() == keeper_q_hash))
        .expect("a record producing keeper_q's real content must exist")
        .id;

    let report = expect_message(
        execute_line(&mut db, r#"PRUNE LINEAGE BEFORE "2099-01-01T00:00:00Z""#, 6),
        "PRUNE LINEAGE",
    );
    assert!(
        report.contains("Pruned 1 of"),
        "expected exactly the orphaned first QR record to be pruned, got: {report}"
    );

    let surviving_ids: Vec<_> = db
        .active_instance()
        .provenance
        .records()
        .iter()
        .map(|r| r.id)
        .collect();
    assert!(
        surviving_ids.contains(&keeper_q_true_id),
        "keeper_q/keeper_r's own TRUE record must survive the prune"
    );

    let _ = fs::remove_dir_all(&data_dir);
}
