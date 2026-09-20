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
