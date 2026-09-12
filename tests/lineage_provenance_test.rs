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
