// tests/group_by_determinism_test.rs
//
// GROUP BY emits groups in first-appearance order. It used to iterate a
// HashMap, whose order is random per map instance, so the same query
// returned its groups in a different order each run, and every content hash
// derived from the result (dataset hashes, provenance) changed with it.
// Found via linal-hub's notebooks after the v0.1.90 release: combined with
// the WAL's replay, it made a GROUP BY dataset lose its lineage after a
// restart (see tests/wal_test.rs).

use linal::core::config::EngineConfig;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

/// 30 groups, inserted in a scrambled (but fixed) order.
fn setup(db: &mut TensorDb) -> Vec<i64> {
    run(db, "DATASET t COLUMNS (g: Int, x: Float)");
    let mut first_seen = Vec::new();
    for i in 0..90i64 {
        let g = (i * 7) % 30;
        if !first_seen.contains(&g) {
            first_seen.push(g);
        }
        run(db, &format!("INSERT INTO t VALUES ({}, {}.5)", g, i));
    }
    first_seen
}

fn group_keys(out: DslOutput) -> Vec<i64> {
    match out {
        DslOutput::Table(ds) => ds
            .rows
            .iter()
            .map(|r| match &r.values[0] {
                linal::core::value::Value::Int(g) => *g,
                other => panic!("expected an Int key, got {:?}", other),
            })
            .collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

#[test]
fn groups_come_out_in_first_appearance_order_every_time() {
    let (_dir, mut db) = db();
    let expected = setup(&mut db);
    for _ in 0..10 {
        let got = group_keys(run(&mut db, "SELECT g, SUM(x) AS s FROM t GROUP BY g"));
        assert_eq!(got, expected);
    }
}

#[test]
fn a_group_by_dataset_has_the_same_content_hash_across_engines() {
    let hash = || {
        let (_dir, mut db) = db();
        setup(&mut db);
        run(&mut db, "DATASET d FROM t GROUP BY g SELECT g, SUM(x) AS s");
        db.get_dataset("d").unwrap().content_hash()
    };
    let first = hash();
    for _ in 0..5 {
        assert_eq!(hash(), first);
    }
}

#[test]
fn order_by_still_wins_over_first_appearance() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let got = group_keys(run(
        &mut db,
        "SELECT g, COUNT(*) AS n FROM t GROUP BY g ORDER BY g",
    ));
    assert_eq!(got, (0..30).collect::<Vec<_>>());
}
