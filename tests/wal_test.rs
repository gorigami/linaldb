// tests/wal_test.rs
//
// Write-ahead log (`[wal] enabled = true`, `engine::wal`): in-memory state
// survives a restart without SAVE, via checkpoint restore + log replay.
// "Crash" here means dropping the TensorDb without saving anything, then
// constructing a fresh one over the same data directory.

use linal::core::config::{EngineConfig, WalConfig, WalSync};
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::path::Path;

fn open(dir: &Path) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.to_path_buf();
    config.wal = WalConfig {
        enabled: true,
        sync: WalSync::Always,
        ..WalConfig::default()
    };
    TensorDb::with_config(config)
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn tensor(db: &mut TensorDb, name: &str) -> Vec<f32> {
    match run(db, &format!("SHOW {}", name)) {
        DslOutput::Tensor(t) => t.to_logical_vec(),
        other => panic!("expected tensor {}, got {:?}", name, other),
    }
}

fn row_count(db: &mut TensorDb, name: &str) -> usize {
    match run(db, &format!("SHOW {}", name)) {
        DslOutput::Table(ds) => ds.rows.len(),
        other => panic!("expected table {}, got {:?}", name, other),
    }
}

fn wal_lines(dir: &Path, db: &str) -> usize {
    std::fs::read_to_string(dir.join(db).join("wal.log"))
        .map(|s| s.lines().count())
        .unwrap_or(0)
}

#[test]
fn unsaved_state_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path());
        for line in [
            "VECTOR a = [1, 2, 3]",
            "VECTOR b = [10, 20, 30]",
            "LET c = a + b",
            "BIND alias TO c",
            "LAZY LET d = ADD a b",
            "DATASET items COLUMNS (id: Int, score: Float)",
            "INSERT INTO items VALUES (1, 0.5)",
            "INSERT INTO items VALUES (2, 0.9)",
            "CREATE INDEX ON items(id)",
            "DEFINE PIPELINE top AS ORDER BY score DESC THEN LIMIT 1",
            "SHOW c",
            "SELECT * FROM items",
        ] {
            run(&mut db, line);
        }
        // Read-only statements are not logged.
        assert_eq!(wal_lines(dir.path(), "default"), 10);
    }

    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "c"), vec![11.0, 22.0, 33.0]);
    assert_eq!(tensor(&mut db, "alias"), vec![11.0, 22.0, 33.0]);
    assert_eq!(tensor(&mut db, "d"), vec![11.0, 22.0, 33.0]);
    assert_eq!(row_count(&mut db, "items"), 2);
    let DslOutput::Message(indexes) = run(&mut db, "SHOW INDEXES items") else {
        panic!()
    };
    assert!(indexes.contains("id"), "{indexes}");
    run(&mut db, "APPLY PIPELINE top ON items INTO best");
    assert_eq!(row_count(&mut db, "best"), 1);
}

#[test]
fn wal_off_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db = TensorDb::with_config(config);
    run(&mut db, "VECTOR a = [1]");
    assert!(!dir.path().join("default").join("wal.log").exists());
    assert!(execute_line(&mut db, "CHECKPOINT", 1).is_err());
}

#[test]
fn checkpoint_truncates_the_log_and_restores_with_aliases_and_views() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path());
        run(&mut db, "MATRIX m = [[1, 2], [3, 4]]");
        run(&mut db, "LET mt = TRANSPOSE m");
        run(&mut db, "BIND same TO m");
        run(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]");
        run(&mut db, "LET ds1 = dataset(\"ds1\")");
        run(&mut db, "ds1.add_column(\"col1\", v1)");
        let DslOutput::Message(msg) = run(&mut db, "CHECKPOINT") else {
            panic!()
        };
        assert!(msg.contains("Checkpoint written"), "{msg}");
        assert_eq!(wal_lines(dir.path(), "default"), 0);
        run(&mut db, "VECTOR after = [7]");
        assert_eq!(wal_lines(dir.path(), "default"), 1);
    }

    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "mt"), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(tensor(&mut db, "same"), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor(&mut db, "after"), vec![7.0]);
    // The transpose view still shares its source's buffer after a restore.
    {
        let m = db.get("m").unwrap().data.clone();
        let mt = db.get("mt").unwrap().data.clone();
        assert!(std::sync::Arc::ptr_eq(&m, &mt));
    }
    assert!(db.get_tensor_dataset("ds1").is_some());
    assert_eq!(
        db.verify_tensor_dataset("ds1").unwrap(),
        Vec::<String>::new()
    );

    // And the sequence continues after the checkpointed records.
    run(&mut db, "VECTOR more = [8]");
    drop(db);
    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "more"), vec![8.0]);
    assert_eq!(tensor(&mut db, "after"), vec![7.0]);
}

#[test]
fn load_modify_save_does_not_double_apply_on_replay() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path());
        run(&mut db, "DATASET items COLUMNS (id: Int)");
        run(&mut db, "INSERT INTO items VALUES (1)");
        run(&mut db, "SAVE DATASET items");
    }
    {
        let mut db = open(dir.path());
        run(&mut db, "LOAD DATASET items");
        run(&mut db, "INSERT INTO items VALUES (2)");
        // Overwrites the package the LOAD above read, then checkpoints.
        run(&mut db, "SAVE DATASET items");
        run(&mut db, "INSERT INTO items VALUES (3)");
    }
    let mut db = open(dir.path());
    assert_eq!(row_count(&mut db, "items"), 3);
}

#[test]
fn replay_fails_loudly_when_an_imported_file_changed() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("input.csv");
    std::fs::write(&csv, "id,x\n1,0.5\n2,0.7\n").unwrap();
    {
        let mut db = open(dir.path());
        run(
            &mut db,
            &format!("IMPORT CSV FROM \"{}\" AS input", csv.display()),
        );
        run(&mut db, "VECTOR other = [1]");
    }
    std::fs::write(&csv, "id,x\n1,0.5\n2,0.7\n3,0.9\n").unwrap();

    let mut db = open(dir.path());
    let err = execute_line(&mut db, "SHOW other", 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("failed WAL recovery"), "{err}");
    assert!(err.contains("has changed"), "{err}");
}

#[test]
fn a_truncated_final_record_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path());
        run(&mut db, "VECTOR a = [1]");
        run(&mut db, "VECTOR b = [2]");
    }
    let wal = dir.path().join("default").join("wal.log");
    let mut contents = std::fs::read_to_string(&wal).unwrap();
    contents.push_str("{\"seq\":3,\"ts\":\"2026-");
    std::fs::write(&wal, contents).unwrap();

    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "b"), vec![2.0]);
    // Appending continues with a fresh, well-formed record after the tail.
    run(&mut db, "VECTOR c = [3]");
    drop(db);
    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "c"), vec![3.0]);
}

#[test]
fn replay_does_not_duplicate_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let provenance = dir.path().join("default").join("provenance.jsonl");
    let count = || {
        std::fs::read_to_string(&provenance)
            .map(|s| s.lines().count())
            .unwrap_or(0)
    };
    {
        let mut db = open(dir.path());
        run(&mut db, "VECTOR a = [1, 2]");
        run(&mut db, "LET b = a * 2");
    }
    let before = count();
    assert!(before > 0);
    let mut db = open(dir.path());
    assert_eq!(tensor(&mut db, "b"), vec![2.0, 4.0]);
    assert_eq!(count(), before);
    run(&mut db, "CHECKPOINT");
    drop(db);
    let _db = open(dir.path());
    assert_eq!(count(), before);
}

#[test]
fn each_database_replays_its_own_log() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path());
        run(&mut db, "CREATE DATABASE other");
        run(&mut db, "USE other");
        run(&mut db, "VECTOR only_other = [5]");
        run(&mut db, "USE default");
        run(&mut db, "VECTOR only_default = [6]");
    }
    let mut db = open(dir.path());
    assert_eq!(db.active_db(), "default");
    assert_eq!(tensor(&mut db, "only_default"), vec![6.0]);
    assert!(execute_line(&mut db, "SHOW only_other", 1).is_err());
    run(&mut db, "USE other");
    assert_eq!(tensor(&mut db, "only_other"), vec![5.0]);
}

#[test]
fn checkpoint_works_with_a_relative_data_dir() {
    // `./data` (relative) is the default data_dir; the checkpoint's dataset
    // packages must still land inside `checkpoint/`, not a nested copy of
    // the data directory.
    let abs = tempfile::tempdir_in(".").unwrap();
    let cwd = std::env::current_dir().unwrap();
    let rel = abs.path().strip_prefix(&cwd).unwrap().to_path_buf();
    assert!(rel.is_relative());
    {
        let mut db = open(&rel);
        run(&mut db, "DATASET items COLUMNS (id: Int)");
        run(&mut db, "INSERT INTO items VALUES (1)");
        run(&mut db, "CHECKPOINT");
        run(&mut db, "INSERT INTO items VALUES (2)");
    }
    assert!(!abs.path().join("default").join("data").exists());
    let mut db = open(&rel);
    assert_eq!(row_count(&mut db, "items"), 2);
}
