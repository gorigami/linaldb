// tests/server_per_db_locking_test.rs
//
// `linal serve`'s per-database locking (`server::engine::SharedEngine`):
// requests against different databases never wait on each other, while the
// session semantics of USE / X-Linal-Database stay exactly what they were
// under the old single global lock.

use linal::core::config::EngineConfig;
use linal::dsl::DslOutput;
use linal::engine::TensorDb;
use linal::server::engine::{Session, SharedEngine};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

fn make_engine(dir: &std::path::Path) -> Arc<SharedEngine> {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.to_path_buf();
    let mut db = TensorDb::with_config(config);
    Arc::new(SharedEngine::from_tensor_db(&mut db))
}

fn message(out: DslOutput) -> String {
    match out {
        DslOutput::Message(m) => m,
        other => panic!("expected a message, got {:?}", other),
    }
}

/// Runs `line` on a background thread and reports whether it finished
/// within `wait`.
fn finishes_within(
    engine: &Arc<SharedEngine>,
    session: Session,
    line: &'static str,
    wait: Duration,
) -> (bool, mpsc::Receiver<()>) {
    let (tx, rx) = mpsc::channel();
    let engine = engine.clone();
    std::thread::spawn(move || {
        let mut session = session;
        engine.execute(&mut session, line, 1).unwrap();
        let _ = tx.send(());
    });
    let done = rx.recv_timeout(wait).is_ok();
    (done, rx)
}

#[test]
fn write_on_one_database_does_not_block_another() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "CREATE DATABASE a", 1).unwrap();
    engine.execute(&mut s, "CREATE DATABASE b", 1).unwrap();

    // Simulate a long-running statement on `a` by holding its write lock.
    let a = engine.database("a").unwrap();
    let a_guard = a.write().unwrap();

    let (done_b, _) = finishes_within(
        &engine,
        Session::Pinned("b".into()),
        "VECTOR v = [1, 2, 3]",
        Duration::from_secs(5),
    );
    assert!(
        done_b,
        "a write on 'b' must not wait for a lock held on 'a'"
    );

    let (done_a, rx_a) = finishes_within(
        &engine,
        Session::Pinned("a".into()),
        "VECTOR v = [1, 2, 3]",
        Duration::from_millis(300),
    );
    assert!(!done_a, "a write on 'a' must wait for 'a's lock");

    drop(a_guard);
    rx_a.recv_timeout(Duration::from_secs(5))
        .expect("the write on 'a' should finish once the lock is released");
}

#[test]
fn show_runs_under_a_read_lock() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "VECTOR v = [1, 2, 3]", 1).unwrap();

    // Another reader holding `default`'s read lock must not block SHOW.
    let default = engine.database("default").unwrap();
    let _reader = default.read().unwrap();
    let (done, _) = finishes_within(&engine, Session::Server, "SHOW v", Duration::from_secs(5));
    assert!(
        done,
        "SHOW of a materialized tensor should only need a read lock"
    );
}

#[test]
fn show_of_a_lazy_tensor_still_materializes_it() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "VECTOR a = [1, 2, 3]", 1).unwrap();
    engine
        .execute(&mut s, "VECTOR b = [10, 20, 30]", 1)
        .unwrap();
    engine.execute(&mut s, "LAZY LET c = ADD a b", 1).unwrap();
    assert!(engine
        .database("default")
        .unwrap()
        .read()
        .unwrap()
        .is_lazy("c"));

    match engine.execute(&mut s, "SHOW c", 1).unwrap() {
        DslOutput::Tensor(t) => assert_eq!(t.data.as_slice(), &[11.0, 22.0, 33.0]),
        other => panic!("expected a tensor, got {:?}", other),
    }
    assert!(!engine
        .database("default")
        .unwrap()
        .read()
        .unwrap()
        .is_lazy("c"));
}

#[test]
fn headerless_use_persists_for_later_requests() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "CREATE DATABASE a", 1).unwrap();
    let msg = message(engine.execute(&mut s, "USE a", 1).unwrap());
    assert_eq!(msg, "Switched to database 'a'");
    assert_eq!(engine.server_active(), "a");

    engine
        .execute(&mut Session::Server, "VECTOR only_in_a = [1]", 1)
        .unwrap();
    assert!(engine
        .execute(&mut Session::Pinned("default".into()), "SHOW only_in_a", 1)
        .is_err());
    assert!(engine
        .execute(&mut Session::Pinned("a".into()), "SHOW only_in_a", 1)
        .is_ok());
}

#[test]
fn pinned_session_never_moves_the_server() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    engine
        .execute(&mut Session::Server, "CREATE DATABASE a", 1)
        .unwrap();

    // What a scheduled task with `target_db` does. It used to switch the
    // global active database and never switch back.
    let mut pinned = Session::Pinned("a".into());
    engine.execute(&mut pinned, "VECTOR v = [1]", 1).unwrap();
    engine.execute(&mut pinned, "USE default", 1).unwrap();
    assert_eq!(engine.server_active(), "default");

    let mut pinned = Session::Pinned("default".into());
    engine.execute(&mut pinned, "USE a", 1).unwrap();
    assert_eq!(engine.server_active(), "default");
    assert_eq!(engine.resolve(&pinned), "a");
}

#[test]
fn batch_use_persists_within_the_batch_and_follows_the_header_rules() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    engine
        .execute(&mut Session::Server, "CREATE DATABASE a", 1)
        .unwrap();

    // Pinned (header) batch: USE applies for the rest of the batch only.
    let mut pinned = Session::Pinned("default".into());
    let results =
        engine.execute_batch(&mut pinned, vec![("USE a", 1), ("VECTOR in_a = [1, 2]", 2)]);
    assert!(results.iter().all(|(_, r)| r.is_ok()));
    assert_eq!(engine.server_active(), "default");
    assert!(engine
        .execute(&mut Session::Pinned("a".into()), "SHOW in_a", 1)
        .is_ok());

    // Headerless batch: USE also persists for later requests.
    let mut following = Session::Following(engine.server_active());
    let results = engine.execute_batch(&mut following, vec![("USE a", 1)]);
    assert!(results.iter().all(|(_, r)| r.is_ok()));
    assert_eq!(engine.server_active(), "a");
}

#[test]
fn batch_stops_at_first_error() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let results = engine.execute_batch(
        &mut Session::Following("default".into()),
        vec![
            ("VECTOR ok = [1]", 1),
            ("SHOW does_not_exist", 2),
            ("VECTOR never = [1]", 3),
        ],
    );
    assert_eq!(results.len(), 2);
    assert!(results[0].1.is_ok());
    assert!(results[1].1.is_err());
}

#[test]
fn catalog_statements_are_answered_across_databases() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "CREATE DATABASE a", 1).unwrap();
    engine.execute(&mut s, "CREATE DATABASE b", 1).unwrap();

    // SHOW DATABASES from inside one database still sees all of them.
    let msg = message(
        engine
            .execute(&mut Session::Pinned("a".into()), "SHOW DATABASES", 1)
            .unwrap(),
    );
    for name in ["a", "b", "default"] {
        assert!(msg.contains(&format!("- {}", name)), "{msg}");
    }

    let msg = message(
        engine
            .execute(&mut s, "CREATE DATABASE IF NOT EXISTS a", 1)
            .unwrap(),
    );
    assert_eq!(msg, "Database 'a' already exists (skipped)");
    assert!(engine.execute(&mut s, "CREATE DATABASE a", 1).is_err());
    assert!(engine.execute(&mut s, "USE nope", 1).is_err());
    assert!(engine.execute(&mut s, "DROP DATABASE default", 1).is_err());
}

#[test]
fn dropping_the_active_database_falls_back_to_default() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "CREATE DATABASE a", 1).unwrap();
    engine.execute(&mut s, "USE a", 1).unwrap();
    assert!(dir.path().join("a").exists());

    engine.execute(&mut s, "DROP DATABASE a", 1).unwrap();
    assert_eq!(engine.server_active(), "default");
    assert!(!dir.path().join("a").exists());
    assert!(engine.database("a").is_err());
}

#[test]
fn pipelines_stay_session_wide_across_databases() {
    let dir = tempfile::tempdir().unwrap();
    let engine = make_engine(dir.path());
    let mut s = Session::Server;
    engine.execute(&mut s, "CREATE DATABASE a", 1).unwrap();
    engine
        .execute(
            &mut s,
            "DEFINE PIPELINE top AS ORDER BY score DESC THEN LIMIT 1",
            1,
        )
        .unwrap();

    let mut in_a = Session::Pinned("a".into());
    for line in [
        "DATASET items COLUMNS (id: Int, score: Float)",
        "INSERT INTO items VALUES (1, 0.2)",
        "INSERT INTO items VALUES (2, 0.9)",
        "APPLY PIPELINE top ON items INTO best",
    ] {
        engine.execute(&mut in_a, line, 1).unwrap();
    }
    match engine.execute(&mut in_a, "SHOW best", 1).unwrap() {
        DslOutput::Table(ds) => assert_eq!(ds.rows.len(), 1),
        other => panic!("expected a table, got {:?}", other),
    }
}
