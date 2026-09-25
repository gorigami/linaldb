// tests/select_read_only_test.rs
//
// SELECT never writes to the database: CTEs and FROM-subqueries live in a
// per-query scope (`LogicalPlan::Values`) instead of being registered as
// datasets. That fixes a leak (a subquery alias became a permanent dataset,
// so the same query failed the second time), and lets `linal serve` run
// SELECT -- and SEARCH without INTO -- under a read lock, concurrently with
// other reads of the same database.

use linal::core::config::EngineConfig;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use linal::server::engine::{Session, SharedEngine};
use std::sync::{mpsc, Arc};
use std::time::Duration;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let db = TensorDb::with_config(config);
    (dir, db)
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn rows(db: &mut TensorDb, line: &str) -> Vec<Vec<String>> {
    match run(db, line) {
        DslOutput::Table(ds) => ds
            .rows
            .iter()
            .map(|r| r.values.iter().map(|v| v.to_string()).collect())
            .collect(),
        other => panic!("`{}`: expected a table, got {:?}", line, other),
    }
}

fn setup(db: &mut TensorDb) {
    for line in [
        "DATASET items COLUMNS (id: Int, cat: Int, score: Float)",
        "INSERT INTO items VALUES (1, 10, 0.5)",
        "INSERT INTO items VALUES (2, 10, 0.9)",
        "INSERT INTO items VALUES (3, 20, 0.7)",
        "DATASET cats COLUMNS (cat: Int, label: String)",
        r#"INSERT INTO cats VALUES (10, "a")"#,
        r#"INSERT INTO cats VALUES (20, "b")"#,
    ] {
        run(db, line);
    }
}

fn datasets(db: &TensorDb) -> Vec<String> {
    let mut names = db.list_dataset_names();
    names.sort();
    names
}

#[test]
fn from_subquery_does_not_leak_and_can_run_twice() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let q = "SELECT * FROM (SELECT id FROM items WHERE score > 0.6) AS sub";
    assert_eq!(rows(&mut db, q).len(), 2);
    // Used to fail: "Dataset name already exists: sub".
    assert_eq!(rows(&mut db, q).len(), 2);
    assert_eq!(datasets(&db), vec!["cats", "items"]);
}

#[test]
fn ctes_are_scoped_to_the_query() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let q = "WITH hi AS (SELECT id, cat FROM items WHERE score > 0.6) SELECT id FROM hi";
    assert_eq!(rows(&mut db, q).len(), 2);
    assert_eq!(rows(&mut db, q).len(), 2);
    assert_eq!(datasets(&db), vec!["cats", "items"]);
}

#[test]
fn a_cte_shadows_a_dataset_with_the_same_name() {
    let (_dir, mut db) = db();
    setup(&mut db);
    // Used to fail: creating the CTE's temp dataset collided with `items`.
    let q = "WITH items AS (SELECT id FROM items WHERE score > 0.8) SELECT id FROM items";
    assert_eq!(rows(&mut db, q), vec![vec!["2".to_string()]]);
    // The real dataset is untouched.
    assert_eq!(rows(&mut db, "SELECT id FROM items").len(), 3);
}

#[test]
fn ctes_are_visible_to_joins_later_ctes_subqueries_and_union() {
    let (_dir, mut db) = db();
    setup(&mut db);

    let joined = rows(
        &mut db,
        "WITH hi AS (SELECT id, cat FROM items WHERE score > 0.6) \
         SELECT * FROM hi JOIN cats ON hi.cat = cats.cat",
    );
    assert_eq!(joined.len(), 2);

    let right_cte = rows(
        &mut db,
        "WITH c AS (SELECT cat, label FROM cats WHERE cat = 10) \
         SELECT * FROM items JOIN c ON items.cat = c.cat",
    );
    assert_eq!(right_cte.len(), 2);

    let chained = rows(
        &mut db,
        "WITH hi AS (SELECT id, score FROM items WHERE score > 0.6), \
         top AS (SELECT id FROM hi WHERE score > 0.8) SELECT id FROM top",
    );
    assert_eq!(chained, vec![vec!["2".to_string()]]);

    let nested = rows(
        &mut db,
        "WITH hi AS (SELECT id, score FROM items WHERE score > 0.6) \
         SELECT * FROM (SELECT id FROM hi WHERE score < 0.8) AS sub",
    );
    assert_eq!(nested, vec![vec!["3".to_string()]]);

    let union = rows(
        &mut db,
        "WITH lo AS (SELECT id FROM items WHERE score < 0.6) \
         SELECT id FROM items WHERE score > 0.8 UNION ALL SELECT id FROM lo",
    );
    assert_eq!(union.len(), 2);
    assert_eq!(datasets(&db), vec!["cats", "items"]);
}

fn engine() -> (tempfile::TempDir, Arc<SharedEngine>) {
    let (dir, mut db) = db();
    setup(&mut db);
    run(&mut db, "DATASET docs COLUMNS (id: Int, emb: Vector(3))");
    run(&mut db, "INSERT INTO docs VALUES (1, [1.0, 0.0, 0.0])");
    run(&mut db, "INSERT INTO docs VALUES (2, [0.0, 1.0, 0.0])");
    run(&mut db, "CREATE VECTOR INDEX ON docs(emb)");
    (dir, Arc::new(SharedEngine::from_tensor_db(&mut db)))
}

/// Runs `line` on another thread and reports whether it finished within
/// `wait`, plus a receiver to wait for it later.
fn finishes_within(
    engine: &Arc<SharedEngine>,
    line: &'static str,
    wait: Duration,
) -> (bool, mpsc::Receiver<()>) {
    let (tx, rx) = mpsc::channel();
    let engine = engine.clone();
    std::thread::spawn(move || {
        engine.execute(&mut Session::Server, line, 1).unwrap();
        let _ = tx.send(());
    });
    (rx.recv_timeout(wait).is_ok(), rx)
}

#[test]
fn select_and_search_run_under_a_read_lock() {
    let (_dir, engine) = engine();
    let default = engine.database("default").unwrap();
    // Another reader holds the database's read lock the whole time.
    let _reader = default.read().unwrap();

    for q in [
        "SELECT id FROM items WHERE score > 0.6",
        "WITH hi AS (SELECT id FROM items WHERE score > 0.6) SELECT id FROM hi",
        "SEARCH docs ON emb QUERY [1.0, 0.0, 0.0] LIMIT 1",
    ] {
        let (done, _) = finishes_within(&engine, q, Duration::from_secs(5));
        assert!(done, "`{q}` should only need a read lock");
    }

    // SEARCH ... INTO writes a dataset, so it still waits for the reader.
    let (done, rx) = finishes_within(
        &engine,
        "SEARCH docs ON emb QUERY [1.0, 0.0, 0.0] LIMIT 1 INTO nearest",
        Duration::from_millis(300),
    );
    assert!(!done, "SEARCH ... INTO must take the write lock");
    drop(_reader);
    rx.recv_timeout(Duration::from_secs(5))
        .expect("SEARCH ... INTO should finish once the reader is gone");
}

#[test]
fn concurrent_selects_on_one_database_all_succeed() {
    let (_dir, engine) = engine();
    let handles: Vec<_> = (0..16)
        .map(|i| {
            let engine = engine.clone();
            std::thread::spawn(move || {
                let q = if i % 2 == 0 {
                    "SELECT cat, COUNT(*) AS n FROM items GROUP BY cat"
                } else {
                    "SELECT * FROM (SELECT id FROM items WHERE score > 0.6) AS sub"
                };
                match engine.execute(&mut Session::Server, q, 1).unwrap() {
                    DslOutput::Table(ds) => ds.rows.len(),
                    other => panic!("expected a table, got {:?}", other),
                }
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), 2);
    }
}
