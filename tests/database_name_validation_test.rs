// tests/database_name_validation_test.rs
//
// Database names become directory names under `data_dir`, and dropping a
// database deletes that directory. The DSL only accepts identifiers, but
// `POST/DELETE /databases/:name` and `linal db create|drop <name>` take any
// string, so a name like `../victim` must be rejected before it touches the
// filesystem.

use linal::core::config::EngineConfig;
use linal::engine::TensorDb;
use linal::server::engine::SharedEngine;
use std::path::Path;

const BAD_NAMES: &[&str] = &[
    "..",
    ".",
    "../victim",
    "a/b",
    "a\\b",
    "/tmp/x",
    "",
    "with space",
];

fn tensor_db(data_dir: &Path) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = data_dir.to_path_buf();
    TensorDb::with_config(config)
}

/// `root/data` is the engine's data dir; `root/victim` sits next to it.
fn layout() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let data_dir = root.path().join("data");
    let victim = root.path().join("victim");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&victim).unwrap();
    std::fs::write(victim.join("keep.txt"), "do not delete").unwrap();
    (root, data_dir, victim)
}

#[test]
fn tensor_db_rejects_path_like_names() {
    let (_root, data_dir, victim) = layout();
    let mut db = tensor_db(&data_dir);

    for name in BAD_NAMES {
        assert!(
            db.create_database(name.to_string()).is_err(),
            "create_database accepted {:?}",
            name
        );
        assert!(
            db.drop_database(name).is_err(),
            "drop_database accepted {:?}",
            name
        );
    }
    assert!(victim.join("keep.txt").exists());
    assert!(!db.list_databases().iter().any(|n| n.contains("victim")));
}

#[test]
fn shared_engine_rejects_path_like_names() {
    let (_root, data_dir, victim) = layout();
    let mut db = tensor_db(&data_dir);
    let engine = SharedEngine::from_tensor_db(&mut db);

    for name in BAD_NAMES {
        assert!(
            engine.create_database(name.to_string()).is_err(),
            "create_database accepted {:?}",
            name
        );
        assert!(
            engine.drop_database(name).is_err(),
            "drop_database accepted {:?}",
            name
        );
    }
    assert!(victim.join("keep.txt").exists());
}

#[test]
fn valid_names_still_work() {
    let (_root, data_dir, _victim) = layout();
    let mut db = tensor_db(&data_dir);
    for name in ["analytics", "casmi_2026", "Run-01"] {
        db.create_database(name.to_string()).unwrap();
        assert!(data_dir.join(name).is_dir());
        db.drop_database(name).unwrap();
        assert!(!data_dir.join(name).exists());
    }
}

/// The real HTTP route: axum percent-decodes `..%2Fvictim` into `../victim`
/// before the handler sees it.
#[tokio::test]
async fn http_routes_reject_path_like_names() {
    let (_root, data_dir, victim) = layout();
    let db = tensor_db(&data_dir);
    let port = 8320;
    let engine = std::sync::Arc::new(SharedEngine::from_tensor_db(&mut { db }));
    tokio::spawn(linal::server::start_server_with_engine(engine, port));
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    for encoded in ["..%2Fvictim", "%2E%2E"] {
        let url = format!("http://localhost:{}/databases/{}", port, encoded);
        let created = client.post(&url).send().await.unwrap();
        // `%2E%2E` may be normalized away by the client (404); either way it
        // must not succeed.
        assert!(
            created.status().is_client_error(),
            "POST {} was accepted",
            encoded
        );
        let dropped = client.delete(&url).send().await.unwrap();
        assert!(
            dropped.status().is_client_error(),
            "DELETE {} was accepted",
            encoded
        );
    }
    assert!(victim.join("keep.txt").exists());
    assert!(data_dir.exists());
}
