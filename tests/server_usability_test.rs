use linal::engine::TensorDb;
use linal::server::start_server;
use reqwest::Client;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

async fn setup_server(port: u16) -> Arc<RwLock<TensorDb>> {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let db_clone = db.clone();
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1500)).await;
    db
}

#[tokio::test]
async fn test_database_lifecycle_api() {
    let port = 8201;
    let _db = setup_server(port).await;
    let client = Client::new();

    // 1. List databases (should have default)
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(dbs.iter().any(|d| d.as_str() == Some("default")));

    // 2. Create a new database
    let resp = client
        .post(format!("http://localhost:{}/databases/test_api_db", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // 3. List again
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(dbs.iter().any(|d| d.as_str() == Some("test_api_db")));

    // 4. Delete the database
    let resp = client
        .delete(format!("http://localhost:{}/databases/test_api_db", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // 5. Verify it's gone
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(!dbs.iter().any(|d| d.as_str() == Some("test_api_db")));
}

#[tokio::test]
async fn test_server_multitenancy() {
    let port = 8202;
    let _db = setup_server(port).await;
    let client = Client::new();

    // Create two databases
    client
        .post(format!("http://localhost:{}/databases/db_x", port))
        .send()
        .await
        .unwrap();
    client
        .post(format!("http://localhost:{}/databases/db_y", port))
        .send()
        .await
        .unwrap();

    // Define 'v' in db_x
    client
        .post(format!("http://localhost:{}/execute", port))
        .header("X-Linal-Database", "db_x")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [100]")
        .send()
        .await
        .unwrap();

    // Define 'v' in db_y as something else
    client
        .post(format!("http://localhost:{}/execute", port))
        .header("X-Linal-Database", "db_y")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [200]")
        .send()
        .await
        .unwrap();

    // Verify db_x has 100
    let resp_x = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "db_x")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_x: serde_json::Value = resp_x.json().await.unwrap();
    assert_eq!(body_x["result"]["Tensor"]["data"][0], 100.0);

    // Verify db_y has 200
    let resp_y = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "db_y")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_y: serde_json::Value = resp_y.json().await.unwrap();
    assert_eq!(body_y["result"]["Tensor"]["data"][0], 200.0);
}

#[tokio::test]
async fn test_server_scheduling() {
    let port = 8203;
    let _db = setup_server(port).await;
    let client = Client::new();

    // Create a schedule that runs every 1 second
    let resp = client
        .post(format!("http://localhost:{}/schedule", port))
        .json(&serde_json::json!({
            "name": "periodic_calc",
            "command": "VECTOR sched_v = [42]",
            "interval_secs": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let task_id = body["id"].as_str().unwrap().to_string();

    // Wait for scheduler to run
    sleep(Duration::from_secs(3)).await;

    // Check if tensor was created
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("SHOW sched_v")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["result"]["Tensor"]["data"][0], 42.0);

    // Remove task
    let resp = client
        .delete(format!("http://localhost:{}/schedule/{}", port, task_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

/// Regression test for a real bug (fixed in v0.1.74): `execute_command`'s
/// "restore previous database to ensure per-request isolation" logic ran
/// *unconditionally* after every command, even for a plain headerless
/// request whose `target_db` was `None` — meaning a `USE <db>` DSL
/// statement sent to `/execute` with no `X-Linal-Database` header would
/// report success but have its own effect silently reverted before the
/// response even went out. Every subsequent headerless request, no matter
/// how many, kept seeing the *old* active database — the entire
/// session-level `USE` workflow was a no-op over HTTP, even though the
/// same command works correctly via the embedded CLI/REPL (which never
/// goes through this restore logic at all). `test_server_multitenancy`
/// above didn't catch this because it always sends the header on every
/// single request — it never exercises the "switch once via a plain `USE`,
/// then rely on that being remembered" pattern a real interactive session
/// (or a Python/R client that issues `USE` without setting `database=`)
/// would actually use.
#[tokio::test]
async fn test_server_use_database_persists_without_header() {
    let port = 8204;
    let _db = setup_server(port).await;
    let client = Client::new();

    client
        .post(format!("http://localhost:{}/databases/use_persist_db", port))
        .send()
        .await
        .unwrap();

    // Headerless USE -- this is the exact case that silently no-op'd.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("USE use_persist_db")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    // Still headerless -- if USE didn't really persist, this creates `v`
    // in `default` instead of `use_persist_db`.
    client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [7]")
        .send()
        .await
        .unwrap();

    // A third, still-headerless request must see the same active database
    // the second request left behind.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"], "ok",
        "USE use_persist_db should still be active for this headerless request"
    );
    assert_eq!(body["result"]["Tensor"]["data"][0], 7.0);

    // And `v` must genuinely be in `use_persist_db`, not `default`.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"], "error",
        "`v` must not exist in `default` -- it belongs in use_persist_db"
    );
}
