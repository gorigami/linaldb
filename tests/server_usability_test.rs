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
        .post(format!(
            "http://localhost:{}/databases/use_persist_db",
            port
        ))
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

/// Regression test for the sibling bug to the one fixed above: `USE <db>`
/// sent to `/execute` *with* `X-Linal-Database` set used to report success
/// (`"Switched to database 'x'"`) while the switch was silently reverted
/// before the response went out -- the header always won again on the next
/// request. Rather than resurrect that misleading-success behavior, a
/// single-statement request combining a header with `USE` is now a clear
/// error: there is no "rest of the request" for `USE` to usefully persist
/// across in a single-statement call. See `test_batch_use_persists_within_one_request`
/// for the case where `USE` legitimately belongs (spanning multiple
/// statements sent as one `/execute/batch` request).
#[tokio::test]
async fn test_server_use_database_errors_with_header() {
    let port = 8205;
    let _db = setup_server(port).await;
    let client = Client::new();

    client
        .post(format!("http://localhost:{}/databases/use_header_db", port))
        .send()
        .await
        .unwrap();

    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body("USE use_header_db")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "error");
    let error = body["error"].as_str().unwrap();
    assert!(
        error.contains("USE") && error.contains("X-Linal-Database"),
        "error should explain the USE + header conflict, got: {error}"
    );

    // And, just as important: the header's own target (`default`) must be
    // completely unaffected -- the rejected statement never ran at all.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body("SHOW ALL DATASETS")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

/// `/execute/batch`: the exact scenario that motivated this whole fix --
/// `CREATE DATABASE`/`USE`/dataset creation, run as one script the way a
/// real `.lnl` file (or the linal-hub playground) would send it. `USE`
/// persists naturally across the rest of this one batch, then is undone
/// once the batch finishes.
#[tokio::test]
async fn test_batch_use_persists_within_one_request() {
    let port = 8206;
    let _db = setup_server(port).await;
    let client = Client::new();

    let script = "CREATE DATABASE IF NOT EXISTS batch_smoke_db\n\
                  USE batch_smoke_db\n\
                  DATASET users COLUMNS (\n    id: INT,\n    age: INT\n)\n\
                  INSERT INTO users VALUES (1, 25)\n\
                  SHOW SCHEMA users";

    let resp = client
        .post(format!(
            "http://localhost:{}/execute/batch?format=json",
            port
        ))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body(script)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok", "full body: {body}");
    let statements = body["statements"].as_array().unwrap();
    assert_eq!(statements.len(), 5);
    for stmt in statements {
        assert_eq!(stmt["status"], "ok", "statement failed: {stmt}");
    }

    // The dataset must genuinely be inside batch_smoke_db, not `default`.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "batch_smoke_db")
        .header("Content-Type", "text/plain")
        .body("SHOW SCHEMA users")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

/// `/execute/batch`: once the batch finishes, the header's previously
/// active database must be restored -- same invariant `/execute` already
/// has for a single statement, just scoped to N statements.
#[tokio::test]
async fn test_batch_restores_header_db_after_completion() {
    let port = 8207;
    let _db = setup_server(port).await;
    let client = Client::new();

    let script = "CREATE DATABASE IF NOT EXISTS batch_restore_db\nUSE batch_restore_db";
    client
        .post(format!("http://localhost:{}/execute/batch", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body(script)
        .send()
        .await
        .unwrap();

    // A follow-up plain request with the same header must still see
    // `default`, not the batch's own internal `USE batch_restore_db`.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body("VECTOR probe = [1]")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "default")
        .header("Content-Type", "text/plain")
        .body("SHOW probe")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"], "ok",
        "probe should be in default, proving the batch's internal USE didn't leak"
    );
}

/// `/execute/batch`: a batch that errors partway through stops there --
/// later statements in the body are never executed.
#[tokio::test]
async fn test_batch_stops_at_first_error() {
    let port = 8208;
    let _db = setup_server(port).await;
    let client = Client::new();

    let script = "VECTOR ok_one = [1]\n\
                  THIS IS NOT VALID DSL\n\
                  VECTOR never_created = [2]";

    let resp = client
        .post(format!(
            "http://localhost:{}/execute/batch?format=json",
            port
        ))
        .header("Content-Type", "text/plain")
        .body(script)
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "error");
    let statements = body["statements"].as_array().unwrap();
    // Only the first (ok) and second (the failing one) statements ran.
    assert_eq!(statements.len(), 2);
    assert_eq!(statements[0]["status"], "ok");
    assert_eq!(statements[1]["status"], "error");

    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("SHOW never_created")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"], "error",
        "never_created must not exist -- the batch should have stopped before it"
    );
}

/// `/execute/batch` sibling of `test_server_multitenancy`: two batches with
/// different headers must not leak into each other, same guarantee as
/// single-statement `/execute`, just at batch granularity.
#[tokio::test]
async fn test_batch_multitenancy_isolated_across_headers() {
    let port = 8209;
    let _db = setup_server(port).await;
    let client = Client::new();

    client
        .post(format!("http://localhost:{}/databases/batch_db_x", port))
        .send()
        .await
        .unwrap();
    client
        .post(format!("http://localhost:{}/databases/batch_db_y", port))
        .send()
        .await
        .unwrap();

    client
        .post(format!("http://localhost:{}/execute/batch", port))
        .header("X-Linal-Database", "batch_db_x")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [100]\nVECTOR w = [1000]")
        .send()
        .await
        .unwrap();
    client
        .post(format!("http://localhost:{}/execute/batch", port))
        .header("X-Linal-Database", "batch_db_y")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [200]\nVECTOR w = [2000]")
        .send()
        .await
        .unwrap();

    let resp_x = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "batch_db_x")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_x: serde_json::Value = resp_x.json().await.unwrap();
    assert_eq!(body_x["result"]["Tensor"]["data"][0], 100.0);

    let resp_y = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "batch_db_y")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_y: serde_json::Value = resp_y.json().await.unwrap();
    assert_eq!(body_y["result"]["Tensor"]["data"][0], 200.0);
}

/// `/schedule`: locks in the current, intentional behavior that a scheduled
/// task's `target_db` switch is permanent (operator-configured recurring
/// tasks are a different contract from per-visitor request isolation) --
/// documented in `docs/ARCHITECTURE.md`, previously untested. A future
/// change to this needs to be a deliberate, visible diff against this test,
/// not a silent surprise.
#[tokio::test]
async fn test_schedule_target_db_switch_is_permanent() {
    let port = 8210;
    let _db = setup_server(port).await;
    let client = Client::new();

    client
        .post(format!(
            "http://localhost:{}/databases/scheduled_target_db",
            port
        ))
        .send()
        .await
        .unwrap();

    let resp = client
        .post(format!("http://localhost:{}/schedule", port))
        .json(&serde_json::json!({
            "name": "switches_active_db",
            "command": "VECTOR sched_marker = [99]",
            "interval_secs": 1,
            "target_db": "scheduled_target_db"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let task_id = body["id"].as_str().unwrap().to_string();

    sleep(Duration::from_secs(2)).await;

    // A plain, headerless request now sees `scheduled_target_db` as active
    // -- the scheduler's switch was never reverted.
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("SHOW sched_marker")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"], "ok",
        "scheduler's target_db switch should still be active"
    );
    assert_eq!(body["result"]["Tensor"]["data"][0], 99.0);

    client
        .delete(format!("http://localhost:{}/schedule/{}", port, task_id))
        .send()
        .await
        .unwrap();
}
