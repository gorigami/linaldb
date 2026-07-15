// tests/server_pipeline_search_test.rs
// Server-level coverage for CONSISTENCY_PLAN.md Track C / C4: no server
// test previously referenced PIPELINE or SEARCH statements at all — the
// generic `/execute` passthrough (src/server/mod.rs) was never verified
// against these DSL surfaces specifically. Follows the same harness pattern
// as tests/server_test.rs.

use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

async fn post_execute(port: u16, dsl: &str) -> serde_json::Value {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body(dsl.to_string())
        .send()
        .await
        .unwrap_or_else(|e| panic!("request failed for `{dsl}`: {e}"));
    assert_eq!(resp.status(), 200, "unexpected status for `{dsl}`");
    let body = resp.text().await.expect("failed to get body");
    serde_json::from_str(&body)
        .unwrap_or_else(|e| panic!("invalid JSON for `{dsl}`: {e}\nbody: {body}"))
}

#[tokio::test]
async fn test_pipeline_lifecycle_over_http() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8300;
    let db_clone = db.clone();
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1000)).await;

    let create = post_execute(port, "DATASET items COLUMNS (id: Int, price: Float)").await;
    assert_eq!(create["status"], "ok");

    let insert1 = post_execute(port, "INSERT INTO items VALUES (1, 5.0)").await;
    assert_eq!(insert1["status"], "ok");
    let insert2 = post_execute(port, "INSERT INTO items VALUES (2, 50.0)").await;
    assert_eq!(insert2["status"], "ok");

    let define = post_execute(port, "DEFINE PIPELINE cheap AS WHERE price < 10").await;
    assert_eq!(define["status"], "ok");

    let show = post_execute(port, "SHOW PIPELINES").await;
    assert_eq!(show["status"], "ok");

    let apply = post_execute(port, "APPLY PIPELINE cheap ON items INTO cheap_items").await;
    assert_eq!(apply["status"], "ok");

    let select = post_execute(port, "SELECT * FROM cheap_items").await;
    assert_eq!(select["status"], "ok");
    let rows = select["result"]["Table"]["rows"]
        .as_array()
        .expect("expected rows array");
    assert_eq!(
        rows.len(),
        1,
        "only the id=1 row (price=5.0 < 10) should survive the filter"
    );

    let drop = post_execute(port, "DROP PIPELINE cheap").await;
    assert_eq!(drop["status"], "ok");

    // Applying a dropped pipeline must now fail, not silently succeed.
    let apply_after_drop = post_execute(port, "APPLY PIPELINE cheap ON items").await;
    assert_eq!(
        apply_after_drop["status"], "error",
        "APPLY on a dropped pipeline should error"
    );
}

#[tokio::test]
async fn test_search_over_http() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8301;
    let db_clone = db.clone();
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1000)).await;

    let create = post_execute(port, "DATASET docs COLUMNS (id: Int, embedding: Vector(3))").await;
    assert_eq!(create["status"], "ok");

    post_execute(port, "INSERT INTO docs VALUES (1, [1.0, 0.0, 0.0])").await;
    post_execute(port, "INSERT INTO docs VALUES (2, [0.9, 0.1, 0.0])").await;
    post_execute(port, "INSERT INTO docs VALUES (3, [0.0, 1.0, 0.0])").await;

    let index = post_execute(port, "CREATE VECTOR INDEX ON docs(embedding)").await;
    assert_eq!(index["status"], "ok");

    let search = post_execute(
        port,
        "SEARCH docs ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 2 INTO search_out",
    )
    .await;
    assert_eq!(search["status"], "ok");

    let select = post_execute(port, "SELECT * FROM search_out").await;
    assert_eq!(select["status"], "ok");
    let rows = select["result"]["Table"]["rows"]
        .as_array()
        .expect("expected rows array");
    assert_eq!(
        rows.len(),
        2,
        "LIMIT 2 should return exactly 2 nearest rows"
    );
}

#[tokio::test]
async fn test_search_without_index_errors_over_http() {
    // SEARCH always requires a prebuilt vector index (DSL_REFERENCE.md §7);
    // confirm that constraint is enforced through the HTTP path too, not
    // just direct DSL execution.
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8302;
    let db_clone = db.clone();
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1000)).await;

    post_execute(
        port,
        "DATASET docs2 COLUMNS (id: Int, embedding: Vector(3))",
    )
    .await;
    post_execute(port, "INSERT INTO docs2 VALUES (1, [1.0, 0.0, 0.0])").await;

    let search = post_execute(
        port,
        "SEARCH docs2 ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 1",
    )
    .await;
    assert_eq!(
        search["status"], "error",
        "SEARCH without a vector index should error, not silently succeed"
    );
}
