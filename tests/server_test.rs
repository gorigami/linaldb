use axum::http::StatusCode;
use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_toon_server_output() {
    // 1. Setup DB and start server in background
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8095; // Use valid test port
    let db_clone = db.clone();

    // Spawn server
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    // Wait for server to be ready
    sleep(Duration::from_millis(1000)).await;

    // 2. Perform Request with raw DSL (new format)
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // 3. Assert Headers
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should be text/toon"
    );

    // 4. Assert Body Content
    let body = resp.text().await.expect("Failed to get body");
    println!("Response Body:\n{}", body);

    // Simple checks for TOON structure
    assert!(body.contains("status: ok"));
    assert!(body.contains("result:"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_toon_dsl_output() {
    // 1. Setup DB and start server
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8096;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // 2. Setup Data
    let client = reqwest::Client::new();
    // Create tensor first
    let resp_create = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2], [3, 4]]"
        }))
        .send()
        .await
        .unwrap();

    let create_body = resp_create.text().await.unwrap();
    println!("Create Response: {}", create_body);
    assert!(
        create_body.contains("status: ok"),
        "Creation failed: {}",
        create_body
    );

    // 3. Query it
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "SHOW m"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    println!("Matrix Body:\n{}", body);

    assert!(body.contains("Tensor:"));
    assert!(body.contains("shape:"));
    assert!(body.contains("dims[2]: 2,2")); // Check if TOON format is roughly as expected
                                            // Note: TOON format for arrays/dims might vary slightly based on toon-format crate version
                                            // but typically it's clean. Adjust assertion if failed.
}

#[tokio::test]
async fn test_json_backward_compatibility() {
    // Test that JSON format still works (with deprecation warning)
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8097;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Send request with JSON format (legacy)
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "VECTOR v = [1, 2, 3]"
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Should still work
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("Failed to get body");
    println!("JSON Backward Compat Response:\n{}", body);

    assert!(body.contains("status: ok"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_json_format_response() {
    // Test JSON format via query parameter
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8098;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request JSON format
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Verify JSON response
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("application/json"),
        "Content-Type should be application/json"
    );

    let body = resp.text().await.expect("Failed to get body");
    println!("JSON Format Response:\n{}", body);

    // Parse as JSON
    let json: serde_json::Value = serde_json::from_str(&body).expect("Should be valid JSON");
    assert_eq!(json["status"], "ok");
    assert!(json["result"].is_object());
}

#[tokio::test]
async fn test_toon_format_explicit() {
    // Test explicit TOON format via query parameter
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8099;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request TOON format explicitly
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=toon", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Verify TOON response
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should be text/toon"
    );

    let body = resp.text().await.expect("Failed to get body");
    println!("TOON Format Response:\n{}", body);

    assert!(body.contains("status: ok"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_invalid_format_defaults_to_toon() {
    // Test that invalid format defaults to TOON
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8100;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request with invalid format
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=xml", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Should default to TOON
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should default to text/toon"
    );

    let body = resp.text().await.expect("Failed to get body");
    assert!(body.contains("status: ok"));
}

/// `?format=arrow` (`PERFORMANCE_OPTIMIZATION_PLAN.md` Phase 3): a real
/// `Table` result comes back as genuine binary Arrow IPC, decodable by a
/// real `arrow` reader -- not just "some bytes with the right header".
#[tokio::test]
async fn test_arrow_format_table_result() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8103;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let base = format!("http://localhost:{}/execute", port);

    for stmt in [
        "DATASET arrow_fmt_t COLUMNS (id: Int, val: Float)",
        "INSERT INTO arrow_fmt_t VALUES (1, 1.5)",
        "INSERT INTO arrow_fmt_t VALUES (2, 2.5)",
    ] {
        let resp = client
            .post(&base)
            .header("Content-Type", "text/plain")
            .body(stmt)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "setup statement failed: {stmt}");
    }

    let resp = client
        .post(format!("{}?format=arrow", base))
        .header("Content-Type", "text/plain")
        .body("SELECT * FROM arrow_fmt_t")
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/vnd.apache.arrow.stream"
    );

    let bytes = resp.bytes().await.expect("Failed to get body");
    let cursor = std::io::Cursor::new(bytes.as_ref());
    let reader =
        arrow::ipc::reader::StreamReader::try_new(cursor, None).expect("valid Arrow IPC stream");
    let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "expected both inserted rows in the Arrow stream"
    );

    let schema = batches[0].schema();
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("val").is_ok());
}

/// A non-tabular result under `?format=arrow` (nothing Arrow-shaped to
/// encode) falls back to a JSON body rather than erroring or returning
/// meaningless bytes -- consistent with this endpoint's existing
/// error-always-falls-back-to-JSON convention regardless of requested
/// format.
#[tokio::test]
async fn test_arrow_format_falls_back_to_json_for_non_tabular_result() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8104;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=arrow", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("application/json"),
        "expected a JSON fallback for a non-tabular result under ?format=arrow"
    );
    let body: serde_json::Value = resp.json().await.expect("valid JSON fallback body");
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_server_validation_empty() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8101;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body("") // Empty body
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("Command cannot be empty"));
}

#[tokio::test]
async fn test_server_validation_length() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8102;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let long_command = "a".repeat(16 * 1024 + 1); // 16KB + 1
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body(long_command)
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("Command too long"));
}
