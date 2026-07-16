use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_matrix_indexing() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8098;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2, 3], [4, 5, 6]]"
        }))
        .send()
        .await
        .unwrap();

    // Test single element indexing
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET x = m[0, 1]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(body.contains("status: ok"), "Indexing failed: {}", body);

    // Verify the value
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW x"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW x output: {}", show_body);

    // x should be a scalar with value 2.0
    assert!(show_body.contains("Tensor:"));
    assert!(show_body.contains("2"));
}

#[tokio::test]
async fn test_vector_indexing() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8099;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a vector
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "VECTOR v = [10, 20, 30, 40]"
        }))
        .send()
        .await
        .unwrap();

    // Test indexing
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET elem = v[2]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("status: ok"),
        "Vector indexing failed: {}",
        body
    );

    // Verify the value
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW elem"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW elem output: {}", show_body);

    // elem should be 30.0
    assert!(show_body.contains("30"));
}

#[tokio::test]
async fn test_indexing_out_of_bounds() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8100;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2], [3, 4]]"
        }))
        .send()
        .await
        .unwrap();

    // Test out of bounds indexing
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET x = m[5, 5]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    println!("Out of bounds response: {}", body);

    // Should contain error
    assert!(body.contains("error") || body.contains("out of bounds") || body.contains("Index"));
}

#[tokio::test]
async fn test_indexing_wrong_dimensions() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8101;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a vector
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "VECTOR v = [1, 2, 3]"
        }))
        .send()
        .await
        .unwrap();

    // Test wrong number of indices (vector needs 1, providing 2)
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET x = v[0, 1]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    println!("Wrong dimensions response: {}", body);

    // Should contain error about dimension mismatch
    assert!(body.contains("error") || body.contains("dimension") || body.contains("mismatch"));
}

#[tokio::test]
async fn test_row_slicing() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8102;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2, 3], [4, 5, 6]]"
        }))
        .send()
        .await
        .unwrap();

    // Extract row using wildcard: m[0, *]
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET row = m[0, *]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(body.contains("status: ok"), "Row slicing failed: {}", body);

    // Verify the result is a vector
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW row"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW row output: {}", show_body);

    // Should be vector [1, 2, 3]
    assert!(show_body.contains("1"));
    assert!(show_body.contains("2"));
    assert!(show_body.contains("3"));
}

#[tokio::test]
async fn test_column_slicing() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8103;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2, 3], [4, 5, 6]]"
        }))
        .send()
        .await
        .unwrap();

    // Extract column using wildcard: m[*, 1]
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET col = m[*, 1]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("status: ok"),
        "Column slicing failed: {}",
        body
    );

    // Verify the result
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW col"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW col output: {}", show_body);

    // Should be vector [2, 5]
    assert!(show_body.contains("2"));
    assert!(show_body.contains("5"));
}

#[tokio::test]
async fn test_range_slicing() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8104;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]"
        }))
        .send()
        .await
        .unwrap();

    // Extract submatrix: m[0:2, 1:3]
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET sub = m[0:2, 1:3]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("status: ok"),
        "Range slicing failed: {}",
        body
    );

    // Verify the result
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW sub"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW sub output: {}", show_body);

    // Should be 2x2 matrix [[2, 3], [5, 6]]
    assert!(show_body.contains("dims[2]: 2,2"));
}

#[tokio::test]
async fn test_colon_wildcard() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8105;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a matrix
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2], [3, 4]]"
        }))
        .send()
        .await
        .unwrap();

    // Extract row using colon: m[1, :]
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET row = m[1, :]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("status: ok"),
        "Colon wildcard failed: {}",
        body
    );

    // Verify the result
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW row"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW row output: {}", show_body);

    // Should be vector [3, 4]
    assert!(show_body.contains("3"));
    assert!(show_body.contains("4"));
}
