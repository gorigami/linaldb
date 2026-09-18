use linal::engine::TensorDb;
use linal::server::start_server;
use serde_json::Value;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_server_background_job_lifecycle() {
    // 1. Setup DB and start server
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8105;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", port);

    // 2. Submit a job
    let resp = client
        .post(format!("{}/jobs", base_url))
        .body("VECTOR v = [10, 20, 30]")
        .send()
        .await
        .expect("Failed to submit job");

    assert_eq!(resp.status(), 202); // Accepted
    let json: Value = resp.json().await.unwrap();
    let job_id = json["job_id"].as_str().expect("job_id missing");
    println!("Submitted job: {}", job_id);

    // 3. Poll for completion
    let mut completed = false;
    for _ in 0..10 {
        let poll_resp = client
            .get(format!("{}/jobs/{}", base_url, job_id))
            .send()
            .await
            .unwrap();

        let poll_json: Value = poll_resp.json().await.unwrap();
        let status = poll_json["job"]["status"].as_str().unwrap();
        println!("Job status: {}", status);

        if status == "Completed" {
            completed = true;
            break;
        }
        sleep(Duration::from_millis(200)).await;
    }

    assert!(completed, "Job did not complete in time");

    // 4. Retrieve result
    let res_resp = client
        .get(format!("{}/jobs/{}/result", base_url, job_id))
        .send()
        .await
        .unwrap();

    let res_json: Value = res_resp.json().await.unwrap();
    println!("Job Result: {:?}", res_json);
    assert_eq!(res_json["status"], "ok");
    assert!(res_json["result"].is_object());
    // Result should be DslOutput::Message("Defined vector: v")
    assert!(res_json["result"]["Message"]
        .as_str()
        .unwrap()
        .contains("Defined vector: v"));

    // 5. Verify it's actually in the DB
    let exec_resp = client
        .post(format!("{}/execute", base_url))
        .body("SHOW v")
        .send()
        .await
        .unwrap();

    let exec_body = exec_resp.text().await.unwrap();
    println!("SHOW v Body: {}", exec_body);
    assert!(exec_body.contains("id:"));
}

#[tokio::test]
async fn test_server_job_not_found() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8106;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let random_id = uuid::Uuid::new_v4();
    let resp = client
        .get(format!("http://localhost:{}/jobs/{}", port, random_id))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 404);
}

/// `/jobs` sibling of `server_usability_test.rs`'s
/// `test_server_use_database_errors_with_header` -- this call site had the
/// identical unconditional-restore bug with no guard and no coverage at
/// all. A `USE <db>` submitted as a job alongside `X-Linal-Database` is
/// rejected up front (before a job is even created), same as `/execute`.
#[tokio::test]
async fn test_server_jobs_use_database_errors_with_header() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8107;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", port);

    client
        .post(format!("{}/databases/jobs_use_header_db", base_url))
        .send()
        .await
        .unwrap();

    let resp = client
        .post(format!("{}/jobs", base_url))
        .header("X-Linal-Database", "default")
        .body("USE jobs_use_header_db")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "error");
    let message = body["message"].as_str().unwrap();
    assert!(
        message.contains("USE") && message.contains("X-Linal-Database"),
        "error should explain the USE + header conflict, got: {message}"
    );
}
