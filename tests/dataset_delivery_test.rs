use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use linal::core::dataset_legacy::{Dataset, DatasetId};
use linal::core::storage::{ParquetStorage, StorageEngine};
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use linal::server::dataset_server::DatasetServer;
use std::fs;
use std::sync::Arc;
use tower::util::ServiceExt; // for `oneshot` and `ready`

/// Helper to create a test dataset
fn create_test_dataset(name: &str) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("val", ValueType::Float),
    ]));

    let mut dataset = Dataset::new(DatasetId(1), schema.clone(), Some(name.to_string()));

    let rows = vec![
        Tuple::new(schema.clone(), vec![Value::Int(1), Value::Float(1.1)]).unwrap(),
        Tuple::new(schema.clone(), vec![Value::Int(2), Value::Float(2.2)]).unwrap(),
    ];

    dataset.rows = rows;
    dataset.metadata.update_stats(&schema, &dataset.rows);
    dataset
}

#[tokio::test]
async fn test_dataset_delivery_http_endpoints() {
    let temp_dir = "/tmp/linal_test_delivery_http";
    let _ = fs::remove_dir_all(temp_dir);
    fs::create_dir_all(temp_dir).unwrap();

    // `/delivery` always resolves `{data_dir}/{database}/datasets/{name}` —
    // see `dataset_dir()` in `src/server/dataset_server.rs` — defaulting to
    // the "default" database when no `X-Linal-Database` header is sent, so
    // the fixture must be saved under that same subdirectory to match what
    // a real `SAVE DATASET` run through `dsl::persistence` would produce
    // (see `test_dataset_delivery_matches_real_save_dataset_path` below for
    // a test that goes through that real path end-to-end instead of
    // constructing the directory layout by hand like this one does).
    let db_dir = format!("{}/default", temp_dir);
    fs::create_dir_all(&db_dir).unwrap();
    let storage = Arc::new(ParquetStorage::new(&db_dir));
    let dataset = create_test_dataset("delivery_test");

    // 1. Save dataset (creates the package structure)
    storage.save_dataset(&dataset).unwrap();

    // 2. Setup DatasetServer, rooted at the outer data_dir (matches how
    // `server::start_server` constructs it — NOT `db_dir`).
    let ds_server = DatasetServer::new(Arc::new(ParquetStorage::new(temp_dir)));
    let app: Router = ds_server.router();

    // 3. Test Manifest Endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/datasets/delivery_test/manifest.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("\"name\":\"delivery_test\""));
    assert!(body_str.contains("\"version\":\"1.0\""));

    // 4. Test Schema Endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/datasets/delivery_test/schema.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // DatasetSchema structure has "columns"
    assert!(body_str.contains("\"columns\""));
    // ValueType::Int serializes to "Int"
    assert!(body_str.contains("\"value_type\":\"Int\""));
    assert!(body_str.contains("\"name\":\"id\""));

    // 5. Test Non-Existent Dataset
    let response = app
        .oneshot(
            Request::builder()
                .uri("/datasets/ghost/manifest.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

/// Regression test for the real bug (fixed in v0.1.73): this drives
/// `SAVE DATASET` through the actual DSL/`TensorDb` path — the same one
/// `linal serve`'s `/execute` uses — instead of hand-constructing the
/// on-disk package layout like `test_dataset_delivery_http_endpoints`
/// above does. That's precisely the gap that let the original bug ship
/// undetected: `dataset_server.rs`'s handlers read
/// `{data_dir}/datasets/{name}/...`, but `dsl::persistence`'s
/// `resolve_persistence_path`/`instance_base_dir` always write to
/// `{data_dir}/{database}/datasets/{name}/...` (the active database name
/// as an extra path segment) — a mismatch invisible to a test that
/// constructs its own fixture directory by hand to match whatever the
/// handler currently expects, found only by exercising the two real,
/// independently-written code paths together.
#[tokio::test]
async fn test_dataset_delivery_matches_real_save_dataset_path() {
    use linal::core::storage::ParquetStorage as PS;
    use linal::dsl::execute_line;
    use linal::engine::TensorDb;

    let temp_dir = "/tmp/linal_test_delivery_real_save_path";
    let _ = fs::remove_dir_all(temp_dir);
    fs::create_dir_all(temp_dir).unwrap();

    // Real DSL path: a fresh TensorDb pointed at temp_dir (mirrors how
    // `linal serve` configures its default data_dir, just pointed at a
    // scratch directory for this test) running the exact same statements
    // `SAVE DATASET` support in `/execute` would run.
    let mut db = TensorDb::new();
    db.config.storage.data_dir = temp_dir.into();
    execute_line(&mut db, "DATASET probe COLUMNS (id: Int, val: Float)", 1).unwrap();
    execute_line(&mut db, "INSERT INTO probe VALUES (1, 1.1)", 1).unwrap();
    execute_line(&mut db, "SAVE DATASET probe", 1).unwrap();

    // Real `/delivery` path: rooted at the same outer data_dir, exactly
    // as `server::start_server` constructs it.
    let ds_server = DatasetServer::new(Arc::new(PS::new(temp_dir)));
    let app: Router = ds_server.router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/datasets/probe/schema.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a dataset saved via the real SAVE DATASET path must be reachable via /delivery \
         with no X-Linal-Database header (defaults to the active database, 'default')"
    );

    let _ = fs::remove_dir_all(temp_dir);
}
