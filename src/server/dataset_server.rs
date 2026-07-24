use crate::core::storage::ParquetStorage;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde_json::Value;
use std::sync::Arc;

const DEFAULT_DATABASE: &str = "default";

/// Lightweight read-only server for delivering datasets.
pub struct DatasetServer {
    pub storage: Arc<ParquetStorage>,
}

impl DatasetServer {
    pub fn new(storage: Arc<ParquetStorage>) -> Self {
        Self { storage }
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/datasets/:name/manifest.json", get(get_manifest))
            .route("/datasets/:name/schema.json", get(get_schema))
            .route("/datasets/:name/stats.json", get(get_stats))
            .route("/datasets/:name/data.parquet", get(get_data))
            .with_state(Arc::new(self))
    }
}

/// Every dataset package actually lives at
/// `{data_dir}/{database}/datasets/{name}/...` — `resolve_persistence_path`/
/// `instance_base_dir` (`src/dsl/persistence.rs`) always insert the active
/// database name as a path segment, e.g. `SAVE DATASET`'s own success
/// message literally says `Saved dataset 'x' (v1) to './data/default'`.
/// This mirrors that convention using the same `X-Linal-Database` header
/// `/execute` already honors (default `"default"`, matching the engine's
/// own default active database) — **fixed in v0.1.73**: before this, every
/// handler here read `{data_dir}/datasets/{name}/...`, missing the
/// database segment entirely, so `/delivery` 404'd for every dataset ever
/// saved through the standard `SAVE DATASET` path, in every database
/// including the default one. Found by driving the Python client's
/// `Dataset.to_arrow()` against a real running server rather than trusting
/// the endpoint worked because it existed and had tests.
fn dataset_dir(server: &DatasetServer, headers: &HeaderMap, name: &str) -> String {
    let db = headers
        .get("X-Linal-Database")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(DEFAULT_DATABASE);
    format!("{}/{}/datasets/{}", server.storage.base_path(), db, name)
}

async fn get_manifest(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let path = format!("{}/manifest.json", dataset_dir(&server, &headers, &name));
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Manifest not found"})),
        ),
    }
}

async fn get_schema(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let path = format!("{}/schema.json", dataset_dir(&server, &headers, &name));
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Schema not found"})),
        ),
    }
}

async fn get_stats(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let path = format!("{}/stats.json", dataset_dir(&server, &headers, &name));
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Stats not found"})),
        ),
    }
}

async fn get_data(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let path = format!("{}/data.parquet", dataset_dir(&server, &headers, &name));
    match std::fs::read(path) {
        Ok(data) => (StatusCode::OK, data).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Data not found").into_response(),
    }
}
