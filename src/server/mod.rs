pub mod dataset_server;
pub mod engine;
pub mod jobs;
pub mod scheduler;

use crate::dsl::DslOutput;
use crate::engine::TensorDb;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use engine::{Session, SharedEngine};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use toon_format::encode_default;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

struct AppState {
    engine: Arc<SharedEngine>,
    scheduler: Arc<scheduler::Scheduler>,
    job_manager: Arc<jobs::JobManager>,
}

const MAX_COMMAND_LENGTH: usize = 16 * 1024; // 16KB
const QUERY_TIMEOUT_SECS: u64 = 30;

#[derive(Deserialize, utoipa::IntoParams)]
struct ExecuteParams {
    /// Format of the output: 'toon' (default), 'json', or 'arrow' (binary
    /// Arrow IPC stream, tabular results only -- see `execute_command`)
    #[serde(default = "default_format")]
    format: String,
}

fn default_format() -> String {
    "toon".to_string()
}

/// Content-type for `?format=arrow`'s binary response body -- the standard
/// MIME type for the Arrow IPC streaming format.
const ARROW_IPC_CONTENT_TYPE: &str = "application/vnd.apache.arrow.stream";

/// Encodes a `dataset_legacy::Dataset` as an Arrow IPC stream
/// (`core::storage::dataset_to_record_batch` -> `arrow::ipc::writer::StreamWriter`).
/// Reuses the exact conversion `/delivery`'s Parquet export and
/// `core::provenance::record_batch_content_hash` already trust.
fn dataset_to_arrow_ipc_bytes(
    dataset: &crate::core::dataset_legacy::Dataset,
) -> Result<Vec<u8>, String> {
    let batch =
        crate::core::storage::dataset_to_record_batch(dataset).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    {
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct ExecuteRequest {
    command: String,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct ExecuteResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<DslOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct BatchStatementResult {
    statement: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<DslOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct BatchExecuteResponse {
    status: String,
    statements: Vec<BatchStatementResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Deserialize)]
pub struct ScheduleRequest {
    pub name: String,
    pub command: String,
    pub interval_secs: u64,
    pub target_db: Option<String>,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        execute_command,
        execute_batch,
        health_check
    ),
    components(
        schemas(ExecuteRequest, ExecuteResponse, BatchStatementResult, BatchExecuteResponse)
    ),
    tags(
        (name = "VectorDB", description = "LINAL Analytical Engine API")
    )
)]
struct ApiDoc;

/// Starts the server on `db`'s databases. The server takes ownership of every
/// database in `db` (each gets its own lock, see `engine::SharedEngine`), so
/// `db` must not be used for execution after this is called.
pub async fn start_server(db: Arc<RwLock<TensorDb>>, port: u16) {
    let engine = Arc::new(SharedEngine::from_tensor_db(&mut db.write().unwrap()));
    start_server_with_engine(engine, port).await
}

pub async fn start_server_with_engine(engine: Arc<SharedEngine>, port: u16) {
    let scheduler = Arc::new(scheduler::Scheduler::new(engine.clone()));
    let job_manager = Arc::new(jobs::JobManager::new());
    let scheduler_handle = scheduler.clone();
    tokio::spawn(async move {
        scheduler_handle.start().await;
    });

    let state = Arc::new(AppState {
        engine,
        scheduler,
        job_manager,
    });

    // Dataset Delivery Routes (Read-Only)
    let storage = Arc::new(crate::core::storage::ParquetStorage::new("./data"));
    let ds_server = dataset_server::DatasetServer::new(storage);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/execute", post(execute_command))
        .route("/execute/batch", post(execute_batch))
        .route("/databases", get(list_databases))
        .route(
            "/databases/:name",
            post(create_database).delete(delete_database),
        )
        .route("/schedule", get(list_schedules).post(create_schedule))
        .route("/schedule/:id", delete(delete_schedule))
        .route("/jobs", get(list_jobs).post(submit_job))
        .route("/jobs/:id", get(get_job).delete(cancel_job))
        .route("/jobs/:id/result", get(get_job_result))
        .with_state(state)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/delivery", ds_server.router()); // Mount dataset server at /delivery

    let addr = format!("0.0.0.0:{}", port);
    println!("Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("Shutdown signal received, starting graceful shutdown...");
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Health check", body = String)
    )
)]
async fn health_check() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

/// A request that supplies `X-Linal-Database` has already pinned this
/// request's execution target for its whole duration -- a `USE <db>`
/// statement sent alongside it has no "rest of the request" left to persist
/// across, so this rejects the combination outright instead of silently
/// reverting the switch and reporting success (the previous behavior: the
/// response claimed `"Switched to database 'x'"`, but the switch never
/// outlived this one request -- see CHANGELOG for the bug this closes).
/// `POST /execute/batch` is deliberately exempt from this check: a `USE`
/// there persists naturally for the rest of that one batch, since nothing
/// restores the active database until the whole batch finishes.
fn reject_use_with_header(command: &str, target_db: &Option<String>) -> Option<String> {
    if target_db.is_none() {
        return None;
    }
    match crate::dsl::parser::parse(command) {
        Ok(crate::dsl::ast::Statement::UseDatabase(_)) => Some(
            "USE has no persisting effect when X-Linal-Database is set on a \
             single-statement request -- the header already pins the execution \
             target for this request. Drop the header if you want USE to control \
             it, or send this as part of a script to POST /execute/batch if you \
             need USE to persist across multiple statements."
                .to_string(),
        ),
        _ => None,
    }
}

#[utoipa::path(
    post,
    path = "/execute",
    request_body = String,
    params(
        ExecuteParams
    ),
    responses(
        (status = 200, description = "Execution result", body = ExecuteResponse)
    )
)]
async fn execute_command(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecuteParams>,
    headers: axum::http::HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Determine if request is JSON (legacy) or plain text (preferred)
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/plain");

    let command = if content_type.contains("application/json") {
        // Legacy JSON format: {"command": "..."}
        // Log deprecation warning
        eprintln!("[DEPRECATED] JSON request format is deprecated. Use Content-Type: text/plain with raw DSL command instead.");

        match serde_json::from_str::<ExecuteRequest>(&body) {
            Ok(req) => req.command,
            Err(_) => {
                // If JSON parsing fails, treat as raw DSL
                body.trim().to_string()
            }
        }
    } else {
        // Preferred: raw DSL text
        body.trim().to_string()
    };

    if command.len() > MAX_COMMAND_LENGTH {
        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&ExecuteResponse {
                status: "error".to_string(),
                result: None,
                error: Some(format!(
                    "Command too long (max {} bytes)",
                    MAX_COMMAND_LENGTH
                )),
            })
            .unwrap(),
        )
            .into_response();
    }

    if command.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&ExecuteResponse {
                status: "error".to_string(),
                result: None,
                error: Some("Command cannot be empty".to_string()),
            })
            .unwrap(),
        )
            .into_response();
    }

    let target_db = headers
        .get("X-Linal-Database")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if let Some(msg) = reject_use_with_header(&command, &target_db) {
        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&ExecuteResponse {
                status: "error".to_string(),
                result: None,
                error: Some(msg),
            })
            .unwrap(),
        )
            .into_response();
    }

    // Wrap execution in timeout and spawn_blocking to keep server responsive.
    // A request with X-Linal-Database is pinned to that database for its
    // whole duration; one without it follows (and, via USE, moves) the
    // server's active database. Either way it only locks its own database.
    let engine = state.engine.clone();
    let command_clone = command.clone();

    let exec_result = tokio::time::timeout(
        std::time::Duration::from_secs(QUERY_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let mut session = match target_db {
                Some(name) => Session::Pinned(name),
                None => Session::Server,
            };
            engine.execute(&mut session, &command_clone, 1)
        }),
    )
    .await;

    let response = match exec_result {
        Ok(Ok(Ok(output))) => {
            let result = match output {
                DslOutput::None => None,
                _ => Some(output),
            };
            ExecuteResponse {
                status: "ok".to_string(),
                result,
                error: None,
            }
        }
        Ok(Ok(Err(e))) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("{}", e)),
        },
        Ok(Err(e)) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("Execution task panicked: {}", e)),
        },
        Err(_) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("Query timed out after {}s", QUERY_TIMEOUT_SECS)),
        },
    };

    // Serialize based on requested format
    match params.format.as_str() {
        "json" => {
            // JSON format (opt-in)
            let body = serde_json::to_string(&response).unwrap_or_else(|e| {
                format!(
                    "{{\"status\": \"error\", \"error\": \"Serialization failed: {}\"}}",
                    e
                )
            });
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response()
        }
        "arrow" => {
            // Binary Arrow IPC stream (opt-in, additive -- see
            // PERFORMANCE_OPTIMIZATION_PLAN.md Phase 3). Only a successful
            // `DslOutput::Table` result can be represented this way; every
            // other case (an execution error, or a non-tabular success like
            // a bare Tensor/Message) falls back to a JSON body, same
            // convention the rest of this endpoint already uses for errors
            // regardless of the requested format.
            match &response.result {
                Some(DslOutput::Table(dataset)) if response.status == "ok" => {
                    match dataset_to_arrow_ipc_bytes(dataset) {
                        Ok(bytes) => (
                            StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, ARROW_IPC_CONTENT_TYPE)],
                            bytes,
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            serde_json::to_string(&ExecuteResponse {
                                status: "error".to_string(),
                                result: None,
                                error: Some(format!("Arrow IPC encoding failed: {e}")),
                            })
                            .unwrap_or_default()
                            .into_bytes(),
                        )
                            .into_response(),
                    }
                }
                _ => (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&response)
                        .unwrap_or_else(|e| {
                            format!(
                                "{{\"status\": \"error\", \"error\": \"Serialization failed: {}\"}}",
                                e
                            )
                        })
                        .into_bytes(),
                )
                    .into_response(),
            }
        }
        _ => {
            // TOON format (default)
            let body = encode_default(&response)
                .unwrap_or_else(|e| format!("status: error\nerror: Serialization failed: {}", e));
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/toon")],
                body,
            )
                .into_response()
        }
    }
}

#[utoipa::path(
    post,
    path = "/execute/batch",
    request_body = String,
    params(
        ExecuteParams
    ),
    responses(
        (status = 200, description = "Batch execution result", body = BatchExecuteResponse)
    )
)]
async fn execute_batch(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecuteParams>,
    headers: axum::http::HeaderMap,
    body: String,
) -> impl IntoResponse {
    if body.len() > MAX_COMMAND_LENGTH {
        return respond_batch(
            &params,
            BatchExecuteResponse {
                status: "error".to_string(),
                statements: vec![],
                error: Some(format!(
                    "Batch body too long (max {} bytes)",
                    MAX_COMMAND_LENGTH
                )),
            },
        );
    }

    let statements = match crate::dsl::script::split_script(&body) {
        Ok(statements) if !statements.is_empty() => statements,
        Ok(_) => {
            return respond_batch(
                &params,
                BatchExecuteResponse {
                    status: "error".to_string(),
                    statements: vec![],
                    error: Some("Batch body contained no statements".to_string()),
                },
            );
        }
        Err(e) => {
            return respond_batch(
                &params,
                BatchExecuteResponse {
                    status: "error".to_string(),
                    statements: vec![],
                    error: Some(e.to_string()),
                },
            );
        }
    };

    let target_db = headers
        .get("X-Linal-Database")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // A USE inside the batch persists for the rest of *this* batch. Without
    // the header it also persists for later requests (the server's active
    // database); with the header it never leaves the batch. Consecutive
    // statements on one database run under a single write-lock hold.
    let engine = state.engine.clone();
    let exec_result = tokio::time::timeout(
        std::time::Duration::from_secs(QUERY_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let mut session = match target_db {
                Some(name) => Session::Pinned(name),
                None => Session::Following(engine.server_active()),
            };
            engine
                .execute_batch(
                    &mut session,
                    statements
                        .iter()
                        .map(|stmt| (stmt.text.as_str(), stmt.start_line)),
                )
                .into_iter()
                .map(|(statement, result)| match result {
                    Ok(output) => BatchStatementResult {
                        statement,
                        status: "ok".to_string(),
                        result: match output {
                            DslOutput::None => None,
                            other => Some(other),
                        },
                        error: None,
                    },
                    Err(e) => BatchStatementResult {
                        statement,
                        status: "error".to_string(),
                        result: None,
                        error: Some(format!("{}", e)),
                    },
                })
                .collect::<Vec<_>>()
        }),
    )
    .await;

    let response = match exec_result {
        Ok(Ok(results)) => {
            let all_ok = results.iter().all(|r| r.status == "ok");
            BatchExecuteResponse {
                status: if all_ok { "ok" } else { "error" }.to_string(),
                statements: results,
                error: None,
            }
        }
        Ok(Err(e)) => BatchExecuteResponse {
            status: "error".to_string(),
            statements: vec![],
            error: Some(format!("Execution task panicked: {}", e)),
        },
        Err(_) => BatchExecuteResponse {
            status: "error".to_string(),
            statements: vec![],
            error: Some(format!("Batch timed out after {}s", QUERY_TIMEOUT_SECS)),
        },
    };

    respond_batch(&params, response)
}

fn respond_batch(
    params: &ExecuteParams,
    response: BatchExecuteResponse,
) -> axum::response::Response {
    match params.format.as_str() {
        "json" => {
            let body = serde_json::to_string(&response).unwrap_or_else(|e| {
                format!(
                    "{{\"status\": \"error\", \"error\": \"Serialization failed: {}\"}}",
                    e
                )
            });
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response()
        }
        _ => {
            let body = encode_default(&response)
                .unwrap_or_else(|e| format!("status: error\nerror: Serialization failed: {}", e));
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/toon")],
                body,
            )
                .into_response()
        }
    }
}

async fn list_schedules(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let tasks = state.scheduler.list_tasks();
    Json(serde_json::json!({ "status": "ok", "tasks": tasks }))
}

async fn create_schedule(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScheduleRequest>,
) -> impl IntoResponse {
    let id = state
        .scheduler
        .add_task(req.name, req.command, req.interval_secs, req.target_db);
    (
        StatusCode::CREATED,
        Json(serde_json::json!({ "status": "ok", "id": id })),
    )
}

async fn delete_schedule(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id_str): axum::extract::Path<String>,
) -> impl IntoResponse {
    match uuid::Uuid::parse_str(&id_str) {
        Ok(id) => {
            if state.scheduler.remove_task(id) {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({ "status": "ok", "message": "Task removed" })),
                )
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "status": "error", "message": "Task not found" })),
                )
            }
        }
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Invalid UUID" })),
        ),
    }
}

async fn list_databases(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let databases = state.engine.list_databases();
    Json(serde_json::json!({ "status": "ok", "databases": databases }))
}

async fn create_database(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    match state.engine.create_database(name.clone()) {
        Ok(_) => (
            StatusCode::CREATED,
            Json(
                serde_json::json!({ "status": "ok", "message": format!("Database '{}' created", name) }),
            ),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "error": format!("{}", e) })),
        ),
    }
}

async fn delete_database(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    match state.engine.drop_database(&name) {
        Ok(_) => (
            StatusCode::OK,
            Json(
                serde_json::json!({ "status": "ok", "message": format!("Database '{}' dropped", name) }),
            ),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "error": format!("{}", e) })),
        ),
    }
}

// --- Job Handlers ---

async fn list_jobs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let jobs = state.job_manager.list_jobs();
    Json(serde_json::json!({ "status": "ok", "jobs": jobs }))
}

async fn submit_job(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: String,
) -> impl IntoResponse {
    let command = body.trim().to_string();
    if command.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Command cannot be empty" })),
        )
            .into_response();
    }

    let target_db = headers
        .get("X-Linal-Database")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if let Some(msg) = reject_use_with_header(&command, &target_db) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": msg })),
        )
            .into_response();
    }

    let job_id = state
        .job_manager
        .create_job(command.clone(), target_db.clone());

    // Spawn background execution. Pinned to the header's database if given,
    // otherwise to the server's active database at the time the job runs.
    // Unlike headerless `/execute`, a headerless job's own `USE` does not
    // outlive the job -- the pre-existing behavior (the job path always
    // restored the previous active database), kept unchanged here.
    let engine = state.engine.clone();
    let mgr = state.job_manager.clone();
    let job_id_clone = job_id;

    tokio::spawn(async move {
        mgr.update_job_status(job_id_clone, jobs::JobStatus::Running);

        let res = tokio::task::spawn_blocking(move || {
            let mut session = Session::Pinned(target_db.unwrap_or_else(|| engine.server_active()));
            engine.execute(&mut session, &command, 1)
        })
        .await;

        match res {
            Ok(Ok(output)) => mgr.finish_job(job_id_clone, Ok(output)),
            Ok(Err(e)) => mgr.finish_job(job_id_clone, Err(format!("{}", e))),
            Err(e) => mgr.finish_job(job_id_clone, Err(format!("Task panicked: {}", e))),
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "status": "ok", "job_id": job_id })),
    )
        .into_response()
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id_str): axum::extract::Path<String>,
) -> impl IntoResponse {
    match uuid::Uuid::parse_str(&id_str) {
        Ok(id) => match state.job_manager.get_job(id) {
            Some(job) => (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "ok", "job": job })),
            )
                .into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "status": "error", "message": "Job not found" })),
            )
                .into_response(),
        },
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Invalid UUID" })),
        )
            .into_response(),
    }
}

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id_str): axum::extract::Path<String>,
) -> impl IntoResponse {
    // Current implementation doesn't support killing threads easily in spawn_blocking
    // We just mark it as failed if it's still pending
    match uuid::Uuid::parse_str(&id_str) {
        Ok(id) => {
            if let Some(job) = state.job_manager.get_job(id) {
                if job.status == jobs::JobStatus::Pending {
                    state
                        .job_manager
                        .update_job_status(id, jobs::JobStatus::Failed);
                    return (StatusCode::OK, Json(serde_json::json!({ "status": "ok", "message": "Pending job cancelled" }))).into_response();
                }
                (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "status": "error", "message": "Cannot cancel running or finished job" }))).into_response()
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "status": "error", "message": "Job not found" })),
                )
                    .into_response()
            }
        }
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Invalid UUID" })),
        )
            .into_response(),
    }
}

async fn get_job_result(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id_str): axum::extract::Path<String>,
) -> impl IntoResponse {
    match uuid::Uuid::parse_str(&id_str) {
        Ok(id) => match state.job_manager.get_job(id) {
            Some(job) => {
                if job.status == jobs::JobStatus::Completed {
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "status": "ok", "result": job.result })),
                    )
                        .into_response()
                } else if job.status == jobs::JobStatus::Failed {
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "status": "error", "error": job.error })),
                    )
                        .into_response()
                } else {
                    (StatusCode::ACCEPTED, Json(serde_json::json!({ "status": "pending", "message": "Job still processing" }))).into_response()
                }
            }
            None => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "status": "error", "message": "Job not found" })),
            )
                .into_response(),
        },
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Invalid UUID" })),
        )
            .into_response(),
    }
}
