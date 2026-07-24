pub mod dataset_server;
pub mod jobs;
pub mod scheduler;

use crate::dsl::{execute_line, execute_line_shared, is_read_only, DslOutput};
use crate::engine::TensorDb;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use toon_format::encode_default;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

struct AppState {
    db: Arc<RwLock<TensorDb>>,
    scheduler: Arc<scheduler::Scheduler>,
    job_manager: Arc<jobs::JobManager>,
}

const MAX_COMMAND_LENGTH: usize = 16 * 1024; // 16KB
const QUERY_TIMEOUT_SECS: u64 = 30;

#[derive(Deserialize, utoipa::IntoParams)]
struct ExecuteParams {
    /// Format of the output: 'toon' (default) or 'json'
    #[serde(default = "default_format")]
    format: String,
}

fn default_format() -> String {
    "toon".to_string()
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
        health_check
    ),
    components(
        schemas(ExecuteRequest, ExecuteResponse)
    ),
    tags(
        (name = "VectorDB", description = "LINAL Analytical Engine API")
    )
)]
struct ApiDoc;

pub async fn start_server(db: Arc<RwLock<TensorDb>>, port: u16) {
    let scheduler = Arc::new(scheduler::Scheduler::new(db.clone()));
    let job_manager = Arc::new(jobs::JobManager::new());
    let scheduler_handle = scheduler.clone();
    tokio::spawn(async move {
        scheduler_handle.start().await;
    });

    let state = Arc::new(AppState {
        db,
        scheduler,
        job_manager,
    });

    // Dataset Delivery Routes (Read-Only)
    let storage = Arc::new(crate::core::storage::ParquetStorage::new("./data"));
    let ds_server = dataset_server::DatasetServer::new(storage);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/execute", post(execute_command))
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

    // Wrap execution in timeout and spawn_blocking to keep server responsive
    let db_arc = state.db.clone();
    let command_clone = command.clone();

    let exec_result = tokio::time::timeout(
        std::time::Duration::from_secs(QUERY_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let read_only = is_read_only(&command_clone);
            let prev_db_name;

            // Determine if we need a write lock for DB switching
            let needs_write_lock_for_db_switch = if let Some(ref db_name) = target_db {
                let db_read = db_arc.read().unwrap();
                prev_db_name = db_read.active_db().to_string();
                db_name != &prev_db_name
            } else {
                let db_read = db_arc.read().unwrap();
                prev_db_name = db_read.active_db().to_string();
                false
            };

            if read_only && !needs_write_lock_for_db_switch {
                let db = db_arc.read().unwrap();
                execute_line_shared(&db, &command_clone, 1)
            } else {
                let mut db = db_arc.write().unwrap();
                // Only a request that itself asked to *borrow* a different
                // active database via X-Linal-Database (`target_db.is_some()`)
                // gets that borrow reverted afterward, so concurrent requests
                // targeting different databases via the header can't clobber
                // each other's active-db state. A request with no header
                // (target_db=None) executes whatever command it was sent
                // as-is — including a `USE <db>` statement, whose entire
                // point is to persist a session-level database switch, the
                // same way it does for the embedded CLI/REPL (which never
                // goes through this restore logic at all). Previously this
                // restore ran unconditionally, so `USE <db>` sent to
                // `/execute` reported success but was silently undone before
                // the response even went out — every subsequent request,
                // headerless or not, still saw the old active database.
                let switched_db = target_db.is_some();
                if let Some(db_name) = target_db {
                    db.use_database(&db_name)
                        .map_err(|e| crate::dsl::DslError::Engine { line: 0, source: e })?;
                }

                let result = execute_line(&mut db, &command_clone, 1);

                if switched_db {
                    let _ = db.use_database(&prev_db_name);
                }
                result
            }
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
    let db = state.db.read().unwrap();
    let databases = db.list_databases();
    Json(serde_json::json!({ "status": "ok", "databases": databases }))
}

async fn create_database(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut db = state.db.write().unwrap();
    match db.create_database(name.clone()) {
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
    let mut db = state.db.write().unwrap();
    match db.drop_database(&name) {
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

    let job_id = state
        .job_manager
        .create_job(command.clone(), target_db.clone());

    // Spawn background execution
    let db_arc = state.db.clone();
    let mgr = state.job_manager.clone();
    let job_id_clone = job_id;

    tokio::spawn(async move {
        mgr.update_job_status(job_id_clone, jobs::JobStatus::Running);

        let read_only = is_read_only(&command);

        let res = tokio::task::spawn_blocking(move || {
            if read_only {
                let db = db_arc.read().unwrap();
                let prev_db = db.active_db().to_string();

                // Shared execution doesn't support changing DB easily if we want purely parallel.
                // But for now we just use a read lock.
                // NOTE: use_database needs a WRITE lock currently.
                // If the user specified a target_db different from actual, we might need a write lock even if command is SHOW.
                // To keep it simple, if target_db is set, we use write lock for now.
                if let Some(db_name) = target_db.as_ref() {
                    if db_name != &prev_db {
                        drop(db); // release read lock
                        let mut db_write = db_arc.write().unwrap();
                        let _ = db_write.use_database(db_name);
                        let exec_res = execute_line_shared(&db_write, &command, 1);
                        let _ = db_write.use_database(&prev_db);
                        return exec_res;
                    }
                }

                execute_line_shared(&db, &command, 1)
            } else {
                let mut db = db_arc.write().unwrap();
                let prev_db = db.active_db().to_string();

                if let Some(db_name) = target_db {
                    let _ = db.use_database(&db_name);
                }

                let exec_res = execute_line(&mut db, &command, 1);

                let _ = db.use_database(&prev_db);
                exec_res
            }
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
