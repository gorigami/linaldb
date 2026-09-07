#' Open an embedded LINALDB engine
#'
#' Wraps a Rust `TensorDb` directly, in-process -- no `linal serve`
#' involved (contrast with `linaldb::linal_connect()`, the HTTP client
#' under `clients/r`). Persistence (`SAVE DATASET`, etc.) writes under
#' `data_dir` exactly like the CLI/REPL does.
#'
#' @param data_dir Optional directory for persisted datasets (`SAVE
#'   DATASET`, etc.). Defaults to `"./data"`, matching the CLI/REPL and
#'   HTTP server's own default (`src/core/config.rs`).
#' @return A `linal_embedded_db` object, passed as the first argument to
#'   `linal_embedded_execute()`/`linal_embedded_query()`/
#'   `linal_embedded_dataset()`.
#' @export
linal_embedded_db <- function(data_dir = NULL) {
  structure(
    list(ptr = Db$new(data_dir)),
    class = "linal_embedded_db"
  )
}

#' Run one DSL command against an embedded engine
#'
#' @param db A `linal_embedded_db` from `linal_embedded_db()`.
#' @param dsl The DSL command string.
#' @return `NULL` (no output), a `character` scalar (`Message`), a
#'   `linal_table_result`, or a `linal_tensor_result`. Raises a
#'   `linal_error` condition on failure.
#' @export
linal_embedded_execute <- function(db, dsl) {
  result <- tryCatch(
    db$ptr$execute(dsl),
    error = function(e) stop(linal_error(conditionMessage(e)))
  )
  unwrap_result(result)
}

#' Run a DSL command expected to return a table and return a `data.frame`
#'
#' @param db A `linal_embedded_db` from `linal_embedded_db()`.
#' @param dsl The DSL command string.
#' @return A `data.frame`. Raises a `linal_error` if the result isn't
#'   table-shaped.
#' @export
linal_embedded_query <- function(db, dsl) {
  result <- linal_embedded_execute(db, dsl)
  if (!inherits(result, "linal_table_result")) {
    stop(linal_error(sprintf(
      "linal_embedded_query() expects a table-shaped result, got %s (use linal_embedded_execute() for non-table results)",
      class(result)[[1]]
    )))
  }
  columns_to_dataframe(result$columns)
}

#' The active database's name
#'
#' @param db A `linal_embedded_db` from `linal_embedded_db()`.
#' @return A `character` scalar.
#' @export
linal_embedded_active_db <- function(db) {
  db$ptr$active_db()
}

#' The engine's configured data directory
#'
#' @param db A `linal_embedded_db` from `linal_embedded_db()`.
#' @return A `character` scalar (a path).
#' @export
linal_embedded_data_dir <- function(db) {
  db$ptr$data_dir()
}
