# Server enforces a 30s query timeout (QUERY_TIMEOUT_SECS in
# src/server/mod.rs) and returns a clean status:error response when hit --
# our own request timeout is set a little above that so we see the
# server's real timeout message instead of cutting the connection first.
DEFAULT_TIMEOUT_SECS <- 35

#' Connect to a running `linal serve` instance
#'
#' @param url Server base URL, e.g. `"http://localhost:8080"`.
#' @param database Optional database name to target via the
#'   `X-Linal-Database` header (see `clients/CONTRACT.md`).
#' @return A `linal_connection` object, passed as the first argument to
#'   `linal_execute()`/`linal_query()`/`linal_dataset()`.
#' @export
linal_connect <- function(url, database = NULL) {
  structure(
    list(url = sub("/+$", "", url), database = database),
    class = "linal_connection"
  )
}

#' Run one DSL command
#'
#' @param conn A `linal_connection` from `linal_connect()`.
#' @param dsl The DSL command string.
#' @return `NULL` (no output), a `character` scalar (`Message`), a
#'   `linal_table_result`, or a `linal_tensor_result`. Raises a
#'   `linal_error` condition on `status: "error"`.
#' @export
linal_execute <- function(conn, dsl) {
  req <- httr2::request(paste0(conn$url, "/execute"))
  req <- httr2::req_url_query(req, format = "json")
  req <- httr2::req_body_raw(req, dsl, type = "text/plain")
  if (!is.null(conn$database)) {
    req <- httr2::req_headers(req, `X-Linal-Database` = conn$database)
  }
  req <- httr2::req_timeout(req, DEFAULT_TIMEOUT_SECS)
  # We check `status` in the JSON body ourselves (contract Sec.4: a 400 for
  # e.g. an empty command still carries the same {"status": "error", ...}
  # shape) rather than let httr2 throw on a non-2xx HTTP status.
  req <- httr2::req_error(req, is_error = function(resp) FALSE)

  resp <- httr2::req_perform(req)

  body <- tryCatch(
    httr2::resp_body_json(resp, simplifyVector = FALSE),
    error = function(e) NULL
  )

  if (is.null(body)) {
    if (httr2::resp_status(resp) >= 400) {
      stop(linal_error(sprintf(
        "HTTP %d from server (non-JSON body): %s",
        httr2::resp_status(resp), httr2::resp_body_string(resp)
      )))
    }
    stop(linal_error(sprintf(
      "Non-JSON response from server (HTTP %d): %s",
      httr2::resp_status(resp), httr2::resp_body_string(resp)
    )))
  }

  if (is.null(body$status) || body$status != "ok") {
    msg <- if (!is.null(body$error)) {
      body$error
    } else {
      sprintf("Unknown server error (HTTP %d)", httr2::resp_status(resp))
    }
    stop(linal_error(msg))
  }

  unwrap_result(body$result)
}

#' Run a DSL command expected to return a table and return a `data.frame`
#'
#' @param conn A `linal_connection` from `linal_connect()`.
#' @param dsl The DSL command string.
#' @return A `data.frame`. Raises a `linal_error` if the result isn't
#'   table-shaped.
#' @export
linal_query <- function(conn, dsl) {
  result <- linal_execute(conn, dsl)
  if (!inherits(result, "linal_table_result")) {
    stop(linal_error(sprintf(
      "linal_query() expects a table-shaped result, got %s (use linal_execute() for non-table results)",
      class(result)[[1]]
    )))
  }
  columns_to_dataframe(result$columns)
}
