# /delivery dataset export -- see clients/CONTRACT.md Sec.2. Mirrors
# clients/python/linaldb/dataset.py; this is checkpoint 4 of
# PYTHON_R_INTEROP_PLAN.md, the R counterpart of checkpoint 2's Python
# saved-dataset Parquet export.

#' A handle to a saved dataset's `/delivery/*` export
#'
#' @param conn A `linal_connection` from `linal_connect()`.
#' @param name Dataset name.
#' @return A `linal_dataset` object, passed to `linal_dataset_schema()`/
#'   `linal_dataset_manifest()`/`linal_dataset_stats()`/
#'   `linal_dataset_to_arrow()`/`linal_dataset_read()`.
#' @export
linal_dataset <- function(conn, name) {
  structure(list(conn = conn, name = name), class = "linal_dataset")
}

.linal_delivery_url <- function(ds, path) {
  sprintf("%s/delivery/datasets/%s/%s", ds$conn$url, ds$name, path)
}

#' /delivery resolves the same per-database subdirectory /execute does via
#' this header (contract Sec.2, engine v0.1.73) -- without it, a
#' connection configured for a non-default database would silently hit
#' the *default* database's copy of a same-named dataset instead (or 404
#' if there isn't one), not the one this connection actually points at.
#' Found while building a real example against a non-default database,
#' not caught by checkpoint 4's tests, which only ever used the default
#' database. Mirrors the equivalent fix in the Python client's dataset.py.
#' @keywords internal
#' @noRd
.linal_delivery_request <- function(ds, path, timeout_secs) {
  req <- httr2::request(.linal_delivery_url(ds, path))
  req <- httr2::req_timeout(req, timeout_secs)
  if (!is.null(ds$conn$database)) {
    req <- httr2::req_headers(req, `X-Linal-Database` = ds$conn$database)
  }
  req
}

.linal_delivery_get_json <- function(ds, path) {
  req <- .linal_delivery_request(ds, path, 30)
  resp <- httr2::req_perform(httr2::req_error(req, is_error = function(resp) FALSE))
  if (httr2::resp_status(resp) != 200) {
    stop(linal_error(sprintf(
      "GET /delivery/datasets/%s/%s failed (HTTP %d): %s",
      ds$name, path, httr2::resp_status(resp), httr2::resp_body_string(resp)
    )))
  }
  httr2::resp_body_json(resp, simplifyVector = FALSE)
}

#' @rdname linal_dataset
#' @param ds A `linal_dataset` from `linal_dataset()`.
#' @export
linal_dataset_manifest <- function(ds) {
  .linal_delivery_get_json(ds, "manifest.json")
}

#' @rdname linal_dataset
#' @export
linal_dataset_schema <- function(ds) {
  .linal_delivery_get_json(ds, "schema.json")
}

#' @rdname linal_dataset
#' @export
linal_dataset_stats <- function(ds) {
  .linal_delivery_get_json(ds, "stats.json")
}

#' Column names whose declared `value_type` (per `schema.json`, contract
#' Sec.2's authoritative source for logical column typing) is Vector or
#' Matrix.
#' @keywords internal
#' @noRd
.vector_or_matrix_columns <- function(schema) {
  names_out <- character(0)
  for (col in schema$columns) {
    vt <- col$value_type
    if (is.list(vt) && (!is.null(vt$Vector) || !is.null(vt$Matrix))) {
      names_out <- c(names_out, col$name)
    }
  }
  names_out
}

#' Unwrap one legacy-fallback-encoded cell (a JSON string like
#' `{"Vector": [1.0,2.0,3.0]}`, contract Sec.2) into a plain R value.
#' @keywords internal
#' @noRd
.unwrap_json_fallback_cell <- function(raw) {
  if (is.na(raw)) {
    return(NA)
  }
  unwrap_value(jsonlite::fromJSON(raw, simplifyVector = FALSE))
}

#' Fetch `data.parquet` and return it as a `data.frame`, transparently
#' unwrapping any column that landed in the legacy JSON-string fallback
#' encoding (contract Sec.2) back into a real list-column -- the caller
#' never sees the raw tagged-JSON text (e.g. Vector wire-tagged as
#' \code{list(Vector = c(...))}).
#'
#' @rdname linal_dataset
#' @export
linal_dataset_read <- function(ds) {
  req <- .linal_delivery_request(ds, "data.parquet", 60)
  resp <- httr2::req_perform(httr2::req_error(req, is_error = function(resp) FALSE))
  if (httr2::resp_status(resp) != 200) {
    stop(linal_error(sprintf(
      "GET /delivery/datasets/%s/data.parquet failed (HTTP %d): %s",
      ds$name, httr2::resp_status(resp), httr2::resp_body_string(resp)
    )))
  }
  raw_bytes <- httr2::resp_body_raw(resp)

  table <- arrow::read_parquet(raw_bytes, as_data_frame = FALSE)
  df <- as.data.frame(table)

  vector_or_matrix <- .vector_or_matrix_columns(linal_dataset_schema(ds))
  for (col_name in vector_or_matrix) {
    if (!(col_name %in% names(df))) {
      next
    }
    field_idx <- match(col_name, table$schema$names) - 1L
    field_type <- table$schema$field(field_idx)$type
    if (inherits(field_type, "Utf8") || inherits(field_type, "LargeUtf8")) {
      # Legacy fallback encoding -- unwrap each cell's JSON text.
      df[[col_name]] <- lapply(df[[col_name]], .unwrap_json_fallback_cell)
    }
    # Otherwise it's already a native FixedSizeList column, which
    # as.data.frame() on the Table already turned into a correct
    # numeric list-column with no help needed.
  }

  df
}

#' `arrow::arrow_table(linal_dataset_read(ds))` -- the fixed-up
#' `data.frame` re-encoded as a real Arrow Table, for callers who want to
#' keep working in Arrow rather than convert to base R types. Rebuilding
#' from the already-unwrapped data.frame (rather than patching the raw
#' Arrow Table's columns directly) keeps a single source of truth with
#' `linal_dataset_read()` instead of two parallel unwrapping
#' implementations that could drift apart.
#'
#' @rdname linal_dataset
#' @export
linal_dataset_to_arrow <- function(ds) {
  arrow::arrow_table(linal_dataset_read(ds))
}
