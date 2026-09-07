# Local-filesystem dataset export -- the embedded counterpart of
# `clients/r/R/dataset.R`'s `/delivery/*` HTTP fetches. A saved dataset's
# package (`data.parquet`/`schema.json`/`stats.json`/`manifest.json`)
# already lives on disk at `Db$dataset_dir(name)`
# (`{data_dir}/{active_db}/datasets/{name}/`, `src/core/storage.rs`) --
# no export step needed, just read the files directly.

#' A handle to a saved dataset's on-disk package
#'
#' @param db A `linal_embedded_db` from `linal_embedded_db()`.
#' @param name Dataset name.
#' @return A `linal_embedded_dataset` object, passed to
#'   `linal_embedded_dataset_schema()`/`linal_embedded_dataset_manifest()`/
#'   `linal_embedded_dataset_stats()`/`linal_embedded_dataset_to_arrow()`/
#'   `linal_embedded_dataset_read()`.
#' @export
linal_embedded_dataset <- function(db, name) {
  structure(list(db = db, name = name), class = "linal_embedded_dataset")
}

.linal_embedded_dataset_dir <- function(ds) {
  ds$db$ptr$dataset_dir(ds$name)
}

.linal_embedded_read_json <- function(ds, filename) {
  path <- file.path(.linal_embedded_dataset_dir(ds), filename)
  if (!file.exists(path)) {
    stop(linal_error(sprintf("No such file: %s (has '%s' been SAVE'd?)", path, ds$name)))
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

#' @rdname linal_embedded_dataset
#' @param ds A `linal_embedded_dataset` from `linal_embedded_dataset()`.
#' @export
linal_embedded_dataset_manifest <- function(ds) {
  .linal_embedded_read_json(ds, "manifest.json")
}

#' @rdname linal_embedded_dataset
#' @export
linal_embedded_dataset_schema <- function(ds) {
  .linal_embedded_read_json(ds, "schema.json")
}

#' @rdname linal_embedded_dataset
#' @export
linal_embedded_dataset_stats <- function(ds) {
  .linal_embedded_read_json(ds, "stats.json")
}

#' Column names whose declared `value_type` (per `schema.json`) is Vector
#' or Matrix. Mirrors `clients/r/R/dataset.R`'s
#' `.vector_or_matrix_columns()`.
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

.unwrap_json_fallback_cell <- function(raw) {
  if (is.na(raw)) {
    return(NA)
  }
  parsed <- jsonlite::fromJSON(raw, simplifyVector = FALSE)
  if (is.list(parsed) && length(parsed) == 1 && !is.null(names(parsed))) {
    key <- names(parsed)[[1]]
    inner <- parsed[[1]]
    if (key == "Vector") {
      return(as.numeric(unlist(inner)))
    }
    if (key == "Matrix") {
      return(lapply(inner, function(row) as.numeric(unlist(row))))
    }
  }
  stop(linal_error(paste0("Unrecognized fallback-encoded cell: ", raw)))
}

#' Read `data.parquet` directly off disk and return it as a `data.frame`,
#' transparently unwrapping any column that landed in the legacy
#' JSON-string fallback encoding back into a real list-column -- the
#' caller never sees the raw tagged-JSON text. Mirrors
#' `clients/r/R/dataset.R`'s `linal_dataset_read()`, minus the HTTP fetch.
#'
#' @rdname linal_embedded_dataset
#' @export
linal_embedded_dataset_read <- function(ds) {
  path <- file.path(.linal_embedded_dataset_dir(ds), "data.parquet")
  if (!file.exists(path)) {
    stop(linal_error(sprintf("No such file: %s (has '%s' been SAVE'd?)", path, ds$name)))
  }
  table <- arrow::read_parquet(path, as_data_frame = FALSE)
  df <- as.data.frame(table)

  vector_or_matrix <- .vector_or_matrix_columns(linal_embedded_dataset_schema(ds))
  for (col_name in vector_or_matrix) {
    if (!(col_name %in% names(df))) {
      next
    }
    field_idx <- match(col_name, table$schema$names) - 1L
    field_type <- table$schema$field(field_idx)$type
    if (inherits(field_type, "Utf8") || inherits(field_type, "LargeUtf8")) {
      df[[col_name]] <- lapply(df[[col_name]], .unwrap_json_fallback_cell)
    }
  }

  df
}

#' `arrow::arrow_table(linal_embedded_dataset_read(ds))` -- for callers
#' who want to keep working in Arrow rather than convert to base R types.
#'
#' @rdname linal_embedded_dataset
#' @export
linal_embedded_dataset_to_arrow <- function(ds) {
  arrow::arrow_table(linal_embedded_dataset_read(ds))
}
