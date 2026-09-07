#' Whether an unwrapped `Value` cell is scalar (safe to put in an atomic
#' vector column) vs. a Vector/Matrix (must stay a list-column). Mirrors
#' `clients/r/R/wire.R`'s `is_scalar_value()` -- same degenerate-case note
#' applies: a single-element Vector column is indistinguishable from a
#' plain numeric column here, and that's fine, only the "this was
#' semantically a Vector" tag is lost for that one case.
#' @keywords internal
#' @noRd
is_scalar_value <- function(v) {
  length(v) == 1 && !is.list(v)
}

#' Convert one `Db$execute()` `list(columns = ...)` payload (each column
#' already a list of per-row `Robj` cells, per
#' `../EMBEDDED_CONTRACT.md`) into a `data.frame`. A column where every
#' value is scalar becomes a normal atomic vector column; a column
#' containing any Vector/Matrix value (or a mix with `NA`) stays a
#' list-column. Mirrors `clients/r/R/wire.R`'s `columns_to_dataframe()`.
#' @keywords internal
#' @noRd
columns_to_dataframe <- function(columns) {
  if (length(columns) == 0) {
    return(data.frame())
  }
  nrows <- length(columns[[1]])
  df <- data.frame(row.names = seq_len(nrows))
  for (name in names(columns)) {
    col <- columns[[name]]
    if (all(vapply(col, is_scalar_value, logical(1)))) {
      df[[name]] <- unlist(lapply(col, function(v) if (length(v) == 1 && is.na(v)) NA else v))
    } else {
      df[[name]] <- col
    }
  }
  df
}

#' Unwrap one `Db$execute()` return value (per `../EMBEDDED_CONTRACT.md`)
#' into `NULL` (no output), a `character` scalar (`Message`), a
#' `linal_table_result`, or a `linal_tensor_result`. Mirrors
#' `clients/r/R/wire.R`'s `unwrap_result()` -- no JSON parsing needed
#' here, the Rust side already returns native R values, but the tagged
#' shape and downstream S3 classes are the same.
#' @keywords internal
#' @noRd
unwrap_result <- function(result) {
  if (is.null(result)) {
    return(NULL)
  }
  if (!is.list(result) || length(result) != 1 || is.null(names(result))) {
    stop(linal_error("Unrecognized embedded result shape"))
  }
  kind <- names(result)[[1]]
  payload <- result[[1]]

  if (kind == "Message") {
    return(payload)
  }
  if (kind == "Table") {
    return(structure(
      list(columns = payload$columns),
      class = "linal_table_result"
    ))
  }
  if (kind == "Tensor") {
    return(structure(
      list(shape = as.integer(payload$shape), data = as.numeric(payload$data)),
      class = "linal_tensor_result"
    ))
  }
  stop(linal_error(paste0("Unknown embedded result variant: ", kind)))
}
