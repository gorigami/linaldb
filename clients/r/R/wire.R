#' Unwrap one tagged `Value` cell (contract Sec.3) into a plain R value.
#'
#' @param value A parsed JSON value (as a nested list, via
#'   `httr2::resp_body_json(resp, simplifyVector = FALSE)`).
#' @return `NA`, a scalar, a numeric vector (Vector), or a list of numeric
#'   vectors (Matrix, one per row).
#' @keywords internal
#' @noRd
unwrap_value <- function(value) {
  if (identical(value, "Null")) {
    return(NA)
  }
  if (is.list(value) && length(value) == 1 && !is.null(names(value))) {
    key <- names(value)[[1]]
    inner <- value[[1]]
    if (key == "Float") {
      return(as.numeric(inner))
    }
    if (key == "Int") {
      return(as.integer(inner))
    }
    if (key == "String") {
      return(as.character(inner))
    }
    if (key == "Bool") {
      return(as.logical(inner))
    }
    if (key == "Vector") {
      return(as.numeric(unlist(inner)))
    }
    if (key == "Matrix") {
      return(lapply(inner, function(row) as.numeric(unlist(row))))
    }
  }
  stop(linal_error(paste0(
    "Unrecognized Value wire shape: ",
    jsonlite::toJSON(value, auto_unbox = TRUE)
  )))
}

#' Whether an unwrapped `Value` is scalar (safe to put in an atomic vector
#' column) vs. a Vector/Matrix (must stay a list-column). Note: a
#' single-element Vector column (`Vector(1)`) is indistinguishable here from
#' a Float column and gets flattened too -- the numeric content is
#' unaffected, only the "this was semantically a Vector" tag is lost for
#' that degenerate case. A column mixing 1-element and longer vectors still
#' correctly stays a list-column, since not every row would be scalar then.
#' @keywords internal
#' @noRd
is_scalar_value <- function(v) {
  length(v) == 1 && !is.list(v)
}

#' Convert a real `Table` payload (contract Sec.1's verified shape --
#' `payload$schema$fields` for column order/names, `payload$rows[[i]]$values`
#' for each row's cells, NOT the row object itself) into a named list of
#' per-row-unwrapped-value lists, one per column.
#' @keywords internal
#' @noRd
table_result_columns <- function(payload) {
  fields <- payload$schema$fields
  col_names <- vapply(fields, function(f) f$name, character(1))
  rows <- payload$rows
  ncols <- length(col_names)

  columns <- vector("list", ncols)
  names(columns) <- col_names
  for (j in seq_len(ncols)) {
    columns[[j]] <- lapply(rows, function(r) unwrap_value(r$values[[j]]))
  }
  columns
}

#' Convert `table_result_columns()`'s output into a `data.frame`. A column
#' where every value is scalar becomes a normal atomic vector column; a
#' column containing any Vector/Matrix value (or a mix with `NA`) stays a
#' list-column, R's standard way to hold non-atomic per-row values in a
#' data.frame.
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

#' Unwrap one `DslOutput` JSON value (contract Sec.1) into `NULL` (no
#' output), a `character` scalar (`Message`), a `linal_table_result`, or a
#' `linal_tensor_result`.
#' @keywords internal
#' @noRd
unwrap_result <- function(result) {
  if (is.null(result)) {
    return(NULL)
  }
  if (!is.list(result) || length(result) != 1 || is.null(names(result))) {
    stop(linal_error("Unrecognized DslOutput wire shape"))
  }
  kind <- names(result)[[1]]
  payload <- result[[1]]

  if (kind == "Message") {
    return(payload)
  }
  if (kind == "Table") {
    return(structure(
      list(columns = table_result_columns(payload)),
      class = "linal_table_result"
    ))
  }
  if (kind == "TensorTable") {
    # Not yet exercised against a real response -- mirrors the Python
    # client's deliberate scope boundary (see PYTHON_R_INTEROP_PLAN.md
    # checkpoint 1 findings). Fail loudly rather than guess the shape.
    stop(linal_error(paste(
      "TensorTable unwrapping is not yet implemented (no verified wire",
      "example) -- see PYTHON_R_INTEROP_PLAN.md checkpoint 3 findings"
    )))
  }
  if (kind %in% c("Tensor", "LazyTensor")) {
    return(structure(
      list(
        shape = as.integer(unlist(payload$shape$dims)),
        data = as.numeric(unlist(payload$data))
      ),
      class = "linal_tensor_result"
    ))
  }
  stop(linal_error(paste0("Unknown DslOutput variant: ", kind)))
}
