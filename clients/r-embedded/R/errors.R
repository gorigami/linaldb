#' Construct a `linal_error` condition
#'
#' Raised for an engine-reported error (a `DslError`) or a result whose
#' shape doesn't match `../EMBEDDED_CONTRACT.md`. Mirrors
#' `clients/r/R/errors.R`'s `linal_error()` -- same condition class, same
#' purpose, just not literally shared code across the two packages.
#'
#' @param message Error message text.
#' @keywords internal
#' @noRd
linal_error <- function(message) {
  structure(
    class = c("linal_error", "error", "condition"),
    list(message = message, call = sys.call(-1))
  )
}
