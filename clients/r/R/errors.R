#' Construct a `linal_error` condition
#'
#' Raised for a server-reported error (`status: "error"`) or a response
#' whose shape doesn't match `clients/CONTRACT.md`. Network-level failures
#' propagate as whatever `httr2` itself raises -- not wrapped here, per the
#' contract's "never swallowed" rule.
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
