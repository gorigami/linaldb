#' linaldb.embedded: Embedded/Native R Binding for LINALDB
#'
#' Wraps the LINALDB engine's Rust `TensorDb` directly via `extendr` --
#' no HTTP server, no `linal serve` process. See `../EMBEDDED_CONTRACT.md`
#' for the Rust/R result-shape contract this package implements against,
#' and the `linaldb` package under `clients/r` for the HTTP-client
#' alternative (talks to a running `linal serve` instance instead).
#'
#' @keywords internal
"_PACKAGE"
