# Shared test server helper. Per PYTHON_R_INTEROP_PLAN.md's testing
# strategy (design decision 7): integration tests launch a real `linal
# serve` subprocess rather than mocking the HTTP layer, mirroring the
# Python client's tests/conftest.py -- fixture JSON alone can't catch a
# wire-shape drift between this client and the real engine.

#' Walk up from `start` looking for the linal-db-rs repo root (marked by
#' its `Cargo.toml`). A fixed relative-path climb (`../../../..` from the
#' test file) works when running via `pkgload::load_all()` +
#' `test_dir()`/`test_check()` from the real source tree, but breaks under
#' `R CMD check`, which copies the package into its own sandboxed
#' `<pkg>.Rcheck/` directory with different nesting -- found by actually
#' running `R CMD check` (not just `pkgload::load_all()`), which every
#' prior fixed-relative-path version of this function had never been
#' exercised against.
find_repo_root <- function(start) {
  dir <- normalizePath(start, mustWork = FALSE)
  for (i in seq_len(10)) {
    if (file.exists(file.path(dir, "Cargo.toml"))) {
      return(dir)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) {
      break
    }
    dir <- parent
  }
  NULL
}

#' Find the built `linal` binary, or skip the calling test. A `skip()`
#' (not a hard failure) is deliberate: this package's integration tests
#' depend on a sibling Rust checkout with a built binary, which will
#' never be true in a generic `R CMD check`/CRAN sandbox or a fresh clone
#' before `cargo build` has run -- that's an environment gap, not a
#' package defect, and should skip quietly rather than fail the suite.
find_linal_binary <- function() {
  root <- find_repo_root(testthat::test_path())
  if (is.null(root)) {
    testthat::skip("Could not locate the linal-db-rs repo root (no Cargo.toml found walking up from the test directory) -- skipping tests that need a real linal server.")
  }
  for (profile in c("debug", "release")) {
    candidate <- file.path(root, "target", profile, "linal")
    if (file.exists(candidate)) {
      return(candidate)
    }
  }
  testthat::skip(sprintf(
    "No `linal` binary found under %s/target/{debug,release}/ -- run `cargo build --bin linal` in the repo root first.",
    root
  ))
}

#' Find a free TCP port. R's `socketConnection(port = 0, server = TRUE)`
#' (the direct analogue of Python's bind-to-0-then-read-back-the-port
#' trick) blocks in `open()` waiting for a client to connect -- it never
#' returns, since nothing ever connects to it (confirmed by hand: it hung
#' indefinitely). `serverSocket()` doesn't block, but also doesn't expose
#' the OS-assigned port when given `port = 0` (checked `attributes()` on
#' the returned connection -- nothing there). So instead: pick a random
#' high port ourselves and confirm it's free by successfully binding
#' `serverSocket()` to it, retrying on failure. Small unavoidable TOCTOU
#' race between this check and `linal serve` actually binding it, same
#' tradeoff any test port allocator without OS cooperation has.
free_port <- function() {
  for (i in seq_len(20)) {
    port <- sample(20000:60000, 1)
    con <- tryCatch(serverSocket(port = port), error = function(e) NULL)
    if (!is.null(con)) {
      close(con)
      return(port)
    }
  }
  stop("Could not find a free port after 20 attempts")
}

#' Start a real `linal serve` subprocess for the duration of one test,
#' in a fresh temp directory (hermetic -- avoids the disk auto-recovery
#' picking up state from a previous run, the same real issue hit while
#' building the Python client's equivalent fixture).
#'
#' @return list(url = ..., process = <processx::process>)
start_linal_server <- function() {
  binary <- find_linal_binary()
  port <- free_port()
  url <- sprintf("http://127.0.0.1:%d", port)
  server_cwd <- tempfile("linal_test_server_")
  dir.create(server_cwd)

  proc <- processx::process$new(
    binary,
    c("serve", "--port", as.character(port)),
    wd = server_cwd,
    stdout = "|",
    stderr = "|"
  )

  deadline <- Sys.time() + 10
  healthy <- FALSE
  while (Sys.time() < deadline) {
    if (!proc$is_alive()) {
      stop(sprintf(
        "linal serve exited early (status %s): %s",
        proc$get_exit_status(), paste(proc$read_all_error_lines(), collapse = "\n")
      ))
    }
    resp <- tryCatch(
      httr2::req_perform(httr2::req_timeout(httr2::request(paste0(url, "/health")), 1)),
      error = function(e) NULL
    )
    if (!is.null(resp) && httr2::resp_status(resp) == 200) {
      healthy <- TRUE
      break
    }
    Sys.sleep(0.1)
  }
  if (!healthy) {
    proc$kill()
    stop(sprintf("linal serve never became healthy on %s", url))
  }

  list(url = url, process = proc)
}

stop_linal_server <- function(server) {
  server$process$kill()
  server$process$wait(2000)
}

# One server for the whole test run (matching the Python client's
# session-scoped pytest fixture) -- memoized here since testthat has no
# built-in session-scoped fixture; torn down via `teardown_env()`, which
# runs once after every test file in the run has completed.
.linal_test_server_env <- new.env(parent = emptyenv())

#' Get (starting if needed) the shared test server's base URL.
test_server_url <- function() {
  if (is.null(.linal_test_server_env$server)) {
    server <- start_linal_server()
    .linal_test_server_env$server <- server
    withr::defer(stop_linal_server(server), envir = testthat::teardown_env())
  }
  .linal_test_server_env$server$url
}

#' Sanitize a testthat test description into a valid DSL identifier, for
#' tests that need a unique dataset name (all tests share one database --
#' `USE <db>` requires a pre-existing database, engine/db.rs's
#' use_database doesn't auto-create -- so per-test isolation is via
#' dataset name, not a fresh database per test).
unique_name <- function(prefix = "t") {
  paste0(prefix, "_", gsub("[^0-9a-zA-Z_]", "_", tempfile("", "")))
}
