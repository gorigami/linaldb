#!/usr/bin/env Rscript
# Real end-to-end example (PYTHON_R_INTEROP_PLAN.md checkpoint 5): starts
# a real `linal serve`, replays the real UCI handwritten-digits
# classification workflow from
# ../../../examples/hdf5_digit_classification.lnl through this client's
# `/execute`, exports the resulting datasets through `/delivery`, and
# independently recomputes the classification in plain base R from the
# exported raw vectors -- confirming the numbers `/execute`'s SQL engine
# reports match the numbers you get from the raw data `/delivery` serves,
# not just "did it run". Mirrors
# clients/python/examples/digit_classification.py exactly, including
# reusing the same real DSL file rather than duplicating literal data.
#
# Usage: Rscript digit_classification.R

suppressPackageStartupMessages({
  library(processx)
  library(httr2)
})

this_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(this_dir) == 0 || this_dir == "") this_dir <- getwd()
pkgload_ok <- requireNamespace("pkgload", quietly = TRUE)
if (pkgload_ok) {
  pkgload::load_all(file.path(this_dir, ".."), quiet = TRUE)
} else {
  library(linaldb)
}

repo_root <- normalizePath(file.path(this_dir, "..", "..", ".."))
lnl_script <- file.path(repo_root, "examples", "hdf5_digit_classification.lnl")
database <- "hdf5_digit_classification"

find_linal_binary <- function() {
  for (profile in c("release", "debug")) {
    candidate <- file.path(repo_root, "target", profile, "linal")
    if (file.exists(candidate)) return(candidate)
  }
  stop(sprintf(
    "No `linal` binary found under %s/target/{debug,release}/. Run `cargo build --bin linal` in the repo root first.",
    repo_root
  ))
}

start_server <- function(port) {
  binary <- find_linal_binary()
  # cwd = repo root so the .lnl script's `examples/data/...` HDF5 path and
  # the DROP/CREATE DATABASE statements resolve exactly like running
  # `linal run` directly from the repo root would.
  proc <- processx::process$new(
    binary, c("serve", "--port", as.character(port)),
    wd = repo_root, stdout = "|", stderr = "|"
  )
  deadline <- Sys.time() + 10
  repeat {
    if (Sys.time() > deadline) {
      proc$kill()
      stop("linal serve never became healthy")
    }
    if (!proc$is_alive()) {
      stop(sprintf("linal serve exited early: %s", paste(proc$read_all_error_lines(), collapse = "\n")))
    }
    resp <- tryCatch(
      httr2::req_perform(httr2::req_timeout(httr2::request(sprintf("http://127.0.0.1:%d/health", port)), 1)),
      error = function(e) NULL
    )
    if (!is.null(resp) && httr2::resp_status(resp) == 200) break
    Sys.sleep(0.1)
  }
  proc
}

# Execute each real DSL statement in a `.lnl` file through the client.
# Mirrors `linal run`'s own multi-line joiner (src/main.rs) and the
# Python example's replay_lnl_file: accumulate lines, track paren
# balance, execute once balance returns to zero -- most statements in
# this file are one physical line, but the `DATASET ... COLUMNS (...)`
# blocks genuinely span several.
replay_lnl_file <- function(conn, path) {
  lines <- readLines(path, warn = FALSE)
  current <- ""
  balance <- 0
  start_lineno <- NULL
  for (i in seq_along(lines)) {
    line <- trimws(lines[i])
    if (current == "") {
      if (line == "" || startsWith(line, "--")) next
      start_lineno <- i
    }
    current <- if (current == "") line else paste(current, line)
    balance <- balance + lengths(regmatches(line, gregexpr("\\(", line))) -
      lengths(regmatches(line, gregexpr("\\)", line)))
    if (balance == 0) {
      preview <- if (nchar(current) > 80) paste0(substr(current, 1, 80), "...") else current
      cat(sprintf("  [%d] %s\n", start_lineno, preview))
      linal_execute(conn, current)
      current <- ""
    }
  }
}

cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))
}

main <- function() {
  port <- 18410
  cat(sprintf("Starting linal serve on port %d (cwd=%s)...\n", port, repo_root))
  proc <- start_server(port)
  on.exit({
    proc$kill()
    proc$wait(2000)
  })

  url <- sprintf("http://127.0.0.1:%d", port)
  replay_conn <- linal_connect(url)

  cat(sprintf("\nReplaying real DSL from %s:\n", lnl_script))
  replay_lnl_file(replay_conn, lnl_script)

  # Independent of the replay connection's active-database state (the
  # .lnl file's own last line does `USE default`) -- scoped explicitly
  # via the database= parameter, decoupled from server-side USE state.
  export_conn <- linal_connect(url, database = database)

  cat(sprintf("\nQuerying the real classification result via /execute (database='%s')...\n", database))
  classify_sql <- paste0(
    "WITH classified AS (",
    "SELECT query_digits.digit_id AS digit_id, query_digits.true_label AS true_label, ",
    "reference_centroids.digit_class AS predicted_label, ",
    "COSINE_SIM(query_digits.pixels, reference_centroids.centroid) AS similarity, ",
    "ROW_NUMBER() OVER (PARTITION BY digit_id ORDER BY similarity DESC) AS rn ",
    "FROM query_digits JOIN reference_centroids ",
    "ON COSINE_SIM(query_digits.pixels, reference_centroids.centroid) > 0.5",
    ") SELECT digit_id, true_label, predicted_label, similarity ",
    "FROM classified WHERE rn = 1 ORDER BY digit_id"
  )
  sql_result <- linal_execute(export_conn, classify_sql)
  digit_ids <- unlist(sql_result$columns$digit_id)
  cat(sprintf("  /execute returned %d classified rows\n", length(digit_ids)))

  cat("\nExporting query_digits and reference_centroids via /delivery...\n")
  query_df <- linal_dataset_read(linal_dataset(export_conn, "query_digits"))
  centroids_df <- linal_dataset_read(linal_dataset(export_conn, "reference_centroids"))
  cat(sprintf("  query_digits: %d rows, reference_centroids: %d rows\n", nrow(query_df), nrow(centroids_df)))

  cat("\nIndependently recomputing classification in base R from the exported raw vectors...\n")
  centroid_by_label <- setNames(centroids_df$centroid, centroids_df$digit_class)

  mismatches <- character(0)
  correct <- 0
  total <- length(digit_ids)
  for (i in seq_along(digit_ids)) {
    digit_id <- digit_ids[i]
    query_row <- query_df[query_df$digit_id == digit_id, ]
    query_vec <- query_row$pixels[[1]]

    sims <- vapply(centroid_by_label, function(c) cosine_similarity(query_vec, c), numeric(1))
    best_label <- as.integer(names(sims)[which.max(sims)])
    best_sim <- max(sims)

    true_label <- sql_result$columns$true_label[[i]]
    sql_predicted <- sql_result$columns$predicted_label[[i]]
    sql_similarity <- sql_result$columns$similarity[[i]]

    if (abs(best_sim - sql_similarity) > 1e-4) {
      mismatches <- c(mismatches, sprintf(
        "%s: R similarity %.6f vs /execute's %.6f", digit_id, best_sim, sql_similarity
      ))
    }
    if (best_label != sql_predicted) {
      mismatches <- c(mismatches, sprintf(
        "%s: R predicted %d vs /execute's %d", digit_id, best_label, sql_predicted
      ))
    }
    if (best_label == true_label) correct <- correct + 1
  }

  cat(sprintf("\nIndependently-recomputed accuracy: %d/%d (%.1f%%)\n", correct, total, 100 * correct / total))
  sql_correct <- sum(unlist(sql_result$columns$true_label) == unlist(sql_result$columns$predicted_label))
  cat(sprintf("/execute-reported accuracy:        %d/%d (%.1f%%)\n", sql_correct, total, 100 * sql_correct / total))

  if (length(mismatches) > 0) {
    cat(sprintf("\nFAIL: %d mismatch(es) between /execute and /delivery-derived numbers:\n", length(mismatches)))
    for (m in mismatches) cat(sprintf("  - %s\n", m))
    quit(status = 1)
  } else if (correct != sql_correct) {
    cat("\nFAIL: aggregate accuracy differs between the two independently-computed paths.\n")
    quit(status = 1)
  } else {
    cat(paste(
      "\nPASS: every per-row similarity, every predicted label, and the aggregate",
      "accuracy computed from /delivery's raw exported vectors exactly match what",
      "/execute's SQL engine reported.\n"
    ))
  }
}

main()
