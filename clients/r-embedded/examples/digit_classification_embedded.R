#!/usr/bin/env Rscript
# Real end-to-end example, embedded/native counterpart of
# clients/r/examples/digit_classification.R: no `linal serve`, no HTTP --
# opens the engine in-process via linal_embedded_db(), replays the real
# UCI handwritten-digits classification workflow from
# ../../../examples/hdf5_digit_classification.lnl directly against it,
# reads the resulting datasets straight off disk (linal_embedded_dataset*
# -- the same package layout `SAVE DATASET` always writes, no export step
# needed), and independently recomputes the classification in base R from
# the raw exported vectors -- confirming the numbers the SQL engine
# reports match the numbers in the raw data it persisted, not just "did
# it run".
#
# Usage: Rscript digit_classification_embedded.R

this_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(this_dir) == 0 || this_dir == "") this_dir <- getwd()
library(linaldb.embedded)

repo_root <- normalizePath(file.path(this_dir, "..", "..", ".."))
lnl_script <- file.path(repo_root, "examples", "hdf5_digit_classification.lnl")
database <- "hdf5_digit_classification"

# Execute each real DSL statement in a `.lnl` file in-process. Mirrors
# `linal run`'s own multi-line joiner (src/main.rs) and the HTTP R
# example's replay_lnl_file: accumulate lines, track paren balance,
# execute once balance returns to zero.
replay_lnl_file <- function(db, path) {
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
      linal_embedded_execute(db, current)
      current <- ""
    }
  }
}

cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))
}

main <- function() {
  # cwd = repo root so the .lnl script's `examples/data/...` HDF5 path and
  # the DROP/CREATE DATABASE statements resolve exactly like `linal run`
  # from the repo root would, and so `data/` (SAVE DATASET's default
  # target) lands in the same place the HTTP example's server would use.
  old_wd <- getwd()
  setwd(repo_root)
  on.exit(setwd(old_wd), add = TRUE)

  db <- linal_embedded_db()

  cat(sprintf("\nReplaying real DSL from %s (in-process, no server):\n", lnl_script))
  replay_lnl_file(db, lnl_script)

  # The .lnl file's own last line does `USE default`, so re-select the
  # target database explicitly for what follows, exactly like the HTTP
  # example's `database=` parameter does for its export connection.
  linal_embedded_execute(db, sprintf("USE %s", database))

  cat(sprintf("\nQuerying the real classification result in-process (database='%s')...\n", database))
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
  sql_result <- linal_embedded_execute(db, classify_sql)
  digit_ids <- unlist(sql_result$columns$digit_id)
  cat(sprintf("  execute() returned %d classified rows\n", length(digit_ids)))

  # The .lnl script itself already ran `SAVE DATASET query_digits` /
  # `SAVE DATASET reference_centroids` (its own "Persistence" section) --
  # read the packages it wrote straight off disk, no export step needed.
  cat("\nReading query_digits and reference_centroids back off disk...\n")
  query_df <- linal_embedded_dataset_read(linal_embedded_dataset(db, "query_digits"))
  centroids_df <- linal_embedded_dataset_read(linal_embedded_dataset(db, "reference_centroids"))
  cat(sprintf("  query_digits: %d rows, reference_centroids: %d rows\n", nrow(query_df), nrow(centroids_df)))

  cat("\nIndependently recomputing classification in base R from the raw persisted vectors...\n")
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
        "%s: R similarity %.6f vs execute()'s %.6f", digit_id, best_sim, sql_similarity
      ))
    }
    if (best_label != sql_predicted) {
      mismatches <- c(mismatches, sprintf(
        "%s: R predicted %d vs execute()'s %d", digit_id, best_label, sql_predicted
      ))
    }
    if (best_label == true_label) correct <- correct + 1
  }

  cat(sprintf("\nIndependently-recomputed accuracy: %d/%d (%.1f%%)\n", correct, total, 100 * correct / total))
  sql_correct <- sum(unlist(sql_result$columns$true_label) == unlist(sql_result$columns$predicted_label))
  cat(sprintf("execute()-reported accuracy:        %d/%d (%.1f%%)\n", sql_correct, total, 100 * sql_correct / total))

  if (length(mismatches) > 0) {
    cat(sprintf("\nFAIL: %d mismatch(es) between execute() and the raw persisted vectors:\n", length(mismatches)))
    for (m in mismatches) cat(sprintf("  - %s\n", m))
    quit(status = 1)
  } else if (correct != sql_correct) {
    cat("\nFAIL: aggregate accuracy differs between the two independently-computed paths.\n")
    quit(status = 1)
  } else {
    cat(paste(
      "\nPASS: every per-row similarity, every predicted label, and the aggregate",
      "accuracy computed from the raw persisted vectors exactly match what the",
      "in-process SQL engine reported -- with no server and no HTTP round trip.\n"
    ))
  }
}

main()
