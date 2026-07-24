# Integration tests for linal_dataset() (/delivery Parquet export)
# against a real `linal serve` subprocess -- deliberately covers BOTH
# Parquet encodings from clients/CONTRACT.md Sec.2: the native
# FixedSizeList path (no NULLs) and the legacy JSON-string fallback path
# (a real NULL present), since a client that only handles one would
# silently mis-read the other. Mirrors
# clients/python/tests/test_dataset_integration.py.

test_that("linal_dataset_read: native Vector column, no NULLs", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(3))", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [1.0, 2.0, 3.0])", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (2, [4.0, 5.0, 6.0])", name))
  linal_execute(conn, sprintf("SAVE DATASET %s", name))

  df <- linal_dataset_read(linal_dataset(conn, name))

  expect_equal(names(df), c("id", "emb"))
  expect_equal(df$emb[[1]], c(1.0, 2.0, 3.0))
  expect_equal(df$emb[[2]], c(4.0, 5.0, 6.0))
})

test_that("linal_dataset_read: legacy fallback Vector column with a real NULL", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(3)?)", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [1.0, 2.0, 3.0])", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (2, null)", name))
  linal_execute(conn, sprintf("SAVE DATASET %s", name))

  df <- linal_dataset_read(linal_dataset(conn, name))

  # The server fell back to JSON-string encoding for this column (a real
  # NULL is present -- see CHANGELOG v0.1.72) -- the client must
  # transparently unwrap it, never leak the raw `{"Vector": [...]}` text.
  expect_equal(names(df), c("id", "emb"))
  expect_equal(df$emb[[1]], c(1.0, 2.0, 3.0))
  expect_true(is.na(df$emb[[2]]))
})

test_that("linal_dataset_read: native Matrix column", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, m: Matrix(2, 2))", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [[1.0, 2.0], [3.0, 4.0]])", name))
  linal_execute(conn, sprintf("SAVE DATASET %s", name))

  df <- linal_dataset_read(linal_dataset(conn, name))

  # arrow's as.data.frame() represents a native FixedSizeList<FixedSizeList>
  # column as its own `arrow_fixed_size_list`/vctrs_list_of S3 class (not a
  # bare R list) -- correct and expected, just needs `ignore_attr` here to
  # compare the actual numeric content rather than the wrapper's class/
  # list_size/ptype attributes.
  expect_equal(df$m[[1]], list(c(1.0, 2.0), c(3.0, 4.0)), ignore_attr = TRUE)
})

test_that("linal_dataset_to_arrow returns a real arrow Table", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(2))", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [0.5, 0.5])", name))
  linal_execute(conn, sprintf("SAVE DATASET %s", name))

  tbl <- linal_dataset_to_arrow(linal_dataset(conn, name))

  expect_s3_class(tbl, "Table")
  expect_equal(tbl$num_rows, 1)
  expect_equal(as.data.frame(tbl)$emb[[1]], c(0.5, 0.5))
})

test_that("linal_dataset_schema/manifest/stats", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(3))", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [1.0, 2.0, 3.0])", name))
  linal_execute(conn, sprintf("SAVE DATASET %s", name))

  ds <- linal_dataset(conn, name)
  schema <- linal_dataset_schema(ds)
  manifest <- linal_dataset_manifest(ds)
  stats <- linal_dataset_stats(ds)

  emb_col <- Filter(function(c) c$name == "emb", schema$columns)[[1]]
  expect_equal(emb_col$value_type, list(Vector = 3))
  expect_equal(manifest$formats$parquet, "data.parquet")
  expect_equal(stats$row_count, 1)
})

test_that("dataset export honors a non-default database", {
  # Regression test: linal_dataset()'s HTTP calls didn't send
  # X-Linal-Database at all, so a connection configured for a non-default
  # database silently fell back to the server's default database for
  # /delivery -- found while building a real example against a
  # non-default database (checkpoint 5), not by checkpoint 4's tests,
  # which only ever used the default database. A same-named dataset in
  # each database, with different data, proves the header is actually
  # honored (not just "it didn't 404").
  base_url <- test_server_url()
  default_conn <- linal_connect(base_url)
  name <- unique_name()
  db_name <- paste0("db_", name)
  linal_execute(default_conn, sprintf("CREATE DATABASE %s", db_name))

  linal_execute(default_conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(2))", name))
  linal_execute(default_conn, sprintf("INSERT INTO %s VALUES (1, [1.0, 1.0])", name))
  linal_execute(default_conn, sprintf("SAVE DATASET %s", name))

  scoped_conn <- linal_connect(base_url, database = db_name)
  linal_execute(scoped_conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(2))", name))
  linal_execute(scoped_conn, sprintf("INSERT INTO %s VALUES (1, [9.0, 9.0])", name))
  linal_execute(scoped_conn, sprintf("SAVE DATASET %s", name))

  default_df <- linal_dataset_read(linal_dataset(default_conn, name))
  scoped_df <- linal_dataset_read(linal_dataset(scoped_conn, name))

  expect_equal(default_df$emb[[1]], c(1.0, 1.0))
  expect_equal(scoped_df$emb[[1]], c(9.0, 9.0))
})
