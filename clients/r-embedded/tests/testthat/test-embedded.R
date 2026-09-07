test_that("execute returns a Message string for USE / DDL-style statements", {
  db <- linal_embedded_db(tempfile())
  result <- linal_embedded_execute(db, "USE default")
  expect_type(result, "character")
  expect_match(result, "default")
})

test_that("execute returns a Message string for DDL-style statements", {
  db <- linal_embedded_db(tempfile())
  result <- linal_embedded_execute(db, "DATASET t COLUMNS (id: Int, v: Vector(3))")
  expect_type(result, "character")
  expect_match(result, "t")
})

test_that("query returns a data.frame with scalar and vector columns", {
  db <- linal_embedded_db(tempfile())
  linal_embedded_execute(db, "DATASET t COLUMNS (id: Int, v: Vector(3))")
  linal_embedded_execute(db, "INSERT INTO t VALUES (1, [1.0, 2.0, 3.0])")
  linal_embedded_execute(db, "INSERT INTO t VALUES (2, [4.0, 5.0, 6.0])")

  df <- linal_embedded_query(db, "SELECT id, v FROM t ORDER BY id")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 2)
  expect_equal(df$id, c(1L, 2L))
  expect_true(is.list(df$v))
  expect_equal(df$v[[1]], c(1, 2, 3))
  expect_equal(df$v[[2]], c(4, 5, 6))
})

test_that("query() rejects a non-table-shaped result with a linal_error", {
  db <- linal_embedded_db(tempfile())
  expect_error(
    linal_embedded_query(db, "USE default"),
    class = "linal_error"
  )
})

test_that("a DSL parse error surfaces as a linal_error with the engine's message", {
  db <- linal_embedded_db(tempfile())
  err <- tryCatch(
    linal_embedded_execute(db, "NOT VALID DSL"),
    error = function(e) e
  )
  expect_s3_class(err, "linal_error")
  expect_match(conditionMessage(err), "Parse error")
})

test_that("active_db() and data_dir() reflect the configured/default state", {
  dir <- tempfile()
  db <- linal_embedded_db(dir)
  expect_equal(linal_embedded_active_db(db), "default")
  expect_equal(linal_embedded_data_dir(db), dir)
})

test_that("dataset_read() round-trips a saved dataset from disk", {
  dir <- tempfile()
  on.exit(unlink(dir, recursive = TRUE))
  db <- linal_embedded_db(dir)

  linal_embedded_execute(db, "DATASET t COLUMNS (id: Int, v: Vector(3))")
  linal_embedded_execute(db, "INSERT INTO t VALUES (1, [1.0, 2.0, 3.0])")
  linal_embedded_execute(db, "INSERT INTO t VALUES (2, [4.0, 5.0, 6.0])")
  linal_embedded_execute(db, "SAVE DATASET t")

  ds <- linal_embedded_dataset(db, "t")
  expect_true(file.exists(file.path(dir, "default", "datasets", "t", "data.parquet")))

  schema <- linal_embedded_dataset_schema(ds)
  expect_true(is.list(schema))
  expect_true(is.list(linal_embedded_dataset_manifest(ds)))
  expect_true(is.list(linal_embedded_dataset_stats(ds)))

  df <- linal_embedded_dataset_read(ds)
  expect_equal(nrow(df), 2)
  expect_true(is.list(df$v))
  expect_equal(sort(unlist(lapply(df$v, sum))), c(6, 15))

  tbl <- linal_embedded_dataset_to_arrow(ds)
  expect_s3_class(tbl, "Table")
})

test_that("dataset accessors error clearly when the dataset was never saved", {
  db <- linal_embedded_db(tempfile())
  ds <- linal_embedded_dataset(db, "does_not_exist")
  expect_error(linal_embedded_dataset_read(ds), class = "linal_error")
  expect_error(linal_embedded_dataset_schema(ds), class = "linal_error")
})
