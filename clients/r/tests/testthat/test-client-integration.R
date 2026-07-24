# Integration tests against a real `linal serve` subprocess (see
# helper-server.R's `test_server_url()`) -- these are the tests that
# would actually catch a wire-shape drift between this client and the
# real engine, which fixture-based unit tests (test-wire.R) can't.
# Mirrors clients/python/tests/test_client_integration.py.

test_that("linal_execute returns a Message result", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  result <- linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, score: Float)", name))
  expect_type(result, "character")
  expect_true(grepl(name, result, fixed = TRUE))
})

test_that("linal_execute returns a Table result with Vector and NULL", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, emb: Vector(3)?)", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (1, [1.0, 2.0, 3.0])", name))
  linal_execute(conn, sprintf("INSERT INTO %s VALUES (2, null)", name))

  result <- linal_execute(conn, sprintf("SELECT * FROM %s ORDER BY id", name))

  expect_s3_class(result, "linal_table_result")
  expect_equal(names(result$columns), c("id", "emb"))
  expect_equal(result$columns$id, list(1L, 2L))
  expect_equal(result$columns$emb[[1]], c(1.0, 2.0, 3.0))
  expect_true(is.na(result$columns$emb[[2]]))
})

test_that("linal_execute raises a linal_error with the real server message", {
  conn <- linal_connect(test_server_url())
  expect_error(
    linal_execute(conn, "SELECT * FROM this_dataset_does_not_exist"),
    regexp = "not found",
    class = "linal_error"
  )
})

test_that("linal_execute returns NULL-shaped Message for USE", {
  conn <- linal_connect(test_server_url())
  db_name <- unique_name("db")
  linal_execute(conn, sprintf("CREATE DATABASE %s", db_name))
  # A headerless USE now genuinely persists across requests on the shared
  # server (see engine v0.1.74's fix) -- every other test in this session
  # sends headerless requests assuming "default", so restore it afterward
  # rather than leak state across tests.
  withr::defer(linal_execute(conn, "USE default"))
  result <- linal_execute(conn, sprintf("USE %s", db_name))
  expect_equal(result, sprintf("Switched to database '%s'", db_name))
})

test_that("linal_query returns a data.frame", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  linal_execute(conn, sprintf("DATASET %s COLUMNS (id: Int, name: String)", name))
  linal_execute(conn, sprintf('INSERT INTO %s VALUES (1, "alice")', name))
  linal_execute(conn, sprintf('INSERT INTO %s VALUES (2, "bob")', name))

  df <- linal_query(conn, sprintf("SELECT * FROM %s ORDER BY id", name))

  expect_s3_class(df, "data.frame")
  expect_equal(names(df), c("id", "name"))
  expect_equal(df$name, c("alice", "bob"))
})

test_that("linal_query rejects a non-table result", {
  conn <- linal_connect(test_server_url())
  name <- unique_name()
  expect_error(
    linal_query(conn, sprintf("DATASET %s COLUMNS (id: Int)", name)),
    regexp = "table-shaped",
    class = "linal_error"
  )
})

test_that("X-Linal-Database header actually isolates two databases", {
  base_url <- test_server_url()
  conn <- linal_connect(base_url)
  db_name <- unique_name("db")
  name <- unique_name()
  linal_execute(conn, sprintf("CREATE DATABASE %s", db_name))

  header_conn <- linal_connect(base_url, database = db_name)
  linal_execute(header_conn, sprintf("DATASET %s COLUMNS (id: Int)", name))
  linal_execute(header_conn, sprintf("INSERT INTO %s VALUES (1)", name))

  # Not visible from the default database's connection.
  expect_error(
    linal_execute(conn, sprintf("SELECT * FROM %s", name)),
    class = "linal_error"
  )

  # Visible via the header-scoped connection.
  result <- linal_execute(header_conn, sprintf("SELECT * FROM %s", name))
  expect_equal(result$columns$id, list(1L))
})
