test_that("unwrap_value handles scalars", {
  expect_equal(unwrap_value(list(Float = 1.5)), 1.5)
  expect_equal(unwrap_value(list(Int = 5L)), 5L)
  expect_equal(unwrap_value(list(String = "x")), "x")
  expect_equal(unwrap_value(list(Bool = TRUE)), TRUE)
})

test_that("unwrap_value: Null is the bare string 'Null', not an object", {
  # Confirmed against a live v0.1.72 server (see clients/CONTRACT.md
  # Sec.3) -- Value::Null is a unit variant, serializes as "Null", not
  # {"Null": ...}. Exactly the case an assumption-only wire contract
  # would get wrong.
  expect_true(is.na(unwrap_value("Null")))
})

test_that("unwrap_value handles Vector and Matrix", {
  expect_equal(unwrap_value(list(Vector = list(1.0, 2.0, 3.0))), c(1.0, 2.0, 3.0))
  result <- unwrap_value(list(Matrix = list(list(1.0, 0.0), list(0.0, 1.0))))
  expect_equal(result, list(c(1.0, 0.0), c(0.0, 1.0)))
})

test_that("unwrap_value rejects an unknown shape", {
  expect_error(unwrap_value(list(Unknown = 1)), class = "linal_error")
  expect_error(unwrap_value(42), class = "linal_error")
})

test_that("unwrap_result handles NULL and Message", {
  expect_null(unwrap_result(NULL))
  expect_equal(
    unwrap_result(list(Message = "Switched to database 'default'")),
    "Switched to database 'default'"
  )
})

# Real payload captured from a live v0.1.72 server (SELECT * FROM probe
# where probe is (id: Int, emb: Vector(3)?) with rows (1, [1,2,3]) and
# (2, NULL)) -- see clients/CONTRACT.md's verified Table shape, and the
# identical fixture used by the Python client's test_wire.py.
real_table_payload <- list(
  id = 0,
  schema = list(
    fields = list(
      list(name = "id", value_type = "Int", nullable = FALSE, is_lazy = FALSE),
      list(name = "emb", value_type = list(Vector = 3), nullable = TRUE, is_lazy = FALSE)
    ),
    field_indices = list(id = 0, emb = 1)
  ),
  rows = list(
    list(schema = list(), values = list(list(Int = 1L), list(Vector = list(1.0, 2.0, 3.0)))),
    list(schema = list(), values = list(list(Int = 2L), "Null"))
  ),
  metadata = list(name = "Query Result", row_count = 2)
)

test_that("unwrap_result Table matches the verified wire shape", {
  result <- unwrap_result(list(Table = real_table_payload))
  expect_s3_class(result, "linal_table_result")
  expect_equal(names(result$columns), c("id", "emb"))
  expect_equal(result$columns$id, list(1L, 2L))
  expect_equal(result$columns$emb[[1]], c(1.0, 2.0, 3.0))
  expect_true(is.na(result$columns$emb[[2]]))
})

test_that("unwrap_result TensorTable is deliberately not implemented", {
  expect_error(
    unwrap_result(list(TensorTable = list(list(), list()))),
    class = "linal_error"
  )
})

test_that("unwrap_result handles Tensor", {
  payload <- list(
    Tensor = list(
      id = "t1",
      shape = list(dims = list(2, 3)),
      data = list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
      strides = list(3, 1),
      offset = 0
    )
  )
  result <- unwrap_result(payload)
  expect_s3_class(result, "linal_tensor_result")
  expect_equal(result$shape, c(2L, 3L))
  expect_equal(result$data, c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
})

test_that("unwrap_result rejects an unknown variant", {
  expect_error(unwrap_result(list(NotARealVariant = list())), class = "linal_error")
})

test_that("columns_to_dataframe builds a scalar column as atomic, keeps Vector as list-column", {
  df <- columns_to_dataframe(list(
    id = list(1L, 2L),
    emb = list(c(1.0, 2.0, 3.0), NA)
  ))
  expect_equal(df$id, c(1L, 2L))
  expect_true(is.list(df$emb))
  expect_equal(df$emb[[1]], c(1.0, 2.0, 3.0))
  expect_true(is.na(df$emb[[2]]))
})
