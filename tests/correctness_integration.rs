// tests/correctness_integration.rs
// Integration tests for v0.1.30 correctness fixes:
//   - Multiple computed columns in one SELECT
//   - Window function with no PARTITION BY
//   - UNION deduplication vs UNION ALL
//   - CAST to BOOL
//   - SUBSTR 2-arg (1-based)
//   - COALESCE with 3+ args
//   - RIGHT JOIN row correctness
//   - FULL OUTER JOIN row correctness
//   - CTE temp dataset cleanup (non-shadowing)
//   - TENSOR column dimension preservation

use linal::core::value::{Value, ValueType};
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

// ─── Helper ──────────────────────────────────────────────────────────────────

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn expect_table(out: DslOutput) -> linal::core::dataset_legacy::Dataset {
    match out {
        DslOutput::Table(ds) => ds,
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ─── Test 1: Multiple computed columns ───────────────────────────────────────

#[test]
fn test_multiple_computed_columns() {
    let mut db = TensorDb::new();

    exec(
        &mut db,
        "DATASET products COLUMNS (name: STRING, price: FLOAT, qty: INT)",
        1,
    );
    exec(
        &mut db,
        r#"INSERT INTO products VALUES ("apple", 1.5, 10)"#,
        2,
    );
    exec(
        &mut db,
        r#"INSERT INTO products VALUES ("banana", 0.8, 25)"#,
        3,
    );

    let ds = expect_table(exec(
        &mut db,
        "SELECT name, price * 2 AS double_price, qty * price AS total_value FROM products",
        4,
    ));

    // Three columns must exist and be distinct
    assert_eq!(ds.schema.fields.len(), 3, "Expected 3 output columns");
    assert_eq!(ds.schema.fields[0].name, "name");
    assert_eq!(ds.schema.fields[1].name, "double_price");
    assert_eq!(ds.schema.fields[2].name, "total_value");

    // Both rows must be present
    assert_eq!(ds.rows.len(), 2);

    // Row 0: apple — double_price=3.0, total_value=15.0
    let row0 = &ds.rows[0];
    assert!(
        matches!(&row0.values[0], Value::String(s) if s == "apple"),
        "row0 name = {:?}",
        row0.values[0]
    );
    // double_price (price * 2 = 1.5 * 2 = 3.0)
    assert!(
        matches!(row0.values[1], Value::Float(f) if (f as f64 - 3.0).abs() < 0.01),
        "row0 double_price = {:?}",
        row0.values[1]
    );
    // total_value (qty * price = 10 * 1.5 = 15.0)
    assert!(
        matches!(row0.values[2], Value::Float(f) if (f as f64 - 15.0).abs() < 0.01),
        "row0 total_value = {:?}",
        row0.values[2]
    );

    // Row 1: banana — double_price=1.6, total_value=20.0
    let row1 = &ds.rows[1];
    assert!(
        matches!(row1.values[1], Value::Float(f) if (f as f64 - 1.6).abs() < 0.05),
        "row1 double_price = {:?}",
        row1.values[1]
    );
    assert!(
        matches!(row1.values[2], Value::Float(f) if (f as f64 - 20.0).abs() < 0.1),
        "row1 total_value = {:?}",
        row1.values[2]
    );
}

// ─── Test 2: Window function with no PARTITION BY ─────────────────────────────

#[test]
fn test_window_no_partition_by() {
    let mut db = TensorDb::new();

    exec(
        &mut db,
        "DATASET items COLUMNS (name: STRING, price: FLOAT)",
        1,
    );
    exec(&mut db, r#"INSERT INTO items VALUES ("c", 3.0)"#, 2);
    exec(&mut db, r#"INSERT INTO items VALUES ("a", 1.0)"#, 3);
    exec(&mut db, r#"INSERT INTO items VALUES ("b", 2.0)"#, 4);

    let ds = expect_table(exec(
        &mut db,
        "SELECT name, ROW_NUMBER() OVER (ORDER BY price ASC) AS rn FROM items",
        5,
    ));

    assert_eq!(ds.rows.len(), 3);

    // Collect (name, rn) pairs
    let pairs: Vec<(&str, i64)> = ds
        .rows
        .iter()
        .map(|row| {
            let name = match &row.values[0] {
                Value::String(s) => s.as_str(),
                _ => panic!("expected string name"),
            };
            let rn = match row.values[1] {
                Value::Int(n) => n,
                _ => panic!("expected Int rn, got {:?}", row.values[1]),
            };
            (name, rn)
        })
        .collect();

    // All row numbers must be unique
    let mut rns: Vec<i64> = pairs.iter().map(|(_, rn)| *rn).collect();
    rns.sort_unstable();
    assert_eq!(rns, vec![1, 2, 3], "Row numbers must be 1,2,3: {:?}", rns);

    // The row ordered first by price (a=1.0) should have rn=1
    let a_rn = pairs.iter().find(|(n, _)| *n == "a").map(|(_, rn)| *rn);
    assert_eq!(a_rn, Some(1), "item 'a' (lowest price) should be rn=1");
}

// ─── Test 3: UNION deduplication ─────────────────────────────────────────────

#[test]
fn test_union_deduplicates() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET ua COLUMNS (x: INT)", 1);
    exec(&mut db, "INSERT INTO ua VALUES (1)", 2);
    exec(&mut db, "INSERT INTO ua VALUES (2)", 3);
    exec(&mut db, "DATASET ub COLUMNS (x: INT)", 4);
    exec(&mut db, "INSERT INTO ub VALUES (2)", 5);
    exec(&mut db, "INSERT INTO ub VALUES (3)", 6);

    let ds = expect_table(exec(&mut db, "SELECT x FROM ua UNION SELECT x FROM ub", 7));

    // UNION should deduplicate: {1, 2, 3}
    assert_eq!(
        ds.rows.len(),
        3,
        "UNION should yield 3 unique rows, got {:?}",
        ds.rows.len()
    );

    let mut vals: Vec<i64> = ds
        .rows
        .iter()
        .map(|r| match r.values[0] {
            Value::Int(n) => n,
            _ => panic!("expected Int"),
        })
        .collect();
    vals.sort_unstable();
    assert_eq!(vals, vec![1, 2, 3]);
}

// ─── Test 4: UNION ALL keeps duplicates ──────────────────────────────────────

#[test]
fn test_union_all_keeps_duplicates() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET uca COLUMNS (x: INT)", 1);
    exec(&mut db, "INSERT INTO uca VALUES (1)", 2);
    exec(&mut db, "INSERT INTO uca VALUES (2)", 3);
    exec(&mut db, "DATASET ucb COLUMNS (x: INT)", 4);
    exec(&mut db, "INSERT INTO ucb VALUES (2)", 5);
    exec(&mut db, "INSERT INTO ucb VALUES (3)", 6);

    let ds = expect_table(exec(
        &mut db,
        "SELECT x FROM uca UNION ALL SELECT x FROM ucb",
        7,
    ));

    // UNION ALL keeps all 4 rows
    assert_eq!(ds.rows.len(), 4, "UNION ALL should yield 4 rows");
}

// ─── Test 5: CAST to BOOL ────────────────────────────────────────────────────

#[test]
fn test_cast_to_bool() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET flags COLUMNS (val: INT)", 1);
    exec(&mut db, "INSERT INTO flags VALUES (0)", 2);
    exec(&mut db, "INSERT INTO flags VALUES (1)", 3);

    let ds = expect_table(exec(
        &mut db,
        "SELECT CAST(val AS BOOL) AS flag FROM flags",
        4,
    ));

    assert_eq!(ds.rows.len(), 2);
    assert_eq!(
        ds.rows[0].values[0],
        Value::Bool(false),
        "val=0 should cast to false"
    );
    assert_eq!(
        ds.rows[1].values[0],
        Value::Bool(true),
        "val=1 should cast to true"
    );
}

// ─── Test: CAST to VECTOR/MATRIX reshapes tensor columns (v0.1.39, D1) ────────
//
// RESHAPE/FLATTEN exist for standalone tensor variables (`LET x = RESHAPE t
// TO [dims]`) but are not usable inside a SQL SELECT expression — there was
// previously no way to reshape a Vector/Matrix *column* inline in a query.
// CAST(expr AS VECTOR(n)) / CAST(expr AS MATRIX(r, c)) fill that gap.

#[test]
fn test_cast_vector_to_matrix() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: INT, v: Vector(4))", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, [1.0, 2.0, 3.0, 4.0])", 2);

    let ds = expect_table(exec(
        &mut db,
        "SELECT CAST(v AS MATRIX(2, 2)) AS m FROM t",
        3,
    ));
    assert_eq!(
        ds.rows[0].values[0],
        Value::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        "Vector(4) cast to Matrix(2,2) should reshape row-major"
    );
}

#[test]
fn test_cast_matrix_to_vector() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: INT, m: Matrix(2, 3))", 1);
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, [[1, 2, 3], [4, 5, 6]])",
        2,
    );

    let ds = expect_table(exec(&mut db, "SELECT CAST(m AS VECTOR(6)) AS v FROM t", 3));
    assert_eq!(
        ds.rows[0].values[0],
        Value::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "Matrix(2,3) cast to Vector(6) should flatten row-major"
    );
}

#[test]
fn test_cast_vector_matrix_dimension_mismatch_is_null_not_crash() {
    // CAST doesn't resize/pad — a dimension mismatch is a graceful Null,
    // matching the existing convention for other invalid CAST combinations
    // (e.g. CAST(vector AS INT)), not a panic or hard error.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: INT, v: Vector(4))", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, [1.0, 2.0, 3.0, 4.0])", 2);

    let ds = expect_table(exec(
        &mut db,
        "SELECT CAST(v AS MATRIX(3, 3)) AS bad FROM t",
        3,
    ));
    assert_eq!(ds.rows[0].values[0], Value::Null);
}

// ─── Test: FLATTEN(expr) works inside SELECT (v0.1.40, Track F / F2) ─────────
//
// FLATTEN(col) previously parsed successfully inside a SELECT list but
// always evaluated to NULL — NORMALIZE/MATMUL/TRANSPOSE had a dual-branch
// (NAME(x) -> SQL VectorFn, bare NAME x -> tensor-DSL CallExpr) that FLATTEN
// never got, so `FLATTEN(v)` fell through to the tensor-DSL CallExpr path,
// which the SQL row evaluator (dsl_expr_to_logical_expr) doesn't handle and
// silently defaults to NULL.

#[test]
fn test_flatten_vector_in_select() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: INT, v: Vector(4))", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, [1.0, 2.0, 3.0, 4.0])", 2);

    let ds = expect_table(exec(&mut db, "SELECT FLATTEN(v) AS fv FROM t", 3));
    assert_eq!(
        ds.rows[0].values[0],
        Value::Vector(vec![1.0, 2.0, 3.0, 4.0]),
        "FLATTEN on an already-flat Vector is a no-op"
    );
}

#[test]
fn test_flatten_matrix_in_select() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: INT, m: Matrix(2, 2))", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, [[1, 2], [3, 4]])", 2);

    let ds = expect_table(exec(&mut db, "SELECT FLATTEN(m) AS fm FROM t", 3));
    assert_eq!(
        ds.rows[0].values[0],
        Value::Vector(vec![1.0, 2.0, 3.0, 4.0]),
        "FLATTEN(matrix) should flatten row-major"
    );
}

#[test]
fn test_flatten_bare_tensor_dsl_still_works() {
    // Guard against the SQL-context fix breaking the pre-existing
    // tensor-DSL form: `LET x = FLATTEN t` (no parens).
    let mut db = TensorDb::new();
    exec(&mut db, "MATRIX mm = [[1, 2], [3, 4]]", 1);
    let out = exec(&mut db, "LET flat_mm = FLATTEN mm", 2);
    match out {
        DslOutput::Message(msg) => assert!(msg.contains("flat_mm")),
        other => panic!("expected Message output, got: {other:?}"),
    }
}

// ─── Test 6: SUBSTR 2-arg form (1-based) ─────────────────────────────────────

#[test]
fn test_substr_two_arg() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET words COLUMNS (w: STRING)", 1);
    exec(&mut db, r#"INSERT INTO words VALUES ("hello")"#, 2);

    let ds = expect_table(exec(&mut db, "SELECT SUBSTR(w, 2) AS s FROM words", 3));

    assert_eq!(ds.rows.len(), 1);
    // 1-based: SUBSTR("hello", 2) → "ello"
    assert_eq!(
        ds.rows[0].values[0],
        Value::String("ello".to_string()),
        "SUBSTR('hello', 2) should be 'ello'"
    );
}

// ─── Test 7: COALESCE with 3+ args ───────────────────────────────────────────

#[test]
fn test_coalesce_three_args() {
    let mut db = TensorDb::new();

    exec(
        &mut db,
        "DATASET nulls COLUMNS (a: INT NULLABLE, b: INT NULLABLE, c: INT NULLABLE)",
        1,
    );
    exec(&mut db, "INSERT INTO nulls VALUES (NULL, NULL, 42)", 2);
    exec(&mut db, "INSERT INTO nulls VALUES (NULL, 7, 99)", 3);
    exec(&mut db, "INSERT INTO nulls VALUES (5, 8, 11)", 4);

    let ds = expect_table(exec(
        &mut db,
        "SELECT COALESCE(a, b, c) AS first_non_null FROM nulls",
        5,
    ));

    assert_eq!(ds.rows.len(), 3);
    // Row 0: (NULL, NULL, 42) → 42
    assert_eq!(ds.rows[0].values[0], Value::Int(42));
    // Row 1: (NULL, 7, 99) → 7
    assert_eq!(ds.rows[1].values[0], Value::Int(7));
    // Row 2: (5, 8, 11) → 5
    assert_eq!(ds.rows[2].values[0], Value::Int(5));
}

// ─── Test 8: RIGHT JOIN correctness ──────────────────────────────────────────

#[test]
fn test_right_join_correctness() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET left_rj COLUMNS (id: INT, val: STRING)", 1);
    exec(&mut db, r#"INSERT INTO left_rj VALUES (1, "a")"#, 2);

    exec(
        &mut db,
        "DATASET right_rj COLUMNS (id: INT, extra: STRING)",
        3,
    );
    exec(&mut db, r#"INSERT INTO right_rj VALUES (1, "x")"#, 4);
    exec(&mut db, r#"INSERT INTO right_rj VALUES (2, "y")"#, 5);

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM left_rj RIGHT JOIN right_rj ON left_rj.id = right_rj.id",
        6,
    ));

    // RIGHT JOIN: right table drives — 2 rows (id=1 matched, id=2 right-only)
    assert_eq!(ds.rows.len(), 2, "RIGHT JOIN should produce 2 rows");

    // Find the row with right-side extra="y" (unmatched) — left columns should be NULL
    let unmatched = ds.rows.iter().find(|row| {
        row.values
            .iter()
            .any(|v| matches!(v, Value::String(s) if s == "y"))
    });
    assert!(
        unmatched.is_some(),
        "Should have an unmatched right row (extra='y')"
    );
    let unmatched = unmatched.unwrap();
    // The left-side val column should be NULL for the unmatched right row
    let has_null_left = unmatched.values.iter().any(|v| matches!(v, Value::Null));
    assert!(
        has_null_left,
        "Unmatched right row should have NULL left columns: {:?}",
        unmatched.values
    );
}

// ─── Test 9: FULL OUTER JOIN correctness ─────────────────────────────────────

#[test]
fn test_full_outer_join_correctness() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET left_fj COLUMNS (id: INT, val: STRING)", 1);
    exec(&mut db, r#"INSERT INTO left_fj VALUES (1, "a")"#, 2);
    exec(&mut db, r#"INSERT INTO left_fj VALUES (3, "c")"#, 3);

    exec(
        &mut db,
        "DATASET right_fj COLUMNS (id: INT, extra: STRING)",
        4,
    );
    exec(&mut db, r#"INSERT INTO right_fj VALUES (1, "x")"#, 5);
    exec(&mut db, r#"INSERT INTO right_fj VALUES (2, "y")"#, 6);

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM left_fj FULL OUTER JOIN right_fj ON left_fj.id = right_fj.id",
        7,
    ));

    // FULL JOIN: id=1 matched, id=2 right-only, id=3 left-only → 3 rows
    assert_eq!(ds.rows.len(), 3, "FULL OUTER JOIN should produce 3 rows");

    // At least one row must be fully matched (no NULLs)
    let matched = ds
        .rows
        .iter()
        .filter(|row| !row.values.iter().any(|v| matches!(v, Value::Null)))
        .count();
    assert!(matched >= 1, "At least one row should be fully matched");

    // At least two rows should have NULLs (the unmatched sides)
    let with_nulls = ds
        .rows
        .iter()
        .filter(|row| row.values.iter().any(|v| matches!(v, Value::Null)))
        .count();
    assert_eq!(
        with_nulls, 2,
        "Two rows (left-only and right-only) should have NULLs"
    );
}

// ─── Test 10: CTE temp dataset is cleaned up after query ─────────────────────

#[test]
fn test_cte_cleanup_after_query() {
    let mut db = TensorDb::new();

    exec(&mut db, "DATASET source_data COLUMNS (x: INT)", 1);
    exec(&mut db, "INSERT INTO source_data VALUES (10)", 2);
    exec(&mut db, "INSERT INTO source_data VALUES (20)", 3);

    // Run a CTE query using a temp name
    let ds = expect_table(exec(
        &mut db,
        "WITH cte_result AS (SELECT x FROM source_data WHERE x > 5) SELECT x FROM cte_result",
        4,
    ));
    assert_eq!(ds.rows.len(), 2, "CTE query should return 2 rows");

    // After the query, the CTE temp dataset must be gone
    let probe = execute_line(&mut db, "SELECT x FROM cte_result", 5);
    assert!(
        probe.is_err(),
        "CTE temp dataset 'cte_result' should not exist after the query"
    );
}

// ─── Test 11: TENSOR column type preserves dimensions ────────────────────────

#[test]
fn test_tensor_column_preserves_dimensions() {
    let mut db = TensorDb::new();

    exec(
        &mut db,
        "DATASET vecs COLUMNS (name: STRING, embedding: TENSOR(128))",
        1,
    );

    let ds = db.get_dataset("vecs").expect("dataset 'vecs' should exist");

    let embedding_field = ds
        .schema
        .fields
        .iter()
        .find(|f| f.name == "embedding")
        .expect("embedding field should exist");

    assert_eq!(
        embedding_field.value_type,
        ValueType::Vector(128),
        "TENSOR(128) should map to Vector(128), got {:?}",
        embedding_field.value_type
    );
}

// ─── Test 12: TENSOR(r, c) maps to Matrix ────────────────────────────────────

#[test]
fn test_tensor_2d_column_maps_to_matrix() {
    let mut db = TensorDb::new();

    exec(
        &mut db,
        "DATASET mats COLUMNS (name: STRING, weights: TENSOR(4, 8))",
        1,
    );

    let ds = db.get_dataset("mats").expect("dataset 'mats' should exist");

    let weights_field = ds
        .schema
        .fields
        .iter()
        .find(|f| f.name == "weights")
        .expect("weights field should exist");

    assert_eq!(
        weights_field.value_type,
        ValueType::Matrix(4, 8),
        "TENSOR(4, 8) should map to Matrix(4, 8), got {:?}",
        weights_field.value_type
    );
}
