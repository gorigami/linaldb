// tests/float64_precision_test.rs
//
// Regression tests for real f64 scalar support (`Value::Float64` / DSL
// `DOUBLE`/`FLOAT64`), added because `Float` was previously f32-only
// end-to-end despite `DOUBLE`/`FLOAT64` being accepted keywords that
// silently aliased to the same f32 storage (see CHANGELOG.md). A real GPS
// timestamp (~1.1e9 with sub-second precision) is the motivating case: it
// already exceeds f32's ~7 significant digits.

use linal::core::value::{Value, ValueType};
use linal::dsl::execute_line;
use linal::{execute_script, TensorDb};

/// A value with enough significant digits that f32 storage would visibly
/// round it (f32 has ~7 significant decimal digits).
const GPS_LIKE: f64 = 1_126_259_462.413_456_7;

#[test]
fn double_column_insert_preserves_full_precision() {
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET events COLUMNS (id: Int, gps_time: Double)
        INSERT INTO events VALUES (1, {GPS_LIKE})
    "#
    );
    execute_script(&mut db, &script).expect("setup failed");

    let ds = db.get_dataset("events").expect("dataset not found");
    assert_eq!(ds.schema.fields[1].value_type, ValueType::Float64);
    match ds.rows[0].values[1] {
        Value::Float64(v) => assert_eq!(
            v, GPS_LIKE,
            "DOUBLE column narrowed a value it shouldn't have"
        ),
        ref other => panic!("expected Value::Float64, got {other:?}"),
    }
}

#[test]
fn float_column_still_narrows_to_f32() {
    // FLOAT/FLOAT32 must keep meaning f32 -- only DOUBLE/FLOAT64 changed.
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET events COLUMNS (id: Int, gps_time: Float)
        INSERT INTO events VALUES (1, {GPS_LIKE})
    "#
    );
    execute_script(&mut db, &script).expect("setup failed");

    let ds = db.get_dataset("events").expect("dataset not found");
    assert_eq!(ds.schema.fields[1].value_type, ValueType::Float);
    match ds.rows[0].values[1] {
        Value::Float(v) => assert_ne!(
            v as f64, GPS_LIKE,
            "a plain FLOAT column should still lose precision on this value"
        ),
        ref other => panic!("expected Value::Float, got {other:?}"),
    }
}

#[test]
fn cast_as_double_preserves_precision_cast_as_float_does_not() {
    let mut db = TensorDb::new();
    execute_script(&mut db, "DATASET t COLUMNS (id: Int, s: String)").expect("setup failed");
    execute_line(
        &mut db,
        &format!(r#"INSERT INTO t VALUES (1, "{GPS_LIKE}")"#),
        0,
    )
    .expect("insert failed");

    execute_line(
        &mut db,
        "DATASET as_double FROM t SELECT CAST(s AS DOUBLE) AS v",
        0,
    )
    .expect("CAST AS DOUBLE failed");
    let double_ds = db.get_dataset("as_double").expect("dataset not found");
    assert_eq!(double_ds.schema.fields[0].value_type, ValueType::Float64);
    assert_eq!(double_ds.rows[0].values[0], Value::Float64(GPS_LIKE));

    execute_line(
        &mut db,
        "DATASET as_float FROM t SELECT CAST(s AS FLOAT) AS v",
        0,
    )
    .expect("CAST AS FLOAT failed");
    let float_ds = db.get_dataset("as_float").expect("dataset not found");
    assert_eq!(float_ds.schema.fields[0].value_type, ValueType::Float);
    match float_ds.rows[0].values[0] {
        Value::Float(v) => assert_ne!(v as f64, GPS_LIKE),
        ref other => panic!("expected Value::Float, got {other:?}"),
    }
}

#[test]
fn cast_as_float64_is_an_alias_for_double() {
    let mut db = TensorDb::new();
    execute_script(&mut db, "DATASET t COLUMNS (id: Int, n: Int)").expect("setup failed");
    execute_line(&mut db, "INSERT INTO t VALUES (1, 5)", 0).expect("insert failed");
    execute_line(
        &mut db,
        "DATASET r FROM t SELECT CAST(n AS FLOAT64) AS v",
        0,
    )
    .expect("CAST AS FLOAT64 failed");
    let ds = db.get_dataset("r").expect("dataset not found");
    assert_eq!(ds.schema.fields[0].value_type, ValueType::Float64);
    assert_eq!(ds.rows[0].values[0], Value::Float64(5.0));
}

#[test]
fn mixed_arithmetic_promotes_to_float64() {
    // A computed column combining a DOUBLE column with a plain Int/Float
    // must promote to Float64 at full precision, not silently narrow.
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET events COLUMNS (id: Int, gps_time: Double, offset: Int)
        INSERT INTO events VALUES (1, {GPS_LIKE}, 1)
    "#
    );
    execute_script(&mut db, &script).expect("setup failed");

    execute_line(
        &mut db,
        "DATASET result FROM events SELECT (gps_time + offset) AS shifted",
        0,
    )
    .expect("computed column failed");

    let ds = db.get_dataset("result").expect("dataset not found");
    assert_eq!(ds.schema.fields[0].value_type, ValueType::Float64);
    match ds.rows[0].values[0] {
        Value::Float64(v) => assert_eq!(v, GPS_LIKE + 1.0),
        ref other => panic!("expected Value::Float64, got {other:?}"),
    }
}

#[test]
fn sum_and_avg_on_double_column_preserve_precision() {
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET t COLUMNS (id: Int, v: Double)
        INSERT INTO t VALUES (1, {GPS_LIKE})
        INSERT INTO t VALUES (2, {GPS_LIKE})
    "#
    );
    execute_script(&mut db, &script).expect("setup failed");

    execute_line(&mut db, "DATASET summed FROM t SELECT SUM(v)", 0).expect("SUM failed");
    let sum_ds = db.get_dataset("summed").expect("dataset not found");
    match sum_ds.rows[0].values[0] {
        Value::Float64(v) => assert!(
            (v - 2.0 * GPS_LIKE).abs() < 1e-6,
            "expected ~{}, got {v}",
            2.0 * GPS_LIKE
        ),
        ref other => panic!("expected Value::Float64 from SUM, got {other:?}"),
    }

    execute_line(&mut db, "DATASET avgd FROM t SELECT AVG(v)", 0).expect("AVG failed");
    let avg_ds = db.get_dataset("avgd").expect("dataset not found");
    match avg_ds.rows[0].values[0] {
        Value::Float64(v) => assert!((v - GPS_LIKE).abs() < 1e-6, "expected ~{GPS_LIKE}, got {v}"),
        ref other => panic!("expected Value::Float64 from AVG, got {other:?}"),
    }
}

#[test]
fn computed_column_with_some_null_rows_shares_one_schema_type() {
    // Regression test: a computed column whose expression is null for some
    // rows (e.g. LAG at the first row) and a real Float64 value for others
    // must get ONE consistent declared type across every row -- a prior bug
    // let each row's schema be decided independently (falling back to a
    // naive static guess only on null), producing rows with different
    // schemas for the same logical column and failing to combine into one
    // dataset at all.
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET t COLUMNS (id: Int, v: Double)
        INSERT INTO t VALUES (1, {GPS_LIKE})
        INSERT INTO t VALUES (2, {GPS_LIKE})
    "#
    );
    execute_script(&mut db, &script).expect("setup failed");

    // LAG(v) is NULL for the first row (by id order) and a real Float64 for
    // the second; the outer SELECT computes v - prev on both rows.
    execute_line(
        &mut db,
        "DATASET ordered FROM t SELECT id, v, LAG(v) OVER (ORDER BY id) AS prev",
        0,
    )
    .expect("window function failed");
    let result = execute_line(
        &mut db,
        "SELECT id, v, prev, (v - prev) AS delta FROM ordered ORDER BY id",
        0,
    )
    .expect("computed column over partially-null column failed");

    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.schema.fields[3].value_type, ValueType::Float64);
        assert_eq!(ds.rows[0].values[3], Value::Null);
        assert_eq!(ds.rows[1].values[3], Value::Float64(0.0));
    } else {
        panic!("expected a table result");
    }
}

#[test]
fn double_column_survives_save_and_load_round_trip() {
    // Real Arrow Float64 write/read path (src/core/storage.rs), not just the
    // in-memory DSL evaluator -- this is the actual precision fix that
    // matters for persisted data (Parquet + the legacy .meta.json sidecar).
    let mut db = TensorDb::new();
    let script = format!(
        r#"
        DATASET float64_roundtrip_test_ds COLUMNS (id: Int, gps_time: Double)
        INSERT INTO float64_roundtrip_test_ds VALUES (1, {GPS_LIKE})
        SAVE DATASET float64_roundtrip_test_ds TO "float64_roundtrip_test.parquet"
    "#
    );
    execute_script(&mut db, &script).expect("save failed");

    let mut db2 = TensorDb::new();
    execute_line(
        &mut db2,
        r#"LOAD DATASET float64_roundtrip_test_ds FROM "float64_roundtrip_test.parquet""#,
        0,
    )
    .expect("load failed");

    let loaded = db2
        .get_dataset("float64_roundtrip_test_ds")
        .expect("dataset not found after reload");
    assert_eq!(loaded.schema.fields[1].value_type, ValueType::Float64);
    assert_eq!(loaded.rows[0].values[1], Value::Float64(GPS_LIKE));
}

#[test]
fn double_field_is_compatible_with_itself() {
    // Regression test for a Field::is_compatible gap: it had no
    // (Float64, Float64) arm and fell through to a wildcard `false`,
    // rejecting a genuinely-matching Double column with a confusing
    // "expected DOUBLE, got DOUBLE" schema error.
    use linal::core::tuple::Field;
    let field = Field::new("x", ValueType::Float64);
    assert!(field.is_compatible(&Value::Float64(GPS_LIKE)));
}

// ── CAST of a bare numeric literal to DOUBLE must not lose precision ───────
//
// Found via linal-hub's pytest suite against the published `linaldb` PyPI
// package: `CAST(1.23456789012345 AS DOUBLE)` returned the f32-rounded
// value merely widened to f64, even though the lexer/parser/AST already
// carry the literal at full f64 precision -- `dsl_expr_to_logical_expr`'s
// generic `Expr::Scalar` arm narrowed to `Value::Float(f32)` before the
// enclosing `Cast{to: Double}` node ever ran.

const LITERAL_TEXT: &str = "1.23456789012345";
const LITERAL_VALUE: f64 = 1.234_567_890_123_45;

#[test]
fn cast_bare_float_literal_as_double_preserves_full_precision() {
    let mut db = TensorDb::new();
    execute_script(&mut db, "DATASET t COLUMNS (id: Int)").expect("setup failed");
    execute_line(&mut db, "INSERT INTO t VALUES (1)", 0).expect("insert failed");

    execute_line(
        &mut db,
        &format!("DATASET r FROM t SELECT CAST({LITERAL_TEXT} AS DOUBLE) AS v"),
        0,
    )
    .expect("CAST AS DOUBLE failed");

    let ds = db.get_dataset("r").expect("dataset not found");
    assert_eq!(ds.schema.fields[0].value_type, ValueType::Float64);
    assert_eq!(ds.rows[0].values[0], Value::Float64(LITERAL_VALUE));
}

#[test]
fn cast_negative_float_literal_as_double_preserves_full_precision() {
    // Unary minus on a numeric literal folds into `Expr::Scalar(-n)` at
    // parse time, hitting the identical code shape as the positive case.
    let mut db = TensorDb::new();
    execute_script(&mut db, "DATASET t COLUMNS (id: Int)").expect("setup failed");
    execute_line(&mut db, "INSERT INTO t VALUES (1)", 0).expect("insert failed");

    execute_line(
        &mut db,
        &format!("DATASET r FROM t SELECT CAST(-{LITERAL_TEXT} AS DOUBLE) AS v"),
        0,
    )
    .expect("CAST AS DOUBLE failed");

    let ds = db.get_dataset("r").expect("dataset not found");
    assert_eq!(ds.rows[0].values[0], Value::Float64(-LITERAL_VALUE));
}
