// tests/case_when_comparison_test.rs
// Regression tests for a silent correctness bug found while building the
// PBMC cell-typing example: `evaluate_expression` (src/query/physical.rs),
// used for CASE WHEN conditions, computed SELECT columns, and aggregate
// inner expressions, had NO handling for comparison operators (=, !=, >, <,
// >=, <=) in its `Expr::BinaryExpr` arm — only arithmetic (+, -, *, /). Any
// comparison silently fell through to `_ => Value::Null`, and since
// `Value::Null` is never `Value::Bool(true)`, every CASE WHEN condition
// using a comparison (the exact form documented in DSL_REFERENCE.md's own
// example: `CASE WHEN score > 90 THEN ...`) always took the ELSE branch,
// no error raised. WHERE clauses were unaffected — they route through a
// separate, correct implementation (`query::planner::evaluate_expr`).
//
// This had zero test coverage anywhere in the suite before this file.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn expect_table(out: DslOutput) -> linal::core::dataset_legacy::Dataset {
    match out {
        DslOutput::Table(ds) => ds,
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ─── The exact example from DSL_REFERENCE.md's CASE section ─────────────────

#[test]
fn test_case_when_documented_grade_example() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET students COLUMNS (id: Int, score: Int)", 1);
    exec(&mut db, "INSERT INTO students VALUES (1, 95)", 2);
    exec(&mut db, "INSERT INTO students VALUES (2, 85)", 3);
    exec(&mut db, "INSERT INTO students VALUES (3, 60)", 4);

    let ds = expect_table(exec(
        &mut db,
        r#"SELECT id, CASE WHEN score > 90 THEN "A" WHEN score > 80 THEN "B" ELSE "C" END AS grade FROM students"#,
        5,
    ));

    let grade_idx = ds.schema.get_field_index("grade").expect("grade column");
    let id_idx = ds.schema.get_field_index("id").expect("id column");
    for row in &ds.rows {
        let id = &row.values[id_idx];
        let grade = &row.values[grade_idx];
        let expected = match id {
            Value::Int(1) => "A", // score 95 > 90
            Value::Int(2) => "B", // score 85 > 80, not > 90
            Value::Int(3) => "C", // score 60, neither
            other => panic!("unexpected id: {other:?}"),
        };
        assert_eq!(
            grade,
            &Value::String(expected.to_string()),
            "id={id:?} got grade {grade:?}, expected {expected} — CASE WHEN comparison must be evaluated, not always fall to ELSE"
        );
    }
}

// ─── Each comparison operator, in a plain computed column ────────────────────

#[test]
fn test_case_when_each_comparison_operator() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int)", 1);
    exec(&mut db, "INSERT INTO t VALUES (5)", 2);

    let cases: &[(&str, bool)] = &[
        ("a = 5", true),
        ("a = 6", false),
        ("a != 5", false),
        ("a != 6", true),
        ("a > 4", true),
        ("a > 5", false),
        ("a < 6", true),
        ("a < 5", false),
        ("a >= 5", true),
        ("a >= 6", false),
        ("a <= 5", true),
        ("a <= 4", false),
    ];

    for (idx, (cond, expected)) in cases.iter().enumerate() {
        let dsl = format!("SELECT CASE WHEN {cond} THEN true ELSE false END AS flag FROM t");
        let ds = expect_table(exec(&mut db, &dsl, 10 + idx));
        let flag_idx = ds.schema.get_field_index("flag").expect("flag column");
        assert_eq!(
            ds.rows[0].values[flag_idx],
            Value::Bool(*expected),
            "condition '{cond}' should evaluate to {expected}"
        );
    }
}

// ─── String comparison (not just numeric) ────────────────────────────────────

#[test]
fn test_case_when_string_equality() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: String, b: String)", 1);
    exec(&mut db, r#"INSERT INTO t VALUES ("x", "x")"#, 2);
    exec(&mut db, r#"INSERT INTO t VALUES ("x", "y")"#, 3);

    let ds = expect_table(exec(
        &mut db,
        r#"SELECT a, b, CASE WHEN a = b THEN "match" ELSE "no match" END AS result FROM t"#,
        4,
    ));

    let result_idx = ds.schema.get_field_index("result").expect("result column");
    let b_idx = ds.schema.get_field_index("b").expect("b column");
    for row in &ds.rows {
        let expected = if row.values[b_idx] == Value::String("x".to_string()) {
            "match"
        } else {
            "no match"
        };
        assert_eq!(row.values[result_idx], Value::String(expected.to_string()));
    }
}

// ─── Comparison inside a bare (non-windowed, non-grouped) aggregate ─────────
// This is the pattern needed to compute an accuracy/match-rate metric —
// exactly what surfaced the bug while building a classification example.

#[test]
fn test_sum_case_when_comparison_bare_aggregate() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET t COLUMNS (predicted: String, actual: String)",
        1,
    );
    exec(&mut db, r#"INSERT INTO t VALUES ("cat", "cat")"#, 2);
    exec(&mut db, r#"INSERT INTO t VALUES ("cat", "dog")"#, 3);
    exec(&mut db, r#"INSERT INTO t VALUES ("dog", "dog")"#, 4);

    let ds = expect_table(exec(
        &mut db,
        "SELECT COUNT(*) AS total, SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) AS correct FROM t",
        5,
    ));

    assert_eq!(ds.rows.len(), 1);
    let total_idx = ds.schema.get_field_index("total").expect("total column");
    let correct_idx = ds
        .schema
        .get_field_index("correct")
        .expect("correct column");
    assert_eq!(ds.rows[0].values[total_idx], Value::Int(3));
    assert_eq!(
        ds.rows[0].values[correct_idx],
        Value::Int(2),
        "2 of 3 rows have predicted = actual"
    );
}

// ─── Comparison inside a GROUP BY aggregate ──────────────────────────────────

#[test]
fn test_sum_case_when_comparison_group_by() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (grp: String, a: Int)", 1);
    exec(&mut db, r#"INSERT INTO t VALUES ("g1", 1)"#, 2);
    exec(&mut db, r#"INSERT INTO t VALUES ("g1", 2)"#, 3);
    exec(&mut db, r#"INSERT INTO t VALUES ("g2", 5)"#, 4);
    exec(&mut db, r#"INSERT INTO t VALUES ("g2", 10)"#, 5);

    let ds = expect_table(exec(
        &mut db,
        "SELECT grp, SUM(CASE WHEN a > 1 THEN 1 ELSE 0 END) AS count_gt1 FROM t GROUP BY grp",
        6,
    ));

    let grp_idx = ds.schema.get_field_index("grp").expect("grp column");
    let count_idx = ds
        .schema
        .get_field_index("count_gt1")
        .expect("count_gt1 column");
    for row in &ds.rows {
        let expected = match &row.values[grp_idx] {
            Value::String(s) if s == "g1" => 1, // only a=2 satisfies a > 1
            Value::String(s) if s == "g2" => 2, // both a=5 and a=10 satisfy
            other => panic!("unexpected group: {other:?}"),
        };
        assert_eq!(row.values[count_idx], Value::Int(expected));
    }
}
