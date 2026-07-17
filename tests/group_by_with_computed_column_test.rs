// tests/group_by_with_computed_column_test.rs
// Regression tests for a silent correctness bug found while building the
// PBMC cell-typing example: a GROUP BY query whose SELECT list also
// contains a `SelectExpr::Computed` item (any qualified column with an
// alias, e.g. `t.col AS col`, or a genuinely computed expression like
// `UPPER(name) AS n`, parses as Computed — not just real window functions)
// made `execute_select`'s window/computed post-processing path run even
// though a GROUP BY was present. That path's `agg_idx` counter (added in
// v0.1.45, Track G3) assumed `base_schema.fields` were *only* the aggregate
// outputs, one per Aggregate SelectExpr in order — true only when there's
// no GROUP BY. With a GROUP BY, `LogicalPlan::Aggregate::schema()` puts the
// group-key fields first, then the aggregates — so `agg_idx` pointed at the
// wrong fields, returning group-key values (duplicated) in place of the
// actual aggregate results, which were silently dropped from the output
// entirely.

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

fn setup(db: &mut TensorDb) {
    exec(db, "DATASET small COLUMNS (sid: String, cond: String)", 1);
    exec(db, r#"INSERT INTO small VALUES ("s1", "healthy")"#, 2);
    exec(db, r#"INSERT INTO small VALUES ("s2", "sick")"#, 3);
    exec(db, "DATASET big COLUMNS (sid: String, val: Int)", 4);
    exec(db, r#"INSERT INTO big VALUES ("s1", 10)"#, 5);
    exec(db, r#"INSERT INTO big VALUES ("s1", 20)"#, 6);
    exec(db, r#"INSERT INTO big VALUES ("s2", 5)"#, 7);
}

#[test]
fn test_group_by_with_qualified_aliased_column_and_aggregate() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let ds = expect_table(exec(
        &mut db,
        "SELECT small.sid AS sid, small.cond AS cond, COUNT(*) AS n, AVG(big.val) AS avg_val \
         FROM big JOIN small ON big.sid = small.sid GROUP BY sid, cond",
        10,
    ));

    // Exactly 4 columns — the bug produced 4 columns too, but with the
    // aggregate columns silently replaced by duplicated group-key values.
    assert_eq!(ds.schema.fields.len(), 4);
    assert!(ds.schema.get_field_index("n").is_some(), "n column missing");
    assert!(
        ds.schema.get_field_index("avg_val").is_some(),
        "avg_val column missing"
    );

    let sid_idx = ds.schema.get_field_index("sid").unwrap();
    let n_idx = ds.schema.get_field_index("n").unwrap();
    let avg_idx = ds.schema.get_field_index("avg_val").unwrap();

    for row in &ds.rows {
        let (expected_n, expected_avg) = match &row.values[sid_idx] {
            Value::String(s) if s == "s1" => (2, 15.0),
            Value::String(s) if s == "s2" => (1, 5.0),
            other => panic!("unexpected sid: {other:?}"),
        };
        assert_eq!(row.values[n_idx], Value::Int(expected_n));
        assert_eq!(row.values[avg_idx], Value::Float(expected_avg));
    }
}

#[test]
fn test_group_by_with_genuinely_computed_column_and_aggregate() {
    // Not just qualified-column aliasing — a real computed expression
    // (UPPER) alongside GROUP BY + an aggregate hits the same code path.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (grp: String, val: Int)", 1);
    exec(&mut db, r#"INSERT INTO t VALUES ("a", 1)"#, 2);
    exec(&mut db, r#"INSERT INTO t VALUES ("a", 3)"#, 3);
    exec(&mut db, r#"INSERT INTO t VALUES ("b", 10)"#, 4);

    let ds = expect_table(exec(
        &mut db,
        r#"SELECT grp, UPPER(grp) AS grp_upper, SUM(val) AS total FROM t GROUP BY grp"#,
        5,
    ));

    let grp_idx = ds.schema.get_field_index("grp").expect("grp column");
    let upper_idx = ds
        .schema
        .get_field_index("grp_upper")
        .expect("grp_upper column");
    let total_idx = ds.schema.get_field_index("total").expect("total column");

    for row in &ds.rows {
        let (expected_upper, expected_total) = match &row.values[grp_idx] {
            Value::String(s) if s == "a" => ("A", 4),
            Value::String(s) if s == "b" => ("B", 10),
            other => panic!("unexpected grp: {other:?}"),
        };
        assert_eq!(
            row.values[upper_idx],
            Value::String(expected_upper.to_string())
        );
        assert_eq!(row.values[total_idx], Value::Int(expected_total));
    }
}

#[test]
fn test_group_by_plain_aggregate_still_works() {
    // Guard against over-broad fixes: a plain GROUP BY + aggregate with no
    // Computed items (the common case, already well-tested elsewhere) must
    // be unaffected — this path doesn't even reach the window/computed
    // post-processing code the bug was in.
    let mut db = TensorDb::new();
    setup(&mut db);

    let ds = expect_table(exec(
        &mut db,
        "SELECT sid, COUNT(*) AS n FROM big GROUP BY sid",
        10,
    ));

    let sid_idx = ds.schema.get_field_index("sid").unwrap();
    let n_idx = ds.schema.get_field_index("n").unwrap();
    for row in &ds.rows {
        let expected = match &row.values[sid_idx] {
            Value::String(s) if s == "s1" => 2,
            Value::String(s) if s == "s2" => 1,
            other => panic!("unexpected sid: {other:?}"),
        };
        assert_eq!(row.values[n_idx], Value::Int(expected));
    }
}
