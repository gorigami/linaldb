// tests/hash_join_test.rs
// Regression tests for the smaller-side hash join refactor
// (NestedLoopJoinExec -> HashJoinExec, src/query/physical.rs).
//
// Background: the equi-join executor was already hash-based (not a true
// nested loop) but always hashed a *fixed* side per JoinType (right for
// Inner/Left/Full, left for Right), regardless of which side was actually
// smaller — `tiny JOIN huge` ended up hashing `huge`. Fixed by always
// building the hash table on whichever materialized side has fewer rows,
// decoupled from JoinType's NULL-padding semantics. Pure unit tests for the
// build-side decision function live next to it in src/query/physical.rs;
// these are DSL-level correctness tests confirming row content/count and
// NULL-padding are correct in both size directions (small-JOIN-large and
// large-JOIN-small), which exercises both branches of the build/probe split.

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

fn setup_small_and_large(db: &mut TensorDb) {
    // small: 3 rows, ids 1..3
    exec(db, "DATASET small COLUMNS (id: INT, tag: STRING)", 1);
    exec(db, r#"INSERT INTO small VALUES (1, "a")"#, 2);
    exec(db, r#"INSERT INTO small VALUES (2, "b")"#, 3);
    exec(db, r#"INSERT INTO small VALUES (3, "c")"#, 4);

    // large: 20 rows, ids 1..20 — only 1..3 overlap with `small`
    exec(db, "DATASET large COLUMNS (id: INT, val: INT)", 5);
    for i in 1..=20 {
        exec(
            db,
            &format!("INSERT INTO large VALUES ({i}, {})", i * 10),
            5 + i,
        );
    }
}

// ─── INNER JOIN: correctness must hold regardless of which side is smaller ──

#[test]
fn test_inner_join_small_left_large_right() {
    let mut db = TensorDb::new();
    setup_small_and_large(&mut db);

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM small JOIN large ON small.id = large.id",
        100,
    ));

    assert_eq!(ds.rows.len(), 3, "expected exactly the 3 overlapping ids");
    for row in &ds.rows {
        // Every matched row must carry both a tag (String) and a val (Int),
        // never a NULL — INNER JOIN never pads.
        assert!(
            row.values.iter().any(|v| matches!(v, Value::String(_))),
            "missing tag column: {:?}",
            row.values
        );
        assert!(
            !row.values.iter().any(|v| matches!(v, Value::Null)),
            "INNER JOIN row must not contain NULL: {:?}",
            row.values
        );
    }
}

#[test]
fn test_inner_join_large_left_small_right() {
    // Same predicate, sides swapped — forces build_side_is_left to flip
    // relative to the test above (large is now materialized as `left`,
    // small as `right`, and small is still the smaller relation).
    let mut db = TensorDb::new();
    setup_small_and_large(&mut db);

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM large JOIN small ON large.id = small.id",
        100,
    ));

    assert_eq!(ds.rows.len(), 3, "expected exactly the 3 overlapping ids");
    for row in &ds.rows {
        assert!(
            row.values.iter().any(|v| matches!(v, Value::String(_))),
            "missing tag column: {:?}",
            row.values
        );
        assert!(
            !row.values.iter().any(|v| matches!(v, Value::Null)),
            "INNER JOIN row must not contain NULL: {:?}",
            row.values
        );
    }
}

// ─── LEFT JOIN: unmatched-row NULL-padding must be correct regardless of
//     which physical side ends up as the hash build side ───────────────────

#[test]
fn test_left_join_small_left_unmatched_row_padded() {
    // `small` (left, smaller — becomes the probe side here since it's the
    // side that must be preserved) has an id with no match in `large`.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET small COLUMNS (id: INT, tag: STRING)", 1);
    exec(&mut db, r#"INSERT INTO small VALUES (1, "a")"#, 2);
    exec(&mut db, r#"INSERT INTO small VALUES (99, "unmatched")"#, 3);

    exec(&mut db, "DATASET large COLUMNS (id: INT, val: INT)", 4);
    for i in 1..=20 {
        exec(
            &mut db,
            &format!("INSERT INTO large VALUES ({i}, {})", i * 10),
            4 + i,
        );
    }

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM small LEFT JOIN large ON small.id = large.id",
        100,
    ));

    assert_eq!(ds.rows.len(), 2, "one matched + one unmatched left row");

    let unmatched = ds
        .rows
        .iter()
        .find(|row| {
            row.values
                .iter()
                .any(|v| matches!(v, Value::String(s) if s == "unmatched"))
        })
        .expect("unmatched left row (id=99) must be present");
    assert!(
        unmatched.values.iter().any(|v| matches!(v, Value::Null)),
        "unmatched left row must have NULL right-side columns: {:?}",
        unmatched.values
    );

    let matched = ds
        .rows
        .iter()
        .find(|row| {
            row.values
                .iter()
                .any(|v| matches!(v, Value::String(s) if s == "a"))
        })
        .expect("matched row (id=1) must be present");
    assert!(
        !matched.values.iter().any(|v| matches!(v, Value::Null)),
        "matched row must not contain NULL: {:?}",
        matched.values
    );
}

#[test]
fn test_left_join_large_left_many_unmatched_rows_padded() {
    // `large` (left, bigger — becomes the probe side, still preserved by
    // LEFT JOIN semantics even though it's NOT the smaller/build side here).
    let mut db = TensorDb::new();
    setup_small_and_large(&mut db);

    let ds = expect_table(exec(
        &mut db,
        "SELECT * FROM large LEFT JOIN small ON large.id = small.id",
        100,
    ));

    assert_eq!(ds.rows.len(), 20, "every large row must be preserved");

    let matched_count = ds
        .rows
        .iter()
        .filter(|row| row.values.iter().any(|v| matches!(v, Value::String(_))))
        .count();
    assert_eq!(matched_count, 3, "only ids 1..3 should have a matching tag");

    let unmatched_count = ds
        .rows
        .iter()
        .filter(|row| row.values.iter().any(|v| matches!(v, Value::Null)))
        .count();
    assert_eq!(
        unmatched_count, 17,
        "the remaining 17 large rows must be NULL-padded on the right"
    );
}
