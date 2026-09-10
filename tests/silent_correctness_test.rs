// tests/silent_correctness_test.rs
// Regression tests for Track A of CONSISTENCY_PLAN.md — silent correctness
// bugs that previously misled users with no error and no warning:
//   - A1: ORDER BY on Vector/Matrix columns silently no-op'd instead of erroring
//   - A2: SUM_VEC/AVG_VEC on dimension-mismatched vectors/matrices silently
//         dropped rows instead of erroring
//   - A3: DELIVER was a hardcoded-success stub regardless of whether the
//         dataset existed or had actually been persisted
//   - A4: SELECT with an aggregate function and no GROUP BY (e.g.
//         `SELECT SUM(price) FROM t`) silently returned the raw,
//         unaggregated table instead of computing the aggregate
//
// Round 2 (Track G):
//   - G1: SUM_VEC/AVG_VEC OVER (...) window aggregates silently returned
//         0.0 instead of an element-wise running aggregate
//   - G2: bare (non-windowed) aggregates with `AS alias` silently ignored
//         the alias, always naming the output column after the function call
//   - G3: a SELECT mixing a bare aggregate with a window function (no
//         GROUP BY) could silently drop the aggregate column from the
//         final projection

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use tempfile::TempDir;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn make_db(data_dir: &std::path::Path) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = data_dir.to_path_buf();
    TensorDb::with_config(config)
}

// ── A1: ORDER BY on Vector/Matrix columns must error ─────────────────────────

#[test]
fn test_order_by_vector_column_errors() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, embedding: Vector(3))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, [1.0, 0.0, 0.0])", 2);

    let result = execute_line(&mut db, "SELECT * FROM docs ORDER BY embedding", 3);
    assert!(
        result.is_err(),
        "ORDER BY on a Vector column must error, not silently no-op the sort"
    );
}

#[test]
fn test_order_by_matrix_column_errors() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET mats COLUMNS (id: INT, m: Matrix(2, 2))",
        1,
    );
    exec(
        &mut db,
        "INSERT INTO mats VALUES (1, [[1.0, 0.0], [0.0, 1.0]])",
        2,
    );

    let result = execute_line(&mut db, "SELECT * FROM mats ORDER BY m", 3);
    assert!(
        result.is_err(),
        "ORDER BY on a Matrix column must error, not silently no-op the sort"
    );
}

#[test]
fn test_order_by_scalar_column_still_works() {
    // Guard against over-broad fixes: scalar ORDER BY must be unaffected.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET items COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO items VALUES (1, 30.0)", 2);
    exec(&mut db, "INSERT INTO items VALUES (2, 10.0)", 3);

    let out = exec(&mut db, "SELECT id FROM items ORDER BY price ASC", 4);
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.len(), 2);
            assert_eq!(ds.rows[0].values[0], linal::core::value::Value::Int(2));
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_window_order_by_vector_column_errors() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, embedding: Vector(3))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, [1.0, 0.0, 0.0])", 2);
    exec(&mut db, "INSERT INTO docs VALUES (2, [0.0, 1.0, 0.0])", 3);

    let result = execute_line(
        &mut db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY embedding) AS rn FROM docs",
        4,
    );
    assert!(
        result.is_err(),
        "Window function ORDER BY on a Vector column must error, not silently no-op the sort"
    );
}

// ── A2: SUM_VEC/AVG_VEC dimension mismatch must error ────────────────────────
//
// A fixed-dimension Vector(N) column rejects mismatched inserts at the type
// level, so the only DSL-reachable way to feed an aggregate two different
// vector dimensions for the "same" group is a wildcard `Vector(0)` column
// (which accepts any dimension per row) combined with GROUP BY, which is
// what actually routes execution through AggregateExec.

#[test]
fn test_sum_vec_dimension_mismatch_errors() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET mixed COLUMNS (id: INT, cat: STRING, v: Vector(0))",
        1,
    );
    exec(&mut db, "INSERT INTO mixed VALUES (1, 'g', [1.0, 2.0])", 2);
    exec(
        &mut db,
        "INSERT INTO mixed VALUES (2, 'g', [1.0, 1.0, 1.0])",
        3,
    );

    let result = execute_line(
        &mut db,
        "SELECT cat, SUM_VEC(v) AS total FROM mixed GROUP BY cat",
        4,
    );
    assert!(
        result.is_err(),
        "SUM_VEC over mismatched vector dimensions must error, not silently drop the mismatched row"
    );
}

#[test]
fn test_avg_vec_dimension_mismatch_errors() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET mixed COLUMNS (id: INT, cat: STRING, v: Vector(0))",
        1,
    );
    exec(&mut db, "INSERT INTO mixed VALUES (1, 'g', [1.0, 2.0])", 2);
    exec(
        &mut db,
        "INSERT INTO mixed VALUES (2, 'g', [1.0, 1.0, 1.0])",
        3,
    );

    let result = execute_line(
        &mut db,
        "SELECT cat, AVG_VEC(v) AS centroid FROM mixed GROUP BY cat",
        4,
    );
    assert!(
        result.is_err(),
        "AVG_VEC over mismatched vector dimensions must error, not silently corrupt the average"
    );
}

#[test]
fn test_sum_vec_matching_dimensions_still_works() {
    // Guard against over-broad fixes: same-dimension SUM_VEC must be unaffected.
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, category: STRING, embedding: Vector(2))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, 'a', [1.0, 2.0])", 2);
    exec(&mut db, "INSERT INTO docs VALUES (2, 'a', [3.0, 4.0])", 3);

    let out = exec(
        &mut db,
        "SELECT category, SUM_VEC(embedding) AS total FROM docs GROUP BY category",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.len(), 1);
            match &ds.rows[0].values[1] {
                linal::core::value::Value::Vector(v) => {
                    assert_eq!(v, &vec![4.0f32, 6.0f32]);
                }
                other => panic!("Expected Vector, got {other:?}"),
            }
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── A3: DELIVER must reflect real state, not a hardcoded success message ─────

#[test]
fn test_deliver_nonexistent_dataset_errors() {
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());

    let result = execute_line(&mut db, "DELIVER ghost", 1);
    assert!(
        result.is_err(),
        "DELIVER on a dataset that doesn't exist must error, not report fake success"
    );
}

#[test]
fn test_deliver_unsaved_dataset_reports_not_persisted() {
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());
    exec(&mut db, "DATASET docs COLUMNS (id: INT)", 1);
    exec(&mut db, "INSERT INTO docs VALUES (1)", 2);

    let out = exec(&mut db, "DELIVER docs", 3);
    match out {
        DslOutput::Message(msg) => {
            assert!(
                msg.contains("has not been persisted") || msg.contains("SAVE DATASET"),
                "expected message to explain the dataset isn't deliverable yet, got: {msg}"
            );
        }
        other => panic!("Expected Message output, got: {other:?}"),
    }
}

#[test]
fn test_deliver_saved_dataset_reports_deliverable() {
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());
    exec(&mut db, "DATASET docs COLUMNS (id: INT)", 1);
    exec(&mut db, "INSERT INTO docs VALUES (1)", 2);
    exec(&mut db, "SAVE DATASET docs", 3);

    let out = exec(&mut db, "DELIVER docs", 4);
    match out {
        DslOutput::Message(msg) => {
            assert!(
                msg.contains("deliverable") || msg.contains("manifest"),
                "expected message to confirm the dataset is deliverable, got: {msg}"
            );
        }
        other => panic!("Expected Message output, got: {other:?}"),
    }
}

// ── A4: bare (non-GROUP-BY) aggregates must actually aggregate ───────────────

#[test]
fn test_bare_sum_aggregates_instead_of_returning_raw_rows() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO nums VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO nums VALUES (2, 20.0)", 3);
    exec(&mut db, "INSERT INTO nums VALUES (3, 30.0)", 4);

    let out = exec(&mut db, "SELECT SUM(price) AS total FROM nums", 5);
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(
                ds.len(),
                1,
                "a bare SUM with no GROUP BY must collapse to a single row, not return all raw rows"
            );
            assert_eq!(ds.rows[0].values[0], linal::core::value::Value::Float(60.0));
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_bare_count_and_avg_aggregate_together() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO nums VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO nums VALUES (2, 20.0)", 3);
    exec(&mut db, "INSERT INTO nums VALUES (3, 30.0)", 4);

    let out = exec(
        &mut db,
        "SELECT COUNT(*) AS n, AVG(price) AS avg_p FROM nums",
        5,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.len(), 1);
            assert_eq!(ds.rows[0].values[0], linal::core::value::Value::Int(3));
            assert_eq!(ds.rows[0].values[1], linal::core::value::Value::Float(20.0));
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_bare_aggregate_on_empty_table_returns_no_rows() {
    // Aggregation over an empty set returns no rows (existing convention,
    // see AggregateExec::execute's early-return for empty input) — this
    // guards against the fix changing that behavior.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);

    let out = exec(&mut db, "SELECT SUM(price) AS total FROM nums", 2);
    match out {
        DslOutput::Table(ds) => assert_eq!(ds.len(), 0),
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── G1: SUM_VEC/AVG_VEC OVER (...) must actually aggregate, not zero out ─────

#[test]
fn test_sum_vec_over_window_is_cumulative_not_zero() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, category: STRING, embedding: Vector(2))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, 'a', [1.0, 2.0])", 2);
    exec(&mut db, "INSERT INTO docs VALUES (2, 'a', [3.0, 4.0])", 3);

    let out = exec(
        &mut db,
        "SELECT id, SUM_VEC(embedding) OVER (PARTITION BY category ORDER BY id) AS running \
         FROM docs",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.len(), 2);
            let running_idx = ds.schema.get_field_index("running").expect("running col");
            match &ds.rows[0].values[running_idx] {
                linal::core::value::Value::Vector(v) => assert_eq!(v, &vec![1.0f32, 2.0f32]),
                other => panic!("Expected Vector, got {other:?} — SUM_VEC OVER silently zeroed"),
            }
            match &ds.rows[1].values[running_idx] {
                linal::core::value::Value::Vector(v) => assert_eq!(v, &vec![4.0f32, 6.0f32]),
                other => panic!("Expected Vector, got {other:?} — SUM_VEC OVER silently zeroed"),
            }
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_avg_vec_over_window_is_running_average_not_zero() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, category: STRING, embedding: Vector(2))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, 'a', [2.0, 4.0])", 2);
    exec(&mut db, "INSERT INTO docs VALUES (2, 'a', [4.0, 8.0])", 3);

    let out = exec(
        &mut db,
        "SELECT id, AVG_VEC(embedding) OVER (PARTITION BY category ORDER BY id) AS running_avg \
         FROM docs",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            let idx = ds.schema.get_field_index("running_avg").expect("col");
            match &ds.rows[1].values[idx] {
                linal::core::value::Value::Vector(v) => assert_eq!(v, &vec![3.0f32, 6.0f32]),
                other => panic!("Expected Vector, got {other:?} — AVG_VEC OVER silently zeroed"),
            }
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_plain_sum_over_window_still_works() {
    // Guard against over-broad fixes: scalar SUM() OVER (...) must be unaffected.
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET items COLUMNS (id: INT, category: STRING, price: FLOAT)",
        1,
    );
    exec(&mut db, "INSERT INTO items VALUES (1, 'a', 10.0)", 2);
    exec(&mut db, "INSERT INTO items VALUES (2, 'a', 20.0)", 3);

    let out = exec(
        &mut db,
        "SELECT id, SUM(price) OVER (PARTITION BY category ORDER BY id) AS running FROM items",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            let idx = ds.schema.get_field_index("running").expect("running col");
            assert_eq!(
                ds.rows[1].values[idx],
                linal::core::value::Value::Float(30.0)
            );
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── G2: bare (non-windowed) aggregates must honor `AS alias` ─────────────────

#[test]
fn test_bare_sum_honors_as_alias() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO nums VALUES (1, 10.0)", 2);

    let out = exec(&mut db, "SELECT SUM(price) AS total FROM nums", 2);
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(
                ds.schema.get_field_index("total"),
                Some(0),
                "aliased bare aggregate must be named 'total', got schema: {:?}",
                ds.schema
            );
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_bare_sum_vec_honors_as_alias() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET docs COLUMNS (id: INT, embedding: Vector(2))",
        1,
    );
    exec(&mut db, "INSERT INTO docs VALUES (1, [1.0, 2.0])", 2);

    let out = exec(
        &mut db,
        "SELECT SUM_VEC(embedding) AS centroid FROM docs",
        3,
    );
    match out {
        DslOutput::Table(ds) => {
            assert!(
                ds.schema.get_field_index("centroid").is_some(),
                "aliased bare SUM_VEC must be named 'centroid', got schema: {:?}",
                ds.schema
            );
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_bare_aggregate_without_alias_still_uses_function_name() {
    // Guard against over-broad fixes: unaliased bare aggregates keep the
    // existing FUNC(col) naming convention.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO nums VALUES (1, 10.0)", 2);

    let out = exec(&mut db, "SELECT SUM(price) FROM nums", 2);
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.schema.fields[0].name, "SUM(price)");
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── G3: mixing a bare aggregate with a window function must not drop the
//        aggregate column from the final projection ────────────────────────

#[test]
fn test_bare_aggregate_survives_alongside_window_function() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET nums COLUMNS (id: INT, price: FLOAT)", 1);
    exec(&mut db, "INSERT INTO nums VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO nums VALUES (2, 20.0)", 3);

    let out = exec(
        &mut db,
        "SELECT SUM(price) AS total, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM nums",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            assert!(
                ds.schema.get_field_index("total").is_some(),
                "bare aggregate 'total' must survive projection alongside a window \
                 function, got schema: {:?}",
                ds.schema
            );
            assert!(ds.schema.get_field_index("rn").is_some());
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── H: HAVING must resolve aliased aggregates and error on unknown columns
//      instead of silently matching zero rows ──────────────────────────────
//
// Found via linal-hub's pytest suite against the published `linaldb` PyPI
// package: `HAVING AVG(score) > 0.5` alongside `AVG(score) AS avg_score`
// silently returned zero rows with no error, because the parser lowers a
// bare aggregate call in HAVING to a literal column-name Ref
// (`"AVG(score)"`), which stops matching once the SELECT list renames that
// same aggregate's real output column via an alias.

#[test]
fn test_having_with_aliased_aggregate_resolves_correctly() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET diagnostics COLUMNS (region: String, score: Float)",
        1,
    );
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"west\", 0.9)", 2);
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"west\", 0.1)", 3);
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"east\", 0.9)", 4);

    let out = exec(
        &mut db,
        "SELECT region, AVG(score) AS avg_score FROM diagnostics GROUP BY region HAVING AVG(score) > 0.5",
        5,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(
                ds.rows.len(),
                1,
                "HAVING on an aliased aggregate must still filter correctly, got rows: {:?}",
                ds.rows
            );
            assert_eq!(ds.rows[0].values[0], Value::String("east".to_string()));
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_having_with_unaliased_aggregate_still_works() {
    // Regression guard: the pre-existing working case (no alias) must keep
    // working after the alias-resolution fix.
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET diagnostics COLUMNS (region: String, score: Float)",
        1,
    );
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"west\", 0.9)", 2);
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"west\", 0.1)", 3);
    exec(&mut db, "INSERT INTO diagnostics VALUES (\"east\", 0.9)", 4);

    let out = exec(
        &mut db,
        "SELECT region, AVG(score) FROM diagnostics GROUP BY region HAVING AVG(score) > 0.5",
        5,
    );
    match out {
        DslOutput::Table(ds) => assert_eq!(ds.rows.len(), 1),
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_having_unknown_column_errors_loudly() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: Int, price: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 10.0)", 2);

    let result = execute_line(
        &mut db,
        "SELECT id, SUM(price) AS total FROM t GROUP BY id HAVING nonexistent_col > 1",
        3,
    );
    assert!(
        result.is_err(),
        "HAVING referencing an unknown column must error, not silently return nothing"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("nonexistent_col"),
        "error should name the unresolved column, got: {msg}"
    );
}

#[test]
fn test_having_without_group_by_on_global_aggregate() {
    // Covers the second HAVING call site (no GROUP BY / global aggregate).
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (id: Int, price: Float)", 1);
    exec(&mut db, "INSERT INTO t VALUES (1, 10.0)", 2);
    exec(&mut db, "INSERT INTO t VALUES (2, 20.0)", 3);

    let out = exec(
        &mut db,
        "SELECT SUM(price) AS total FROM t HAVING SUM(price) > 100",
        4,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.rows.len(), 0, "SUM(price)=30 must not pass HAVING > 100");
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }

    let out = exec(
        &mut db,
        "SELECT SUM(price) AS total FROM t HAVING SUM(price) > 10",
        5,
    );
    match out {
        DslOutput::Table(ds) => {
            assert_eq!(ds.rows.len(), 1, "SUM(price)=30 must pass HAVING > 10");
        }
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

// ── I: INSERT must reject a value/column-name count that doesn't match the
//      target schema, instead of silently truncating or dropping extras ────
//
// Found via linal-hub's pytest suite: `INSERT INTO t VALUES (1, 2, 3)` into
// a 1-column dataset succeeded silently as `(1,)` -- `Iterator::zip` drops
// the excess values before `Tuple::new`/`Schema::validate` (which DOES
// correctly catch too-FEW values) ever sees the mismatch.

#[test]
fn test_insert_positional_too_many_values_errors() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int)", 1);

    let result = execute_line(&mut db, "INSERT INTO t VALUES (1, 2, 3)", 2);
    assert!(
        result.is_err(),
        "INSERT with more values than columns must error, not silently truncate"
    );

    let out = exec(&mut db, "SELECT * FROM t", 3);
    match out {
        DslOutput::Table(ds) => assert_eq!(
            ds.rows.len(),
            0,
            "the failed INSERT must not leave a silently-truncated partial row"
        ),
        other => panic!("Expected Table output, got: {other:?}"),
    }
}

#[test]
fn test_insert_positional_too_few_values_still_errors() {
    // Regression guard for the pre-existing under-supply error path.
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int, b: Int)", 1);

    let result = execute_line(&mut db, "INSERT INTO t VALUES (1)", 2);
    assert!(
        result.is_err(),
        "INSERT with fewer values than columns must still error"
    );
}

#[test]
fn test_insert_named_unknown_column_errors() {
    let mut db = TensorDb::new();
    exec(&mut db, "DATASET t COLUMNS (a: Int)", 1);

    let result = execute_line(&mut db, "INSERT INTO t (a = 1, bogus = 2)", 2);
    assert!(
        result.is_err(),
        "INSERT naming an unknown column must error, not silently drop it"
    );
}
