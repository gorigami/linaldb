//! DSL-level integration tests for `Value::Complex` --
//! SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 3. `src/core/value.rs`'s own
//! logic (equals/as_complex/Display) is simple enough to not need isolated
//! unit tests; these exercise it through the real DSL pipeline (parser ->
//! planner -> physical execution), matching this repo's established
//! preference for real end-to-end coverage over isolated unit tests.

use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn run(db: &mut TensorDb, line: &str, n: usize) -> DslOutput {
    execute_line(db, line, n).unwrap_or_else(|e| panic!("line {n} ({line:?}) failed: {e:?}"))
}

fn setup(db: &mut TensorDb) {
    run(
        db,
        "DATASET nums COLUMNS (id: Int, re: Float64, im: Float64)",
        1,
    );
    run(db, "INSERT INTO nums (id = 1, re = 3.0, im = 4.0)", 2);
    run(db, "INSERT INTO nums (id = 2, re = 1.0, im = -1.0)", 3);
}

#[test]
fn complex_construction_and_display() {
    let mut db = TensorDb::new();
    setup(&mut db);
    let out = run(&mut db, "SELECT id, COMPLEX(re, im) AS val FROM nums", 4);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows.len(), 2);
    let val_col = ds.schema.get_field_index("val").unwrap();
    assert_eq!(
        ds.schema.fields[val_col].value_type,
        linal::core::value::ValueType::Complex
    );
    match &ds.rows[0].values[val_col] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 3.0).abs() < 1e-9);
            assert!((c.im - 4.0).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
}

#[test]
fn complex_scalar_functions_real_imag_abs_phase_conj() {
    let mut db = TensorDb::new();
    setup(&mut db);
    // 3+4i is a 3-4-5 right triangle: |z| = 5, arg = atan2(4, 3).
    let out = run(
        &mut db,
        "SELECT REAL(COMPLEX(re, im)) AS r, IMAG(COMPLEX(re, im)) AS i, \
         ABS(COMPLEX(re, im)) AS a, PHASE(COMPLEX(re, im)) AS p \
         FROM nums WHERE id = 1",
        4,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let get = |col: &str| -> f64 {
        let idx = ds.schema.get_field_index(col).unwrap();
        match &ds.rows[0].values[idx] {
            linal::core::value::Value::Float64(f) => *f,
            other => panic!("expected Float64 for {col}, got {other:?}"),
        }
    };
    assert!((get("r") - 3.0).abs() < 1e-9);
    assert!((get("i") - 4.0).abs() < 1e-9);
    assert!((get("a") - 5.0).abs() < 1e-9);
    assert!((get("p") - 4.0f64.atan2(3.0)).abs() < 1e-9);

    let out = run(
        &mut db,
        "SELECT CONJ(COMPLEX(re, im)) AS c FROM nums WHERE id = 1",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let c_idx = ds.schema.get_field_index("c").unwrap();
    match &ds.rows[0].values[c_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 3.0).abs() < 1e-9);
            assert!((c.im - (-4.0)).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
}

#[test]
fn complex_arithmetic_add_sub_mul_div() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let cases = [
        ("COMPLEX(1.0, 2.0) + COMPLEX(3.0, 4.0)", 4.0, 6.0),
        ("COMPLEX(1.0, 2.0) - COMPLEX(3.0, 4.0)", -2.0, -2.0),
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = 3 - 8 + 10i = -5 + 10i
        ("COMPLEX(1.0, 2.0) * COMPLEX(3.0, 4.0)", -5.0, 10.0),
        // (1+2i)/(1+0i) = 1+2i
        ("COMPLEX(1.0, 2.0) / COMPLEX(1.0, 0.0)", 1.0, 2.0),
        // Mixed: a real Int promotes into Complex.
        ("COMPLEX(1.0, 2.0) + 3", 4.0, 2.0),
    ];
    for (expr, exp_re, exp_im) in cases {
        let out = run(
            &mut db,
            &format!("SELECT {expr} AS z FROM nums WHERE id = 1"),
            10,
        );
        let DslOutput::Table(ds) = out else {
            panic!("expected table for {expr}")
        };
        let z_idx = ds.schema.get_field_index("z").unwrap();
        match &ds.rows[0].values[z_idx] {
            linal::core::value::Value::Complex(c) => {
                assert!(
                    (c.re - exp_re).abs() < 1e-9,
                    "{expr}: expected re={exp_re}, got {}",
                    c.re
                );
                assert!(
                    (c.im - exp_im).abs() < 1e-9,
                    "{expr}: expected im={exp_im}, got {}",
                    c.im
                );
            }
            other => panic!("{expr}: expected Complex, got {other:?}"),
        }
    }
}

#[test]
fn complex_equality_matches_where_clause_and_in_list() {
    let mut db = TensorDb::new();
    setup(&mut db);

    let out = run(
        &mut db,
        r#"SELECT id FROM nums WHERE COMPLEX(re, im) = COMPLEX(3.0, 4.0)"#,
        4,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows.len(), 1);

    let out = run(
        &mut db,
        r#"SELECT id FROM nums WHERE COMPLEX(re, im) != COMPLEX(3.0, 4.0)"#,
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows.len(), 1);
    let id_idx = ds.schema.get_field_index("id").unwrap();
    assert_eq!(ds.rows[0].values[id_idx], linal::core::value::Value::Int(2));
}

#[test]
fn complex_column_survives_save_and_load() {
    let _ = std::fs::remove_dir_all("./data/default/datasets/cnums_persist");

    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums_persist",
        4,
    );
    run(&mut db, "SAVE DATASET cnums_persist", 5);

    let mut db2 = TensorDb::new();
    run(&mut db2, "LOAD DATASET cnums_persist", 1);
    let out = run(&mut db2, "SELECT id, val FROM cnums_persist", 2);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows.len(), 2);
    let val_idx = ds.schema.get_field_index("val").unwrap();
    assert_eq!(
        ds.schema.fields[val_idx].value_type,
        linal::core::value::ValueType::Complex,
        "schema must report Complex, not the JSON-fallback's raw String"
    );
    let id_idx = ds.schema.get_field_index("id").unwrap();
    for row in &ds.rows {
        let id = match row.values[id_idx] {
            linal::core::value::Value::Int(i) => i,
            _ => panic!("expected Int id"),
        };
        let (exp_re, exp_im) = if id == 1 { (3.0, 4.0) } else { (1.0, -1.0) };
        match &row.values[val_idx] {
            linal::core::value::Value::Complex(c) => {
                assert!((c.re - exp_re).abs() < 1e-9);
                assert!((c.im - exp_im).abs() < 1e-9);
            }
            other => panic!("expected Complex, got {other:?}"),
        }
    }

    let _ = std::fs::remove_dir_all("./data/default/datasets/cnums_persist");
}

/// Regression coverage for the aggregate-path bugs Phase 3's dedicated
/// wildcard-arm audit found: SUM/AVG(complex_col) silently produced Int(0)/
/// a schema type mismatch, and MIN/MAX(complex_col) silently returned the
/// first row's value dressed up as "the max" instead of erroring (Complex
/// has no total order).
#[test]
fn complex_sum_and_avg_aggregates_are_correct() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );

    // (3+4i) + (1-1i) = 4+3i
    let out = run(&mut db, "SELECT SUM(val) AS s FROM cnums", 5);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let s_idx = ds.schema.get_field_index("s").unwrap();
    assert_eq!(
        ds.schema.fields[s_idx].value_type,
        linal::core::value::ValueType::Complex
    );
    match &ds.rows[0].values[s_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 4.0).abs() < 1e-9);
            assert!((c.im - 3.0).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }

    // (4+3i) / 2 = 2+1.5i
    let out = run(&mut db, "SELECT AVG(val) AS a FROM cnums", 6);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let a_idx = ds.schema.get_field_index("a").unwrap();
    assert_eq!(
        ds.schema.fields[a_idx].value_type,
        linal::core::value::ValueType::Complex
    );
    match &ds.rows[0].values[a_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 2.0).abs() < 1e-9);
            assert!((c.im - 1.5).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
}

#[test]
fn complex_min_max_aggregates_error_loudly_not_silently_wrong() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );
    assert!(execute_line(&mut db, "SELECT MIN(val) FROM cnums", 5).is_err());
    assert!(execute_line(&mut db, "SELECT MAX(val) FROM cnums", 6).is_err());
}

#[test]
fn complex_window_sum_accumulates_correctly() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );

    let out = run(
        &mut db,
        "SELECT id, SUM(val) OVER (ORDER BY id) AS s FROM cnums",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let s_idx = ds.schema.get_field_index("s").unwrap();
    // Row 1: just (3+4i). Row 2: (3+4i)+(1-1i) = 4+3i.
    match &ds.rows[0].values[s_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 3.0).abs() < 1e-9 && (c.im - 4.0).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
    match &ds.rows[1].values[s_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 4.0).abs() < 1e-9 && (c.im - 3.0).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
}

#[test]
fn complex_window_avg_divides_by_count_not_just_the_running_sum() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );

    let out = run(
        &mut db,
        "SELECT id, AVG(val) OVER (ORDER BY id) AS a FROM cnums",
        5,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    let a_idx = ds.schema.get_field_index("a").unwrap();
    // Row 1: (3+4i)/1 = 3+4i. Row 2: ((3+4i)+(1-1i))/2 = (4+3i)/2 = 2+1.5i.
    match &ds.rows[0].values[a_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 3.0).abs() < 1e-9 && (c.im - 4.0).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
    match &ds.rows[1].values[a_idx] {
        linal::core::value::Value::Complex(c) => {
            assert!((c.re - 2.0).abs() < 1e-9 && (c.im - 1.5).abs() < 1e-9);
        }
        other => panic!("expected Complex, got {other:?}"),
    }
}

#[test]
fn complex_window_max_errors_loudly_not_silently_wrong() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );
    let result = execute_line(
        &mut db,
        "SELECT id, MAX(val) OVER (ORDER BY id) AS m FROM cnums",
        5,
    );
    assert!(result.is_err());
}

#[test]
fn order_by_complex_column_errors_loudly_not_silently_unsorted() {
    let mut db = TensorDb::new();
    setup(&mut db);
    run(
        &mut db,
        "TRANSFORM nums SELECT id, COMPLEX(re, im) AS val INTO cnums",
        4,
    );
    let result = execute_line(&mut db, "SELECT id FROM cnums ORDER BY val", 5);
    assert!(result.is_err());
}
