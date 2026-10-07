// tests/spectral_similarity_test.rs
//
// Peak-list spectra in `Matrix(2, *)` columns and SPEC_COSINE /
// SPEC_COSINE_MOD / SPEC_MATCHES (CASMI_WORKLOADS_PLAN.md, P5). Agreement
// with matchms itself is checked in clients/python-embedded's tests; these
// cover the DSL, storage and error paths with hand-computable cases.

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn run_err(db: &mut TensorDb, line: &str) -> String {
    match execute_line(db, line, 1) {
        Ok(out) => panic!("`{}` should have failed, got {:?}", line, out),
        Err(e) => e.to_string(),
    }
}

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

fn f64v(v: &Value) -> f64 {
    match v {
        Value::Float64(x) => *x,
        other => panic!("expected Float64, got {:?}", other),
    }
}

fn setup(db: &mut TensorDb) {
    run(
        db,
        "DATASET lib COLUMNS (id: Int, pm: DOUBLE, spec: Matrix(2, *))",
    );
    // Same peaks; a 3-peak spectrum; one shifted by +20 above m/z 150; a 1-peak one.
    run(
        db,
        "INSERT INTO lib VALUES (1, 300.0, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]])",
    );
    run(
        db,
        "INSERT INTO lib VALUES (2, 300.0, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]])",
    );
    run(
        db,
        "INSERT INTO lib VALUES (3, 320.0, [[100.0, 170.0, 220.0], [1.0, 1.0, 1.0]])",
    );
    run(db, "INSERT INTO lib VALUES (4, 300.0, [[100.0], [2.0]])");
    run(
        db,
        "DATASET q COLUMNS (k: Int, qpm: DOUBLE, qspec: Matrix(2, *))",
    );
    run(
        db,
        "INSERT INTO q VALUES (1, 300.0, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]])",
    );
    run(db, "DATASET one COLUMNS (k: Int)");
    run(db, "INSERT INTO one VALUES (1)");
}

#[test]
fn cosine_and_modified_cosine_by_hand() {
    let (_dir, mut db) = db();
    setup(&mut db);
    let r = rows(run(
        &mut db,
        "SELECT id, SPEC_COSINE(spec, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]], 0.1) AS c, SPEC_COSINE_MOD(spec, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]], 0.1, pm - 300.0) AS m, SPEC_MATCHES(spec, [[100.0, 150.0, 200.0], [1.0, 1.0, 1.0]], 0.1) AS n FROM lib ORDER BY id",
    ));
    // identical
    assert!((f64v(&r[0][1]) - 1.0).abs() < 1e-12);
    // spectrum 3 shares only m/z 100 unshifted: 1 / (sqrt3 * sqrt3) = 1/3
    assert!((f64v(&r[2][1]) - 1.0 / 3.0).abs() < 1e-12);
    // with the +20 precursor shift, 170 and 220 match 150 and 200: 3/3
    assert!((f64v(&r[2][2]) - 1.0).abs() < 1e-12);
    assert_eq!(r[2][3], Value::Int(1));
    // one peak of intensity 2 vs three of 1: 2 / (2 * sqrt3)
    assert!((f64v(&r[3][1]) - 1.0 / 3f64.sqrt()).abs() < 1e-12);
}

#[test]
fn rank_library_spectra_against_queries_with_a_join() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(
        &mut db,
        "DATASET lib2 COLUMNS (id: Int, k: Int, spec: Matrix(2, *))",
    );
    for (id, s) in [
        (1, "[[100.0, 150.0], [1.0, 1.0]]"),
        (2, "[[100.0, 151.0], [1.0, 1.0]]"),
        (3, "[[99.0, 150.0], [1.0, 0.1]]"),
    ] {
        run(
            &mut db,
            &format!("INSERT INTO lib2 VALUES ({}, 1, {})", id, s),
        );
    }
    let r = rows(run(
        &mut db,
        "SELECT id, SPEC_COSINE(spec, qspec, 0.1) AS c FROM lib2 JOIN q ON lib2.k = q.k WHERE SPEC_COSINE(spec, qspec, 0.1) > 0.0 ORDER BY c DESC",
    ));
    let ids: Vec<Value> = r.iter().map(|row| row[0].clone()).collect();
    assert_eq!(ids, vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
}

#[test]
fn variable_width_spectra_round_trip_through_parquet() {
    let (dir, mut db) = db();
    setup(&mut db);
    run(&mut db, "SAVE DATASET lib");
    let before = rows(run(&mut db, "SELECT * FROM lib ORDER BY id"));

    let file = std::fs::File::open(dir.path().join("default/datasets/lib/data.parquet")).unwrap();
    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let field = reader.schema().field_with_name("spec").unwrap().clone();
    assert!(
        matches!(field.data_type(), arrow::datatypes::DataType::FixedSizeList(inner, 2) if matches!(inner.data_type(), arrow::datatypes::DataType::List(_))),
        "expected a native FixedSizeList<List<Float32>, 2>, got {:?}",
        field.data_type()
    );

    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db2 = TensorDb::with_config(config);
    run(&mut db2, "LOAD DATASET lib");
    assert_eq!(rows(run(&mut db2, "SELECT * FROM lib ORDER BY id")), before);
    let types: Vec<String> = db2
        .get_dataset("lib")
        .unwrap()
        .schema
        .fields
        .iter()
        .map(|f| f.value_type.to_string())
        .collect();
    assert_eq!(types[2], "MATRIX[2, 0]");
}

#[test]
fn loud_errors() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(
        &mut db,
        "INSERT INTO lib VALUES (9, 300.0, [[200.0, 100.0], [1.0, 1.0]])",
    );
    let e = run_err(
        &mut db,
        "SELECT id, SPEC_COSINE(spec, spec, 0.1) AS c FROM lib",
    );
    assert!(
        e.contains("SPEC_COSINE: first spectrum: m/z values must be sorted ascending"),
        "{}",
        e
    );
    let e = run_err(
        &mut db,
        "SELECT id FROM lib WHERE SPEC_COSINE(spec, spec, 0.1) > 0.5",
    );
    assert!(e.contains("sorted ascending"), "{}", e);
    let e = run_err(
        &mut db,
        "SELECT SPEC_COSINE(spec, spec, -1.0) AS c FROM lib WHERE id = 1",
    );
    assert!(
        e.contains("tolerance must be a finite, non-negative number"),
        "{}",
        e
    );
    let e = run_err(&mut db, "SELECT SPEC_COSINE(spec, spec) AS c FROM lib");
    assert!(e.contains("takes 3 to 5 arguments"), "{}", e);
    let e = run_err(&mut db, "SELECT SPEC_COSINE(id, spec, 0.1) AS c FROM lib");
    assert!(e.contains("argument 1 must be a peak list"), "{}", e);
    let e = run_err(
        &mut db,
        "INSERT INTO lib VALUES (10, 1.0, [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])",
    );
    assert!(e.to_lowercase().contains("mismatch"), "{}", e);
    // A failing UPDATE changes nothing.
    let before = rows(run(&mut db, "SELECT pm FROM lib ORDER BY id"));
    let e = run_err(&mut db, "UPDATE lib SET pm = SPEC_COSINE(spec, spec, 0.1)");
    assert!(e.contains("sorted ascending"), "{}", e);
    assert_eq!(rows(run(&mut db, "SELECT pm FROM lib ORDER BY id")), before);
}
