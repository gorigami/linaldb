// tests/memory_load_test.rs
//
// TensorDb::load_record_batch: creating a dataset straight from in-memory
// Arrow data (CASMI_WORKLOADS_PLAN.md, P1) -- the path the Python binding's
// load_numpy/load_arrow use.

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeStringArray, ListArray, StringArray,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::sync::Arc;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn vector_column(values: &[f32], dim: i32) -> ArrayRef {
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    Arc::new(
        FixedSizeListArray::try_new(
            field,
            dim,
            Arc::new(Float32Array::from(values.to_vec())),
            None,
        )
        .unwrap(),
    )
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), a.null_count() > 0))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

/// Values chosen to catch any lossy conversion: subnormals, max, -0.0,
/// and digits a decimal round trip would mangle.
fn tricky_vectors() -> Vec<f32> {
    vec![
        0.1,
        -0.0,
        f32::MAX,
        f32::MIN_POSITIVE,
        1.0e-40,
        3.4028e38,
        1.0 / 3.0,
        -7.25,
        0.0,
        2.5e-7,
        123456.79,
        -1.1754942e-38,
    ]
}

#[test]
fn round_trip_is_bit_exact_and_types_map() {
    let (_dir, mut db) = db();
    let v = tricky_vectors();
    let b = batch(vec![
        (
            "id",
            Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
        ),
        (
            "small",
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        ),
        (
            "mass",
            Arc::new(Float64Array::from(vec![180.0634, 0.1, 1e300])) as ArrayRef,
        ),
        (
            "f",
            Arc::new(Float32Array::from(vec![0.5, -0.0, 1.0e-40])) as ArrayRef,
        ),
        (
            "name",
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ),
        (
            "big",
            Arc::new(LargeStringArray::from(vec!["x", "y", "z"])) as ArrayRef,
        ),
        (
            "ok",
            Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
        ),
        ("e", vector_column(&v, 4)),
    ]);
    assert_eq!(db.load_record_batch("spec", &b, "test").unwrap(), 3);

    let ds = db.get_dataset("spec").unwrap();
    let types: Vec<String> = ds
        .schema
        .fields
        .iter()
        .map(|f| format!("{:?}", f.value_type))
        .collect();
    assert_eq!(
        types,
        vec![
            "Int",
            "Int",
            "Float64",
            "Float",
            "String",
            "String",
            "Bool",
            "Vector(4)"
        ]
    );
    for (r, row) in ds.rows.iter().enumerate() {
        match &row.values[7] {
            Value::Vector(got) => {
                let want = &v[r * 4..(r + 1) * 4];
                let bits = |x: &[f32]| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(got), bits(want), "row {}", r);
            }
            other => panic!("expected a vector, got {:?}", other),
        }
    }
    assert_eq!(ds.rows[2].values[2], Value::Float64(1e300));
    assert_eq!(ds.rows[1].values[3], Value::Float(-0.0));
    assert_eq!(ds.rows[0].values[5], Value::String("x".into()));

    // Immediately queryable through the DSL.
    match run(&mut db, "SELECT id FROM spec WHERE mass > 1.0 ORDER BY id") {
        DslOutput::Table(t) => assert_eq!(t.rows.len(), 2),
        other => panic!("{:?}", other),
    }
}

#[test]
fn matrix_column_and_nulls() {
    let (_dir, mut db) = db();
    let inner = Arc::new(Field::new("item", DataType::Float32, false));
    let rows_of_2 = FixedSizeListArray::try_new(
        inner,
        2,
        Arc::new(Float32Array::from(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ])),
        None,
    )
    .unwrap();
    let outer_field = Arc::new(Field::new("item", rows_of_2.data_type().clone(), false));
    let matrices: ArrayRef =
        Arc::new(FixedSizeListArray::try_new(outer_field, 2, Arc::new(rows_of_2), None).unwrap());
    let b = batch(vec![
        ("m", matrices),
        (
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), None])) as ArrayRef,
        ),
    ]);
    db.load_record_batch("mats", &b, "test").unwrap();
    let ds = db.get_dataset("mats").unwrap();
    assert_eq!(
        ds.rows[1].values[0],
        Value::Matrix(vec![vec![5.0, 6.0], vec![7.0, 8.0]])
    );
    assert_eq!(ds.rows[1].values[1], Value::Null);
    assert!(ds.schema.fields[1].nullable);
}

#[test]
fn rejects_nan_inf_unsupported_types_and_duplicates() {
    let (_dir, mut db) = db();
    let err = |db: &mut TensorDb, b: &RecordBatch| {
        db.load_record_batch("x", b, "test")
            .unwrap_err()
            .to_string()
    };

    let e = err(
        &mut db,
        &batch(vec![("e", vector_column(&[1.0, 2.0, f32::NAN, 4.0], 2))]),
    );
    assert!(e.contains("column 'e' row 1") && e.contains("NaN"), "{}", e);

    let e = err(
        &mut db,
        &batch(vec![(
            "m",
            Arc::new(Float64Array::from(vec![1.0, f64::INFINITY])) as ArrayRef,
        )]),
    );
    assert!(e.contains("column 'm' row 1"), "{}", e);

    let f64_list: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float64, false)),
            2,
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            None,
        )
        .unwrap(),
    );
    let e = err(&mut db, &batch(vec![("e", f64_list)]));
    assert!(e.contains("cast to float32"), "{}", e);

    let ragged: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
        Some(vec![Some(1.0)]),
        Some(vec![Some(1.0), Some(2.0)]),
    ]));
    let e = err(&mut db, &batch(vec![("e", ragged)]));
    assert!(e.contains("fixed size"), "{}", e);

    let ok = batch(vec![(
        "id",
        Arc::new(Int64Array::from(vec![1])) as ArrayRef,
    )]);
    db.load_record_batch("x", &ok, "test").unwrap();
    let e = err(&mut db, &ok);
    assert!(e.contains("already exists"), "{}", e);
}

#[test]
fn lineage_records_the_origin_and_hash() {
    let (_dir, mut db) = db();
    let b = batch(vec![(
        "id",
        Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
    )]);
    db.load_record_batch("loaded", &b, "numpy").unwrap();
    run(&mut db, "DATASET derived FROM loaded FILTER id > 1");
    let text = match run(&mut db, "EXPLAIN LINEAGE derived AS JSON") {
        DslOutput::Message(m) => m,
        other => format!("{:?}", other),
    };
    assert!(text.contains("LOAD FROM MEMORY"), "{}", text);
    assert!(text.contains("numpy"), "{}", text);
    let hash = db.get_dataset("loaded").unwrap().content_hash();
    assert!(text.contains(&hash), "{}", text);
}

#[test]
fn matches_import_then_load_of_the_same_parquet() {
    let (dir, mut db) = db();
    let v: Vec<f32> = (0..24).map(|i| (i as f32) * 0.37 - 3.1).collect();
    let b = batch(vec![
        (
            "id",
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
        ),
        (
            "mass",
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])) as ArrayRef,
        ),
        ("e", vector_column(&v, 4)),
    ]);
    let path = dir.path().join("spectra.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, b.schema(), None).unwrap();
    writer.write(&b).unwrap();
    writer.close().unwrap();

    db.load_record_batch("from_memory", &b, "arrow").unwrap();
    run(
        &mut db,
        &format!("IMPORT DATASET FROM \"{}\" AS from_file", path.display()),
    );
    run(&mut db, "LOAD DATASET from_file");

    let a = db.get_dataset("from_memory").unwrap();
    let f = db.get_dataset("from_file").unwrap();
    let types = |d: &linal::Dataset| {
        d.schema
            .fields
            .iter()
            .map(|x| (x.name.clone(), x.value_type.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(types(a), types(f));
    let values = |d: &linal::Dataset| d.rows.iter().map(|r| r.values.clone()).collect::<Vec<_>>();
    assert_eq!(values(a), values(f));
}

#[test]
fn with_the_wal_a_load_survives_a_restart() {
    use linal::core::config::{WalConfig, WalSync};
    let dir = tempfile::tempdir().unwrap();
    let open = || {
        let mut config = EngineConfig::default();
        config.storage.data_dir = dir.path().to_path_buf();
        config.wal = WalConfig {
            enabled: true,
            sync: WalSync::Always,
            ..WalConfig::default()
        };
        TensorDb::with_config(config)
    };
    let v: Vec<f32> = vec![0.25, -1.5, 3.0, 4.75];
    {
        let mut db = open();
        let b = batch(vec![
            ("id", Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef),
            ("e", vector_column(&v, 2)),
        ]);
        db.load_record_batch("loaded", &b, "numpy").unwrap();
        run(&mut db, "INSERT INTO loaded VALUES (3, [9.0, 9.5])");
    }
    let mut db = open();
    let ds = db.get_dataset("loaded").unwrap();
    assert_eq!(ds.rows.len(), 3);
    assert_eq!(ds.rows[0].values[1], Value::Vector(vec![0.25, -1.5]));
    assert_eq!(ds.rows[2].values[1], Value::Vector(vec![9.0, 9.5]));
    match run(&mut db, "SELECT id FROM loaded") {
        DslOutput::Table(t) => assert_eq!(t.rows.len(), 3),
        other => panic!("{:?}", other),
    }
}
