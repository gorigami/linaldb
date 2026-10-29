// tests/drop_dataset_test.rs
//
// DROP DATASET [IF EXISTS] <name>: frees a dataset (rows, indexes, mmap'd
// columns) so the name can be loaded again in the same session -- the
// workflow `load_record_batch` already pointed users to ("DROP it first").
// A saved package on disk is not touched.

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use linal::core::config::{EngineConfig, WalConfig, WalSync};
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::path::Path;
use std::sync::Arc;

fn open(dir: &Path, wal: bool) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.to_path_buf();
    if wal {
        config.wal = WalConfig {
            enabled: true,
            sync: WalSync::Always,
            ..WalConfig::default()
        };
    }
    TensorDb::with_config(config)
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn message(out: DslOutput) -> String {
    match out {
        DslOutput::Message(m) => m,
        other => panic!("expected a message, got {:?}", other),
    }
}

/// `n` rows of (id, 4-dim vector).
fn spectra(n: usize) -> RecordBatch {
    let ids: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
    let values: Vec<f32> = (0..n * 4).map(|i| (i % 7) as f32 + 1.0).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vecs: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(item, 4, Arc::new(Float32Array::from(values)), None).unwrap(),
    );
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("e", vecs.data_type().clone(), false),
    ]);
    RecordBatch::try_new(Arc::new(schema), vec![ids, vecs]).unwrap()
}

/// Names that appear in SHOW MEMORY (datasets and their indexes).
fn memory_names(db: &mut TensorDb) -> Vec<String> {
    match run(db, "SHOW MEMORY") {
        DslOutput::Table(ds) => ds
            .rows
            .iter()
            .filter_map(|r| match &r.values[1] {
                Value::String(name) => Some(name.clone()),
                _ => None,
            })
            .collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

#[test]
fn drop_frees_the_name_and_its_indexes_for_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path(), false);

    db.load_record_batch("lib", &spectra(50), "test").unwrap();
    run(&mut db, "CREATE VECTOR INDEX ON lib(e) USING HNSW");
    let err = db
        .load_record_batch("lib", &spectra(10), "test")
        .unwrap_err();
    assert!(err.to_string().contains("DROP DATASET lib"), "{err}");
    assert!(memory_names(&mut db).contains(&"lib".to_string()));
    let before = message(run(&mut db, "SHOW INDEXES lib"));
    assert!(before.contains("HNSW"), "{before}");

    assert_eq!(
        message(run(&mut db, "DROP DATASET lib")),
        "Dropped dataset: lib"
    );
    assert!(!memory_names(&mut db).contains(&"lib".to_string()));
    assert!(execute_line(&mut db, "SELECT * FROM lib", 1).is_err());

    // Reload under the same name: new rows, no stale index.
    db.load_record_batch("lib", &spectra(10), "test").unwrap();
    match run(&mut db, "SELECT * FROM lib") {
        DslOutput::Table(ds) => assert_eq!(ds.rows.len(), 10),
        other => panic!("expected a table, got {:?}", other),
    }
    let indexes = message(run(&mut db, "SHOW INDEXES lib"));
    assert!(!indexes.contains("HNSW"), "{indexes}");
}

#[test]
fn if_exists_skips_a_missing_dataset_and_plain_drop_fails() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path(), false);

    let skipped = message(run(&mut db, "DROP DATASET IF EXISTS nope"));
    assert!(skipped.contains("not found (skipped)"), "{skipped}");
    let err = execute_line(&mut db, "DROP DATASET nope", 1).unwrap_err();
    assert!(err.to_string().contains("nope"), "{err}");
}

#[test]
fn drop_leaves_the_saved_package_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path(), false);

    run(&mut db, "DATASET items COLUMNS (id: Int, score: Float)");
    run(&mut db, "INSERT INTO items VALUES (1, 0.5)");
    run(&mut db, "SAVE DATASET items");
    run(&mut db, "DROP DATASET items");
    assert!(execute_line(&mut db, "SELECT * FROM items", 1).is_err());

    run(&mut db, "LOAD DATASET items");
    match run(&mut db, "SELECT * FROM items") {
        DslOutput::Table(ds) => assert_eq!(ds.rows.len(), 1),
        other => panic!("expected a table, got {:?}", other),
    }
}

#[test]
fn drop_removes_a_tensor_first_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path(), false);

    run(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]");
    run(&mut db, "LET ds = dataset(\"built\")");
    run(&mut db, "ds.add_column(\"vec1\", v1)");
    assert!(db.get_tensor_dataset("built").is_some());

    run(&mut db, "DROP DATASET built");
    assert!(db.get_tensor_dataset("built").is_none());
    assert!(execute_line(&mut db, "SELECT * FROM built", 1).is_err());
}

#[test]
fn drop_survives_a_restart_with_the_wal() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = open(dir.path(), true);
        run(&mut db, "DATASET items COLUMNS (id: Int, score: Float)");
        run(&mut db, "INSERT INTO items VALUES (1, 0.5)");
        run(&mut db, "DATASET kept COLUMNS (id: Int)");
        run(&mut db, "DROP DATASET items");
    }
    let mut db = open(dir.path(), true);
    assert!(execute_line(&mut db, "SELECT * FROM items", 1).is_err());
    run(&mut db, "SELECT * FROM kept");
}
