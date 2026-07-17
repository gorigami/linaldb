use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::fs;

#[test]
fn test_use_dataset_from_csv() {
    let mut db = TensorDb::new();

    // 1. Create a dummy CSV file
    let temp_dir = std::env::temp_dir();
    let csv_path_buf = temp_dir.join("test_ingestion_use.csv");
    let csv_path = csv_path_buf.to_str().unwrap();
    let csv_content = "val1,val2\n1.0,2.0\n3.0,4.0\n";
    fs::write(csv_path, csv_content).unwrap();

    // 2. Test USE DATASET FROM
    let use_cmd = format!(r#"USE DATASET FROM "{}" AS use_ds"#, csv_path);
    let out = execute_line(&mut db, &use_cmd, 1).expect("Failed to execute USE DATASET FROM");

    // Verify it returns a table (materialized)
    match out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 2);
            assert_eq!(table.schema.fields.len(), 2);
            assert_eq!(table.schema.fields[0].name, "val1");
            assert_eq!(table.schema.fields[1].name, "val2");
        }
        _ => panic!("Expected Table output, got {:?}", out),
    }

    // Verify tensors are registered in store
    let names = db.active_instance().list_names();
    assert!(names.contains(&"use_ds_val1".to_string()));
    assert!(names.contains(&"use_ds_val2".to_string()));

    // Cleanup
    let _ = fs::remove_file(csv_path);
}

#[test]
fn test_import_dataset_from_csv() {
    let mut db = TensorDb::new();
    let _ = fs::remove_dir_all("./data/default/datasets/import_ds");

    // 1. Create a dummy CSV file
    let temp_dir = std::env::temp_dir();
    let csv_path_buf = temp_dir.join("test_ingestion_import.csv");
    let csv_path = csv_path_buf.to_str().unwrap();
    let csv_content = "val1,val2\n10.0,20.0\n";
    fs::write(csv_path, csv_content).unwrap();

    // 2. Test IMPORT DATASET FROM
    let import_cmd = format!(r#"IMPORT DATASET FROM "{}" AS import_ds"#, csv_path);
    let out = execute_line(&mut db, &import_cmd, 1).expect("Failed to execute IMPORT DATASET FROM");

    match out {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Imported dataset 'import_ds'"));
        }
        _ => panic!("Expected Message output, got {:?}", out),
    }

    // Regression test: IMPORT DATASET FROM used to write a package that
    // LOAD DATASET could never find (missing legacy .meta.json sidecar) —
    // it reported success but the dataset was invisible to SHOW ALL
    // DATASETS / LOAD DATASET despite the data being correctly on disk.
    let load_out =
        execute_line(&mut db, "LOAD DATASET import_ds", 2).expect("LOAD DATASET should succeed");
    match load_out {
        DslOutput::Message(msg) => {
            assert!(msg.contains("import_ds"));
        }
        other => panic!("Expected Message output, got {other:?}"),
    }

    let loaded = db
        .get_dataset("import_ds")
        .expect("import_ds should be loaded into the active instance");
    assert_eq!(loaded.rows.len(), 1);
    assert_eq!(loaded.schema.fields.len(), 2);
    assert_eq!(loaded.schema.fields[0].name, "val1");
    assert_eq!(loaded.schema.fields[1].name, "val2");

    // Cleanup
    let _ = fs::remove_file(csv_path);
    let _ = fs::remove_dir_all("./data/default/datasets/import_ds");
}

/// Regression test for a silent-data-loss bug found while auditing all
/// connectors after the v0.1.55/v0.1.56 fixes: `read_dataset` can skip
/// individual fields it couldn't reconcile (shape/length mismatch, or an
/// unsupported dtype) with zero indication -- `import_dataset_core` used to
/// swallow that entirely. It now appends a warning to the returned
/// `Message`, exercised here end-to-end through the DSL (connector-level
/// coverage of the underlying skip logic itself lives in
/// `tests/connector_silent_skip_test.rs`).
#[test]
fn test_import_dataset_from_hdf5_surfaces_skip_warning() {
    use ndarray::{Array1, Array2};

    let mut db = TensorDb::new();
    let _ = fs::remove_dir_all("./data/default/datasets/import_mismatch_ds");

    let temp_dir = std::env::temp_dir();
    let h5_path_buf = temp_dir.join("test_ingestion_import_mismatch.h5");
    let h5_path = h5_path_buf.to_str().unwrap();

    let file = hdf5::File::create(h5_path).unwrap();
    let a: Array2<f32> = Array2::from_shape_fn((5, 3), |(r, c)| (r * 3 + c) as f32);
    file.new_dataset::<f32>()
        .shape((5, 3))
        .create("matrix_a")
        .unwrap()
        .write(&a)
        .unwrap();
    let b: Array1<f32> = Array1::from_shape_fn(7, |i| i as f32);
    file.new_dataset::<f32>()
        .shape(7)
        .create("vector_b")
        .unwrap()
        .write(&b)
        .unwrap();
    drop(file);

    let import_cmd = format!(r#"IMPORT DATASET FROM "{}" AS import_mismatch_ds"#, h5_path);
    let out = execute_line(&mut db, &import_cmd, 1).expect("Failed to execute IMPORT DATASET FROM");

    match out {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Imported dataset 'import_mismatch_ds'"));
            assert!(
                msg.contains("field(s) skipped during import"),
                "message should surface the skipped field, not silently succeed: {msg}"
            );
            assert!(
                msg.contains("matrix_a") || msg.contains("vector_b"),
                "message should name the skipped field: {msg}"
            );
        }
        other => panic!("Expected Message output, got {other:?}"),
    }

    let _ = fs::remove_file(h5_path);
    let _ = fs::remove_dir_all("./data/default/datasets/import_mismatch_ds");
}
