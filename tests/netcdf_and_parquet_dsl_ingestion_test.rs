//! End-to-end DSL-level ingestion for the two new Phase 1 connectors --
//! SCIENTIFIC_ENGINE_EXPANSION_PLAN.md. `tests/netcdf_connector_test.rs` and
//! `tests/parquet_connector_test.rs` exercise the raw `Connector` trait directly; this proves
//! the same files work through the real `USE DATASET FROM`/`IMPORT DATASET FROM` DSL statements,
//! same as `tests/ingestion_test.rs` does for CSV.

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hdf5::types::VarLenUnicode;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use parquet::arrow::arrow_writer::ArrowWriter;
use std::str::FromStr;
use std::sync::Arc;

#[test]
fn use_dataset_from_netcdf_ingests_and_applies_cf_decoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.nc");

    let file = hdf5::File::create(&path).unwrap();
    let raw: Vec<f32> = vec![0.0, 100.0];
    let ds = file
        .new_dataset::<f32>()
        .shape(raw.len())
        .create("temperature")
        .unwrap();
    ds.write(&raw).unwrap();
    ds.new_attr::<f64>()
        .create("scale_factor")
        .unwrap()
        .write_scalar(&0.1)
        .unwrap();
    ds.new_attr::<f64>()
        .create("add_offset")
        .unwrap()
        .write_scalar(&273.15)
        .unwrap();
    ds.new_attr::<VarLenUnicode>()
        .create("units")
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str("kelvin").unwrap())
        .unwrap();
    drop(ds);
    drop(file);

    let mut db = TensorDb::new();
    let cmd = format!(
        r#"USE DATASET FROM "{}" AS climate"#,
        path.to_str().unwrap()
    );
    let out = execute_line(&mut db, &cmd, 1).expect("USE DATASET FROM .nc should succeed");

    match out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 2);
        }
        other => panic!("expected Table output, got {other:?}"),
    }

    // Same 0.1 scale_factor / 273.15 add_offset applied by the connector test --
    // 0.0 -> 273.15, 100.0 -> 283.15.
    let names = db.active_instance().list_names();
    assert!(names.contains(&"climate_temperature".to_string()));
    let tensor = db.get("climate_temperature").unwrap();
    let data = tensor.to_logical_vec();
    assert!((data[0] - 273.15).abs() < 1e-3);
    assert!((data[1] - 283.15).abs() < 1e-3);
}

#[test]
fn import_dataset_from_parquet_round_trips_through_load_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let mut db = TensorDb::new();
    let _ = std::fs::remove_dir_all("./data/default/datasets/parquet_import_ds");

    let cmd = format!(
        r#"IMPORT DATASET FROM "{}" AS parquet_import_ds"#,
        path.to_str().unwrap()
    );
    let out = execute_line(&mut db, &cmd, 1).expect("IMPORT DATASET FROM .parquet should succeed");
    match out {
        DslOutput::Message(msg) => assert!(msg.contains("Imported dataset 'parquet_import_ds'")),
        other => panic!("expected Message output, got {other:?}"),
    }

    let load_out = execute_line(&mut db, "LOAD DATASET parquet_import_ds", 2)
        .expect("LOAD DATASET should find the imported package");
    match load_out {
        DslOutput::Message(msg) => assert!(msg.contains("parquet_import_ds")),
        other => panic!("expected Message output, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all("./data/default/datasets/parquet_import_ds");
}
