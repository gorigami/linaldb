//! External Parquet ingestion connector integration tests --
//! SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 1. Writes a real Parquet file via the same
//! `arrow`/`parquet` writer this engine's own `ParquetStorage` uses (the connector under test
//! reads it back through `USE DATASET FROM`/`IMPORT DATASET FROM`'s connector-registry path,
//! a completely different code path from `ParquetStorage`/`LOAD DATASET`).

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use linal::dsl::persistence;
use parquet::arrow::arrow_writer::ArrowWriter;
use std::sync::Arc;

fn write_sample_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "carol"])),
            Arc::new(Float64Array::from(vec![9.5, 7.25, 8.0])),
        ],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn parquet_extension_routes_to_parquet_connector() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");
    write_sample_parquet(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry
        .find_connector(path.to_str().unwrap())
        .expect("a connector should be found for .parquet");
    assert_eq!(connector.name(), "parquet");
}

#[test]
fn parquet_connector_reads_generic_external_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");
    write_sample_parquet(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let (batch, _lineage) = connector
        .read_dataset(path.to_str().unwrap(), None)
        .expect("should read the .parquet file");

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);

    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["id", "name", "score"]);
}

#[test]
fn parquet_connector_honors_fields_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");
    write_sample_parquet(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let fields = vec!["score".to_string(), "id".to_string()];
    let (batch, _lineage) = connector
        .read_dataset(path.to_str().unwrap(), Some(&fields))
        .expect("should read with a FIELDS selection");

    assert_eq!(batch.num_columns(), 2);
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["score", "id"]);
}

#[test]
fn parquet_connector_fields_selection_rejects_unknown_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");
    write_sample_parquet(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let fields = vec!["does_not_exist".to_string()];
    let result = connector.read_dataset(path.to_str().unwrap(), Some(&fields));
    assert!(result.is_err());
}

#[test]
fn parquet_connector_inspect_matches_schema_without_reading_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.parquet");
    write_sample_parquet(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let schema = connector
        .inspect(path.to_str().unwrap())
        .expect("inspect should succeed");
    assert_eq!(schema.columns.len(), 3);
}
