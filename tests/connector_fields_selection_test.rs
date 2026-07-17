// Tests for the `FIELDS (name1, name2, ...)` clause added to
// `IMPORT DATASET FROM` / `USE DATASET FROM`, which lets a caller
// explicitly pick which named fields/arrays to ingest from a source that
// bundles several of different shapes -- the follow-up to v0.1.58's
// warn-and-drop fix (tests/connector_silent_skip_test.rs). Without
// `FIELDS`, a connector still keeps whichever fields share the
// first-encountered field's shape and warns about the rest; `FIELDS` makes
// the choice explicit, and turns "the requested fields don't share a
// combinable shape" into a hard error rather than a warned skip, since the
// caller has no fallback expectation once they've named exactly what they
// want.

use linal::core::connectors::csv_connector::CsvConnector;
use linal::core::connectors::hdf5_connector::Hdf5Connector;
use linal::core::connectors::zarr_connector::ZarrConnector;
use linal::core::connectors::Connector;
use linal::dsl::ast::Statement;
use linal::dsl::parser;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn make_mismatched_h5(path: &str) {
    let file = hdf5::File::create(path).unwrap();
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
}

#[test]
fn test_parser_import_dataset_fields_clause() {
    let stmt = parser::parse(r#"IMPORT DATASET FROM "x.h5" AS y FIELDS (a, b)"#).unwrap();
    match stmt {
        Statement::Import(s) => {
            assert!(!s.ephemeral);
            assert_eq!(s.name.as_deref(), Some("y"));
            assert_eq!(s.fields, Some(vec!["a".to_string(), "b".to_string()]));
        }
        other => panic!("Expected Import statement, got {other:?}"),
    }
}

#[test]
fn test_parser_use_dataset_fields_clause() {
    let stmt = parser::parse(r#"USE DATASET FROM "x.h5" AS y FIELDS (only_one)"#).unwrap();
    match stmt {
        Statement::Import(s) => {
            assert!(s.ephemeral);
            assert_eq!(s.fields, Some(vec!["only_one".to_string()]));
        }
        other => panic!("Expected Import statement, got {other:?}"),
    }
}

#[test]
fn test_parser_import_dataset_without_fields_clause_is_none() {
    let stmt = parser::parse(r#"IMPORT DATASET FROM "x.h5" AS y"#).unwrap();
    match stmt {
        Statement::Import(s) => assert_eq!(s.fields, None),
        other => panic!("Expected Import statement, got {other:?}"),
    }
}

#[test]
fn test_hdf5_fields_selects_previously_dropped_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.h5");
    let path_str = path.to_str().unwrap();
    make_mismatched_h5(path_str);

    // Without FIELDS: only matrix_a (first-encountered) survives, vector_b
    // is dropped with a warning (v0.1.58 behavior, unchanged).
    let (default_batch, default_lineage) = Hdf5Connector.read_dataset(path_str, None).unwrap();
    assert_eq!(default_batch.num_columns(), 1);
    assert_eq!(default_lineage.warnings.len(), 1);

    // With FIELDS (vector_b): the previously-dropped field is explicitly
    // selected and now included, with no warnings at all.
    let requested = vec!["vector_b".to_string()];
    let (batch, lineage) = Hdf5Connector
        .read_dataset(path_str, Some(&requested))
        .unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.schema().field(0).name(), "vector_b");
    assert_eq!(batch.num_rows(), 7);
    assert!(lineage.warnings.is_empty());
}

#[test]
fn test_hdf5_fields_missing_name_is_hard_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.h5");
    let path_str = path.to_str().unwrap();
    make_mismatched_h5(path_str);

    let requested = vec!["does_not_exist".to_string()];
    let err = Hdf5Connector
        .read_dataset(path_str, Some(&requested))
        .expect_err("requesting a nonexistent field must be a hard error, not an empty result");
    assert!(format!("{err}").contains("does_not_exist"));
}

#[test]
fn test_hdf5_fields_incompatible_lengths_is_hard_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.h5");
    let path_str = path.to_str().unwrap();
    make_mismatched_h5(path_str);

    // Explicitly requesting both fields together is a hard error (15 vs 7
    // elements can't become one RecordBatch), not a silent/warned skip --
    // the caller named exactly what they wanted.
    let requested = vec!["matrix_a".to_string(), "vector_b".to_string()];
    let err = Hdf5Connector
        .read_dataset(path_str, Some(&requested))
        .expect_err("explicitly requesting incompatible-shape fields must be a hard error");
    let msg = format!("{err}");
    assert!(msg.contains("FIELDS"));
    assert!(msg.contains('7') && msg.contains("15"));
}

#[test]
fn test_zarr_fields_selects_previously_dropped_array() {
    use zarrs::array::{ArrayBuilder, DataType, FillValue};
    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::GroupBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.zarr");
    let path_str = path.to_str().unwrap();

    let store = Arc::new(FilesystemStore::new(path_str).unwrap());
    let group = GroupBuilder::new().build(store.clone(), "/").unwrap();
    group.store_metadata().unwrap();

    let array_a = ArrayBuilder::new(
        vec![6],
        DataType::Float32,
        vec![6].try_into().unwrap(),
        FillValue::from(0.0f32),
    )
    .build(store.clone(), "/array_a")
    .unwrap();
    array_a.store_metadata().unwrap();
    let data_a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    array_a
        .store_array_subset_elements(&ArraySubset::new_with_shape(vec![6]), &data_a)
        .unwrap();

    let array_b = ArrayBuilder::new(
        vec![10],
        DataType::Float32,
        vec![10].try_into().unwrap(),
        FillValue::from(0.0f32),
    )
    .build(store.clone(), "/array_b")
    .unwrap();
    array_b.store_metadata().unwrap();
    let data_b: Vec<f32> = (0..10).map(|i| i as f32).collect();
    array_b
        .store_array_subset_elements(&ArraySubset::new_with_shape(vec![10]), &data_b)
        .unwrap();

    let requested = vec!["array_b".to_string()];
    let (batch, lineage) = ZarrConnector
        .read_dataset(path_str, Some(&requested))
        .unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.schema().field(0).name(), "array_b");
    assert!(lineage.warnings.is_empty());
}

#[test]
fn test_csv_fields_projects_to_requested_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    std::fs::write(&path, "a,b,c\n1,2,3\n4,5,6\n").unwrap();
    let path_str = path.to_str().unwrap();

    let requested = vec!["c".to_string(), "a".to_string()];
    let (batch, _lineage) = CsvConnector::new()
        .read_dataset(path_str, Some(&requested))
        .unwrap();

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.schema().field(0).name(), "c");
    assert_eq!(batch.schema().field(1).name(), "a");
}

#[test]
fn test_csv_fields_missing_column_is_hard_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    std::fs::write(&path, "a,b\n1,2\n").unwrap();
    let path_str = path.to_str().unwrap();

    let requested = vec!["not_a_column".to_string()];
    let err = CsvConnector::new()
        .read_dataset(path_str, Some(&requested))
        .expect_err("requesting a nonexistent column must be a hard error");
    assert!(format!("{err}").contains("not_a_column"));
}

#[test]
fn test_use_dataset_from_fields_clause_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.h5");
    let path_str = path.to_str().unwrap();
    make_mismatched_h5(path_str);

    let mut db = TensorDb::new();
    let cmd = format!(r#"USE DATASET FROM "{path_str}" AS m FIELDS (vector_b)"#);
    let out = execute_line(&mut db, &cmd, 1).expect("USE DATASET FROM ... FIELDS should succeed");

    match out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 7, "vector_b has 7 elements");
            assert_eq!(table.schema.fields.len(), 1);
            assert_eq!(table.schema.fields[0].name, "vector_b");
        }
        other => panic!("Expected Table output, got {other:?}"),
    }
}
