// Regression tests for a real, previously-undetected silent-data-loss bug
// found while auditing all four scientific/tabular connectors for issues
// similar to the v0.1.55/v0.1.56 IMPORT/LOAD and shape-flattening bugs:
// the HDF5, NPZ, and Zarr connectors all tracked the expected element count
// from the first dataset/array they happened to iterate, then silently
// dropped every subsequent one that didn't match -- either a genuine
// shape/length mismatch, or a dtype `read` failure -- with zero warning,
// while the overall command still reported success. A real multi-array
// scientific file (data + labels + metadata, extremely common) could lose
// most of its actual content this way with no indication anything was
// wrong. Fixed by threading a `warnings: Vec<String>` through each
// connector into the returned `DatasetLineage`, surfaced immediately by
// `import_dataset_core` (appended to its success `Message`) and
// `use_dataset_core` (printed to stderr, since `DslOutput::Table` has no
// message slot).
//
// Also covers the accompanying `inspect()` staleness fix: `inspect()` used
// to hardcode a flat `Shape::new(vec![batch.num_rows()])` for every field,
// never reading the shape-preservation metadata the v0.1.56 fix added --
// stale/wrong for any genuinely multi-dimensional field, even though the
// underlying ingestion path (`record_batch_to_tensors`) got it right.

use linal::core::connectors::hdf5_connector::Hdf5Connector;
use linal::core::connectors::numpy_connector::NumpyConnector;
use linal::core::connectors::zarr_connector::ZarrConnector;
use linal::core::connectors::Connector;
use ndarray::{Array1, Array2};
use std::sync::Arc;

#[test]
fn test_hdf5_shape_mismatch_produces_warning_not_silent_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.h5");
    let path_str = path.to_str().unwrap();

    let file = hdf5::File::create(path_str).unwrap();
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

    let (batch, lineage) = Hdf5Connector.read_dataset(path_str, None).unwrap();

    // Exactly one of the two datasets survives (whichever HDF5 iterates
    // first); the other must be reported as a warning, not silently gone.
    assert_eq!(
        batch.num_columns() + lineage.warnings.len(),
        2,
        "one dataset should be ingested, the other explicitly warned about \
         -- neither silently vanishing nor both surviving into one invalid batch"
    );
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(lineage.warnings.len(), 1);

    let warning = &lineage.warnings[0];
    assert!(
        warning.contains("matrix_a") || warning.contains("vector_b"),
        "warning should name the skipped dataset: {warning}"
    );
    assert!(
        warning.contains("15") && warning.contains('7'),
        "warning should mention both the skipped dataset's element count (15 or 7) \
         and the expected count it didn't match: {warning}"
    );
}

#[test]
fn test_hdf5_inspect_reports_correct_multidim_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_4x3.h5");
    let path_str = path.to_str().unwrap();

    let file = hdf5::File::create(path_str).unwrap();
    let a: Array2<f32> = Array2::from_shape_fn((4, 3), |(r, c)| (r * 3 + c) as f32);
    file.new_dataset::<f32>()
        .shape((4, 3))
        .create("m")
        .unwrap()
        .write(&a)
        .unwrap();
    drop(file);

    let schema = Hdf5Connector.inspect(path_str).unwrap();
    assert_eq!(schema.columns.len(), 1);
    assert_eq!(
        schema.columns[0].shape.dims,
        vec![4, 3],
        "inspect() must report the real (4, 3) shape, not the flattened \
         element count [12] it used to hardcode"
    );
}

#[test]
fn test_numpy_inspect_reports_correct_multidim_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_4x3.npy");
    let path_str = path.to_str().unwrap();

    let a: Array2<f32> = Array2::from_shape_fn((4, 3), |(r, c)| (r * 3 + c) as f32);
    ndarray_npy::write_npy(path_str, &a).unwrap();

    let schema = NumpyConnector.inspect(path_str).unwrap();
    assert_eq!(schema.columns.len(), 1);
    assert_eq!(schema.columns[0].shape.dims, vec![4, 3]);
}

#[test]
fn test_npz_dtype_mismatch_produces_warning_not_silent_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_dtype.npz");
    let path_str = path.to_str().unwrap();

    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ndarray_npy::NpzWriter::new(file);
    let valid: Array1<f32> = Array1::from_shape_fn(5, |i| i as f32);
    writer.add_array("valid_float", &valid).unwrap();
    let invalid: Array1<i32> = Array1::from_shape_fn(5, |i| i as i32);
    writer.add_array("invalid_int", &invalid).unwrap();
    writer.finish().unwrap();

    let (batch, lineage) = NumpyConnector.read_dataset(path_str, None).unwrap();

    assert_eq!(batch.num_columns(), 1);
    assert_eq!(
        batch.schema().field(0).name(),
        "valid_float",
        "the f32-readable array must still be ingested"
    );
    assert_eq!(lineage.warnings.len(), 1);
    assert!(
        lineage.warnings[0].contains("invalid_int"),
        "warning should name the array that couldn't be read as f32: {}",
        lineage.warnings[0]
    );
}

#[test]
fn test_npz_length_mismatch_produces_warning_not_silent_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.npz");
    let path_str = path.to_str().unwrap();

    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ndarray_npy::NpzWriter::new(file);
    let a: Array1<f32> = Array1::from_shape_fn(5, |i| i as f32);
    writer.add_array("array_a", &a).unwrap();
    let b: Array1<f32> = Array1::from_shape_fn(9, |i| i as f32);
    writer.add_array("array_b", &b).unwrap();
    writer.finish().unwrap();

    let (batch, lineage) = NumpyConnector.read_dataset(path_str, None).unwrap();

    assert_eq!(batch.num_columns() + lineage.warnings.len(), 2);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(lineage.warnings.len(), 1);
    let warning = &lineage.warnings[0];
    assert!(warning.contains("array_a") || warning.contains("array_b"));
    assert!(warning.contains('5') && warning.contains('9'));
}

#[test]
fn test_zarr_shape_mismatch_produces_warning_not_silent_drop() {
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

    let (batch, lineage) = ZarrConnector.read_dataset(path_str, None).unwrap();

    assert_eq!(batch.num_columns() + lineage.warnings.len(), 2);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(lineage.warnings.len(), 1);
    let warning = &lineage.warnings[0];
    assert!(warning.contains("array_a") || warning.contains("array_b"));
    assert!(warning.contains('6') && warning.contains("10"));
}
