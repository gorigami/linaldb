use linal::dsl::persistence;
use std::path::Path;

#[tokio::test]
async fn test_numpy_ingestion() {
    let path = "test_data.npy";
    assert!(Path::new(path).exists(), "test_data.npy must exist");

    let registry = persistence::get_connector_registry();
    let connector = registry
        .find_connector(path)
        .expect("Numpy connector should be found for .npy");

    let (batch, _lineage) = connector
        .read_dataset(path, None)
        .expect("Should read NPY dataset");

    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 4);
}

#[tokio::test]
async fn test_hdf5_ingestion() {
    let path = "test_data.h5";
    assert!(Path::new(path).exists(), "test_data.h5 must exist");

    let registry = persistence::get_connector_registry();
    let connector = registry
        .find_connector(path)
        .expect("HDF5 connector should be found for .h5");

    let (batch, _lineage) = connector
        .read_dataset(path, None)
        .expect("Should read HDF5 dataset");

    assert!(batch.num_columns() >= 1);
    assert_eq!(batch.num_rows(), 4);
}

#[tokio::test]
async fn test_zarr_ingestion() {
    let path = "test_data.zarr";
    assert!(Path::new(path).exists(), "test_data.zarr must exist");

    let registry = persistence::get_connector_registry();
    let connector = registry
        .find_connector(path)
        .expect("Zarr connector should be found for .zarr");

    let (batch, _lineage) = connector
        .read_dataset(path, None)
        .expect("Should read Zarr dataset");

    assert!(batch.num_columns() >= 1);
    assert_eq!(batch.num_rows(), 4);
}
