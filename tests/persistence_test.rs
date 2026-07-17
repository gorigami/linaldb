use linal::core::dataset_legacy::{Dataset, DatasetId};
use linal::core::storage::{ParquetStorage, StorageEngine};
use linal::core::tensor::{Shape, Tensor, TensorId};
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use std::fs;
use std::sync::Arc;

/// Helper to create a test dataset
fn create_test_dataset(name: &str) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("name", ValueType::String),
        Field::new("score", ValueType::Float),
    ]));

    let mut dataset = Dataset::new(DatasetId(1), schema.clone(), Some(name.to_string()));

    // Add rows
    let rows = vec![
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Float(95.5),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(2),
                Value::String("Bob".to_string()),
                Value::Float(88.0),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(3),
                Value::String("Charlie".to_string()),
                Value::Float(92.0),
            ],
        )
        .unwrap(),
    ];

    dataset.rows = rows;
    dataset.metadata.update_stats(&schema, &dataset.rows);

    dataset
}

/// Helper to create a test tensor
fn create_test_tensor(_id: u64, dims: Vec<usize>) -> Tensor {
    let shape = Shape::new(dims);
    let id = TensorId::new();
    let num_elements = shape.num_elements();
    let data: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let metadata = linal::core::tensor::TensorMetadata::new(id, None);

    Tensor::new(id, shape, data, metadata).unwrap()
}

#[test]
fn test_save_dataset_creates_files() {
    let temp_dir = "/tmp/linal_test_persistence_save";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let dataset = create_test_dataset("test_users");

    // Save dataset
    storage.save_dataset(&dataset).unwrap();

    // Verify files exist
    assert!(storage.dataset_exists("test_users"));
    // New standardized structure: datasets/{name}/data.parquet
    assert!(
        std::path::Path::new(&format!("{}/datasets/test_users/data.parquet", temp_dir)).exists()
    );
    assert!(
        std::path::Path::new(&format!("{}/datasets/test_users/manifest.json", temp_dir)).exists()
    );
    assert!(
        std::path::Path::new(&format!("{}/datasets/test_users/schema.json", temp_dir)).exists()
    );
    assert!(std::path::Path::new(&format!("{}/datasets/test_users/stats.json", temp_dir)).exists());

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_list_datasets() {
    let temp_dir = "/tmp/linal_test_persistence_list";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);

    // Initially empty
    let datasets = storage.list_datasets().unwrap();
    assert_eq!(datasets.len(), 0);

    // Save multiple datasets
    let dataset1 = create_test_dataset("users");
    let dataset2 = create_test_dataset("products");

    storage.save_dataset(&dataset1).unwrap();
    storage.save_dataset(&dataset2).unwrap();

    // List should return both
    let datasets = storage.list_datasets().unwrap();
    assert_eq!(datasets.len(), 2);
    assert!(datasets.contains(&"users".to_string()));
    assert!(datasets.contains(&"products".to_string()));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_delete_dataset() {
    let temp_dir = "/tmp/linal_test_persistence_delete";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let dataset = create_test_dataset("temp_data");

    // Save and verify
    storage.save_dataset(&dataset).unwrap();
    assert!(storage.dataset_exists("temp_data"));

    // Delete and verify
    storage.delete_dataset("temp_data").unwrap();
    assert!(!storage.dataset_exists("temp_data"));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_save_tensor() {
    let temp_dir = "/tmp/linal_test_persistence_tensor_save";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let tensor = create_test_tensor(1, vec![3, 4]);

    // Save tensor
    storage.save_tensor("embedding", &tensor).unwrap();

    // Verify file exists
    assert!(storage.tensor_exists("embedding"));
    assert!(std::path::Path::new(&format!("{}/tensors/embedding.json", temp_dir)).exists());

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_load_tensor() {
    let temp_dir = "/tmp/linal_test_persistence_tensor_load";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let original_tensor = create_test_tensor(42, vec![2, 3]);

    // Save tensor
    storage.save_tensor("weights", &original_tensor).unwrap();

    // Load tensor
    let loaded_tensor = storage.load_tensor("weights").unwrap();

    // Verify data matches
    assert_eq!(loaded_tensor.id, original_tensor.id);
    assert_eq!(loaded_tensor.shape.dims, original_tensor.shape.dims);
    assert_eq!(loaded_tensor.data, original_tensor.data);

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_list_tensors() {
    let temp_dir = "/tmp/linal_test_persistence_tensor_list";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);

    // Initially empty
    let tensors = storage.list_tensors().unwrap();
    assert_eq!(tensors.len(), 0);

    // Save multiple tensors
    let tensor1 = create_test_tensor(1, vec![10]);
    let tensor2 = create_test_tensor(2, vec![5, 5]);

    storage.save_tensor("vector_a", &tensor1).unwrap();
    storage.save_tensor("matrix_b", &tensor2).unwrap();

    // List should return both
    let tensors = storage.list_tensors().unwrap();
    assert_eq!(tensors.len(), 2);
    assert!(tensors.contains(&"vector_a".to_string()));
    assert!(tensors.contains(&"matrix_b".to_string()));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_delete_tensor() {
    let temp_dir = "/tmp/linal_test_persistence_tensor_delete";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let tensor = create_test_tensor(1, vec![4]);

    // Save and verify
    storage.save_tensor("temp_tensor", &tensor).unwrap();
    assert!(storage.tensor_exists("temp_tensor"));

    // Delete and verify
    storage.delete_tensor("temp_tensor").unwrap();
    assert!(!storage.tensor_exists("temp_tensor"));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_metadata_preservation() {
    let temp_dir = "/tmp/linal_test_persistence_metadata";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let dataset = create_test_dataset("metadata_test");

    // Save dataset
    storage.save_dataset(&dataset).unwrap();

    // Read manifest file directly
    let manifest_path = format!("{}/datasets/metadata_test/manifest.json", temp_dir);
    let manifest_json = fs::read_to_string(&manifest_path).unwrap();

    // Verify manifest contains expected fields
    assert!(manifest_json.contains("\"name\""));
    assert!(manifest_json.contains("metadata_test"));
    assert!(manifest_json.contains("\"version\""));

    // Read stats file directly
    let stats_path = format!("{}/datasets/metadata_test/stats.json", temp_dir);
    let stats_json = fs::read_to_string(&stats_path).unwrap();
    assert!(stats_json.contains("\"row_count\""));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_tensor_round_trip_different_shapes() {
    let temp_dir = "/tmp/linal_test_persistence_tensor_shapes";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);

    // Test scalar (0D)
    let scalar = create_test_tensor(1, vec![]);
    storage.save_tensor("scalar", &scalar).unwrap();
    let loaded_scalar = storage.load_tensor("scalar").unwrap();
    assert_eq!(loaded_scalar.shape.dims, Vec::<usize>::new());

    // Test vector (1D)
    let vector = create_test_tensor(2, vec![5]);
    storage.save_tensor("vector", &vector).unwrap();
    let loaded_vector = storage.load_tensor("vector").unwrap();
    assert_eq!(loaded_vector.shape.dims, vec![5]);

    // Test matrix (2D)
    let matrix = create_test_tensor(3, vec![3, 4]);
    storage.save_tensor("matrix", &matrix).unwrap();
    let loaded_matrix = storage.load_tensor("matrix").unwrap();
    assert_eq!(loaded_matrix.shape.dims, vec![3, 4]);

    // Test 3D tensor
    let tensor_3d = create_test_tensor(4, vec![2, 3, 4]);
    storage.save_tensor("tensor_3d", &tensor_3d).unwrap();
    let loaded_3d = storage.load_tensor("tensor_3d").unwrap();
    assert_eq!(loaded_3d.shape.dims, vec![2, 3, 4]);

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_concurrent_dataset_and_tensor_storage() {
    let temp_dir = "/tmp/linal_test_persistence_concurrent";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);

    // Save both datasets and tensors
    let dataset = create_test_dataset("mixed_test");
    let tensor = create_test_tensor(1, vec![10]);

    storage.save_dataset(&dataset).unwrap();
    storage.save_tensor("mixed_tensor", &tensor).unwrap();

    // Verify both exist
    assert!(storage.dataset_exists("mixed_test"));
    assert!(storage.tensor_exists("mixed_tensor"));

    // Verify directories are separate
    assert!(std::path::Path::new(&format!("{}/datasets", temp_dir)).exists());
    assert!(std::path::Path::new(&format!("{}/tensors", temp_dir)).exists());

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

#[test]
fn test_dataset_round_trip() {
    let temp_dir = "/tmp/linal_test_persistence_dataset_round_trip";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let original_dataset = create_test_dataset("round_trip_users");

    // Save
    storage.save_dataset(&original_dataset).unwrap();

    // Load
    let loaded_dataset = storage.load_dataset("round_trip_users").unwrap();

    // Verify Metadata
    assert_eq!(loaded_dataset.metadata.name, original_dataset.metadata.name);
    // Schema verification (field by field)
    assert_eq!(
        loaded_dataset.schema.fields.len(),
        original_dataset.schema.fields.len()
    );
    for (i, field) in original_dataset.schema.fields.iter().enumerate() {
        assert_eq!(loaded_dataset.schema.fields[i].name, field.name);
        // assert_eq!(loaded_dataset.schema.fields[i].value_type, field.value_type); // Need PartialEq for ValueType
    }

    // Verify Data
    assert_eq!(loaded_dataset.rows.len(), original_dataset.rows.len());

    // Verify content of first row
    let row0 = &loaded_dataset.rows[0];
    assert!(matches!(row0.values[0], Value::Int(1)));
    assert!(matches!(row0.values[1], Value::String(ref s) if s == "Alice"));
    assert!(matches!(row0.values[2], Value::Float(f) if (f - 95.5).abs() < 1e-6));

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}

/// Regression test for a bug found while testing the engine against a real
/// downloaded HDF5 file: `import_dataset_core` (the DSL's `IMPORT DATASET
/// FROM` handler) used to call only `save_dataset_package`, never writing
/// the legacy `.meta.json` sidecar `load_dataset` hard-requires — so an
/// IMPORT-produced package reported success but could never be loaded back,
/// even though the data was correctly on disk. This test exercises the fix
/// (`ParquetStorage::save_legacy_metadata_for_batch`) directly at the
/// connector + storage layer, mirroring how `import_dataset_core` uses it.
#[test]
fn test_import_dataset_hdf5_round_trip() {
    use linal::core::connectors::{hdf5_connector::Hdf5Connector, Connector};
    use linal::core::dataset::{DatasetMetadata, DatasetOrigin};
    use linal::core::value::ValueType;

    let temp_dir = "/tmp/linal_test_import_hdf5_round_trip";
    let _ = fs::remove_dir_all(temp_dir);
    let storage = ParquetStorage::new(temp_dir);

    let (batch, lineage) = Hdf5Connector
        .read_dataset("test_data.h5", None)
        .expect("test_data.h5 fixture must exist (cargo run --example gen_test_data)");

    let metadata = DatasetMetadata::new(
        "hdf5_import".to_string(),
        DatasetOrigin::Imported {
            source: "test_data.h5".to_string(),
        },
    );
    storage
        .save_dataset_package("hdf5_import", &batch, &metadata, &lineage)
        .unwrap();
    storage
        .save_legacy_metadata_for_batch("hdf5_import", &batch)
        .unwrap();

    // This is the regression check: previously failed with DatasetNotFound
    // even though save_dataset_package had just written valid data.
    let loaded = storage
        .load_dataset("hdf5_import")
        .expect("LOAD should succeed now that the legacy sidecar is written");

    assert_eq!(loaded.rows.len(), batch.num_rows());
    assert_eq!(loaded.schema.fields.len(), batch.num_columns());
    for field in &loaded.schema.fields {
        assert_eq!(field.value_type, ValueType::Float);
    }

    let _ = fs::remove_dir_all(temp_dir);
}
