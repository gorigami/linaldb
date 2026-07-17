use linal::core::connectors::hdf5_connector::Hdf5Connector;
use linal::core::connectors::numpy_connector::NumpyConnector;
use linal::core::connectors::Connector;
use linal::core::storage::record_batch_to_tensors;
use linal::core::value::ValueType;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use ndarray::Array2;

/// The checked-in `test_data.h5`/`test_data.npy` fixtures are a constant 2x2
/// array — insufficient to catch a transpose/ordering bug (any permutation
/// of 4 identical values still "looks" correct). These tests build a
/// non-square, non-constant 4x3 array instead, with values equal to their
/// row-major flat index (0..12), so both shape and element ordering are
/// independently verifiable.
fn sample_4x3() -> Array2<f32> {
    Array2::from_shape_fn((4, 3), |(r, c)| (r * 3 + c) as f32)
}

#[test]
fn test_hdf5_2d_shape_preserved_in_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_4x3.h5");
    let path_str = path.to_str().unwrap();

    let data = sample_4x3();
    let file = hdf5::File::create(path_str).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape((4, 3))
        .create("dataset1")
        .unwrap();
    ds.write(&data).unwrap();
    drop(file);

    let (batch, _lineage) = Hdf5Connector.read_dataset(path_str).unwrap();
    let tensors = record_batch_to_tensors(&batch).unwrap();

    assert_eq!(tensors.len(), 1);
    let (_name, tensor) = &tensors[0];
    assert_eq!(
        tensor.shape.dims,
        vec![4, 3],
        "shape must be preserved as [4, 3], not flattened to [12]"
    );
    let expected: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(tensor.data.as_ref(), &expected);
}

#[test]
fn test_numpy_2d_shape_preserved_in_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_4x3.npy");
    let path_str = path.to_str().unwrap();

    let data = sample_4x3();
    ndarray_npy::write_npy(path_str, &data).unwrap();

    let (batch, _lineage) = NumpyConnector.read_dataset(path_str).unwrap();
    let tensors = record_batch_to_tensors(&batch).unwrap();

    assert_eq!(tensors.len(), 1);
    let (_name, tensor) = &tensors[0];
    assert_eq!(
        tensor.shape.dims,
        vec![4, 3],
        "shape must be preserved as [4, 3], not flattened to [12]"
    );
    let expected: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(tensor.data.as_ref(), &expected);
}

#[test]
fn test_use_dataset_from_hdf5_reports_matrix_type_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_4x3.h5");
    let path_str = path.to_str().unwrap();

    let data = sample_4x3();
    let file = hdf5::File::create(path_str).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape((4, 3))
        .create("dataset1")
        .unwrap();
    ds.write(&data).unwrap();
    drop(file);

    let mut db = TensorDb::new();
    let use_cmd = format!(r#"USE DATASET FROM "{}" AS matrix_ds"#, path_str);
    let out = execute_line(&mut db, &use_cmd, 1).expect("USE DATASET FROM should succeed");

    match out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 4, "4x3 matrix must materialize as 4 rows");
            assert_eq!(table.schema.fields.len(), 1);
            assert_eq!(table.schema.fields[0].value_type, ValueType::Vector(3));
            for row in &table.rows {
                match &row.values[0] {
                    linal::core::value::Value::Vector(v) => assert_eq!(v.len(), 3),
                    other => panic!("Expected each row to be a Vector(3), got {:?}", other),
                }
            }
        }
        other => panic!("Expected Table output, got {:?}", other),
    }
}

/// Documents the accepted, known limitation: the shape-preservation
/// mechanism itself works at the tensor level for rank > 2, but
/// materialization (`USE DATASET FROM` / `materialize_tensor_dataset`) is
/// still explicitly out of scope for rank > 2 and must fail loudly rather
/// than silently return flattened, mislabeled data.
#[test]
fn test_hdf5_3d_shape_preserved_at_tensor_level_but_not_materializable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape_2x3x4.h5");
    let path_str = path.to_str().unwrap();

    let data: ndarray::Array3<f32> =
        ndarray::Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f32);
    let file = hdf5::File::create(path_str).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape((2, 3, 4))
        .create("dataset1")
        .unwrap();
    ds.write(&data).unwrap();
    drop(file);

    let (batch, _lineage) = Hdf5Connector.read_dataset(path_str).unwrap();
    let tensors = record_batch_to_tensors(&batch).unwrap();

    assert_eq!(tensors.len(), 1);
    let (_name, tensor) = &tensors[0];
    assert_eq!(
        tensor.shape.dims,
        vec![2, 3, 4],
        "the shape-metadata mechanism must preserve rank-3 shape, even though \
         materialization for it is out of scope"
    );

    let mut db = TensorDb::new();
    let use_cmd = format!(r#"USE DATASET FROM "{}" AS cube_ds"#, path_str);
    let err = execute_line(&mut db, &use_cmd, 1)
        .expect_err("USE DATASET FROM on a rank-3 array must fail loudly, not silently flatten");
    let msg = format!("{}", err);
    assert!(
        msg.contains("rank > 2"),
        "expected a rank>2 materialization error, got: {}",
        msg
    );
}
