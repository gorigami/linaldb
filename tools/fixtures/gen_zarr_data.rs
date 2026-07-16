use std::sync::Arc;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "test_data.zarr";
    if std::path::Path::new(path).exists() {
        std::fs::remove_dir_all(path)?;
    }

    let store = Arc::new(FilesystemStore::new(path)?);

    let group = GroupBuilder::new().build(store.clone(), "/")?;
    group.store_metadata()?;

    let array_path = "/data";
    let array = ArrayBuilder::new(
        vec![4], // shape
        DataType::Float32,
        vec![4].try_into()?, // chunk shape
        FillValue::from(0.0f32),
    )
    .build(store.clone(), array_path)?;
    array.store_metadata()?;

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let subset = ArraySubset::new_with_shape(vec![4]);
    array.store_array_subset_elements(&subset, &data)?;

    println!("Created test_data.zarr");
    Ok(())
}
