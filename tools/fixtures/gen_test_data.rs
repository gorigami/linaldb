use hdf5::File;
use ndarray::Array2;
use ndarray_npy::write_npy;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create NPY
    let data: Array2<f32> = Array2::from_elem((2, 2), 1.0);
    write_npy("test_data.npy", &data)?;
    println!("Created test_data.npy");

    // Create HDF5
    let file = File::create("test_data.h5")?;
    let ds = file.new_dataset::<f32>().shape((2, 2)).create("dataset1")?;
    ds.write(&data)?;
    println!("Created test_data.h5");

    Ok(())
}
