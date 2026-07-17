//! Downloads the real UCI "Optical Recognition of Handwritten Digits" dataset
//! and derives a small, genuinely-real showcase fixture from it:
//!
//! - `examples/data/digits_centroids.h5`: one HDF5 dataset "centroids" of
//!   shape (10, 64) — row `i` is the real, empirically-averaged pixel
//!   centroid for digit class `i`, computed from real downloaded samples
//!   (never including the held-out query samples below). This is a genuine
//!   rank-2 array, used to exercise both the IMPORT/LOAD round-trip fix and
//!   the shape-preservation fix in examples/hdf5_digit_classification.lnl.
//! - Prints `INSERT INTO query_digits VALUES (...)` lines for a handful of
//!   real, held-out sample images per class, meant to be pasted directly
//!   into that example script (the same pattern `examples/pbmc_cell_typing.lnl`
//!   uses for its synthetic cells, except these pixel values are real).
//!
//! Run with `cargo run --example gen_digits_data`.

use hdf5::File;
use ndarray::Array2;
use std::collections::HashMap;

const SOURCE_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra";
const QUERIES_PER_CLASS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Downloading real UCI digits dataset from {SOURCE_URL} ...");
    let body = reqwest::blocking::get(SOURCE_URL)?.text()?;

    let mut by_class: HashMap<u8, Vec<[f32; 64]>> = HashMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<i64> = line
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<_, _>>()?;
        if parts.len() != 65 {
            continue;
        }
        let label = parts[64] as u8;
        let mut pixels = [0f32; 64];
        for (i, p) in parts[..64].iter().enumerate() {
            pixels[i] = *p as f32;
        }
        by_class.entry(label).or_default().push(pixels);
    }

    std::fs::create_dir_all("examples/data")?;

    let mut centroid_rows: Vec<f32> = Vec::with_capacity(10 * 64);
    let mut query_inserts = String::new();

    for class in 0u8..10 {
        let samples = by_class
            .get(&class)
            .unwrap_or_else(|| panic!("no downloaded samples for digit class {class}"));
        assert!(
            samples.len() > QUERIES_PER_CLASS,
            "class {class} has too few samples ({}) to hold out {QUERIES_PER_CLASS} queries",
            samples.len()
        );

        // Hold out the first QUERIES_PER_CLASS real samples as queries; the
        // *remaining* real samples are averaged into the reference centroid,
        // so no query image ever contributes to its own class centroid.
        let (queries, rest) = samples.split_at(QUERIES_PER_CLASS);

        let mut centroid = [0f32; 64];
        for s in rest {
            for i in 0..64 {
                centroid[i] += s[i];
            }
        }
        for v in centroid.iter_mut() {
            *v /= rest.len() as f32;
        }
        centroid_rows.extend_from_slice(&centroid);

        for (qi, q) in queries.iter().enumerate() {
            let values = q
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            query_inserts.push_str(&format!(
                "INSERT INTO query_digits VALUES (\"digit_{class}_{qi}\", {class}, [{values}])\n"
            ));
        }
    }

    let h5_path = "examples/data/digits_centroids.h5";
    let centroids = Array2::from_shape_vec((10, 64), centroid_rows.clone())?;
    let file = File::create(h5_path)?;
    let ds = file
        .new_dataset::<f32>()
        .shape((10, 64))
        .create("centroids")?;
    ds.write(&centroids)?;

    // Same real centroid values as digits_centroids.h5 above, re-expressed as
    // literal INSERTs into a labeled relational table (mirroring
    // pbmc_cell_typing.lnl's reference_profiles pattern) so the
    // classification step below can reference each centroid by its digit
    // class without needing a row-order-dependent join against the
    // unlabeled HDF5 tensor.
    let mut centroid_inserts = String::new();
    for class in 0u8..10 {
        let row = &centroid_rows[(class as usize) * 64..(class as usize + 1) * 64];
        let values = row
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        centroid_inserts.push_str(&format!(
            "INSERT INTO reference_centroids VALUES ({class}, [{values}])\n"
        ));
    }

    println!(
        "Wrote {h5_path} (10 real per-class centroids, {} samples/class averaged, {QUERIES_PER_CLASS} held out as queries/class)",
        by_class.values().next().map(Vec::len).unwrap_or(0) - QUERIES_PER_CLASS
    );
    println!("\n--- paste into examples/hdf5_digit_classification.lnl: reference_centroids ---\n");
    println!("{centroid_inserts}");
    println!("\n--- paste into examples/hdf5_digit_classification.lnl: query_digits ---\n");
    println!("{query_inserts}");

    Ok(())
}
