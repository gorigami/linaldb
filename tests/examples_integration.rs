// tests/examples_integration.rs
//
// Integration tests for example .lnl files
// These tests verify that the example scripts execute correctly end-to-end

use linal::{execute_script, TensorDb};
use std::fs;

#[test]
fn test_example_vdb_integration() {
    let mut db = TensorDb::new();

    // Load and execute the example.lnl file
    let script =
        fs::read_to_string("examples/example.lnl").expect("Failed to read examples/example.lnl");

    // Execute the script - should not panic
    execute_script(&mut db, &script).expect("Example script execution failed");

    // Verify vectors were created
    let v1 = db.get("v1").expect("v1 should exist");
    assert_eq!(v1.shape.dims, vec![3]);
    assert_eq!(v1.data_ref(), vec![1.0, 2.0, 3.0]);

    let v2 = db.get("v2").expect("v2 should exist");
    assert_eq!(v2.data_ref(), vec![4.0, 5.0, 6.0]);

    // Verify vector addition result
    let v3 = db.get("v3").expect("v3 should exist");
    assert_eq!(v3.data_ref(), vec![5.0, 7.0, 9.0]);

    // Verify similarity was computed
    let sim = db.get("sim").expect("sim should exist");
    assert_eq!(sim.shape.dims, Vec::<usize>::new()); // Similarity returns a scalar (empty dims)

    // Verify matrices were created
    let m1 = db.get("m1").expect("m1 should exist");
    assert_eq!(m1.shape.dims, vec![2, 2]);
    assert_eq!(m1.data_ref(), vec![1.0, 2.0, 3.0, 4.0]);

    let m2 = db.get("m2").expect("m2 should exist");
    assert_eq!(m2.shape.dims, vec![2, 2]);

    // Verify matrix multiplication result
    let m3 = db.get("m3").expect("m3 should exist");
    assert_eq!(m3.shape.dims, vec![2, 2]);
    // m1 * m2 = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(m3.data_ref(), vec![19.0, 22.0, 43.0, 50.0]);

    // Verify transpose
    let m3_t = db.get("m3_t").expect("m3_t should exist");
    assert_eq!(m3_t.shape.dims, vec![2, 2]);
    // Verify logical correctness of transpose
    // m3 was [[19, 22], [43, 50]]
    // m3_t is [[19, 43], [22, 50]]

    // Explicitly check elements because zero-copy transpose preserves original data layout
    // but changes strides to provide correct "view".
    use linal::engine::kernels::index;
    assert_eq!(index(m3_t, &[0, 0]).unwrap(), 19.0);
    assert_eq!(index(m3_t, &[0, 1]).unwrap(), 43.0);
    assert_eq!(index(m3_t, &[1, 0]).unwrap(), 22.0);
    assert_eq!(index(m3_t, &[1, 1]).unwrap(), 50.0);

    // Verify flatten
    let flat = db.get("flat").expect("flat should exist");
    assert_eq!(flat.shape.dims, vec![4]); // Flattened 2x2 matrix

    // Verify dataset was created
    let users = db.get_dataset("users").expect("users dataset should exist");
    assert_eq!(users.len(), 5);
    assert_eq!(users.schema.len(), 5);

    // Verify filtered datasets
    let active_users = db
        .get_dataset("active_users")
        .expect("active_users should exist");
    assert_eq!(active_users.len(), 3); // Alice, Charlie, Eve

    let target_group = db
        .get_dataset("target_group")
        .expect("target_group should exist");
    assert_eq!(target_group.len(), 3); // Alice(30), Charlie(35), Eve(32) - all > 28

    let final_list = db
        .get_dataset("final_list")
        .expect("final_list should exist");
    assert_eq!(final_list.len(), 3);
    assert_eq!(final_list.schema.len(), 2); // name, score
}

#[test]
fn test_features_demo_vdb_integration() {
    let mut db = TensorDb::new();

    // Load and execute the features_demo.lnl file
    let script = fs::read_to_string("examples/features_demo.lnl")
        .expect("Failed to read examples/features_demo.lnl");

    // Execute the script - should not panic
    execute_script(&mut db, &script).expect("Features demo script execution failed");

    // Verify dataset was created with correct schema
    let analytics_data = db
        .get_dataset("analytics_data")
        .expect("analytics_data dataset should exist");
    assert_eq!(analytics_data.len(), 4);
    assert_eq!(analytics_data.schema.len(), 4);

    // Verify indices were created
    let indices = db.list_indices();
    assert!(indices.len() >= 2, "Should have at least 2 indices");

    let has_region_idx = indices
        .iter()
        .any(|(ds, col, t)| ds == "analytics_data" && col == "region" && t == "HASH");
    assert!(has_region_idx, "region_idx should exist");

    let has_emb_idx = indices
        .iter()
        .any(|(ds, col, t)| ds == "analytics_data" && col == "embedding" && t == "VECTOR");
    assert!(has_emb_idx, "emb_idx should exist");
}

#[test]
fn test_pbmc_cell_typing_integration() {
    // Unlike the other examples in this file, this script calls SAVE
    // DATASET and actually writes to ./data/pbmc_celltyping — clean up
    // before and after so a failed run doesn't leave stray files behind.
    let _ = fs::remove_dir_all("./data/pbmc_celltyping");

    let mut db = TensorDb::new();

    let script = fs::read_to_string("examples/pbmc_cell_typing.lnl")
        .expect("Failed to read examples/pbmc_cell_typing.lnl");

    execute_script(&mut db, &script).expect("PBMC cell-typing script execution failed");

    // The script ends with `USE default` — switch back to the instance it
    // actually created its datasets in before inspecting them.
    db.use_database("pbmc_celltyping")
        .expect("pbmc_celltyping database should exist");

    let reference_profiles = db
        .get_dataset("reference_profiles")
        .expect("reference_profiles dataset should exist");
    assert_eq!(reference_profiles.len(), 6, "6 canonical PBMC cell types");

    let query_cells = db
        .get_dataset("query_cells")
        .expect("query_cells dataset should exist");
    assert_eq!(query_cells.len(), 240, "240 synthetic cells generated");

    let filtered_cells = db
        .get_dataset("filtered_cells")
        .expect("filtered_cells dataset should exist (QC pipeline output)");
    assert_eq!(
        filtered_cells.len(),
        237,
        "QC filter (n_genes_detected >= 200) should drop exactly the 3 low-quality cells"
    );

    let observed_profiles = db
        .get_dataset("observed_profiles")
        .expect("observed_profiles dataset should exist (GROUP BY + AVG_VEC output)");
    assert_eq!(
        observed_profiles.len(),
        6,
        "one empirical centroid per cell type"
    );

    // Verify the vector index used for classification exists.
    let indices = db.list_indices();
    let has_vector_idx = indices
        .iter()
        .any(|(ds, col, t)| ds == "reference_profiles" && col == "markers" && t == "VECTOR");
    assert!(
        has_vector_idx,
        "reference_profiles should have a VECTOR index on markers"
    );

    let _ = fs::remove_dir_all("./data/pbmc_celltyping");
}
