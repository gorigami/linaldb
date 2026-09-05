use linal::engine::TensorDb;

#[test]
fn test_indexing_workflow() {
    let mut db = TensorDb::new();

    // 1. Create Dataset
    // Need to use DSL or Engine methods. Using Engine for precise control initially?
    // Actually, integration tests usually test the DSL execution flow.
    // But direct Engine access is also fine.
    // Let's use `execute_script` helper if available? It's in `dsl` module.
    // But I can't access private modules easily in integration tests unless they are public.
    // `linal::dsl::execute_script` is public.

    // Test Script
    let script = r#"
    DATASET items COLUMNS (id: Int, category: String, embedding: Vector(3))

    CREATE INDEX cat_idx ON items(category)
    CREATE VECTOR INDEX vec_idx ON items(embedding)

    INSERT INTO items VALUES (1, "A", [1.0, 0.0, 0.0])
    INSERT INTO items VALUES (2, "B", [0.0, 1.0, 0.0])
    INSERT INTO items VALUES (3, "A", [0.0, 0.0, 1.0])

    SHOW INDEXES
    "#;

    linal::dsl::execute_script(&mut db, script).expect("Script execution failed");

    // Verify indices exist via public API or by checking output (harder)
    // We can use db.list_indices() if it's public.
    let indices = db.list_indices();
    assert_eq!(indices.len(), 2);

    let has_cat_idx = indices
        .iter()
        .any(|(ds, col, type_)| ds == "items" && col == "category" && type_ == "HASH");
    let has_vec_idx = indices
        .iter()
        .any(|(ds, col, type_)| ds == "items" && col == "embedding" && type_ == "VECTOR");

    assert!(has_cat_idx, "Hash index not found");
    assert!(has_vec_idx, "Vector index not found");

    // Note: We are not testing SEARCH yet as SELECT/FIND is not updated to use indices.
    // But we are testing CREATE and INSERT maintenance.
}

#[test]
fn test_index_definitions_survive_save_and_load() {
    let mut db = TensorDb::new();

    let save_script = r#"
    DATASET items COLUMNS (id: Int, category: String, embedding: Vector(3))

    CREATE INDEX cat_idx ON items(category)
    CREATE VECTOR INDEX vec_idx ON items(embedding)

    INSERT INTO items VALUES (1, "A", [1.0, 0.0, 0.0])
    INSERT INTO items VALUES (2, "B", [0.0, 1.0, 0.0])
    INSERT INTO items VALUES (3, "A", [0.0, 0.0, 1.0])

    SAVE DATASET items TO "test_index_persistence.parquet"
    "#;
    linal::dsl::execute_script(&mut db, save_script).expect("Save script failed");

    // Indices are in-memory only (`Dataset.indices` is `#[serde(skip)]`), so
    // a *fresh* engine standing in for "the process restarted" must recover
    // them purely from what was written to disk by SAVE DATASET above.
    let mut db2 = TensorDb::new();
    let output = linal::dsl::execute_line(
        &mut db2,
        r#"LOAD DATASET items FROM "test_index_persistence.parquet""#,
        1,
    )
    .expect("Load statement failed");
    let output_text = output.to_string();
    assert!(
        output_text.contains("indices restored on"),
        "expected load output to mention restored indices, got: {}",
        output_text
    );

    let indices = db2.list_indices();
    assert_eq!(indices.len(), 2, "expected both indices to be restored");

    let has_cat_idx = indices
        .iter()
        .any(|(ds, col, type_)| ds == "items" && col == "category" && type_ == "HASH");
    let has_vec_idx = indices
        .iter()
        .any(|(ds, col, type_)| ds == "items" && col == "embedding" && type_ == "VECTOR");
    assert!(
        has_cat_idx,
        "Hash index was not restored after LOAD DATASET"
    );
    assert!(
        has_vec_idx,
        "Vector index was not restored after LOAD DATASET"
    );

    // The restored hash index must actually work, not just exist: it should
    // reflect the reloaded rows' row ids, not stale ones from the old
    // in-memory dataset.
    let dataset = db2.get_dataset("items").expect("dataset should be loaded");
    let index = dataset
        .get_index("category")
        .expect("category index should be present");
    let matches = index
        .lookup(&linal::core::value::Value::String("A".to_string()))
        .expect("lookup should succeed");
    assert_eq!(matches.len(), 2, "expected 2 rows with category 'A'");
}

/// End-to-end (DSL -> planner -> `CosineFilterExec` -> clustered
/// `VectorIndex`) check that `WHERE COSINE_SIM(...) > threshold` still
/// returns exactly the right rows once the index is large enough to
/// actually cluster (see `MIN_VECTORS_TO_CLUSTER` in
/// `core::index::vector`), not just on the tiny datasets the other tests
/// here use (which never leave the brute-force fallback path).
#[test]
fn cosine_threshold_query_is_exact_on_a_clustered_vector_index() {
    let mut db = TensorDb::new();

    const DIM: usize = 4;
    const PER_CLUSTER: usize = 80;
    const NUM_CLUSTERS: usize = 3; // total rows = 240, comfortably past the clustering threshold

    let mut script = String::from("DATASET big_vecs COLUMNS (id: Int, embedding: Vector(4))\n");
    script.push_str("CREATE VECTOR INDEX ON big_vecs(embedding)\n");
    for axis in 0..NUM_CLUSTERS {
        for j in 0..PER_CLUSTER {
            let mut v = [0.0f32; DIM];
            v[axis] = 1.0;
            v[(axis + 1) % DIM] += 0.01 * (j as f32 / PER_CLUSTER as f32);
            let row_id = axis * PER_CLUSTER + j;
            script.push_str(&format!(
                "INSERT INTO big_vecs VALUES ({}, [{}])\n",
                row_id,
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    linal::dsl::execute_script(&mut db, &script).expect("setup script failed");

    // Query for the exact centroid of cluster axis=1; only its ~80 members
    // should pass a 0.99 threshold.
    let query = format!(
        "SELECT id FROM big_vecs WHERE COSINE_SIM(embedding, [{}]) > 0.99",
        (0..DIM)
            .map(|d| if d == 1 { "1.0" } else { "0.0" })
            .collect::<Vec<_>>()
            .join(", ")
    );
    let output = linal::dsl::execute_line(&mut db, &query, 1).expect("query failed");
    let row_count = match output {
        linal::dsl::DslOutput::Table(ds) => ds.len(),
        other => panic!("expected a Table result, got {:?}", other),
    };
    assert_eq!(
        row_count, PER_CLUSTER,
        "expected exactly cluster axis=1's {} members, got {}",
        PER_CLUSTER, row_count
    );
}
