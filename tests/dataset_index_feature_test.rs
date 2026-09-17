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
fn test_search_without_into_returns_inline_table() {
    // DSL_REFERENCE.md §7: "INTO <target> materializes the results as a new
    // dataset instead of returning them inline" -- meaning SEARCH without
    // INTO should return the top-k rows directly. Previously it always
    // materialized into a `search_results` dataset (or the named target) and
    // returned only a status Message, even with no INTO clause, contradicting
    // the documented default. Found while building a real recommender-system
    // notebook that queried SEARCH without INTO and expected an inline table.
    let mut db = TensorDb::new();
    let script = r#"
    DATASET items COLUMNS (id: Int, embedding: Vector(3))
    CREATE VECTOR INDEX ON items(embedding)
    INSERT INTO items VALUES (1, [1.0, 0.0, 0.0])
    INSERT INTO items VALUES (2, [0.0, 1.0, 0.0])
    INSERT INTO items VALUES (3, [0.9, 0.1, 0.0])
    "#;
    linal::dsl::execute_script(&mut db, script).expect("setup script failed");

    // No INTO: must come back as an inline Table, not a Message, and must
    // NOT create a `search_results` dataset as a side effect.
    let output = linal::dsl::execute_line(
        &mut db,
        "SEARCH items ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 2",
        1,
    )
    .expect("search failed");
    match output {
        linal::dsl::DslOutput::Table(ds) => {
            assert_eq!(ds.len(), 2, "expected top-2 rows inline");
        }
        other => panic!("expected an inline Table result, got {:?}", other),
    }
    assert!(
        db.get_dataset("search_results").is_err(),
        "SEARCH without INTO should not materialize a `search_results` dataset"
    );

    // With INTO: unchanged behavior -- a status Message, and the named
    // dataset is materialized and queryable afterward.
    let output = linal::dsl::execute_line(
        &mut db,
        "SEARCH items ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 2 INTO nearest",
        2,
    )
    .expect("search into failed");
    match output {
        linal::dsl::DslOutput::Message(_) => {}
        other => panic!(
            "expected a Message result for SEARCH...INTO, got {:?}",
            other
        ),
    }
    let nearest = db.get_dataset("nearest").expect("`nearest` should exist");
    assert_eq!(nearest.len(), 2);
}

#[test]
fn test_search_filter_clause_post_filters_the_top_k_results() {
    // SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 2: SEARCH's modern syntax
    // gains an optional FILTER <predicate> clause, distinct from WHERE
    // (already claimed by the legacy alternate query-vector syntax).
    let mut db = TensorDb::new();
    let script = r#"
    DATASET docs COLUMNS (id: Int, category: String, embedding: Vector(3))
    CREATE VECTOR INDEX ON docs(embedding)
    INSERT INTO docs VALUES (1, "a", [1.0, 0.0, 0.0])
    INSERT INTO docs VALUES (2, "b", [0.9, 0.1, 0.0])
    INSERT INTO docs VALUES (3, "a", [0.8, 0.2, 0.0])
    "#;
    linal::dsl::execute_script(&mut db, script).expect("setup script failed");

    // Unfiltered: all 3 rows are within the top-3.
    let output = linal::dsl::execute_line(
        &mut db,
        "SEARCH docs ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 3",
        1,
    )
    .expect("search failed");
    let linal::dsl::DslOutput::Table(ds) = output else {
        panic!("expected inline Table")
    };
    assert_eq!(ds.len(), 3);

    // FILTER category = "a": only ids 1 and 3 survive, id 2 (category "b")
    // is dropped even though it's within the top-3 by similarity.
    let output = linal::dsl::execute_line(
        &mut db,
        r#"SEARCH docs ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 3 FILTER category = "a""#,
        2,
    )
    .expect("filtered search failed");
    let linal::dsl::DslOutput::Table(ds) = output else {
        panic!("expected inline Table")
    };
    assert_eq!(ds.len(), 2);
    let id_col = ds.schema.get_field_index("id").unwrap();
    let mut ids: Vec<i64> = ds
        .rows
        .iter()
        .map(|row| match &row.values[id_col] {
            linal::core::value::Value::Int(i) => *i,
            other => panic!("expected Int id, got {other:?}"),
        })
        .collect();
    ids.sort();
    assert_eq!(ids, vec![1, 3]);

    // FILTER + INTO together still works.
    let output = linal::dsl::execute_line(
        &mut db,
        r#"SEARCH docs ON embedding QUERY [1.0, 0.0, 0.0] LIMIT 3 FILTER category = "a" INTO filtered"#,
        3,
    )
    .expect("filtered search into failed");
    assert!(matches!(output, linal::dsl::DslOutput::Message(_)));
    let filtered = db.get_dataset("filtered").expect("`filtered` should exist");
    assert_eq!(filtered.len(), 2);
}

#[test]
fn test_search_legacy_forms_unaffected_by_filter_clause() {
    // The plan explicitly scopes FILTER to the modern syntax only -- both
    // legacy forms (FROM...ON...K=, and WHERE...~=...LIMIT) must keep
    // parsing and executing exactly as before.
    let mut db = TensorDb::new();
    let script = r#"
    DATASET docs COLUMNS (id: Int, embedding: Vector(3))
    CREATE VECTOR INDEX ON docs(embedding)
    INSERT INTO docs VALUES (1, [1.0, 0.0, 0.0])
    INSERT INTO docs VALUES (2, [0.0, 1.0, 0.0])
    "#;
    linal::dsl::execute_script(&mut db, script).expect("setup script failed");

    linal::dsl::execute_line(
        &mut db,
        "SEARCH legacy_results FROM docs QUERY [1.0, 0.0, 0.0] ON embedding K=2",
        1,
    )
    .expect("legacy FROM syntax should still work");
    assert_eq!(
        db.get_dataset("legacy_results")
            .expect("legacy_results should exist")
            .len(),
        2
    );

    let output = linal::dsl::execute_line(
        &mut db,
        "SEARCH docs WHERE embedding ~= [1.0, 0.0, 0.0] LIMIT 2",
        2,
    )
    .expect("legacy WHERE syntax should still work");
    let linal::dsl::DslOutput::Table(ds) = output else {
        panic!("expected inline Table")
    };
    assert_eq!(ds.len(), 2);
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

/// SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 2: index persistence. Once a
/// vector index is large enough to actually cluster, SAVE DATASET should
/// persist that clustering (`vector_index_clusters.json`) and LOAD DATASET
/// should restore it directly -- no k-means recomputation -- rather than
/// rebuilding from scratch every time.
#[test]
fn vector_index_clustering_survives_save_and_load_without_rebuilding() {
    let _ = std::fs::remove_dir_all("./data/default/datasets/snap_vecs");

    const N: usize = 120; // comfortably past MIN_VECTORS_TO_CLUSTER (64)

    let mut db = TensorDb::new();
    let mut script = String::from("DATASET snap_vecs COLUMNS (id: Int, embedding: Vector(4))\n");
    for i in 0..N {
        let angle = i as f32;
        let v = [
            angle.sin(),
            angle.cos(),
            (angle * 0.5).sin(),
            (angle * 0.5).cos(),
        ];
        script.push_str(&format!(
            "INSERT INTO snap_vecs VALUES ({}, [{}])\n",
            i,
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    script.push_str("CREATE VECTOR INDEX ON snap_vecs(embedding)\n");
    script.push_str("SAVE DATASET snap_vecs\n");
    linal::dsl::execute_script(&mut db, &script).expect("setup+save script failed");

    let snapshot_path = "./data/default/datasets/snap_vecs/vector_index_clusters.json";
    assert!(
        std::path::Path::new(snapshot_path).exists(),
        "expected SAVE DATASET to write a vector_index_clusters.json"
    );
    let raw = std::fs::read_to_string(snapshot_path).unwrap();
    assert!(
        raw.contains("\"clustered_count\": 120"),
        "expected the persisted snapshot to reflect all 120 clustered vectors, got: {raw}"
    );

    // Fresh engine, simulating a process restart -- must restore from the
    // persisted snapshot, not rebuild via k-means.
    let mut db2 = TensorDb::new();
    let output = linal::dsl::execute_line(&mut db2, "LOAD DATASET snap_vecs", 1)
        .expect("load failed")
        .to_string();
    assert!(
        output.contains("from snapshot: embedding") || output.contains("from snapshot: emb"),
        "expected the load message to report a snapshot-restored vector index, got: {output}"
    );

    // The restored index must actually work: an exact self-query returns
    // the matching row as its top hit.
    let query = format!(
        "SEARCH snap_vecs ON embedding QUERY [{}] LIMIT 1",
        [0.0f32.sin(), 0.0f32.cos(), 0.0f32.sin(), 0.0f32.cos()]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let out = linal::dsl::execute_line(&mut db2, &query, 2).expect("search failed");
    let linal::dsl::DslOutput::Table(ds) = out else {
        panic!("expected inline Table")
    };
    assert_eq!(ds.len(), 1);
    let id_col = ds.schema.get_field_index("id").unwrap();
    match &ds.rows[0].values[id_col] {
        linal::core::value::Value::Int(0) => {}
        other => panic!("expected row id 0 as the exact nearest match, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all("./data/default/datasets/snap_vecs");
}

/// A persisted snapshot whose content hash no longer matches the reloaded
/// column's data (e.g. `data.parquet` edited independently) must be
/// rejected, not silently trusted -- LOAD DATASET falls back to a full
/// k-means rebuild instead, and the index is still correct afterward.
#[test]
fn stale_vector_index_snapshot_is_rejected_and_falls_back_to_rebuild() {
    let _ = std::fs::remove_dir_all("./data/default/datasets/stale_vecs");

    const N: usize = 80;

    let mut db = TensorDb::new();
    let mut script = String::from("DATASET stale_vecs COLUMNS (id: Int, embedding: Vector(4))\n");
    for i in 0..N {
        let angle = i as f32;
        let v = [angle.sin(), angle.cos(), 0.0, 0.0];
        script.push_str(&format!(
            "INSERT INTO stale_vecs VALUES ({}, [{}])\n",
            i,
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    script.push_str("CREATE VECTOR INDEX ON stale_vecs(embedding)\n");
    script.push_str("SAVE DATASET stale_vecs\n");
    linal::dsl::execute_script(&mut db, &script).expect("setup+save script failed");
    // Corrupt the persisted content hash in place, simulating drift between
    // the snapshot and the actual column data.
    let snapshot_path = "./data/default/datasets/stale_vecs/vector_index_clusters.json";
    let raw = std::fs::read_to_string(snapshot_path).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    value["embedding"]["content_hash"] = serde_json::Value::String("stale-hash".to_string());
    std::fs::write(snapshot_path, serde_json::to_string_pretty(&value).unwrap()).unwrap();

    let mut db2 = TensorDb::new();
    let output = linal::dsl::execute_line(&mut db2, "LOAD DATASET stale_vecs", 1)
        .expect("load failed")
        .to_string();
    assert!(
        !output.contains("from snapshot"),
        "a content-hash mismatch must not be silently trusted, got: {output}"
    );
    assert!(
        output.contains("indices restored on"),
        "the index should still be rebuilt (just not from the stale snapshot), got: {output}"
    );

    // The rebuilt index must still be correct.
    let query = format!(
        "SEARCH stale_vecs ON embedding QUERY [{}] LIMIT 1",
        [0.0f32.sin(), 0.0f32.cos(), 0.0, 0.0]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let out = linal::dsl::execute_line(&mut db2, &query, 2).expect("search failed");
    let linal::dsl::DslOutput::Table(ds) = out else {
        panic!("expected inline Table")
    };
    assert_eq!(ds.len(), 1);
    let id_col = ds.schema.get_field_index("id").unwrap();
    match &ds.rows[0].values[id_col] {
        linal::core::value::Value::Int(0) => {}
        other => panic!("expected row id 0 as the exact nearest match, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all("./data/default/datasets/stale_vecs");
}
