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
