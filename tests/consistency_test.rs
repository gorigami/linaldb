use linal::core::tensor::Shape;
use linal::dsl::execute_line;
use linal::engine::TensorDb;

#[test]
fn test_show_lineage() {
    let mut db = TensorDb::new();

    // 1. Create base tensors
    db.insert_named("a", Shape::new(vec![2]), vec![1.0, 2.0])
        .unwrap();
    db.insert_named("b", Shape::new(vec![2]), vec![3.0, 4.0])
        .unwrap();

    // 2. Derive tensors
    execute_line(&mut db, "LET c = a + b", 1).unwrap();
    execute_line(&mut db, "DERIVE d FROM c * 2.0", 2).unwrap();

    // 3. Show lineage
    let output = execute_line(&mut db, "SHOW LINEAGE d", 3).unwrap();

    if let linal::dsl::DslOutput::Message(msg) = output {
        println!("{}", msg);
        // "Lineage for 'd'" (not "... for tensor 'd'"): SHOW LINEAGE's
        // message is entity-agnostic now that it resolves tensors and
        // datasets through the same ProvenanceStore-backed path (see
        // LINEAGE_AND_LINALG_PLAN.md's Phase 0 outcome).
        assert!(msg.contains("Lineage for 'd'"));
        assert!(msg.contains("MULTIPLY (d)"));
        assert!(msg.contains("ADD (c)"));
        assert!(msg.contains("ROOT (a)"));
        assert!(msg.contains("ROOT (b)"));
    } else {
        panic!("Expected Message output");
    }
}

#[test]
fn test_audit_dataset() {
    let mut db = TensorDb::new();

    // 1. Create tensor and dataset
    db.insert_named("vec1", Shape::new(vec![3]), vec![0.1, 0.2, 0.3])
        .unwrap();
    execute_line(&mut db, "LET ds1 = dataset(\"ds1\")", 1).unwrap();
    execute_line(&mut db, "ATTACH vec1 TO ds1.emb", 2).unwrap();

    // 2. Audit (should pass)
    let output = execute_line(&mut db, "AUDIT DATASET ds1", 3).unwrap();
    if let linal::dsl::DslOutput::Message(msg) = output {
        assert!(msg.contains("Audit PASSED"));
    } else {
        panic!("Expected Message output");
    }

    // 3. Delete tensor and audit (should fail)
    db.active_instance_mut().remove_tensor("vec1");
    let output = execute_line(&mut db, "AUDIT DATASET ds1", 4).unwrap();
    if let linal::dsl::DslOutput::Message(msg) = output {
        assert!(msg.contains("Audit FAILED"));
        assert!(msg.contains("emb"));
    } else {
        panic!("Expected Message output");
    }
}
