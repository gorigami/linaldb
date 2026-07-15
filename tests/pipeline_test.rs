use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::{execute_script, TensorDb};

fn setup_products(db: &mut TensorDb) {
    execute_script(
        db,
        r#"
        DATASET products COLUMNS (id: Int, name: String, score: Float, active: Int)
        INSERT INTO products VALUES (1, "alpha", 0.9, 1)
        INSERT INTO products VALUES (2, "beta", 0.5, 0)
        INSERT INTO products VALUES (3, "gamma", 0.7, 1)
        INSERT INTO products VALUES (4, "delta", 0.3, 1)
        INSERT INTO products VALUES (5, "epsilon", 0.8, 0)
    "#,
    )
    .expect("setup failed");
}

// ── 1. DEFINE + SHOW PIPELINES ────────────────────────────────────────────────

#[test]
fn test_define_pipeline_registers_name() {
    let mut db = TensorDb::new();
    execute_line(&mut db, "DEFINE PIPELINE my_pipe AS SELECT id, score", 1).expect("define failed");

    let result = execute_line(&mut db, "SHOW PIPELINES", 2).expect("show failed");
    if let DslOutput::Message(msg) = result {
        assert!(msg.contains("my_pipe"), "Expected 'my_pipe' in: {}", msg);
    } else {
        panic!("Expected Message output");
    }
}

// ── 2. SHOW PIPELINES when empty ──────────────────────────────────────────────

#[test]
fn test_show_pipelines_empty() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "SHOW PIPELINES", 1).expect("show failed");
    if let DslOutput::Message(msg) = result {
        assert!(
            msg.contains("No pipelines"),
            "Expected empty msg, got: {}",
            msg
        );
    } else {
        panic!("Expected Message output");
    }
}

// ── 2b. LIST PIPELINES — alias for SHOW PIPELINES (CONSISTENCY_PLAN.md D7) ────

#[test]
fn test_list_pipelines_is_alias_for_show_pipelines() {
    let mut db = TensorDb::new();
    execute_line(&mut db, "DEFINE PIPELINE my_pipe AS SELECT id, score", 1).expect("define failed");

    let show = execute_line(&mut db, "SHOW PIPELINES", 2).expect("show failed");
    let list = execute_line(&mut db, "LIST PIPELINES", 3).expect("list failed");
    let (DslOutput::Message(show_msg), DslOutput::Message(list_msg)) = (show, list) else {
        panic!("expected Message output from both SHOW PIPELINES and LIST PIPELINES");
    };
    assert_eq!(
        show_msg, list_msg,
        "LIST PIPELINES should produce identical output to SHOW PIPELINES"
    );
}

// ── 3. DESCRIBE PIPELINE ──────────────────────────────────────────────────────

#[test]
fn test_describe_pipeline() {
    let mut db = TensorDb::new();
    execute_line(
        &mut db,
        "DEFINE PIPELINE describe_me AS SELECT id THEN WHERE active = 1 THEN LIMIT 10",
        1,
    )
    .expect("define failed");

    let result =
        execute_line(&mut db, "DESCRIBE PIPELINE describe_me", 2).expect("describe failed");
    if let DslOutput::Message(msg) = result {
        assert!(msg.contains("describe_me"), "Expected name in: {}", msg);
        assert!(msg.contains("SELECT"), "Expected SELECT in: {}", msg);
        assert!(msg.contains("WHERE"), "Expected WHERE in: {}", msg);
        assert!(msg.contains("LIMIT"), "Expected LIMIT in: {}", msg);
    } else {
        panic!("Expected Message output");
    }
}

// ── 4. APPLY PIPELINE basic filter ────────────────────────────────────────────

#[test]
fn test_apply_pipeline_filter() {
    let mut db = TensorDb::new();
    setup_products(&mut db);

    execute_line(
        &mut db,
        "DEFINE PIPELINE active_only AS WHERE active = 1",
        1,
    )
    .expect("define failed");

    execute_line(
        &mut db,
        "APPLY PIPELINE active_only ON products INTO active_products",
        2,
    )
    .expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM active_products", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 3, "Expected 3 active rows, got {}", ds.len());
    } else {
        panic!("Expected Table output");
    }
}

// ── 5. APPLY PIPELINE with SELECT projection ──────────────────────────────────

#[test]
fn test_apply_pipeline_select_projection() {
    let mut db = TensorDb::new();
    setup_products(&mut db);

    execute_line(&mut db, "DEFINE PIPELINE id_score AS SELECT id, score", 1)
        .expect("define failed");

    execute_line(
        &mut db,
        "APPLY PIPELINE id_score ON products INTO projected",
        2,
    )
    .expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM projected", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.schema.fields.len(), 2, "Expected 2 columns");
        assert_eq!(ds.len(), 5);
    } else {
        panic!("Expected Table output");
    }
}

// ── 6. APPLY PIPELINE with LIMIT ──────────────────────────────────────────────

#[test]
fn test_apply_pipeline_limit() {
    let mut db = TensorDb::new();
    setup_products(&mut db);

    execute_line(&mut db, "DEFINE PIPELINE top3 AS LIMIT 3", 1).expect("define failed");

    execute_line(&mut db, "APPLY PIPELINE top3 ON products INTO limited", 2).expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM limited", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 3, "Expected 3 rows, got {}", ds.len());
    } else {
        panic!("Expected Table output");
    }
}

// ── 7. APPLY PIPELINE multi-step ──────────────────────────────────────────────

#[test]
fn test_apply_pipeline_multi_step() {
    let mut db = TensorDb::new();
    setup_products(&mut db);

    execute_line(
        &mut db,
        "DEFINE PIPELINE top_active AS WHERE active = 1 THEN ORDER BY score DESC THEN LIMIT 2",
        1,
    )
    .expect("define failed");

    execute_line(
        &mut db,
        "APPLY PIPELINE top_active ON products INTO top2",
        2,
    )
    .expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM top2", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 2, "Expected 2 rows");
        let first_score = &ds.rows[0].values[2];
        assert!(
            matches!(first_score, Value::Float(f) if *f > 0.8),
            "Expected top score first, got {:?}",
            first_score
        );
    } else {
        panic!("Expected Table output");
    }
}

// ── 8. APPLY PIPELINE without INTO (in-place) ─────────────────────────────────

#[test]
fn test_apply_pipeline_in_place() {
    let mut db = TensorDb::new();
    setup_products(&mut db);

    execute_line(
        &mut db,
        "DEFINE PIPELINE filter_active AS WHERE active = 1",
        1,
    )
    .expect("define failed");

    execute_line(&mut db, "APPLY PIPELINE filter_active ON products", 2).expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM products", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 3, "Expected 3 rows after in-place filter");
    } else {
        panic!("Expected Table output");
    }
}

// ── 9. DROP PIPELINE ──────────────────────────────────────────────────────────

#[test]
fn test_drop_pipeline() {
    let mut db = TensorDb::new();
    execute_line(&mut db, "DEFINE PIPELINE to_drop AS LIMIT 5", 1).expect("define failed");

    execute_line(&mut db, "DROP PIPELINE to_drop", 2).expect("drop failed");

    let result = execute_line(&mut db, "SHOW PIPELINES", 3).expect("show failed");
    if let DslOutput::Message(msg) = result {
        assert!(
            !msg.contains("to_drop"),
            "Dropped pipeline should not appear: {}",
            msg
        );
    } else {
        panic!("Expected Message output");
    }
}

// ── 10. NORMALIZE step ────────────────────────────────────────────────────────

#[test]
fn test_apply_pipeline_normalize_col() {
    let mut db = TensorDb::new();

    execute_script(
        &mut db,
        r#"
        DATASET vecs COLUMNS (id: Int, embedding: Vector(3))
        INSERT INTO vecs VALUES (1, [3.0, 0.0, 0.0])
        INSERT INTO vecs VALUES (2, [0.0, 4.0, 0.0])
    "#,
    )
    .expect("setup failed");

    execute_line(
        &mut db,
        "DEFINE PIPELINE norm_pipe AS NORMALIZE embedding",
        1,
    )
    .expect("define failed");

    execute_line(&mut db, "APPLY PIPELINE norm_pipe ON vecs INTO normed", 2).expect("apply failed");

    let result = execute_line(&mut db, "SELECT * FROM normed", 3).expect("select failed");
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 2);
        let row0_emb = &ds.rows[0].values[1];
        if let Value::Vector(v) = row0_emb {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Expected unit norm, got {}",
                norm
            );
        } else {
            panic!("Expected Vector value, got {:?}", row0_emb);
        }
    } else {
        panic!("Expected Table output");
    }
}
