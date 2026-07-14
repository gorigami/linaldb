// tests/pipeline_vector_engine_test.rs
// Integration coverage for CONSISTENCY_PLAN.md Track C / C2: pipelines
// (v0.1.33/34) and the vector engine / tensor-SQL bridge (v0.1.31/32) were
// previously tested in total isolation — no test chained COSINE_SIM, MATMUL,
// or index-aware vector search inside a pipeline step. This file exercises
// those combinations end-to-end.

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn exec(db: &mut TensorDb, dsl: &str, line: usize) -> DslOutput {
    execute_line(db, dsl, line).unwrap_or_else(|e| panic!("DSL error at line {line}: {e:?}"))
}

fn ids(ds: &linal::core::dataset_legacy::Dataset) -> Vec<i64> {
    let mut v: Vec<i64> = ds
        .rows
        .iter()
        .map(|r| match r.get("id").unwrap() {
            Value::Int(n) => *n,
            other => panic!("expected Int id, got {other:?}"),
        })
        .collect();
    v.sort();
    v
}

fn setup_docs(db: &mut TensorDb) {
    exec(
        db,
        "DATASET docs COLUMNS (id: Int, category: String, embedding: Vector(3))",
        1,
    );
    exec(db, "INSERT INTO docs VALUES (1, 'a', [1.0, 0.0, 0.0])", 2);
    exec(db, "INSERT INTO docs VALUES (2, 'a', [0.9, 0.1, 0.0])", 3);
    exec(db, "INSERT INTO docs VALUES (3, 'b', [0.0, 1.0, 0.0])", 4);
}

// ── COSINE_SIM inside a pipeline WHERE step ────────────────────────────────

#[test]
fn test_pipeline_filters_by_cosine_similarity() {
    let mut db = TensorDb::new();
    setup_docs(&mut db);

    exec(
        &mut db,
        "DEFINE PIPELINE near AS WHERE COSINE_SIM(embedding, [1.0, 0.0, 0.0]) > 0.8",
        5,
    );
    exec(&mut db, "APPLY PIPELINE near ON docs INTO near_results", 6);

    let out = exec(&mut db, "SELECT * FROM near_results", 7);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    // Row 3 (orthogonal, sim=0.0) must be filtered out; rows 1 and 2 kept.
    assert_eq!(ids(&ds), vec![1, 2]);
}

#[test]
fn test_pipeline_where_cosine_sim_still_correct_with_vector_index_present() {
    // Same as above, but with a CREATE VECTOR INDEX on the filtered column
    // — exercises the index-aware COSINE_SIM path (v0.1.32) inside a
    // pipeline step, not just the generic FilterExec predicate path.
    let mut db = TensorDb::new();
    setup_docs(&mut db);
    exec(&mut db, "CREATE VECTOR INDEX ON docs(embedding)", 5);

    exec(
        &mut db,
        "DEFINE PIPELINE near AS WHERE COSINE_SIM(embedding, [1.0, 0.0, 0.0]) > 0.8",
        6,
    );
    exec(
        &mut db,
        "APPLY PIPELINE near ON docs INTO indexed_results",
        7,
    );

    let out = exec(&mut db, "SELECT * FROM indexed_results", 8);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ids(&ds), vec![1, 2]);
}

// ── COSINE_SIM as a computed column inside a pipeline SELECT step ─────────

#[test]
fn test_pipeline_select_computes_cosine_similarity_column() {
    let mut db = TensorDb::new();
    setup_docs(&mut db);

    exec(
        &mut db,
        "DEFINE PIPELINE scored AS SELECT id, COSINE_SIM(embedding, [1.0, 0.0, 0.0]) AS score",
        5,
    );
    exec(
        &mut db,
        "APPLY PIPELINE scored ON docs INTO scored_results",
        6,
    );

    let out = exec(&mut db, "SELECT * FROM scored_results", 7);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 3);
    for row in &ds.rows {
        let id = row.get("id").unwrap().clone();
        let score = row.get("score").unwrap().clone();
        match id {
            Value::Int(1) => assert_eq!(score, Value::Float(1.0)),
            Value::Int(2) => {
                let Value::Float(f) = score else {
                    panic!("expected float score")
                };
                assert!((f - 0.9938837).abs() < 1e-5, "unexpected score {f}");
            }
            Value::Int(3) => assert_eq!(score, Value::Float(0.0)),
            other => panic!("unexpected id {other:?}"),
        }
    }
}

// ── Chained pipeline: vector filter + NORMALIZE step ───────────────────────

#[test]
fn test_pipeline_chains_cosine_filter_then_normalize() {
    let mut db = TensorDb::new();
    setup_docs(&mut db);

    exec(
        &mut db,
        "DEFINE PIPELINE near_norm AS WHERE COSINE_SIM(embedding, [1.0, 0.0, 0.0]) > 0.5 THEN NORMALIZE embedding",
        5,
    );
    exec(
        &mut db,
        "APPLY PIPELINE near_norm ON docs INTO near_norm_results",
        6,
    );

    let out = exec(&mut db, "SELECT * FROM near_norm_results", 7);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ids(&ds), vec![1, 2]);
    for row in &ds.rows {
        let Value::Vector(v) = row.get("embedding").unwrap() else {
            panic!("expected vector")
        };
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding should be unit-normalized after NORMALIZE step, got norm={norm}"
        );
    }
}

// ── MATMUL inside a pipeline SELECT step ────────────────────────────────────

#[test]
fn test_pipeline_select_computes_matmul_column() {
    let mut db = TensorDb::new();
    exec(
        &mut db,
        "DATASET mats COLUMNS (id: Int, m: Matrix(2, 2))",
        1,
    );
    exec(&mut db, "INSERT INTO mats VALUES (1, [[1, 0], [0, 1]])", 2);
    exec(&mut db, "INSERT INTO mats VALUES (2, [[2, 0], [0, 2]])", 3);

    exec(
        &mut db,
        "DEFINE PIPELINE transform AS SELECT id, MATMUL(m, m) AS m2",
        4,
    );
    exec(
        &mut db,
        "APPLY PIPELINE transform ON mats INTO mats_transformed",
        5,
    );

    let out = exec(&mut db, "SELECT * FROM mats_transformed", 6);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 2);
    for row in &ds.rows {
        let id = row.get("id").unwrap().clone();
        let Value::Matrix(m2) = row.get("m2").unwrap() else {
            panic!("expected matrix")
        };
        match id {
            // identity squared = identity
            Value::Int(1) => assert_eq!(m2, &vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            // (2I)^2 = 4I
            Value::Int(2) => assert_eq!(m2, &vec![vec![4.0, 0.0], vec![0.0, 4.0]]),
            other => panic!("unexpected id {other:?}"),
        }
    }
}
