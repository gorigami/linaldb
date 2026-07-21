use linal::core::value::Value;
use linal::dsl::{execute_line, execute_script, DslOutput};
use linal::engine::TensorDb;

fn db_with_embeddings() -> TensorDb {
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
DATASET docs COLUMNS (id: Int, category: String, embedding: Vector(3))
INSERT INTO docs VALUES (1, "ml", [0.9, 0.1, 0.0])
INSERT INTO docs VALUES (2, "ml", [0.8, 0.2, 0.0])
INSERT INTO docs VALUES (3, "db", [0.0, 0.1, 0.9])
INSERT INTO docs VALUES (4, "db", [0.1, 0.0, 0.8])
"#,
    )
    .expect("Setup failed");
    db
}

fn query_rows(db: &mut TensorDb, sql: &str) -> Vec<Vec<Value>> {
    match execute_line(db, sql, 0).expect("Query failed") {
        DslOutput::Table(ds) => ds.rows.into_iter().map(|t| t.values).collect(),
        other => panic!("Expected Table output, got: {}", other),
    }
}

#[test]
fn test_vec_literal_in_select() {
    let mut db = db_with_embeddings();
    let rows = query_rows(&mut db, "SELECT [1.0, 2.0, 3.0] AS v FROM docs LIMIT 1");
    assert_eq!(rows.len(), 1);
    assert!(
        matches!(&rows[0][0], Value::Vector(v) if v.len() == 3),
        "Expected 3-dim vector, got {:?}",
        rows[0][0]
    );
}

#[test]
fn test_cosine_sim_with_literal() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, COSINE_SIM(embedding, [1.0, 0.0, 0.0]) AS score FROM docs ORDER BY id",
    );
    assert_eq!(rows.len(), 4);
    let score1 = match rows[0][1] {
        Value::Float(f) => f,
        ref v => panic!("Expected Float for score, got {:?}", v),
    };
    let score3 = match rows[2][1] {
        Value::Float(f) => f,
        ref v => panic!("Expected Float for score, got {:?}", v),
    };
    assert!(
        score1 > score3,
        "ml doc should score higher on [1,0,0] query"
    );
}

#[test]
fn test_l2_norm_returns_float() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, L2_NORM(embedding) AS norm FROM docs ORDER BY id",
    );
    assert_eq!(rows.len(), 4);
    for row in &rows {
        assert!(
            matches!(row[1], Value::Float(_)),
            "L2_NORM must return Float, got {:?}",
            row[1]
        );
    }
}

#[test]
fn test_dot_product() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, DOT(embedding, [1.0, 0.0, 0.0]) AS d FROM docs ORDER BY id",
    );
    assert_eq!(rows.len(), 4);
    if let Value::Float(d) = rows[0][1] {
        assert!(
            (d - 0.9_f32).abs() < 1e-4,
            "DOT([0.9,0.1,0],[1,0,0]) should be ~0.9, got {}",
            d
        );
    }
}

#[test]
fn test_distance_sql_form() {
    // DISTANCE(a, b) inside a SELECT list -- previously only usable as the
    // standalone `LET x = DISTANCE a TO b` tensor-DSL keyword; "DISTANCE"
    // lexes to a dedicated keyword token (unlike COSINE_SIM/DOT, which have
    // none), so the SQL-callable form needed its own parser arm rather than
    // reaching the existing generic-identifier dispatch.
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, DISTANCE(embedding, [1.0, 0.0, 0.0]) AS d FROM docs ORDER BY id",
    );
    assert_eq!(rows.len(), 4);
    if let Value::Float(d) = rows[0][1] {
        let expected = ((0.9_f32 - 1.0).powi(2) + 0.1_f32.powi(2)).sqrt();
        assert!(
            (d - expected).abs() < 1e-4,
            "DISTANCE([0.9,0.1,0],[1,0,0]) should be ~{}, got {}",
            expected,
            d
        );
    } else {
        panic!("Expected Float for d, got {:?}", rows[0][1]);
    }

    // The standalone keyword form must still work unaffected.
    execute_script(
        &mut db,
        r#"
VECTOR a = [1.0, 0.0, 0.0]
VECTOR b = [0.0, 1.0, 0.0]
LET d = DISTANCE a TO b
"#,
    )
    .expect("standalone DISTANCE a TO b should still parse and execute");
}

#[test]
fn test_normalize_returns_unit_vector() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, NORMALIZE(embedding) AS n FROM docs ORDER BY id",
    );
    assert_eq!(rows.len(), 4);
    for row in &rows {
        if let Value::Vector(v) = &row[1] {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "NORMALIZE result should be unit vector, norm={}",
                norm
            );
        } else {
            panic!("Expected Vector from NORMALIZE, got {:?}", row[1]);
        }
    }
}

#[test]
fn test_vec_add() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, VEC_ADD(embedding, [0.0, 0.0, 0.1]) AS shifted FROM docs LIMIT 2",
    );
    assert_eq!(rows.len(), 2);
    if let Value::Vector(v) = &rows[0][1] {
        // doc1 embedding[2] = 0.0 + 0.1 = 0.1
        assert!((v[2] - 0.1_f32).abs() < 1e-4, "VEC_ADD third dim: {}", v[2]);
    } else {
        panic!("Expected Vector from VEC_ADD");
    }
}

#[test]
fn test_vec_scale() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id, VEC_SCALE(embedding, 2.0) AS scaled FROM docs LIMIT 1",
    );
    assert_eq!(rows.len(), 1);
    if let Value::Vector(v) = &rows[0][1] {
        // doc1 embedding = [0.9, 0.1, 0.0], scaled by 2 = [1.8, 0.2, 0.0]
        let expected = [1.8_f32, 0.2, 0.0];
        for (s, e) in v.iter().zip(expected.iter()) {
            assert!((s - e).abs() < 1e-4, "VEC_SCALE mismatch: {} vs {}", s, e);
        }
    } else {
        panic!("Expected Vector from VEC_SCALE");
    }
}

#[test]
fn test_cosine_sim_where_filter() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT id FROM docs WHERE COSINE_SIM(embedding, [1.0, 0.0, 0.0]) > 0.7 ORDER BY id",
    );
    assert!(!rows.is_empty(), "Expected at least one result");
    for row in &rows {
        let id = match row[0] {
            Value::Int(i) => i,
            _ => panic!("Expected Int id"),
        };
        assert!(
            id == 1 || id == 2,
            "Unexpected id {} passed cosine filter",
            id
        );
    }
}

#[test]
fn test_avg_vec_group_by() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT category, AVG_VEC(embedding) AS centroid FROM docs GROUP BY category ORDER BY category",
    );
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert!(
            matches!(&row[1], Value::Vector(v) if v.len() == 3),
            "AVG_VEC should return a 3-dim vector, got {:?}",
            row[1]
        );
    }
}

#[test]
fn test_plain_avg_and_sum_work_on_vector_columns() {
    // CONSISTENCY_PLAN.md Track D / D4: the executor already merges
    // Sum/SumVec and Avg/AvgVec into the same accumulator logic, and SUM
    // already worked without the _VEC suffix — but AVG's schema-inference
    // hardcoded Float regardless of input type, so `AVG(vector_col)` used
    // to fail with "Type mismatch... expected FLOAT, got VECTOR" even
    // though the executor could compute it fine. Fixed by inferring AVG's
    // result type the same way SUM/MIN/MAX do.
    let mut db = db_with_embeddings();

    let sum_rows = query_rows(
        &mut db,
        "SELECT category, SUM(embedding) AS total FROM docs GROUP BY category ORDER BY category",
    );
    assert_eq!(sum_rows.len(), 2);
    for row in &sum_rows {
        assert!(
            matches!(&row[1], Value::Vector(v) if v.len() == 3),
            "plain SUM should return a 3-dim vector, got {:?}",
            row[1]
        );
    }

    let avg_rows = query_rows(
        &mut db,
        "SELECT category, AVG(embedding) AS centroid FROM docs GROUP BY category ORDER BY category",
    );
    assert_eq!(avg_rows.len(), 2);
    for row in &avg_rows {
        assert!(
            matches!(&row[1], Value::Vector(v) if v.len() == 3),
            "plain AVG should return a 3-dim vector, got {:?}",
            row[1]
        );
    }
}

#[test]
fn test_avg_on_scalar_column_still_returns_float_not_int() {
    // Guard against the D4 fix over-broadening: AVG on a scalar Int column
    // must still produce a Float (fractional averages), not an Int.
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
DATASET nums COLUMNS (id: Int, cat: String, price: Int)
INSERT INTO nums VALUES (1, "a", 10)
INSERT INTO nums VALUES (2, "a", 21)
"#,
    )
    .expect("setup failed");

    let rows = query_rows(
        &mut db,
        "SELECT cat, AVG(price) AS avg FROM nums GROUP BY cat",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][1], Value::Float(15.5));
}

#[test]
fn test_sum_vec_group_by() {
    let mut db = db_with_embeddings();
    let rows = query_rows(
        &mut db,
        "SELECT category, SUM_VEC(embedding) AS total FROM docs GROUP BY category ORDER BY category",
    );
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert!(
            matches!(&row[1], Value::Vector(v) if v.len() == 3),
            "SUM_VEC should return a 3-dim vector, got {:?}",
            row[1]
        );
    }
}

#[test]
fn test_l2_norm_of_literal_vector() {
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
DATASET vecs COLUMNS (id: Int)
INSERT INTO vecs VALUES (1)
"#,
    )
    .expect("Setup");
    let rows = query_rows(
        &mut db,
        "SELECT id, L2_NORM([3.0, 4.0]) AS five FROM vecs LIMIT 1",
    );
    assert_eq!(rows.len(), 1);
    if let Value::Float(f) = rows[0][1] {
        assert!(
            (f - 5.0_f32).abs() < 1e-4,
            "L2_NORM([3,4]) should be 5, got {}",
            f
        );
    } else {
        panic!("Expected Float from L2_NORM, got {:?}", rows[0][1]);
    }
}
