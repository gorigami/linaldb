use linal::core::dataset_legacy::Dataset;
use linal::dsl::execute_line_with_context;
use linal::dsl::DslOutput;
use linal::engine::TensorDb;

fn db() -> TensorDb {
    TensorDb::new()
}

fn exec(db: &mut TensorDb, line: &str) {
    execute_line_with_context(db, line, 1, None)
        .unwrap_or_else(|e| panic!("failed `{}`: {}", line, e));
}

fn as_table(out: DslOutput) -> Dataset {
    match out {
        DslOutput::Table(ds) => ds,
        other => panic!("expected Table, got {:?}", other),
    }
}

// ─── Feature 2: Matrix SQL expressions ───────────────────────────────────────

#[test]
fn parse_matrix_literal_in_select() {
    let mut db = db();
    exec(&mut db, "DATASET t COLUMNS (x: Int)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    let out = execute_line_with_context(
        &mut db,
        "SELECT [[1.0, 2.0], [3.0, 4.0]] AS m FROM t",
        1,
        None,
    )
    .unwrap();
    let table = as_table(out);
    assert_eq!(table.rows.len(), 1);
    let val = table.rows[0].get("m").unwrap();
    assert!(matches!(val, linal::core::value::Value::Matrix(_)));
    if let linal::core::value::Value::Matrix(m) = val {
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
        assert!((m[0][0] - 1.0_f32).abs() < 1e-6);
        assert!((m[1][1] - 4.0_f32).abs() < 1e-6);
    }
}

#[test]
fn mat_shape_returns_string() {
    let mut db = db();
    exec(&mut db, "DATASET mats COLUMNS (m: Matrix(2, 3))");
    exec(&mut db, "INSERT INTO mats (m = [[1, 2, 3], [4, 5, 6]])");
    let out = execute_line_with_context(&mut db, "SELECT MAT_SHAPE(m) AS shape FROM mats", 1, None)
        .unwrap();
    let table = as_table(out);
    let val = table.rows[0].get("shape").unwrap();
    assert_eq!(val, &linal::core::value::Value::String("2x3".to_string()));
}

#[test]
fn transpose_flips_rows_and_cols() {
    let mut db = db();
    exec(&mut db, "DATASET mats COLUMNS (m: Matrix(2, 3))");
    exec(&mut db, "INSERT INTO mats (m = [[1, 2, 3], [4, 5, 6]])");
    let out =
        execute_line_with_context(&mut db, "SELECT TRANSPOSE(m) AS t FROM mats", 1, None).unwrap();
    let table = as_table(out);
    let val = table.rows[0].get("t").unwrap();
    if let linal::core::value::Value::Matrix(m) = val {
        // transposed: 3 rows × 2 cols
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].len(), 2);
        assert!((m[0][0] - 1.0_f32).abs() < 1e-6);
        assert!((m[0][1] - 4.0_f32).abs() < 1e-6);
    } else {
        panic!("expected Matrix, got {:?}", val);
    }
}

#[test]
fn matmul_matrix_times_vector() {
    let mut db = db();
    exec(
        &mut db,
        "DATASET ops COLUMNS (m: Matrix(2, 2), v: Vector(2))",
    );
    exec(
        &mut db,
        "INSERT INTO ops (m = [[1, 0], [0, 1]], v = [3.0, 4.0])",
    );
    let out = execute_line_with_context(&mut db, "SELECT MATMUL(m, v) AS result FROM ops", 1, None)
        .unwrap();
    let table = as_table(out);
    let val = table.rows[0].get("result").unwrap();
    if let linal::core::value::Value::Vector(v) = val {
        assert_eq!(v.len(), 2);
        assert!((v[0] - 3.0_f32).abs() < 1e-6);
        assert!((v[1] - 4.0_f32).abs() < 1e-6);
    } else {
        panic!("expected Vector, got {:?}", val);
    }
}

#[test]
fn matmul_matrix_times_matrix() {
    let mut db = db();
    exec(
        &mut db,
        "DATASET ops COLUMNS (a: Matrix(2, 2), b: Matrix(2, 2))",
    );
    exec(
        &mut db,
        "INSERT INTO ops (a = [[1, 2], [3, 4]], b = [[1, 0], [0, 1]])",
    );
    let out = execute_line_with_context(&mut db, "SELECT MATMUL(a, b) AS result FROM ops", 1, None)
        .unwrap();
    let table = as_table(out);
    let val = table.rows[0].get("result").unwrap();
    if let linal::core::value::Value::Matrix(m) = val {
        // a @ I = a
        assert!((m[0][0] - 1.0_f32).abs() < 1e-6);
        assert!((m[0][1] - 2.0_f32).abs() < 1e-6);
        assert!((m[1][0] - 3.0_f32).abs() < 1e-6);
        assert!((m[1][1] - 4.0_f32).abs() < 1e-6);
    } else {
        panic!("expected Matrix, got {:?}", val);
    }
}

// ─── Feature 3: TRANSFORM statement ──────────────────────────────────────────

#[test]
fn transform_into_new_dataset() {
    let mut db = db();
    exec(
        &mut db,
        "DATASET employees COLUMNS (name: String, salary: Float)",
    );
    exec(
        &mut db,
        "INSERT INTO employees (name = \"Alice\", salary = 80000.0)",
    );
    exec(
        &mut db,
        "INSERT INTO employees (name = \"Bob\", salary = 50000.0)",
    );
    exec(
        &mut db,
        "INSERT INTO employees (name = \"Carol\", salary = 120000.0)",
    );

    exec(
        &mut db,
        "TRANSFORM employees SELECT name WHERE salary > 70000.0 INTO senior",
    );

    let out = execute_line_with_context(&mut db, "SELECT * FROM senior", 1, None).unwrap();
    let table = as_table(out);
    assert_eq!(table.rows.len(), 2);
}

#[test]
fn transform_in_place_replaces_source() {
    let mut db = db();
    exec(
        &mut db,
        "DATASET prices COLUMNS (item: String, price: Float)",
    );
    exec(&mut db, "INSERT INTO prices (item = \"A\", price = 10.0)");
    exec(&mut db, "INSERT INTO prices (item = \"B\", price = 200.0)");
    exec(&mut db, "INSERT INTO prices (item = \"C\", price = 50.0)");

    exec(&mut db, "TRANSFORM prices SELECT * WHERE price > 100.0");

    let out = execute_line_with_context(&mut db, "SELECT * FROM prices", 1, None).unwrap();
    let table = as_table(out);
    assert_eq!(table.rows.len(), 1);
    let val = table.rows[0].get("item").unwrap();
    assert_eq!(val, &linal::core::value::Value::String("B".to_string()));
}

#[test]
fn transform_select_star() {
    let mut db = db();
    exec(&mut db, "DATASET src COLUMNS (x: Int, y: Float)");
    exec(&mut db, "INSERT INTO src VALUES (1, 1.5)");
    exec(&mut db, "INSERT INTO src VALUES (2, 2.5)");
    exec(&mut db, "INSERT INTO src VALUES (3, 3.5)");

    exec(&mut db, "TRANSFORM src SELECT * WHERE x > 1 INTO dst");

    let out = execute_line_with_context(&mut db, "SELECT * FROM dst", 1, None).unwrap();
    let table = as_table(out);
    assert_eq!(table.rows.len(), 2);
}

// ─── Feature 1: Index-aware COSINE_SIM ───────────────────────────────────────

#[test]
fn cosine_sim_threshold_without_index_falls_back_to_scan() {
    let mut db = db();
    exec(&mut db, "DATASET vecs COLUMNS (id: Int, emb: Vector(3))");
    exec(&mut db, "INSERT INTO vecs (id = 1, emb = [1.0, 0.0, 0.0])");
    exec(&mut db, "INSERT INTO vecs (id = 2, emb = [0.0, 1.0, 0.0])");
    exec(&mut db, "INSERT INTO vecs (id = 3, emb = [0.9, 0.1, 0.0])");

    let out = execute_line_with_context(
        &mut db,
        "SELECT id FROM vecs WHERE COSINE_SIM(emb, [1.0, 0.0, 0.0]) > 0.8",
        1,
        None,
    )
    .unwrap();
    let table = as_table(out);
    // Row 1 (cosine=1.0) and row 3 (cosine≈0.994) should pass; row 2 (cosine=0.0) should not
    assert_eq!(table.rows.len(), 2);
}

#[test]
fn cosine_sim_threshold_with_vector_index() {
    let mut db = db();
    exec(&mut db, "DATASET vecs COLUMNS (id: Int, emb: Vector(3))");
    exec(&mut db, "INSERT INTO vecs (id = 1, emb = [1.0, 0.0, 0.0])");
    exec(&mut db, "INSERT INTO vecs (id = 2, emb = [0.0, 1.0, 0.0])");
    exec(
        &mut db,
        "INSERT INTO vecs (id = 3, emb = [0.9, 0.436, 0.0])",
    );
    exec(&mut db, "CREATE VECTOR INDEX ON vecs(emb)");

    let out = execute_line_with_context(
        &mut db,
        "SELECT id FROM vecs WHERE COSINE_SIM(emb, [1.0, 0.0, 0.0]) > 0.8",
        1,
        None,
    )
    .unwrap();
    let table = as_table(out);
    // Row 1 (cosine=1.0) and row 3 should pass; row 2 should not
    assert_eq!(table.rows.len(), 2);
}

#[test]
fn transform_parse_ok() {
    let stmt =
        linal::dsl::parser::parse("TRANSFORM employees SELECT name WHERE salary > 50000 INTO rich")
            .expect("parse failed");
    assert!(matches!(stmt, linal::dsl::ast::Statement::Transform(_)));
}
