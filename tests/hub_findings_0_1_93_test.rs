// tests/hub_findings_0_1_93_test.rs
//
// Five engine issues found by re-running linal-hub's notebooks on the
// published linaldb 0.1.17 and building notebook 16 (MS/MS identification
// on MassBank + GNPS). Each test pins the fixed behavior.

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn open(dir: &std::path::Path) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.to_path_buf();
    TensorDb::with_config(config)
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

/// 1. Exact score ties (duplicate vectors) go in row order with every
///    index type. IVF used to put rows added after the index first, HNSW
///    the lower row id, so the top-1 depended on the index type.
#[test]
fn ties_go_in_row_order_for_every_index_type() {
    for using in ["", " USING HNSW"] {
        for n in [20, 200] {
            let dir = tempfile::tempdir().unwrap();
            let mut db = open(dir.path());
            run(&mut db, "DATASET t COLUMNS (id: Int, e: Vector(3))");
            for i in 0..n {
                let a = i as f32 * 0.37;
                run(
                    &mut db,
                    &format!(
                        "INSERT INTO t VALUES ({}, [{}, {}, {}])",
                        i,
                        a.sin(),
                        a.cos(),
                        0.5
                    ),
                );
            }
            run(&mut db, &format!("CREATE VECTOR INDEX ON t(e){}", using));
            // A duplicate of row 7, added after the index was built.
            let a = 7.0f32 * 0.37;
            run(
                &mut db,
                &format!(
                    "INSERT INTO t VALUES (9999, [{}, {}, {}])",
                    a.sin(),
                    a.cos(),
                    0.5
                ),
            );
            let q = format!("[{}, {}, {}]", a.sin(), a.cos(), 0.5);
            let r = rows(run(&mut db, &format!("SEARCH t ON e QUERY {} LIMIT 2", q)));
            let ids: Vec<Value> = r.iter().map(|x| x[0].clone()).collect();
            assert_eq!(
                ids,
                vec![Value::Int(7), Value::Int(9999)],
                "index{} n={}",
                using,
                n
            );
            let exact = rows(run(
                &mut db,
                &format!("SEARCH t ON e QUERY {} PREFILTER true LIMIT 2", q),
            ));
            assert_eq!(exact.iter().map(|x| x[0].clone()).collect::<Vec<_>>(), ids);
        }
    }
}

/// 2. stats.json (and the .meta.json stats) are byte-identical for
///    identical data; their column order used to change run to run.
#[test]
fn saved_stats_are_deterministic() {
    let save = || {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open(dir.path());
        run(
            &mut db,
            "DATASET s COLUMNS (zeta: Int, alpha: Float, mid: String, beta: Int)",
        );
        for i in 0..20 {
            run(
                &mut db,
                &format!(
                    "INSERT INTO s VALUES ({}, {}.5, \"m{}\", {})",
                    i,
                    i,
                    i % 3,
                    i * 2
                ),
            );
        }
        run(&mut db, "SAVE DATASET s");
        let stats =
            std::fs::read_to_string(dir.path().join("default/datasets/s/stats.json")).unwrap();
        (dir, stats)
    };
    let (_d1, first) = save();
    for _ in 0..4 {
        assert_eq!(save().1, first);
    }
    let order: Vec<usize> = ["alpha", "beta", "mid", "zeta"]
        .iter()
        .map(|c| first.find(&format!("\"{}\"", c)).unwrap())
        .collect();
    assert!(
        order.windows(2).all(|w| w[0] < w[1]),
        "columns in name order: {}",
        first
    );
}

/// 3. A global aggregate (no GROUP BY) over no rows returns one row: COUNT
///    0, everything else NULL. It used to return no rows at all.
#[test]
fn global_aggregate_over_no_rows_returns_one_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path());
    run(&mut db, "DATASET t COLUMNS (id: Int, x: Float)");
    let q = "SELECT COUNT(*) AS n, SUM(x) AS s, AVG(x) AS a, MAX(x) AS m, ARG_MAX(id, x) AS am, RRF(id) AS r FROM t";
    let empty = rows(run(&mut db, q));
    assert_eq!(
        empty,
        vec![vec![
            Value::Int(0),
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null
        ]]
    );
    run(&mut db, "INSERT INTO t VALUES (1, 2.0)");
    let none_match = rows(run(&mut db, "SELECT COUNT(*) AS n FROM t WHERE id > 100"));
    assert_eq!(none_match, vec![vec![Value::Int(0)]]);
    let grouped = rows(run(
        &mut db,
        "SELECT id, COUNT(*) AS n FROM t WHERE id > 100 GROUP BY id",
    ));
    assert!(grouped.is_empty(), "with GROUP BY, no rows means no groups");
    let one = rows(run(&mut db, "SELECT COUNT(*) AS n, SUM(x) AS s FROM t"));
    assert_eq!(one, vec![vec![Value::Int(1), Value::Float(2.0)]]);
}

/// 4. SEARCH ... INTO records lineage: the searched dataset and the
///    queries' dataset are the result's parents (it used to be a root).
#[test]
fn search_into_records_lineage() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path());
    run(&mut db, "DATASET lib COLUMNS (id: Int, e: Vector(2))");
    run(&mut db, "INSERT INTO lib VALUES (1, [1.0, 0.0])");
    run(&mut db, "INSERT INTO lib VALUES (2, [0.0, 1.0])");
    run(&mut db, "CREATE VECTOR INDEX ON lib(e)");
    run(&mut db, "DATASET q COLUMNS (qid: Int, qe: Vector(2))");
    run(&mut db, "INSERT INTO q VALUES (5, [1.0, 0.1])");
    run(
        &mut db,
        "SEARCH lib ON e QUERY [1.0, 0.0] LIMIT 1 INTO single",
    );
    run(
        &mut db,
        "SEARCH lib ON e QUERIES q.qe KEY qid LIMIT 1 INTO batch",
    );
    let lineage = |db: &mut TensorDb, n: &str| match run(db, &format!("EXPLAIN LINEAGE {}", n)) {
        DslOutput::Message(m) => m,
        other => format!("{:?}", other),
    };
    let single = lineage(&mut db, "single");
    assert!(
        single.contains("SEARCH (single)") && single.contains("(lib)"),
        "{}",
        single
    );
    let batch = lineage(&mut db, "batch");
    assert!(
        batch.contains("SEARCH (batch)") && batch.contains("(lib)") && batch.contains("(q)"),
        "{}",
        batch
    );
}

/// 5. SHOW SCHEMA lists variable-width matrices and quantized vectors in
///    DSL syntax, not as `Matrix(2, 0)` / `QVector(4, F16)`.
#[test]
fn show_schema_uses_dsl_type_names() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open(dir.path());
    run(
        &mut db,
        "DATASET s COLUMNS (spec: Matrix(2, *), e: Vector(4, F16), m: Matrix(2, 3))",
    );
    let schema = match run(&mut db, "SHOW SCHEMA s") {
        DslOutput::Message(m) => m,
        other => format!("{:?}", other),
    };
    assert!(schema.contains("Matrix(2, *)"), "{}", schema);
    assert!(schema.contains("Vector(4, F16)"), "{}", schema);
    assert!(schema.contains("Matrix(2, 3)"), "{}", schema);
    assert!(
        !schema.contains("QVector") && !schema.contains("Matrix(2, 0)"),
        "{}",
        schema
    );
}
