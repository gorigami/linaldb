// tests/show_memory_test.rs
//
// SHOW MEMORY [<dataset>] (CASMI_WORKLOADS_PLAN.md, P7 memory report):
// estimated bytes per dataset's rows, per index and per tensor.

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

/// (kind, name, column, detail, rows, bytes)
fn report(db: &mut TensorDb, line: &str) -> Vec<(String, String, String, String, i64, i64)> {
    match run(db, line) {
        DslOutput::Table(ds) => ds
            .rows
            .iter()
            .map(|r| match &r.values[..] {
                [Value::String(k), Value::String(n), Value::String(c), Value::String(d), Value::Int(rows), Value::Int(b)] => {
                    (k.clone(), n.clone(), c.clone(), d.clone(), *rows, *b)
                }
                other => panic!("unexpected row {:?}", other),
            })
            .collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

fn setup(db: &mut TensorDb, n: usize, dim: usize) {
    run(
        db,
        &format!(
            "DATASET lib COLUMNS (id: Int, tag: String, e: Vector({}))",
            dim
        ),
    );
    for i in 0..n {
        let v: Vec<String> = (0..dim)
            .map(|j| format!("{}.0", (i * 7 + j * 3) % 11 + 1))
            .collect();
        run(
            db,
            &format!(
                "INSERT INTO lib VALUES ({}, \"t{}\", [{}])",
                i,
                i % 3,
                v.join(", ")
            ),
        );
    }
}

#[test]
fn reports_datasets_indexes_and_tensors() {
    let (_dir, mut db) = db();
    let (n, dim) = (100, 16);
    setup(&mut db, n, dim);
    run(&mut db, "CREATE VECTOR INDEX ON lib(e) USING HNSW");
    run(&mut db, "CREATE INDEX ON lib(tag)");
    run(&mut db, "VECTOR v = [1.0, 2.0, 3.0]");

    let r = report(&mut db, "SHOW MEMORY");
    let vector_bytes = (n * dim * 4) as i64;

    let ds = r.iter().find(|x| x.0 == "dataset" && x.1 == "lib").unwrap();
    assert_eq!(ds.4, n as i64);
    assert!(
        ds.5 > vector_bytes,
        "dataset rows hold at least the vectors: {:?}",
        ds
    );

    let hnsw = r.iter().find(|x| x.0 == "index" && x.2 == "e").unwrap();
    assert_eq!(hnsw.3, "Hnsw");
    // Two copies of every vector (index list + graph points) plus the graph.
    assert!(hnsw.5 > 2 * vector_bytes, "{:?}", hnsw);

    let hash = r.iter().find(|x| x.0 == "index" && x.2 == "tag").unwrap();
    assert_eq!(hash.3, "Hash");
    assert!(hash.5 > 0);

    let t = r.iter().find(|x| x.0 == "tensor" && x.1 == "v").unwrap();
    assert_eq!(t.4, 3);
    assert!(t.5 >= 12);
}

#[test]
fn ivf_index_counts_its_vector_copies() {
    let (_dir, mut db) = db();
    setup(&mut db, 80, 8);
    run(&mut db, "CREATE VECTOR INDEX ON lib(e)");
    let r = report(&mut db, "SHOW MEMORY lib");
    assert!(
        r.iter().all(|x| x.1 == "lib"),
        "filter keeps only lib: {:?}",
        r
    );
    let ivf = r.iter().find(|x| x.0 == "index").unwrap();
    assert_eq!(ivf.3, "Vector");
    assert!(ivf.5 > (80 * 8 * 4) as i64, "{:?}", ivf);
}

#[test]
fn unknown_dataset_is_an_error_and_memory_named_object_still_shows() {
    let (_dir, mut db) = db();
    assert!(execute_line(&mut db, "SHOW MEMORY nope", 1).is_err());
    run(&mut db, "VECTOR MEMORY = [1.0, 2.0]");
    match run(&mut db, "SHOW MEMORY") {
        DslOutput::Tensor(t) => assert_eq!(t.data.len(), 2),
        other => panic!(
            "SHOW MEMORY should show the tensor named MEMORY, got {:?}",
            other
        ),
    }
}
