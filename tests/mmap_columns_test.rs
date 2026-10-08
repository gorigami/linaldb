// tests/mmap_columns_test.rs
//
// Memory-mapped column files for BitVector / Vector(d, F16|I8) columns
// (CASMI_WORKLOADS_PLAN_2.md, P14): `SAVE DATASET ... MMAP` writes them,
// `LOAD DATASET ... MMAP` maps them. Every operation must give exactly what
// the same dataset loaded into the heap gives.

use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::path::Path;

fn db_at(dir: &Path, mmap_columns: bool) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.to_path_buf();
    config.storage.mmap_columns = mmap_columns;
    TensorDb::with_config(config)
}

fn run(db: &mut TensorDb, line: &str) -> DslOutput {
    execute_line(db, line, 1).unwrap_or_else(|e| panic!("`{}` failed: {}", line, e))
}

fn run_err(db: &mut TensorDb, line: &str) -> String {
    match execute_line(db, line, 1) {
        Ok(out) => panic!("`{}` should have failed, got {:?}", line, out),
        Err(e) => e.to_string(),
    }
}

fn message(out: DslOutput) -> String {
    match out {
        DslOutput::Message(m) => m,
        other => panic!("expected a message, got {:?}", other),
    }
}

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

fn bits(seed: u64, n: usize) -> String {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x % 3 == 0 {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

fn vec_lit(seed: u64, d: usize) -> String {
    let v: Vec<String> = (0..d)
        .map(|i| {
            format!(
                "{:.3}",
                (((seed * 31 + i as u64 * 17) % 200) as f64 - 100.0) / 37.0
            )
        })
        .collect();
    format!("[{}]", v.join(", "))
}

/// 60 rows: a 130-bit fingerprint (3 words, the last partial), F16 and I8
/// vectors (the I8 one nullable, every 7th row NULL), a mass for windows.
fn build(db: &mut TensorDb) {
    run(
        db,
        "DATASET lib COLUMNS (id: Int, mass: DOUBLE, fp: BitVector(130), e: Vector(5, F16), e8: Vector(5, I8)?)",
    );
    for i in 0..60u64 {
        let e8 = if i % 7 == 0 {
            "NULL".to_string()
        } else {
            vec_lit(i + 100, 5)
        };
        run(
            db,
            &format!(
                "INSERT INTO lib VALUES ({}, {}.5, \"{}\", {}, {})",
                i,
                100 + i * 3,
                bits(i, 130),
                vec_lit(i, 5),
                e8
            ),
        );
    }
    run(db, "DATASET q COLUMNS (qid: Int, mass: DOUBLE, fp: BitVector(130), z: Vector(130), e: Vector(5))");
    for i in 0..4u64 {
        let z: Vec<String> = (0..130)
            .map(|k| format!("{:.2}", ((k * (i + 3)) % 10) as f64 / 10.0))
            .collect();
        run(
            db,
            &format!(
                "INSERT INTO q VALUES ({}, {}.0, \"{}\", [{}], {})",
                i,
                150 + i * 40,
                bits(1000 + i, 130),
                z.join(", "),
                vec_lit(500 + i, 5)
            ),
        );
    }
}

const QUERIES: &[&str] = &[
    "SELECT * FROM lib ORDER BY id",
    "SELECT id, TANIMOTO(fp, CAST(\"1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010\" AS BITVECTOR(130))) AS t, BIT_COUNT(fp) AS n FROM lib ORDER BY id",
    "SELECT id, COSINE_SIM(e, [1.0, 0.5, -0.5, 0.25, 2.0]) AS c, L2_NORM(e8) AS n FROM lib ORDER BY id",
    "SELECT COUNT(*) AS n, AVG(BIT_COUNT(fp)) AS b FROM lib WHERE e8 IS NOT NULL",
    "SEARCH lib ON fp QUERIES q.fp KEY qid USING TANIMOTO(fp, q.fp) PREFILTER mass BETWEEN q.mass - 60 AND q.mass + 60 RETURN id LIMIT 5",
    "SEARCH lib ON fp QUERIES q.z KEY qid USING DOT(fp, q.z) RETURN id LIMIT 5",
    "SEARCH lib ON e QUERIES q.e KEY qid PREFILTER mass > 120.0 RETURN id LIMIT 5",
    "SEARCH lib ON e QUERIES q.e KEY qid USING HAMMING(fp, q.fp) ASC CANDIDATES 10 RERANK USING COSINE_SIM(e, q.e) RETURN id LIMIT 3",
];

fn answers(db: &mut TensorDb) -> Vec<Vec<Vec<Value>>> {
    QUERIES.iter().map(|q| rows(run(db, q))).collect()
}

#[test]
fn mapped_columns_answer_exactly_like_heap_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = db_at(dir.path(), false);
    build(&mut db);
    let original = answers(&mut db);
    let saved = message(run(&mut db, "SAVE DATASET lib MMAP"));
    assert!(saved.contains("column files for: fp, e, e8"), "{}", saved);
    drop(db);

    // Same query dataset in both.
    let mut heap = {
        let mut d = db_at(dir.path(), false);
        build_queries_only(&mut d);
        run(&mut d, "LOAD DATASET lib");
        d
    };
    let mut mapped = {
        let mut d = db_at(dir.path(), false);
        build_queries_only(&mut d);
        let loaded = message(run(&mut d, "LOAD DATASET lib MMAP"));
        assert!(loaded.contains("memory-mapped: fp, e, e8"), "{}", loaded);
        d
    };
    let from_heap = answers(&mut heap);
    let from_map = answers(&mut mapped);
    assert_eq!(from_heap, original);
    assert_eq!(from_map, original);

    // SHOW MEMORY reports the mapped bytes per column, not as heap.
    let mem = rows(run(&mut mapped, "SHOW MEMORY lib"));
    let mapped_rows: Vec<(String, i64)> = mem
        .iter()
        .filter(|r| r[0] == Value::String("mapped".into()))
        .map(|r| match (&r[2], &r[5]) {
            (Value::String(c), Value::Int(b)) => (c.clone(), *b),
            other => panic!("{:?}", other),
        })
        .collect();
    // fp: 60 rows x 3 words x 8 bytes; e: 60 x 5 x 2; e8: 51 non-NULL x 5.
    assert_eq!(
        mapped_rows,
        [("fp".into(), 1440), ("e".into(), 600), ("e8".into(), 255)]
    );
    let heap_bytes = |db: &mut TensorDb| match &rows(run(db, "SHOW MEMORY lib"))[0][5] {
        Value::Int(b) => *b,
        other => panic!("{:?}", other),
    };
    assert!(heap_bytes(&mut mapped) < heap_bytes(&mut heap));
}

fn build_queries_only(db: &mut TensorDb) {
    let scratch = tempfile::tempdir().unwrap();
    let mut tmp = db_at(scratch.path(), false);
    build(&mut tmp);
    let q = rows(run(&mut tmp, "SELECT * FROM q ORDER BY qid"));
    run(db, "DATASET q COLUMNS (qid: Int, mass: DOUBLE, fp: BitVector(130), z: Vector(130), e: Vector(5))");
    for r in q {
        let lit = |v: &Value| match v {
            Value::Int(x) => x.to_string(),
            Value::Float64(x) => format!("{:?}", x),
            Value::BitVector(b) => format!("\"{}\"", b),
            Value::Vector(v) => format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{:?}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            other => panic!("{:?}", other),
        };
        let vals: Vec<String> = r.iter().map(lit).collect();
        run(db, &format!("INSERT INTO q VALUES ({})", vals.join(", ")));
    }
}

#[test]
fn writes_after_mapping_copy_the_cell_and_save_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = db_at(dir.path(), false);
    build(&mut db);
    run(&mut db, "SAVE DATASET lib MMAP");
    drop(db);
    let mut db = db_at(dir.path(), false);
    run(&mut db, "LOAD DATASET lib MMAP");
    let ones = "1".repeat(130);
    run(
        &mut db,
        &format!(
            "UPDATE lib SET fp = CAST(\"{}\" AS BITVECTOR(130)) WHERE id = 3",
            ones
        ),
    );
    run(&mut db, "DELETE FROM lib WHERE id = 4");
    run(
        &mut db,
        &format!(
            "INSERT INTO lib VALUES (99, 1.0, \"{}\", [1.0, 2.0, 3.0, 4.0, 5.0], NULL)",
            "0".repeat(130)
        ),
    );
    let expected = rows(run(
        &mut db,
        "SELECT id, BIT_COUNT(fp) AS n, e, e8 FROM lib ORDER BY id",
    ));
    assert_eq!(
        expected.iter().find(|r| r[0] == Value::Int(3)).unwrap()[1],
        Value::Int(130)
    );
    // Re-saving over the mapped files (replaced by rename) while mapped.
    run(&mut db, "SAVE DATASET lib MMAP");
    assert_eq!(
        rows(run(
            &mut db,
            "SELECT id, BIT_COUNT(fp) AS n, e, e8 FROM lib ORDER BY id"
        )),
        expected
    );
    drop(db);
    let mut db = db_at(dir.path(), false);
    run(&mut db, "LOAD DATASET lib MMAP");
    assert_eq!(
        rows(run(
            &mut db,
            "SELECT id, BIT_COUNT(fp) AS n, e, e8 FROM lib ORDER BY id"
        )),
        expected
    );
    // A plain SAVE drops the column files, so they can't go stale.
    run(&mut db, "SAVE DATASET lib");
    let e = run_err(&mut db, "LOAD DATASET lib MMAP");
    assert!(e.contains("no column files"), "{}", e);
}

#[test]
fn a_copied_package_maps_elsewhere_and_damage_is_caught() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = db_at(dir.path(), false);
    build(&mut db);
    let expected = rows(run(&mut db, "SELECT * FROM lib ORDER BY id"));
    run(&mut db, "SAVE DATASET lib MMAP");
    drop(db);

    // Another "machine": the package copied to an unrelated directory.
    // (The package directory plus its `<name>.meta.json` sidecar.)
    let src = dir.path().join("default").join("datasets");
    let other = tempfile::tempdir().unwrap();
    copy_dir(&src, &other.path().join("datasets"));
    let dst = other.path().join("datasets").join("lib");
    let fresh = tempfile::tempdir().unwrap();
    let mut db = db_at(fresh.path(), false);
    run(
        &mut db,
        &format!("LOAD DATASET lib FROM '{}' MMAP", other.path().display()),
    );
    assert_eq!(
        rows(run(&mut db, "SELECT * FROM lib ORDER BY id")),
        expected
    );
    drop(db);

    // A flipped byte: the content hash catches it.
    let fp = dst.join("columns").join("fp.lcol");
    let mut bytes = std::fs::read(&fp).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&fp, &bytes).unwrap();
    let mut db = db_at(fresh.path(), false);
    let e = run_err(
        &mut db,
        &format!("LOAD DATASET lib FROM '{}' MMAP", other.path().display()),
    );
    assert!(
        e.contains("fp.lcol") && e.contains("content hash mismatch"),
        "{}",
        e
    );
    // Truncated.
    std::fs::write(&fp, &bytes[..bytes.len() - 8]).unwrap();
    let e = run_err(
        &mut db,
        &format!("LOAD DATASET lib FROM '{}' MMAP", other.path().display()),
    );
    assert!(e.contains("truncated or extended"), "{}", e);
    // Not a column file at all.
    std::fs::write(&fp, b"hello").unwrap();
    let e = run_err(
        &mut db,
        &format!("LOAD DATASET lib FROM '{}' MMAP", other.path().display()),
    );
    assert!(e.contains("bad header"), "{}", e);
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
}

#[test]
fn config_switch_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    // [storage] mmap_columns = true: every SAVE writes column files and
    // every LOAD maps them.
    let mut db = db_at(dir.path(), true);
    build(&mut db);
    let saved = message(run(&mut db, "SAVE DATASET lib"));
    assert!(saved.contains("column files for"), "{}", saved);
    // ...and a dataset with nothing to map saves normally.
    run(&mut db, "DATASET plain COLUMNS (id: Int, v: Vector(2))");
    run(&mut db, "INSERT INTO plain VALUES (1, [1.0, 2.0])");
    let saved = message(run(&mut db, "SAVE DATASET plain"));
    assert!(!saved.contains("column files"), "{}", saved);
    drop(db);
    let mut db = db_at(dir.path(), true);
    let loaded = message(run(&mut db, "LOAD DATASET lib"));
    assert!(loaded.contains("memory-mapped: fp, e, e8"), "{}", loaded);
    let loaded = message(run(&mut db, "LOAD DATASET plain"));
    assert!(!loaded.contains("memory-mapped"), "{}", loaded);

    // Explicit MMAP where there is nothing to map.
    let e = run_err(&mut db, "SAVE DATASET plain MMAP");
    assert!(
        e.contains("no BitVector or Vector(d, F16|I8) column"),
        "{}",
        e
    );
    let e = run_err(&mut db, "LOAD DATASET plain MMAP");
    assert!(e.contains("no column files"), "{}", e);
    let e = run_err(&mut db, "SAVE TENSOR t MMAP");
    assert!(
        e.contains("MMAP applies to SAVE / LOAD DATASET only"),
        "{}",
        e
    );
}
