// tests/bitvector_test.rs
//
// BitVector(N) columns and TANIMOTO / JACCARD / HAMMING / BIT_COUNT
// (CASMI_WORKLOADS_PLAN.md, P4): molecular fingerprints stored as bits and
// compared with set similarity. Expected values are computed independently
// below from the bit strings; tests/../clients/python-embedded checks them
// against RDKit too.

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

fn open(dir: &tempfile::TempDir) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
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

fn rows(out: DslOutput) -> Vec<Vec<Value>> {
    match out {
        DslOutput::Table(ds) => ds.rows.iter().map(|r| r.values.clone()).collect(),
        other => panic!("expected a table, got {:?}", other),
    }
}

/// Deterministic 70-bit strings (not a multiple of 8 or 64).
fn bits(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..70)
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

fn tanimoto(a: &str, b: &str) -> f64 {
    let (mut and, mut or) = (0, 0);
    for (x, y) in a.chars().zip(b.chars()) {
        and += (x == '1' && y == '1') as u32;
        or += (x == '1' || y == '1') as u32;
    }
    if or == 0 {
        1.0
    } else {
        and as f64 / or as f64
    }
}

fn setup(db: &mut TensorDb) -> Vec<String> {
    run(db, "DATASET fp COLUMNS (id: Int, bits: BitVector(70))");
    let all: Vec<String> = (1..=12).map(bits).collect();
    for (i, b) in all.iter().enumerate() {
        run(db, &format!("INSERT INTO fp VALUES ({}, \"{}\")", i, b));
    }
    all
}

#[test]
fn functions_match_independent_computation() {
    let (_dir, mut db) = db();
    let all = setup(&mut db);
    let q = &all[3];
    let r = rows(run(
        &mut db,
        &format!(
            "SELECT id, TANIMOTO(bits, CAST(\"{q}\" AS BITVECTOR(70))) AS t, JACCARD(bits, CAST(\"{q}\" AS BITVECTOR)) AS j, HAMMING(bits, CAST(\"{q}\" AS BITVECTOR(70))) AS h, BIT_COUNT(bits) AS c FROM fp ORDER BY id"
        ),
    ));
    for (i, row) in r.iter().enumerate() {
        let b = &all[i];
        let expected_t = tanimoto(b, q);
        assert_eq!(row[1], Value::Float64(expected_t), "row {}", i);
        assert_eq!(row[2], Value::Float64(expected_t));
        let ham = b.chars().zip(q.chars()).filter(|(x, y)| x != y).count() as i64;
        assert_eq!(row[3], Value::Int(ham));
        assert_eq!(row[4], Value::Int(b.matches('1').count() as i64));
    }
}

#[test]
fn bit_vectors_in_where_order_by_and_aggregates() {
    let (_dir, mut db) = db();
    let all = setup(&mut db);
    let q = &all[0];
    let mut expected: Vec<(usize, f64)> = all
        .iter()
        .enumerate()
        .map(|(i, b)| (i, tanimoto(b, q)))
        .collect();
    expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));

    let top = rows(run(
        &mut db,
        &format!("SELECT id, TANIMOTO(bits, CAST(\"{q}\" AS BITVECTOR(70))) AS t FROM fp ORDER BY t DESC, id LIMIT 3"),
    ));
    let got: Vec<i64> = top
        .iter()
        .map(|r| match r[0] {
            Value::Int(i) => i,
            _ => panic!(),
        })
        .collect();
    let want: Vec<i64> = expected.iter().take(3).map(|(i, _)| *i as i64).collect();
    assert_eq!(got, want);

    let threshold = expected[4].1;
    let passing = rows(run(
        &mut db,
        &format!(
            "SELECT id FROM fp WHERE TANIMOTO(bits, CAST(\"{q}\" AS BITVECTOR(70))) >= {threshold}"
        ),
    ));
    assert_eq!(
        passing.len(),
        expected.iter().filter(|(_, t)| *t >= threshold).count()
    );

    let agg = rows(run(
        &mut db,
        &format!("SELECT MAX(TANIMOTO(bits, CAST(\"{q}\" AS BITVECTOR(70)))) AS best FROM fp WHERE id > 0"),
    ));
    let best = expected
        .iter()
        .filter(|(i, _)| *i > 0)
        .map(|(_, t)| *t)
        .fold(0.0, f64::max);
    assert_eq!(agg[0][0], Value::Float64(best));

    // Equality on bit vectors works; CAST back to text and to Vector.
    let eq = rows(run(&mut db, &format!("SELECT id, CAST(bits AS TEXT) AS s, CAST(bits AS VECTOR(70)) AS v FROM fp WHERE bits = CAST(\"{}\" AS BITVECTOR(70))", all[5])));
    assert_eq!(eq.len(), 1);
    assert_eq!(eq[0][1], Value::String(all[5].clone()));
    match &eq[0][2] {
        Value::Vector(v) => assert_eq!(
            v.iter().filter(|x| **x == 1.0).count(),
            all[5].matches('1').count()
        ),
        other => panic!("{:?}", other),
    }
}

#[test]
fn save_and_load_round_trips_with_nulls_and_odd_lengths() {
    let (dir, mut db) = db();
    let all = setup(&mut db);
    run(&mut db, "DATASET n COLUMNS (id: Int, bits: BitVector(70)?)");
    run(
        &mut db,
        &format!("INSERT INTO n VALUES (1, \"{}\")", all[0]),
    );
    run(&mut db, "INSERT INTO n VALUES (2, null)");
    run(&mut db, "INSERT INTO n VALUES (3, [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])");
    run(&mut db, "SAVE DATASET fp");
    run(&mut db, "SAVE DATASET n");
    let before_fp = rows(run(&mut db, "SELECT * FROM fp ORDER BY id"));
    let before_n = rows(run(&mut db, "SELECT * FROM n ORDER BY id"));

    let mut db2 = open(&dir);
    run(&mut db2, "LOAD DATASET fp");
    run(&mut db2, "LOAD DATASET n");
    assert_eq!(
        rows(run(&mut db2, "SELECT * FROM fp ORDER BY id")),
        before_fp
    );
    assert_eq!(rows(run(&mut db2, "SELECT * FROM n ORDER BY id")), before_n);
    let types: Vec<String> = db2
        .get_dataset("n")
        .unwrap()
        .schema
        .fields
        .iter()
        .map(|f| f.value_type.to_string())
        .collect();
    assert_eq!(types, vec!["INT", "BITVECTOR[70]"]);
    let r = rows(run(
        &mut db2,
        "SELECT BIT_COUNT(bits) AS c FROM n WHERE id = 3",
    ));
    assert_eq!(r[0][0], Value::Int(3));
}

#[test]
fn loud_errors() {
    let (_dir, mut db) = db();
    setup(&mut db);
    run(
        &mut db,
        "DATASET other COLUMNS (id: Int, bits: BitVector(64), v: Vector(70))",
    );
    run(
        &mut db,
        &format!(
            "INSERT INTO other VALUES (1, \"{}\", [{}])",
            "1".repeat(64),
            vec!["0.5"; 70].join(", ")
        ),
    );

    let e = run_err(&mut db, "INSERT INTO fp VALUES (99, \"0101\")");
    assert!(e.contains("BitVector(70), got 4 bits"), "{}", e);
    let e = run_err(
        &mut db,
        &format!("INSERT INTO fp VALUES (99, \"{}2\")", "0".repeat(69)),
    );
    assert!(e.contains("only contain '0' and '1'"), "{}", e);
    let e = run_err(&mut db, "INSERT INTO other VALUES (2, [0.5], [1])");
    assert!(e.contains("only 0 and 1"), "{}", e);

    // Lengths and types are checked before any row is evaluated.
    let e = run_err(
        &mut db,
        "SELECT TANIMOTO(a.bits, b.bits) AS t FROM fp a JOIN other b ON a.id = b.id",
    );
    assert!(e.contains("lengths differ (70 vs 64)"), "{}", e);
    let e = run_err(&mut db, "SELECT TANIMOTO(v, v) AS t FROM other");
    assert!(
        e.contains("TANIMOTO expects BitVector arguments, got VECTOR[70]"),
        "{}",
        e
    );
    let e = run_err(
        &mut db,
        "SELECT id FROM other WHERE HAMMING(bits, CAST(\"0101\" AS BITVECTOR(4))) > 1",
    );
    assert!(
        e.contains("HAMMING: BitVector lengths differ (64 vs 4)"),
        "{}",
        e
    );
    let e = run_err(&mut db, "SELECT BIT_COUNT(id) AS c FROM fp");
    assert!(e.contains("BIT_COUNT expects BitVector"), "{}", e);
    let e = run_err(&mut db, "DATASET z COLUMNS (b: BitVector(0))");
    assert!(e.contains("at least 1 bit"), "{}", e);
}

#[test]
fn http_json_shape_is_the_bit_string() {
    let v = Value::BitVector(linal::core::bitvec::BitVec::from_bit_string("1010").unwrap());
    assert_eq!(
        serde_json::to_string(&v).unwrap(),
        r#"{"BitVector":"1010"}"#
    );
}
