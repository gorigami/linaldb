// tests/filtered_hnsw_test.rs
//
// SEARCH ... PREFILTER <pred> APPROX (CASMI_WORKLOADS_PLAN.md, large tier):
// the passing rows are ranked through the HNSW graph instead of an exact
// scan. Compared with the exact PREFILTER on the same data.

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

const N: usize = 6000;
const D: usize = 32;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

fn vectors(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n * D).map(|_| lcg(&mut s)).collect()
}

fn batch(ids: Vec<i64>, mass: Vec<f64>, flat: Vec<f32>) -> RecordBatch {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let e: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(item, D as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("mass", DataType::Float64, false),
        Field::new("e", e.data_type().clone(), false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(mass)),
            e,
        ],
    )
    .unwrap()
}

fn db(hnsw: bool) -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db = TensorDb::with_config(config);
    let mass: Vec<f64> = (0..N)
        .map(|i| 100.0 + (i * 7919 % N) as f64 * 0.05)
        .collect();
    db.load_record_batch(
        "lib",
        &batch((0..N as i64).collect(), mass, vectors(N, 3)),
        "test",
    )
    .unwrap();
    let nq = 20;
    let qmass: Vec<f64> = (0..nq).map(|j| 150.0 + j as f64 * 5.0).collect();
    db.load_record_batch(
        "q",
        &batch((0..nq as i64).collect(), qmass, vectors(nq, 9)),
        "test",
    )
    .unwrap();
    if hnsw {
        execute_line(&mut db, "CREATE VECTOR INDEX ON lib(e) USING HNSW", 1).unwrap();
    }
    (dir, db)
}

fn hits(db: &mut TensorDb, q: &str) -> BTreeMap<i64, Vec<i64>> {
    let out = execute_line(db, q, 1).unwrap_or_else(|e| panic!("{}: {}", q, e));
    let DslOutput::Table(t) = out else {
        panic!("{:?}", out)
    };
    let mut m = BTreeMap::new();
    for r in &t.rows {
        let (Value::Int(qid), Value::Int(id)) = (&r.values[0], &r.values[4]) else {
            panic!("{:?}", r.values)
        };
        m.entry(*qid).or_insert_with(Vec::new).push(*id);
    }
    m
}

#[test]
fn broad_filter_through_the_graph_has_high_recall_and_full_k() {
    let (_dir, mut db) = db(true);
    // About half the library passes: well past the exact-fallback size.
    let pred = "PREFILTER mass < 250.0";
    let exact = hits(
        &mut db,
        &format!("SEARCH lib ON e QUERIES q.e KEY id {} LIMIT 10", pred),
    );
    let approx = hits(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERIES q.e KEY id {} APPROX LIMIT 10",
            pred
        ),
    );
    let (mut found, mut total) = (0, 0);
    for (qid, truth) in &exact {
        let got = &approx[qid];
        assert_eq!(got.len(), 10, "query {} must get k rows", qid);
        let truth: HashSet<_> = truth.iter().collect();
        found += got.iter().filter(|id| truth.contains(id)).count();
        total += 10;
    }
    let recall = found as f64 / total as f64;
    eprintln!("filtered HNSW recall@10 = {:.3}", recall);
    assert!(recall >= 0.9, "recall@10 = {}", recall);

    // Every APPROX hit passes the filter.
    let out = execute_line(
        &mut db,
        &format!(
            "SEARCH lib ON e QUERIES q.e KEY id {} APPROX LIMIT 10",
            pred
        ),
        1,
    )
    .unwrap();
    let DslOutput::Table(t) = out else { panic!() };
    assert!(t
        .rows
        .iter()
        .all(|r| matches!(r.values[5], Value::Float64(m) if m < 250.0)));
}

#[test]
fn narrow_filter_falls_back_to_the_exact_answer() {
    let (_dir, mut db) = db(true);
    let pred = "PREFILTER mass BETWEEN q.mass - 2.0 AND q.mass + 2.0";
    let exact = hits(
        &mut db,
        &format!("SEARCH lib ON e QUERIES q.e KEY id {} LIMIT 5", pred),
    );
    let approx = hits(
        &mut db,
        &format!("SEARCH lib ON e QUERIES q.e KEY id {} APPROX LIMIT 5", pred),
    );
    assert_eq!(exact, approx);
}

#[test]
fn explain_and_missing_index() {
    let (_dir, mut with_index) = db(true);
    let plan = format!(
        "{:?}",
        execute_line(
            &mut with_index,
            "EXPLAIN SEARCH lib ON e QUERIES q.e PREFILTER mass < 250.0 APPROX LIMIT 3",
            1
        )
        .unwrap()
    );
    assert!(
        plan.contains("HNSW graph over rows passing PREFILTER (APPROX)"),
        "{}",
        plan
    );

    let (_dir2, mut plain) = db(false);
    let e = execute_line(
        &mut plain,
        "SEARCH lib ON e QUERIES q.e PREFILTER mass < 250.0 APPROX LIMIT 3",
        1,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("APPROX needs an HNSW index"), "{}", e);
}
