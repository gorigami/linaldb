// tests/quantized_vector_test.rs
//
// Opt-in quantized vector columns, Vector(d, F16) / Vector(d, I8)
// (CASMI_WORKLOADS_PLAN.md, large tier): less memory, values rounded once
// on entry, every score consistent with COSINE_SIM on the stored values,
// and top-k recall measured against the full-precision answer.

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use linal::core::config::EngineConfig;
use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::collections::HashSet;
use std::sync::Arc;

fn db() -> (tempfile::TempDir, TensorDb) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    (dir, TensorDb::with_config(config))
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

fn vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n * d)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// `n` vectors of dimension `d` loaded three times: as f32, F16 and I8.
fn load(db: &mut TensorDb, n: usize, d: usize) -> Vec<f32> {
    let flat = vectors(n, d, 21);
    for (name, meta) in [("full", None), ("half", Some("F16")), ("byte", Some("I8"))] {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let e: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                item,
                d as i32,
                Arc::new(Float32Array::from(flat.clone())),
                None,
            )
            .unwrap(),
        );
        let mut field = Field::new("e", e.data_type().clone(), false);
        if let Some(enc) = meta {
            field = field.with_metadata(
                [(
                    "linal.logical_value_type".to_string(),
                    format!("QVector:{},{}", d, enc),
                )]
                .into(),
            );
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            field,
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
                e,
            ],
        )
        .unwrap();
        db.load_record_batch(name, &batch, "test").unwrap();
    }
    flat
}

fn lit(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[test]
fn types_scores_and_consistency() {
    let (_dir, mut db) = db();
    let d = 16;
    let flat = load(&mut db, 200, d);
    let types: Vec<String> = ["full", "half", "byte"]
        .iter()
        .map(|n| {
            db.get_dataset(n).unwrap().schema.fields[1]
                .value_type
                .to_string()
        })
        .collect();
    assert_eq!(
        types,
        vec!["VECTOR[16]", "VECTOR[16, F16]", "VECTOR[16, I8]"]
    );

    let q = lit(&flat[5 * d..6 * d]);
    run(&mut db, &format!("MATRIX qm = [{}]", q));
    for name in ["half", "byte"] {
        // The stored value is the quantized one; expressions see it as f32.
        let r = rows(run(
            &mut db,
            &format!("SELECT id, COSINE_SIM(e, {q}) AS c, CAST(e AS VECTOR({d})) AS v FROM {name} ORDER BY id"),
        ));
        for row in &r {
            let Value::Vector(v) = &row[2] else { panic!() };
            let qv: Vec<f32> = flat[5 * d..6 * d].to_vec();
            let dot: f32 = qv.iter().zip(v).map(|(a, b)| a * b).sum();
            let na: f32 = qv.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_eq!(row[1], Value::Float(dot / (na * nb)));
        }
        // Index scores (IVF and HNSW) are bit-identical to COSINE_SIM.
        for using in ["", " USING HNSW"] {
            run(
                &mut db,
                &format!("DATASET {name}_{} FROM {name} SELECT id, e", using.len()),
            );
            run(
                &mut db,
                &format!("CREATE VECTOR INDEX ON {name}_{}(e){using}", using.len()),
            );
            let hits = rows(run(
                &mut db,
                &format!("SEARCH {name}_{} ON e QUERIES qm LIMIT 5", using.len()),
            ));
            assert_eq!(hits.len(), 5);
            for h in &hits {
                let Value::Int(id) = h[3] else { panic!() };
                let c = &r[id as usize][1];
                assert_eq!(
                    &Value::Float(match h[2] {
                        Value::Float(f) => f,
                        _ => panic!(),
                    }),
                    c
                );
            }
        }
    }
}

#[test]
fn recall_against_full_precision() {
    let (_dir, mut db) = db();
    let (n, d) = (4000, 64);
    let flat = load(&mut db, n, d);
    for name in ["half", "byte"] {
        run(
            &mut db,
            &format!("CREATE VECTOR INDEX ON {name}(e) USING HNSW"),
        );
    }
    let queries = vectors(30, d, 77);
    let mut found = [0usize; 2];
    for qi in 0..30 {
        let q = &queries[qi * d..(qi + 1) * d];
        let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut exact: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &flat[i * d..(i + 1) * d];
                let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                (
                    i,
                    q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() / (qn * vn),
                )
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: HashSet<i64> = exact.iter().take(10).map(|(i, _)| *i as i64).collect();
        for (k, name) in ["half", "byte"].iter().enumerate() {
            let r = rows(run(
                &mut db,
                &format!("SEARCH {name} ON e QUERY {} LIMIT 10", lit(q)),
            ));
            found[k] += r
                .iter()
                .filter(|row| matches!(row[0], Value::Int(i) if truth.contains(&i)))
                .count();
        }
    }
    let recall_f16 = found[0] as f64 / 300.0;
    let recall_i8 = found[1] as f64 / 300.0;
    eprintln!(
        "recall@10 vs f32 truth: F16 {:.3}, I8 {:.3}",
        recall_f16, recall_i8
    );
    assert!(recall_f16 >= 0.95, "F16 recall@10 {}", recall_f16);
    assert!(recall_i8 >= 0.90, "I8 recall@10 {}", recall_i8);
}

#[test]
fn memory_save_load_and_errors() {
    let (dir, mut db) = db();
    let d = 64;
    load(&mut db, 1000, d);
    let report = rows(run(&mut db, "SHOW MEMORY"));
    let bytes = |name: &str| {
        report
            .iter()
            .find(|r| r[0] == Value::String("dataset".into()) && r[1] == Value::String(name.into()))
            .map(|r| match r[5] {
                Value::Int(b) => b as f64,
                _ => panic!(),
            })
            .unwrap()
    };
    let (full, half, byte) = (bytes("full"), bytes("half"), bytes("byte"));
    eprintln!("row bytes: f32 {} F16 {} I8 {}", full, half, byte);
    // The vector payload shrinks from 4 bytes per element to 2 (F16) or 1
    // (I8; its scale fits in the value's inline slot); per-row overhead
    // (the id, value slots) is unchanged.
    assert_eq!(full - half, (1000 * d * 2) as f64);
    assert_eq!(full - byte, (1000 * d * 3) as f64);
    let mut index = |name: &str| {
        run(
            &mut db,
            &format!("CREATE VECTOR INDEX ON {name}(e) USING HNSW"),
        );
        let r = rows(run(&mut db, &format!("SHOW MEMORY {name}")));
        r.iter()
            .find(|r| r[0] == Value::String("index".into()))
            .map(|r| match r[5] {
                Value::Int(b) => b as f64,
                _ => panic!(),
            })
            .unwrap()
    };
    let (fi, hi, bi) = (index("full"), index("half"), index("byte"));
    eprintln!("index bytes: f32 {} F16 {} I8 {}", fi, hi, bi);
    // Index sizes count Vec capacity (grown by doubling), so within 5%.
    let near = |got: f64, want: f64| (got - want).abs() <= 0.05 * want;
    assert!(near(fi - hi, (1000 * d * 2) as f64), "{} {}", fi, hi);
    assert!(
        near(fi - bi, (1000 * d * 3 - 1000 * 4) as f64),
        "{} {}",
        fi,
        bi
    ); // + a 4-byte I8 scale

    run(&mut db, "DATASET n COLUMNS (id: Int, e: Vector(3, I8)?)");
    run(&mut db, "INSERT INTO n VALUES (1, [0.5, -1.0, 0.25])");
    run(&mut db, "INSERT INTO n VALUES (2, null)");
    for name in ["half", "byte", "n"] {
        run(&mut db, &format!("SAVE DATASET {name}"));
    }
    let before: Vec<_> = ["half", "byte", "n"]
        .iter()
        .map(|n| rows(run(&mut db, &format!("SELECT * FROM {n} ORDER BY id"))))
        .collect();
    let mut config = EngineConfig::default();
    config.storage.data_dir = dir.path().to_path_buf();
    let mut db2 = TensorDb::with_config(config);
    for (k, name) in ["half", "byte", "n"].iter().enumerate() {
        run(&mut db2, &format!("LOAD DATASET {name}"));
        assert_eq!(
            rows(run(&mut db2, &format!("SELECT * FROM {name} ORDER BY id"))),
            before[k]
        );
    }

    run(&mut db, "DATASET h COLUMNS (e: Vector(2, F16))");
    let e = execute_line(&mut db, "INSERT INTO h VALUES ([1.0, 100000.0])", 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("outside the F16 range"), "{}", e);
    let e = execute_line(&mut db, "INSERT INTO h VALUES ([1.0])", 1)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("Vector(2, F16), got a vector of length 1"),
        "{}",
        e
    );
    let e = execute_line(&mut db, "DATASET z COLUMNS (e: Vector(2, F8))", 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown vector encoding 'F8'"), "{}", e);
    let r = rows(run(
        &mut db,
        "SELECT CAST([0.5, -1.0] AS VECTOR(2, I8)) AS q",
    ));
    assert!(matches!(&r[0][0], Value::QVector(_)));
}
