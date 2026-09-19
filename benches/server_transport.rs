//! Phase 3 benchmark spike (`PERFORMANCE_OPTIMIZATION_PLAN.md`): measures the
//! actual cost of `/execute`'s current JSON response serialization against a
//! hand-rolled Arrow IPC encode of the same data, for representative query
//! result sizes -- the design-decision #1 gate that must run *before* any
//! server transport code changes. Not wired into any endpoint; this bench
//! exists purely to produce a real number to decide Phase 3's scope with.

use arrow::ipc::writer::StreamWriter;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linal::core::storage::dataset_to_record_batch;
use linal::dsl::execute_line;
use linal::engine::db::TensorDb;
use toon_format::encode_default;

/// A dataset shaped like a realistic `SEARCH`/bulk query result: an `id`
/// column, a `label` column, and a `Vector(128)` embedding column (a common
/// real embedding width, e.g. matches notebook 09's item-embedding size
/// order of magnitude) -- the shape where Arrow's columnar/binary float
/// encoding should matter most against JSON's per-float text encoding.
fn build_dataset(db: &mut TensorDb, rows: usize) {
    execute_line(
        db,
        "DATASET bench_ds COLUMNS (id: Int, label: String, embedding: Vector(128))",
        1,
    )
    .unwrap();
    let mut script = String::new();
    for i in 0..rows {
        let embedding: Vec<String> = (0..128)
            .map(|d| format!("{:.4}", ((i * 128 + d) as f32 * 0.001).sin()))
            .collect();
        script.push_str(&format!(
            "INSERT INTO bench_ds VALUES ({}, \"item_{}\", [{}])\n",
            i,
            i,
            embedding.join(", ")
        ));
    }
    linal::dsl::execute_script(db, &script).unwrap();
}

fn arrow_ipc_encode(batch: &arrow::record_batch::RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
    buf
}

fn json_vs_arrow_ipc(c: &mut Criterion) {
    let mut group = c.benchmark_group("execute_response_encoding");

    for &rows in &[100usize, 1_000, 10_000] {
        let mut db = TensorDb::new();
        build_dataset(&mut db, rows);
        let dataset = db.get_dataset("bench_ds").unwrap().clone();
        let batch = dataset_to_record_batch(&dataset).unwrap();

        group.bench_with_input(
            BenchmarkId::new("json_serialize", rows),
            &dataset,
            |b, dataset| {
                b.iter(|| black_box(serde_json::to_string(dataset).unwrap()));
            },
        );

        // The *actual* production default (`/execute` with no `?format=`
        // query param) -- JSON above is the legacy opt-in path. Comparing
        // Arrow IPC against this, not just JSON, is what actually decides
        // whether the real default response path has a real bottleneck.
        group.bench_with_input(
            BenchmarkId::new("toon_encode_default", rows),
            &dataset,
            |b, dataset| {
                b.iter(|| black_box(encode_default(dataset).unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("arrow_ipc_encode", rows),
            &batch,
            |b, batch| {
                b.iter(|| black_box(arrow_ipc_encode(batch)));
            },
        );

        // Also record the raw byte-size difference, printed once per size
        // (not part of the timed loop) -- payload size matters for network
        // transport cost independently of CPU encode time.
        let json_bytes = serde_json::to_string(&dataset).unwrap().len();
        let toon_bytes = encode_default(&dataset).unwrap().len();
        let arrow_bytes = arrow_ipc_encode(&batch).len();
        eprintln!(
            "[server_transport bench] rows={rows}: JSON={json_bytes} bytes, TOON={toon_bytes} bytes, \
             Arrow IPC={arrow_bytes} bytes (JSON/Arrow ratio {:.2}x, TOON/Arrow ratio {:.2}x)",
            json_bytes as f64 / arrow_bytes as f64,
            toon_bytes as f64 / arrow_bytes as f64
        );
    }

    group.finish();
}

criterion_group!(benches, json_vs_arrow_ipc);
criterion_main!(benches);
