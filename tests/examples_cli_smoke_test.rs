// tests/examples_cli_smoke_test.rs
//
// Smoke tests for every example .lnl file that isn't already covered by a
// deeper assertion-based test in examples_integration.rs. Each one just
// verifies `linal run examples/<name>.lnl` exits cleanly (no parse/engine
// errors), so a broken example can never sit in the repo unnoticed.
//
// See examples/README.md for the convention new examples should follow.

use std::fs;
use std::process::Command;
use std::sync::Mutex;

// Several examples run in the shared "default" instance or share cleanup
// paths under ./data; serialize them so they don't race each other or
// cli_hardening_test.rs's full ./data wipe.
static DATA_DIR_LOCK: Mutex<()> = Mutex::new(());

fn get_bin() -> String {
    "target/debug/linal".to_string()
}

fn assert_example_runs_clean(name: &str) {
    let output = Command::new(get_bin())
        .arg("run")
        .arg(format!("examples/{name}.lnl"))
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute run command for {name}: {e}"));

    assert!(
        output.status.success(),
        "{name}.lnl failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_example_benchmark_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/benchmark_db");
    assert_example_runs_clean("benchmark");
    let _ = fs::remove_dir_all("./data/benchmark_db");
}

#[test]
fn test_example_export_import_csv_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/phase1_test");
    assert_example_runs_clean("export_import_csv");
    let _ = fs::remove_dir_all("./data/phase1_test");
}

#[test]
fn test_example_pipelines_and_search_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/pipelines_and_search");
    assert_example_runs_clean("pipelines_and_search");
    let _ = fs::remove_dir_all("./data/pipelines_and_search");
}

#[test]
fn test_example_persistence_demo_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/persistence_demo");
    assert_example_runs_clean("persistence_demo");
    let _ = fs::remove_dir_all("./data/persistence_demo");
}

#[test]
fn test_example_metadata_demo_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/meta_test");
    assert_example_runs_clean("metadata_demo");
    let _ = fs::remove_dir_all("./data/meta_test");
}

#[test]
fn test_example_smoke_test_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    let _ = fs::remove_dir_all("./data/smoke_db_final");
    assert_example_runs_clean("smoke_test");
    let _ = fs::remove_dir_all("./data/smoke_db_final");
}

#[test]
fn test_example_reference_graph_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    assert_example_runs_clean("reference_graph");
    let _ = fs::remove_file("./data/default/datasets/raw_materialized.parquet");
}

#[test]
fn test_example_tensor_datasets_runs_clean() {
    let _guard = DATA_DIR_LOCK.lock().unwrap();
    assert_example_runs_clean("tensor_datasets");
    let _ = fs::remove_file("./data/default/datasets/sales_analytics.parquet");
}

#[test]
fn test_example_advanced_analytics_runs_clean() {
    assert_example_runs_clean("advanced_analytics");
}

#[test]
fn test_example_introspection_demo_runs_clean() {
    assert_example_runs_clean("introspection_demo");
}

#[test]
fn test_example_managed_service_demo_runs_clean() {
    assert_example_runs_clean("managed_service_demo");
}

#[test]
fn test_example_matrix_operations_runs_clean() {
    assert_example_runs_clean("matrix_operations");
}

#[test]
fn test_example_test_matrix_math_runs_clean() {
    assert_example_runs_clean("test_matrix_math");
}

#[test]
fn test_example_test_multiline_runs_clean() {
    assert_example_runs_clean("test_multiline");
}
