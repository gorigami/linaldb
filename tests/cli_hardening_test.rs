// tests/cli_hardening_test.rs
// Integration tests for CLI features like 'init', 'run' (multiline), and 'load'.

use std::fs;
use std::path::Path;
use std::process::Command;

fn get_bin() -> String {
    "target/debug/linal".to_string()
}

#[test]
fn test_cli_init() {
    // Ensure clean state
    let _ = fs::remove_file("linal.toml");
    let _ = fs::remove_dir_all("./data");

    let output = Command::new(get_bin())
        .arg("init")
        .output()
        .expect("Failed to execute init command");

    assert!(output.status.success());
    assert!(Path::new("linal.toml").exists());
    assert!(Path::new("./data").exists());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Initialization complete"));

    // Cleanup
    let _ = fs::remove_file("linal.toml");
    let _ = fs::remove_dir_all("./data");
}

#[test]
fn test_cli_run_multiline() {
    // Clean up leftover state from previous runs so CREATE DATABASE is idempotent.
    let _ = fs::remove_dir_all("./data/hardening_db");

    let output = Command::new(get_bin())
        .arg("run")
        .arg("examples/hardening_test.lnl")
        .output()
        .expect("Failed to execute run command");

    assert!(
        output.status.success(),
        "script failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for success markers
    assert!(stdout.contains("Database 'hardening_db' created"));
    assert!(stdout.contains("Created dataset: test_ds"));
    assert!(stdout.contains("Query Result (rows: 2, columns: 3)"));

    // Cleanup
    let _ = fs::remove_dir_all("./data/hardening_db");
}

#[test]
fn test_cli_serve_alias() {
    // Just check help to see if serve is listed
    let output = Command::new(get_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("HTTP server"));
}
