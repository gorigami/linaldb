use linal::core::config::{EngineConfig, StorageConfig};
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::fs;
use std::path::PathBuf;

fn setup_test_db(temp_dir: &str) -> TensorDb {
    let _ = fs::remove_dir_all(temp_dir);
    fs::create_dir_all(temp_dir).unwrap();

    let config = EngineConfig {
        storage: StorageConfig {
            data_dir: PathBuf::from(temp_dir),
            default_db: "default".to_string(),
        },
        wal: Default::default(),
        compute: Default::default(),
    };
    TensorDb::with_config(config)
}

#[test]
fn test_database_creation_and_switching() {
    let temp_dir = "/tmp/linal_test_db_creation";
    let mut db = setup_test_db(temp_dir);

    // 1. Check initial databases
    assert!(db.list_databases().contains(&"default".to_string()));

    // 2. Create a new database
    execute_line(&mut db, "CREATE DATABASE analytics", 1).unwrap();
    assert!(db.list_databases().contains(&"analytics".to_string()));

    // 3. Switch to it
    let output = execute_line(&mut db, "USE analytics", 2).unwrap();
    if let DslOutput::Message(msg) = output {
        assert!(msg.contains("Switched to database 'analytics'"));
    } else {
        panic!("Expected message output");
    }

    // 4. Create data in analytics
    execute_line(&mut db, "DATASET users COLUMNS (id: INT, name: STRING)", 3).unwrap();
    execute_line(&mut db, "INSERT INTO users VALUES (1, \"Alice\")", 4).unwrap();

    // 5. Verify it's there
    let ds_output = execute_line(&mut db, "SHOW ALL DATASETS", 5).unwrap();
    if let DslOutput::Message(msg) = ds_output {
        assert!(msg.contains("Dataset: users"));
    } else {
        panic!("Expected message output from SHOW ALL DATASETS");
    }

    // 6. Switch back to default and verify it's empty
    execute_line(&mut db, "USE default", 6).unwrap();
    let ds_output_default = execute_line(&mut db, "SHOW ALL DATASETS", 7).unwrap();
    if let DslOutput::Message(msg) = ds_output_default {
        // Output should just be header and footer if empty
        assert!(msg.contains("--- ALL DATASETS ---"));
        assert!(!msg.contains("Dataset: users"));
    } else {
        panic!("Expected message output");
    }

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_database_isolation() {
    let temp_dir = "/tmp/linal_test_db_isolation";
    let mut db = setup_test_db(temp_dir);

    // Create users in default (schema A)
    execute_line(&mut db, "DATASET users COLUMNS (id: INT, email: STRING)", 1).unwrap();
    execute_line(
        &mut db,
        "INSERT INTO users VALUES (1, \"admin@linal.ai\")",
        2,
    )
    .unwrap();

    // Create users in analytics (schema B)
    execute_line(&mut db, "CREATE DATABASE analytics", 3).unwrap();
    execute_line(&mut db, "USE analytics", 4).unwrap();
    execute_line(&mut db, "DATASET users COLUMNS (id: INT, name: STRING)", 5).unwrap();
    execute_line(&mut db, "INSERT INTO users VALUES (1, \"Alice\")", 6).unwrap();

    // Verify analytics users
    let output = execute_line(&mut db, "SELECT * FROM users", 7).unwrap();
    if let DslOutput::Table(ds) = output {
        assert_eq!(ds.schema.len(), 2);
        assert!(ds.schema.fields.iter().any(|f| f.name == "name"));
    } else {
        panic!("Expected table output");
    }

    // Switch back and verify default users
    execute_line(&mut db, "USE default", 8).unwrap();
    let output = execute_line(&mut db, "SELECT * FROM users", 9).unwrap();
    if let DslOutput::Table(ds) = output {
        assert_eq!(ds.schema.len(), 2);
        assert!(ds.schema.fields.iter().any(|f| f.name == "email"));
    } else {
        panic!("Expected table output");
    }

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_database_recovery() {
    let temp_dir = "/tmp/linal_test_db_recovery";
    let _ = fs::remove_dir_all(temp_dir);

    {
        let mut db = setup_test_db(temp_dir);
        execute_line(&mut db, "CREATE DATABASE persistent_db", 1).unwrap();
        execute_line(&mut db, "USE persistent_db", 2).unwrap();
        execute_line(&mut db, "DATASET vault COLUMNS (secret: STRING)", 3).unwrap();
        execute_line(&mut db, "INSERT INTO vault VALUES (\"hidden\")", 4).unwrap();

        // Save it to establish directory structure
        execute_line(&mut db, "SAVE DATASET vault", 5).unwrap();
    }

    // Re-instantiate engine with same directory
    let config = EngineConfig {
        storage: StorageConfig {
            data_dir: PathBuf::from(temp_dir),
            default_db: "default".to_string(),
        },
        wal: Default::default(),
        compute: Default::default(),
    };
    let mut db2 = TensorDb::with_config(config);

    // Should have discovered persistent_db
    assert!(db2.list_databases().contains(&"persistent_db".to_string()));

    // Should be able to LOAD it
    execute_line(&mut db2, "USE persistent_db", 1).unwrap();
    execute_line(&mut db2, "LOAD DATASET vault", 2).unwrap();

    let output = execute_line(&mut db2, "SELECT * FROM vault", 3).unwrap();
    if let DslOutput::Table(ds) = output {
        assert_eq!(ds.len(), 1);
    } else {
        panic!("Expected table output after recovery");
    }

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_drop_database() {
    let temp_dir = "/tmp/linal_test_db_drop";
    let mut db = setup_test_db(temp_dir);

    execute_line(&mut db, "CREATE DATABASE to_drop", 1).unwrap();
    assert!(db.list_databases().contains(&"to_drop".to_string()));

    execute_line(&mut db, "DROP DATABASE to_drop", 2).unwrap();
    assert!(!db.list_databases().contains(&"to_drop".to_string()));

    // Dropping non-existent
    let result = execute_line(&mut db, "DROP DATABASE non_existent", 3);
    assert!(result.is_err());

    // Dropping default
    let result_default = execute_line(&mut db, "DROP DATABASE default", 4);
    assert!(result_default.is_err());

    fs::remove_dir_all(temp_dir).unwrap();
}
