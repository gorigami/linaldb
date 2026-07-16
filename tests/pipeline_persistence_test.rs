use linal::core::config::EngineConfig;
use linal::dsl::{execute_line, DslOutput};
use linal::{execute_script, TensorDb, Value};
use tempfile::TempDir;

fn make_db(data_dir: &std::path::Path) -> TensorDb {
    let mut config = EngineConfig::default();
    config.storage.data_dir = data_dir.to_path_buf();
    TensorDb::with_config(config)
}

// ── 1. Simple save / load roundtrip ──────────────────────────────────────────

#[test]
fn test_pipeline_save_load_roundtrip() {
    let dir = TempDir::new().unwrap();

    {
        let mut db = make_db(dir.path());
        execute_line(&mut db, "DEFINE PIPELINE simple AS SELECT id, score", 1).unwrap();
        execute_line(&mut db, "SAVE PIPELINE simple", 2).unwrap();
    }

    let mut db = make_db(dir.path());
    execute_line(&mut db, "LOAD PIPELINE simple", 1).unwrap();

    assert!(
        db.pipelines.contains_key("simple"),
        "pipeline should be restored"
    );
    assert_eq!(db.pipelines["simple"].steps.len(), 1);
    assert!(
        db.pipelines["simple"].source.contains("SELECT"),
        "source should include SELECT"
    );
}

// ── 2. SAVE TO / LOAD FROM explicit path ─────────────────────────────────────

#[test]
fn test_pipeline_save_load_explicit_path() {
    let dir = TempDir::new().unwrap();
    let pipe_path = dir.path().join("ep_pipe.json");
    let pipe_path_str = pipe_path.to_str().unwrap();

    let mut db = make_db(dir.path());
    execute_line(
        &mut db,
        "DEFINE PIPELINE ep_pipe AS WHERE active = 1 THEN LIMIT 5",
        1,
    )
    .unwrap();
    execute_line(
        &mut db,
        &format!("SAVE PIPELINE ep_pipe TO '{}'", pipe_path_str),
        2,
    )
    .unwrap();

    assert!(
        pipe_path.exists(),
        "pipeline file should exist at explicit path"
    );

    execute_line(
        &mut db,
        &format!("LOAD PIPELINE ep_pipe FROM '{}'", pipe_path_str),
        3,
    )
    .unwrap();

    assert!(db.pipelines.contains_key("ep_pipe"));
    assert_eq!(
        db.pipelines["ep_pipe"].steps.len(),
        2,
        "WHERE + LIMIT = 2 steps"
    );
}

// ── 3. LOAD non-existent pipeline returns an error ───────────────────────────

#[test]
fn test_pipeline_load_nonexistent_returns_error() {
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());

    let result = execute_line(&mut db, "LOAD PIPELINE ghost", 1);
    assert!(
        result.is_err(),
        "expected Err loading a pipeline that was never saved"
    );
}

// ── 5. LOAD PIPELINE overwrites the in-memory definition ─────────────────────

#[test]
fn test_pipeline_load_overwrites_in_memory() {
    let dir = TempDir::new().unwrap();

    {
        let mut db = make_db(dir.path());
        execute_line(
            &mut db,
            "DEFINE PIPELINE overwrite_me AS SELECT id, score",
            1,
        )
        .unwrap();
        execute_line(&mut db, "SAVE PIPELINE overwrite_me", 2).unwrap();
    }

    let mut db = make_db(dir.path());
    execute_line(
        &mut db,
        "DEFINE PIPELINE overwrite_me AS WHERE active = 1 THEN LIMIT 999",
        1,
    )
    .unwrap();
    assert!(
        db.pipelines["overwrite_me"].source.contains("LIMIT 999"),
        "in-memory definition should have LIMIT 999 before load"
    );

    execute_line(&mut db, "LOAD PIPELINE overwrite_me", 2).unwrap();

    let source = &db.pipelines["overwrite_me"].source;
    assert!(
        source.contains("SELECT"),
        "loaded pipeline should restore the saved SELECT definition, got: {}",
        source
    );
    assert!(
        !source.contains("LIMIT 999"),
        "LIMIT 999 definition should have been overwritten"
    );
}

// ── 6. LOAD PIPELINE stores under the requested name, not the name embedded
//       in the file's source (e.g. after copying/renaming a saved pipeline file) ──

#[test]
fn test_pipeline_load_uses_requested_name_not_source_name() {
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());

    execute_line(&mut db, "DEFINE PIPELINE original AS SELECT id", 1).unwrap();
    let saved_path = dir.path().join("renamed.json");
    execute_line(
        &mut db,
        &format!(
            "SAVE PIPELINE original TO '{}'",
            saved_path.to_str().unwrap()
        ),
        2,
    )
    .unwrap();

    execute_line(
        &mut db,
        &format!(
            "LOAD PIPELINE renamed FROM '{}'",
            saved_path.to_str().unwrap()
        ),
        3,
    )
    .unwrap();

    assert!(
        db.pipelines.contains_key("renamed"),
        "pipeline should be stored under the requested name 'renamed'"
    );
    assert_eq!(db.pipelines["renamed"].steps.len(), 1);
}

// ── 7. Pipeline with NORMALIZE step survives roundtrip and runs correctly ─────

#[test]
fn test_pipeline_vector_ops_roundtrip() {
    let dir = TempDir::new().unwrap();

    {
        let mut db = make_db(dir.path());
        execute_line(
            &mut db,
            "DEFINE PIPELINE norm_pipe AS NORMALIZE embedding",
            1,
        )
        .unwrap();
        execute_line(&mut db, "SAVE PIPELINE norm_pipe", 2).unwrap();
    }

    let mut db = make_db(dir.path());
    execute_script(
        &mut db,
        r#"
        DATASET vecs COLUMNS (id: Int, embedding: Vector(3))
        INSERT INTO vecs VALUES (1, [3.0, 0.0, 0.0])
        INSERT INTO vecs VALUES (2, [0.0, 4.0, 0.0])
        "#,
    )
    .unwrap();

    execute_line(&mut db, "LOAD PIPELINE norm_pipe", 1).unwrap();
    execute_line(&mut db, "APPLY PIPELINE norm_pipe ON vecs INTO normed", 2).unwrap();

    let result = execute_line(&mut db, "SELECT * FROM normed", 3).unwrap();
    if let DslOutput::Table(ds) = result {
        assert_eq!(ds.len(), 2, "expected 2 normalized rows");
        if let Value::Vector(v) = &ds.rows[0].values[1] {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "expected unit-norm vector, got norm={}",
                norm
            );
        } else {
            panic!(
                "expected Vector in embedding column, got {:?}",
                ds.rows[0].values[1]
            );
        }
    } else {
        panic!("expected Table output from SELECT");
    }
}

// ── 7. Relative TO/FROM paths resolve against data_dir, like TENSOR/DATASET ──

#[test]
fn test_pipeline_relative_explicit_path_resolves_against_data_dir() {
    // CONSISTENCY_PLAN.md Track D / D5: pipeline's explicit TO/FROM paths
    // used to be taken as-is (CWD-relative), unlike TENSOR/DATASET, whose
    // explicit relative paths resolve against `<data_dir>/<db>/`. This
    // pins the now-consistent behavior.
    let dir = TempDir::new().unwrap();
    let mut db = make_db(dir.path());

    execute_line(
        &mut db,
        "DEFINE PIPELINE rel AS WHERE active = 1 THEN LIMIT 5",
        1,
    )
    .unwrap();
    execute_line(&mut db, "SAVE PIPELINE rel TO 'backups/rel.json'", 2).unwrap();

    let expected_path = dir.path().join("default").join("backups").join("rel.json");
    assert!(
        expected_path.exists(),
        "relative TO path should resolve under <data_dir>/<db>/, expected file at {:?}",
        expected_path
    );

    let mut db2 = make_db(dir.path());
    execute_line(&mut db2, "LOAD PIPELINE rel FROM 'backups/rel.json'", 3).unwrap();
    assert!(db2.pipelines.contains_key("rel"));
}
