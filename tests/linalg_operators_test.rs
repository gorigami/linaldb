//! DSL-level integration tests for LINEAGE_AND_LINALG_PLAN.md Phase 8's
//! linear algebra operators. `src/core/linalg.rs`'s own unit tests cover the
//! numerics in isolation; these exercise the same operators through the
//! real DSL pipeline (lexer -> parser -> executor -> engine), including the
//! multi-output `LET a, b[, c] = ...` binding syntax (Phase 8.4) and that
//! every operator's provenance is real (Phase 8's own "every operator emits
//! a real ProvenanceRecord from the moment it's implemented" rule).

use linal::dsl::{execute_line, DslError, DslOutput};
use linal::engine::TensorDb;

fn run(db: &mut TensorDb, line: &str, n: usize) -> DslOutput {
    execute_line(db, line, n).unwrap_or_else(|e| panic!("line {n} ({line:?}) failed: {e:?}"))
}

fn run_err(db: &mut TensorDb, line: &str, n: usize) -> DslError {
    execute_line(db, line, n).expect_err(&format!("line {n} ({line:?}) should have failed"))
}

fn tensor_data(db: &TensorDb, name: &str) -> Vec<f32> {
    db.get(name).unwrap().to_logical_vec()
}

#[test]
fn scalar_operators_hand_computable() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX m = [[4, 7], [2, 6]]", 1);

    run(&mut db, "LET tr = TRACE m", 2);
    assert_eq!(tensor_data(&db, "tr"), vec![10.0]);

    run(&mut db, "LET det = DETERMINANT m", 3);
    assert_eq!(tensor_data(&db, "det"), vec![10.0]);

    run(&mut db, "LET r = RANK m", 4);
    assert_eq!(tensor_data(&db, "r"), vec![2.0]);

    run(&mut db, "LET inv = INVERSE m", 5);
    let inv = tensor_data(&db, "inv");
    assert!((inv[0] - 0.6).abs() < 1e-4);
    assert!((inv[1] - (-0.7)).abs() < 1e-4);
    assert!((inv[2] - (-0.2)).abs() < 1e-4);
    assert!((inv[3] - 0.4).abs() < 1e-4);

    run(&mut db, "VECTOR b = [4, 6]", 6);
    run(&mut db, "LET x = SOLVE m b", 7);
    let x = tensor_data(&db, "x");
    assert!((x[0] - (-1.8)).abs() < 1e-4);
    assert!((x[1] - 1.6).abs() < 1e-4);

    run(&mut db, "MATRIX sym = [[5, 0], [0, 3]]", 8);
    run(&mut db, "LET eigs = EIGENVALUES sym", 9);
    let mut eigs = tensor_data(&db, "eigs");
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((eigs[0] - 3.0).abs() < 1e-4);
    assert!((eigs[1] - 5.0).abs() < 1e-4);
}

#[test]
fn error_paths_are_loud_not_nan() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX singular = [[1, 2], [2, 4]]", 1);
    let err = run_err(&mut db, "LET i = INVERSE singular", 2);
    assert!(format!("{err:?}").contains("singular"));

    run(&mut db, "VECTOR b2 = [1, 2]", 3);
    let err = run_err(&mut db, "LET x2 = SOLVE singular b2", 4);
    assert!(format!("{err:?}").contains("singular"));

    run(&mut db, "MATRIX nonsquare = [[1, 2, 3], [4, 5, 6]]", 5);
    let err = run_err(&mut db, "LET t2 = TRACE nonsquare", 6);
    assert!(format!("{err:?}").contains("square"));

    run(&mut db, "MATRIX asym = [[1, 2], [0, 1]]", 7);
    let err = run_err(&mut db, "LET e2 = EIGENVALUES asym", 8);
    assert!(format!("{err:?}").contains("symmetric"));
}

#[test]
fn multi_output_let_binds_decompositions() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX a = [[4, 7], [2, 6]]", 1);

    run(&mut db, "LET q, r = QR a", 2);
    let (q, rr) = (tensor_data(&db, "q"), tensor_data(&db, "r"));
    // Reconstruct Q @ R and check it equals A.
    let qm = |i: usize, j: usize| q[i * 2 + j];
    let rm = |i: usize, j: usize| rr[i * 2 + j];
    let a00 = qm(0, 0) * rm(0, 0) + qm(0, 1) * rm(1, 0);
    let a01 = qm(0, 0) * rm(0, 1) + qm(0, 1) * rm(1, 1);
    assert!((a00 - 4.0).abs() < 1e-3);
    assert!((a01 - 7.0).abs() < 1e-3);

    run(
        &mut db,
        "MATRIX piv = [[0, 2, 1], [1, 1, 1], [2, 0, 1]]",
        3,
    );
    run(&mut db, "LET p, l, u = LU piv", 4);
    assert_eq!(tensor_data(&db, "p").len(), 9);
    assert_eq!(tensor_data(&db, "l").len(), 9);
    assert_eq!(tensor_data(&db, "u").len(), 9);

    run(&mut db, "MATRIX sym2 = [[2, 1], [1, 2]]", 5);
    run(&mut db, "LET vals, vecs = EIGEN sym2", 6);
    let mut vals = tensor_data(&db, "vals");
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((vals[0] - 1.0).abs() < 1e-4);
    assert!((vals[1] - 3.0).abs() < 1e-4);
    assert_eq!(tensor_data(&db, "vecs").len(), 4);

    run(&mut db, "MATRIX rect = [[1, 0, 0], [0, 1, 0]]", 7);
    run(&mut db, "LET su, ss, svt = SVD rect", 8);
    let mut s = tensor_data(&db, "ss");
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((s[0] - 1.0).abs() < 1e-4);
    assert!((s[1] - 1.0).abs() < 1e-4);
    assert_eq!(tensor_data(&db, "su").len(), 4);
    assert_eq!(tensor_data(&db, "svt").len(), 6);
}

#[test]
fn cholesky_is_single_output_not_multi() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX psd = [[4, 2], [2, 3]]", 1);
    run(&mut db, "LET l = CHOLESKY psd", 2);
    let l = tensor_data(&db, "l");
    // L * L^T ~= psd: [[4,2],[2,3]]
    assert!((l[0] * l[0] - 4.0).abs() < 1e-3);
    assert!((l[2] * l[0] - 2.0).abs() < 1e-3);
    assert!((l[2] * l[2] + l[3] * l[3] - 3.0).abs() < 1e-3);

    // Using multi-output LET for a genuinely single-output op is a clear
    // error, not silently binding garbage into a second name.
    let err = run_err(&mut db, "LET l2, junk = CHOLESKY psd", 3);
    assert!(format!("{err:?}").contains("single output"));
}

#[test]
fn pca_projects_to_requested_components() {
    let mut db = TensorDb::new();
    run(
        &mut db,
        "MATRIX data = [[1,2,3],[4,5,6],[7,8,9],[1,0,1]]",
        1,
    );
    run(&mut db, "LET proj = PCA data COMPONENTS 2", 2);
    assert_eq!(tensor_data(&db, "proj").len(), 8); // 4 rows x 2 components
}

#[test]
fn single_output_op_cannot_be_used_in_multi_output_let() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX m = [[1, 0], [0, 1]]", 1);
    let err = run_err(&mut db, "LET a, b = TRACE m", 2);
    assert!(format!("{err:?}").contains("single output"));
}

#[test]
fn multi_output_op_cannot_be_used_in_single_output_let() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX m = [[1, 0], [0, 1]]", 1);
    let err = run_err(&mut db, "LET q = QR m", 2);
    assert!(format!("{err:?}").contains("QR"));
}

#[test]
fn decomposition_result_has_real_provenance() {
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX a = [[4, 7], [2, 6]]", 1);
    run(&mut db, "LET p, l, u = LU a", 2);

    let tree = match run(&mut db, "EXPLAIN LINEAGE l", 3) {
        DslOutput::Message(msg) => msg,
        other => panic!("expected Message, got {other:?}"),
    };
    assert!(tree.contains("LU"));
    assert!(tree.contains("ROOT"));

    let json = match run(&mut db, "EXPLAIN LINEAGE l AS JSON", 4) {
        DslOutput::Message(msg) => msg,
        other => panic!("expected Message, got {other:?}"),
    };
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["operation"], "LU");
}
