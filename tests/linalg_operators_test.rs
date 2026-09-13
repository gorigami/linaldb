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

    run(&mut db, "MATRIX piv = [[0, 2, 1], [1, 1, 1], [2, 0, 1]]", 3);
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

// ── The 12 linalg-operator keywords are non-reserved: usable as a plain
//    identifier (column name, `AS` alias, ordinary reference) anywhere the
//    grammar expects one -- not just as the operator they otherwise start.
//
// Found 2026-09-13 while building a real-data notebook outside this repo:
// `SELECT rank, storage_ratio FROM t` failed to parse ("expected `(`,
// found `,`"), because the ranking-window-function check in
// `parse_select_expr` (added when `RANK` first became a keyword, to keep
// `RANK() OVER (...)` parsing) had no lookahead for `(` before committing --
// unlike the equivalent, correctly-gated `Sum`/`Distance` precedent
// elsewhere in this parser. Investigating further found the same class of
// gap in two more places: `eat_ident` (the ~100+ call sites for column
// declarations, `AS` aliases, etc.) never accepted these keyword tokens at
// all, and `parse_expr_atom`'s primary-expression dispatch routed every one
// of them unconditionally into the operator parser even with no operand
// following (so `WHERE RANK > 0` or `SELECT x + RANK` also failed). All
// three fixed together; see `src/dsl/parser/mod.rs`'s `advance_if_ident`/
// `keyword_token_as_ident`, `src/dsl/parser/expr.rs`'s operand-lookahead
// guard, and `src/dsl/parser/dataset.rs`'s `parse_select_expr` fix.

#[test]
fn keyword_names_usable_as_ordinary_column_declarations_and_references() {
    let mut db = TensorDb::new();
    // Declaring columns literally named after every one of the 12 keywords
    // (plus COMPONENTS, PCA's own keyword) used to fail at `eat_ident`.
    run(
        &mut db,
        "DATASET t COLUMNS (trace: Int, determinant: Int, rank: Int, inverse: Int, \
         solve: Int, eigenvalues: Int, qr: Int, lu: Int, cholesky: Int, eigen: Int, \
         svd: Int, pca: Int, components: Int)",
        1,
    );
    run(
        &mut db,
        "INSERT INTO t VALUES (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)",
        2,
    );

    // Referencing them all back in a plain SELECT -- the originally
    // reported regression's exact shape.
    let out = run(
        &mut db,
        "SELECT trace, determinant, rank, inverse, solve, eigenvalues, qr, lu, \
         cholesky, eigen, svd, pca, components FROM t",
        3,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 1);
    for (i, v) in ds.rows[0].values.iter().enumerate() {
        assert_eq!(*v, linal::core::value::Value::Int(i as i64 + 1));
    }
}

#[test]
fn rank_keyword_usable_as_select_alias_and_in_where_and_arithmetic() {
    let mut db = TensorDb::new();
    run(&mut db, "DATASET t COLUMNS (id: Int, x: Float)", 1);
    run(&mut db, "INSERT INTO t VALUES (1, 5.0)", 2);

    // `AS RANK` -- an alias, not a declaration; a different `eat_ident` call
    // site than the COLUMNS test above.
    let out = run(&mut db, "SELECT id AS RANK FROM t", 3);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.schema.fields[0].name, "RANK");
    assert_eq!(ds.rows[0].values[0], linal::core::value::Value::Int(1));

    // A bare uppercase `RANK` reference in WHERE/arithmetic used to hit a
    // parse error (`parse_expr_atom` committing to the operator parser with
    // no valid operand following); it must at least *parse* now, resolving
    // to a real value when a same-named column exists.
    run(&mut db, "DATASET u COLUMNS (rank: Int, y: Int)", 4);
    run(&mut db, "INSERT INTO u VALUES (10, 1)", 5);
    let out = run(&mut db, "SELECT rank + y AS total FROM u WHERE rank > 5", 6);
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.rows[0].values[0], linal::core::value::Value::Int(11));
}

#[test]
fn rank_operator_and_rank_over_window_function_still_work() {
    // The two legitimate uses of the `RANK` keyword the fix above must not
    // regress: the standalone linear-algebra operator, and the SQL window
    // function it was originally added to disambiguate against.
    let mut db = TensorDb::new();
    run(&mut db, "MATRIX m = [[4, 7], [2, 6]]", 1);
    run(&mut db, "LET r = RANK m", 2);
    assert_eq!(tensor_data(&db, "r"), vec![2.0]);

    run(&mut db, "DATASET scores COLUMNS (id: Int, score: Float)", 3);
    run(&mut db, "INSERT INTO scores VALUES (1, 10.0)", 4);
    run(&mut db, "INSERT INTO scores VALUES (2, 20.0)", 5);
    let out = run(
        &mut db,
        "SELECT id, RANK() OVER (ORDER BY score DESC) AS r FROM scores",
        6,
    );
    let DslOutput::Table(ds) = out else {
        panic!("expected table")
    };
    assert_eq!(ds.rows[0].values[1], linal::core::value::Value::Int(2));
    assert_eq!(ds.rows[1].values[1], linal::core::value::Value::Int(1));
}
