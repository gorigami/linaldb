use linal::dsl::execute_line;
use linal::engine::TensorDb;

#[test]
fn test_statistical_aggregations() {
    let mut db = TensorDb::new();

    // 1. Setup vectors
    execute_line(&mut db, "VECTOR v1 = [1, 2, 3, 4]", 1).unwrap();
    execute_line(&mut db, "VECTOR v2 = [1, 1, 1, 1]", 1).unwrap();

    // 2. Test SUM
    execute_line(&mut db, "LET s1 = SUM v1", 1).unwrap();
    let s1 = db.get("s1").unwrap();
    assert_eq!(s1.data_ref()[0], 10.0);
    // A true scalar (rank-0), not a rank-1 Vector(1) -- a [1]-shaped result here used to make
    // `vector - SUM(vector)` (and other elementwise ops against a reduction) silently corrupt
    // every element past index 0 by falling into the differing-length-vector padding path
    // instead of scalar broadcast. See kernels::sum_with_timestamp.
    assert_eq!(s1.shape.dims, Vec::<usize>::new());
    assert_eq!(s1.shape.rank(), 0);

    // 3. Test MEAN
    execute_line(&mut db, "LET m1 = MEAN v1", 1).unwrap();
    let m1 = db.get("m1").unwrap();
    assert_eq!(m1.data_ref()[0], 2.5);

    execute_line(&mut db, "LET m2 = MEAN v2", 1).unwrap();
    let m2 = db.get("m2").unwrap();
    assert_eq!(m2.data_ref()[0], 1.0);

    // 4. Test STDEV
    // v2 = [1, 1, 1, 1], mean = 1, diffs = [0, 0, 0, 0], stdev = 0
    execute_line(&mut db, "LET sd2 = STDEV v2", 1).unwrap();
    let sd2 = db.get("sd2").unwrap();
    assert_eq!(sd2.data_ref()[0], 0.0);

    // v1 = [1, 2, 3, 4], mean = 2.5
    // diffs = [-1.5, -0.5, 0.5, 1.5]
    // squared = [2.25, 0.25, 0.25, 2.25]
    // sum = 5.0
    // variance = 5.0 / 4 = 1.25
    // stdev = sqrt(1.25) approx 1.118
    execute_line(&mut db, "LET sd1 = STDEV v1", 1).unwrap();
    let sd1 = db.get("sd1").unwrap();
    let expected_sd1 = (1.25f32).sqrt();
    assert!((sd1.data_ref()[0] - expected_sd1).abs() < 1e-5);

    // 5. Test LAZY evaluation
    execute_line(&mut db, "LAZY LET s_lazy = SUM v1", 1).unwrap();
    // SHOW triggers evaluation
    let output = execute_line(&mut db, "SHOW s_lazy", 1).unwrap();
    match output {
        linal::dsl::DslOutput::Tensor(t) => {
            assert_eq!(t.data_ref()[0], 10.0);
        }
        _ => panic!("Expected Tensor output from SHOW"),
    }

    // 6. Test VARIANCE -- same v1, should be STDEV's value squared (1.25).
    execute_line(&mut db, "LET var1 = VARIANCE v1", 1).unwrap();
    let var1 = db.get("var1").unwrap();
    assert!((var1.data_ref()[0] - 1.25).abs() < 1e-5);
    assert_eq!(var1.shape.rank(), 0);

    // LAZY VARIANCE should also work, mirroring SUM/MEAN/STDEV.
    execute_line(&mut db, "LAZY LET var_lazy = VARIANCE v1", 1).unwrap();
    let output = execute_line(&mut db, "SHOW var_lazy", 1).unwrap();
    match output {
        linal::dsl::DslOutput::Tensor(t) => {
            assert!((t.data_ref()[0] - 1.25).abs() < 1e-5);
        }
        _ => panic!("Expected Tensor output from SHOW"),
    }

    // 7. Test MEDIAN -- v1 = [1,2,3,4], even count -> average of 2,3 = 2.5.
    execute_line(&mut db, "LET med1 = MEDIAN v1", 1).unwrap();
    let med1 = db.get("med1").unwrap();
    assert_eq!(med1.data_ref()[0], 2.5);

    execute_line(&mut db, "VECTOR v3 = [5, 1, 3]", 1).unwrap();
    execute_line(&mut db, "LET med3 = MEDIAN v3", 1).unwrap();
    let med3 = db.get("med3").unwrap();
    assert_eq!(med3.data_ref()[0], 3.0);

    // 8. Test QUANTILE -- AT 0.0 / 1.0 are the min/max; AT 0.5 matches MEDIAN.
    execute_line(&mut db, "LET q0 = QUANTILE v1 AT 0.0", 1).unwrap();
    assert_eq!(db.get("q0").unwrap().data_ref()[0], 1.0);
    execute_line(&mut db, "LET q1 = QUANTILE v1 AT 1.0", 1).unwrap();
    assert_eq!(db.get("q1").unwrap().data_ref()[0], 4.0);
    execute_line(&mut db, "LET q50 = QUANTILE v1 AT 0.5", 1).unwrap();
    assert_eq!(db.get("q50").unwrap().data_ref()[0], 2.5);

    // 9. Test COVARIANCE -- v1 with itself is its own variance.
    execute_line(&mut db, "LET cov_self = COVARIANCE v1 WITH v1", 1).unwrap();
    assert!((db.get("cov_self").unwrap().data_ref()[0] - 1.25).abs() < 1e-5);

    // v4 = -v1 (perfectly anti-correlated) -> covariance = -variance(v1)
    execute_line(&mut db, "VECTOR v4 = [-1, -2, -3, -4]", 1).unwrap();
    execute_line(&mut db, "LET cov_neg = COVARIANCE v1 WITH v4", 1).unwrap();
    assert!((db.get("cov_neg").unwrap().data_ref()[0] - (-1.25)).abs() < 1e-5);

    // 10. Test COVARIANCE MATRIX -- symmetric, diagonal matches per-column
    // sample variance (n - 1 denominator, unlike the population COVARIANCE
    // above -- see core::linalg::covariance_matrix's doc comment).
    execute_line(
        &mut db,
        "MATRIX cm_data = [[1, 2], [2, 1], [3, 4], [4, 3]]",
        1,
    )
    .unwrap();
    execute_line(&mut db, "LET cm = COVARIANCE MATRIX cm_data", 1).unwrap();
    let cm = db.get("cm").unwrap();
    assert_eq!(cm.shape.dims, vec![2, 2]);
    let cm_data = cm.to_logical_vec();
    // Column 0 = [1,2,3,4], mean 2.5, sample variance = 5/3.
    assert!((cm_data[0] - (5.0 / 3.0)).abs() < 1e-4);
    // Symmetric: cm[0][1] == cm[1][0]
    assert!((cm_data[1] - cm_data[2]).abs() < 1e-5);
}

/// Regression test for a silent-correctness bug found via a real end-to-end example
/// (leukemia gene-expression classification in linal-hub): `SUM`/`MEAN`/`STDEV` used to return
/// a rank-1 `Vector(1)` instead of a true scalar (rank-0), which made `vector - MEAN(vector)`
/// (and other elementwise ops against a reduction result) silently fall into the
/// differing-length-vector *padding* branch instead of scalar broadcast -- only index 0 got the
/// real operation, every other element passed through untouched.
#[test]
fn test_vector_minus_own_mean_centers_every_element() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "VECTOR v = [1, 2, 3, 4]", 1).unwrap();
    execute_line(&mut db, "LET m = MEAN v", 1).unwrap();
    execute_line(&mut db, "LET centered = v - m", 1).unwrap();

    let centered = db.get("centered").unwrap();
    assert_eq!(centered.data_ref(), &[-1.5, -0.5, 0.5, 1.5]);

    // Same shape-broadcast path, exercised via ADD/MULTIPLY/DIVIDE against SUM/STDEV results too.
    execute_line(&mut db, "LET s = SUM v", 1).unwrap();
    execute_line(&mut db, "LET plus_sum = v + s", 1).unwrap();
    let plus_sum = db.get("plus_sum").unwrap();
    assert_eq!(plus_sum.data_ref(), &[11.0, 12.0, 13.0, 14.0]); // s = 10

    execute_line(&mut db, "LET sd = STDEV v", 1).unwrap();
    execute_line(&mut db, "LET scaled = v / sd", 1).unwrap();
    let scaled = db.get("scaled").unwrap();
    let expected_sd = 1.25f32.sqrt();
    for (actual, original) in scaled.data_ref().iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - original / expected_sd).abs() < 1e-5);
    }
}

#[test]
fn test_matrix_aggregation() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "MATRIX m = [[1, 2], [3, 4]]", 1).unwrap();
    execute_line(&mut db, "LET s = SUM m", 1).unwrap();
    let s = db.get("s").unwrap();
    assert_eq!(s.data_ref()[0], 10.0);
}
