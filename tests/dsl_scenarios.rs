// tests/dsl_scenarios.rs

use linal::{execute_script, TensorDb};

#[test]
fn dsl_basic_math_and_similarity() {
    let mut db = TensorDb::new();

    let script = r#"
# Creamos dos vectores 3D
DEFINE a AS TENSOR [3] VALUES [1, 2, 3]
DEFINE b AS TENSOR [3] VALUES [4, 5, 6]

LET sum      = ADD a b
LET prod     = MULTIPLY a b
LET ratio    = DIVIDE b a
LET corr     = CORRELATE a WITH b
LET sim      = SIMILARITY a WITH b
LET dist     = DISTANCE a TO b
LET a_norm   = NORMALIZE a
"#;

    execute_script(&mut db, script).expect("script should run");

    let sum = db.get("sum").unwrap();
    assert_eq!(sum.shape.dims, vec![3]);
    assert_eq!(sum.data_ref(), vec![5.0, 7.0, 9.0]);

    let prod = db.get("prod").unwrap();
    assert_eq!(prod.data_ref(), vec![4.0, 10.0, 18.0]);

    let ratio = db.get("ratio").unwrap();
    assert_eq!(ratio.data_ref(), vec![4.0, 2.5, 2.0]);

    let corr = db.get("corr").unwrap();
    assert_eq!(corr.shape.rank(), 0);
    assert_eq!(corr.data.len(), 1);
    // b = a + 3 is a perfect positive affine relationship -> Pearson correlation of 1.0
    // (CORRELATE computes true Pearson correlation, not a raw dot product -- see
    // kernels::pearson_correlation_1d).
    assert!((corr.data[0] - 1.0).abs() < 1e-6);

    let sim = db.get("sim").unwrap();
    assert_eq!(sim.shape.rank(), 0);
    let expected_sim = {
        let dot = 32.0;
        let norm_a = (1.0f32 + 4.0 + 9.0).sqrt(); // sqrt(14)
        let norm_b = (16.0f32 + 25.0 + 36.0).sqrt(); // sqrt(77)
        dot / (norm_a * norm_b)
    };
    assert!((sim.data[0] - expected_sim).abs() < 1e-6);

    let dist = db.get("dist").unwrap();
    let expected_dist =
        ((1.0f32 - 4.0f32).powi(2) + (2.0f32 - 5.0f32).powi(2) + (3.0f32 - 6.0f32).powi(2)).sqrt(); // sqrt(27)
    assert!((dist.data[0] - expected_dist).abs() < 1e-6);

    let a_norm = db.get("a_norm").unwrap();
    // ||a_norm|| ≈ 1
    let len_sq: f32 = a_norm.data.iter().map(|x| x * x).sum();
    assert!((len_sq.sqrt() - 1.0).abs() < 1e-6);
}
