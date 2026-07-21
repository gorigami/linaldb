// Checkpoints 1-3 of SIGNAL_PROCESSING_PLAN.md: FFT/IFFT/MAGNITUDE/PSD DSL
// keyword forms. Exercises them through the actual DSL layer (lexer ->
// parser -> AST -> executor -> engine), not just the raw core::signal
// functions (already covered by src/core/signal.rs's own unit tests).

use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn fft_produces_matrix_2_by_n_half_plus_1() {
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
VECTOR sig = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
LET spectrum = FFT sig
"#,
    )
    .expect("FFT should succeed on a rank-1 Vector");

    let spectrum = db.get("spectrum").expect("spectrum should exist");
    assert_eq!(spectrum.shape.dims, vec![2, 5], "N=8 -> Matrix(2, 8/2+1)");
}

#[test]
fn fft_of_pure_sine_is_purely_imaginary_at_its_bin() {
    // A pure sine wave is an odd function -- its DFT is purely imaginary,
    // concentrated at the bin matching its frequency. For a unit-amplitude
    // sine over N=8 samples at 2 cycles/window, theory gives magnitude N/2=4
    // at bin 2, zero everywhere else (both real and imaginary parts).
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
VECTOR sig = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
LET spectrum = FFT sig
"#,
    )
    .expect("FFT should succeed");

    let spectrum = db.get("spectrum").expect("spectrum should exist");
    let data = spectrum.to_logical_vec();
    // Row-major Matrix(2, 5): real parts data[0..5], imaginary parts data[5..10].
    let re = &data[0..5];
    let im = &data[5..10];

    for (i, &r) in re.iter().enumerate() {
        assert!(r.abs() < 1e-4, "real part at bin {i} should be ~0, got {r}");
    }
    for (i, &v) in im.iter().enumerate() {
        if i == 2 {
            assert!(
                (v.abs() - 4.0).abs() < 1e-3,
                "expected |imaginary part| ~4.0 at bin 2, got {v}"
            );
        } else {
            assert!(
                v.abs() < 1e-3,
                "imaginary part at bin {i} should be ~0, got {v}"
            );
        }
    }
}

#[test]
fn ifft_round_trips_fft() {
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
VECTOR sig = [0.3, -1.2, 4.5, 0.0, -2.7, 3.1, 1.0, -0.5]
LET spectrum = FFT sig
LET recovered = IFFT spectrum
"#,
    )
    .expect("FFT/IFFT round trip should succeed");

    let original = db.get("sig").expect("sig should exist").to_logical_vec();
    let recovered = db
        .get("recovered")
        .expect("recovered should exist")
        .to_logical_vec();

    assert_eq!(recovered.len(), original.len());
    for (a, b) in original.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-3, "round-trip mismatch: {a} vs {b}");
    }
}

#[test]
fn fft_rejects_non_vector_input() {
    let mut db = TensorDb::new();
    let result = execute_script(
        &mut db,
        r#"
MATRIX m = [[1, 2], [3, 4]]
LET bad = FFT m
"#,
    );
    assert!(
        result.is_err(),
        "FFT on a Matrix should be a hard error, not silently wrong output"
    );
}

#[test]
fn ifft_rejects_non_matrix_2_n_input() {
    let mut db = TensorDb::new();
    let result = execute_script(
        &mut db,
        r#"
VECTOR v = [1.0, 2.0, 3.0]
LET bad = IFFT v
"#,
    );
    assert!(
        result.is_err(),
        "IFFT on a plain Vector (not a Matrix(2,M) spectrum) should be a hard error"
    );
}

#[test]
fn magnitude_matches_theory_for_pure_sine() {
    // Same signal as fft_of_pure_sine_is_purely_imaginary_at_its_bin:
    // unit-amplitude sine over N=8 at bin 2 -> magnitude spectrum should
    // be exactly [0, 0, 4, 0, 0] (theory: magnitude N/2=4 at that bin).
    let mut db = TensorDb::new();
    execute_script(
        &mut db,
        r#"
VECTOR sig = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
LET spectrum = FFT sig
LET mag = MAGNITUDE spectrum
"#,
    )
    .expect("FFT + MAGNITUDE should succeed");

    let mag = db.get("mag").expect("mag should exist");
    assert_eq!(mag.shape.dims, vec![5]);
    let data = mag.to_logical_vec();
    let expected = [0.0f32, 0.0, 4.0, 0.0, 0.0];
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "magnitude at bin {i}: expected {exp}, got {got}"
        );
    }
}

#[test]
fn magnitude_rejects_non_matrix_2_n_input() {
    let mut db = TensorDb::new();
    let result = execute_script(
        &mut db,
        r#"
VECTOR v = [1.0, 2.0, 3.0]
LET bad = MAGNITUDE v
"#,
    );
    assert!(
        result.is_err(),
        "MAGNITUDE on a plain Vector (not a Matrix(2,M) spectrum) should be a hard error"
    );
}

#[test]
fn psd_peaks_at_expected_bin_for_repeated_sine() {
    // 8 repeated 64-sample windows of a unit-amplitude sine wave at bin 5
    // -> PSD peak should land at bin 5, with power == (window/2)^2 = 1024
    // (magnitude N/2=32 at that bin, squared).
    let window = 64usize;
    let target_bin = 5;
    let values: Vec<String> = (0..window * 8)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * target_bin as f64 * (i % window) as f64
                / window as f64;
            phase.sin().to_string()
        })
        .collect();
    let script = format!(
        "VECTOR sig = [{}]\nLET spectrum = PSD sig WINDOW {}\n",
        values.join(", "),
        window
    );

    let mut db = TensorDb::new();
    execute_script(&mut db, &script).expect("PSD should succeed");

    let spectrum = db.get("spectrum").expect("spectrum should exist");
    assert_eq!(spectrum.shape.dims, vec![window / 2 + 1]);
    let data = spectrum.to_logical_vec();
    let (peak_bin, &peak_val) = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(
        peak_bin, target_bin,
        "expected PSD peak at bin {target_bin}"
    );
    assert!(
        (peak_val - 1024.0).abs() < 1.0,
        "expected peak power ~1024.0, got {peak_val}"
    );
}

#[test]
fn psd_rejects_signal_shorter_than_window() {
    let mut db = TensorDb::new();
    let result = execute_script(
        &mut db,
        r#"
VECTOR v = [1.0, 2.0, 3.0]
LET bad = PSD v WINDOW 64
"#,
    );
    assert!(
        result.is_err(),
        "PSD with WINDOW larger than the signal should be a hard error, not a panic"
    );
}

#[test]
fn psd_rejects_non_vector_input() {
    let mut db = TensorDb::new();
    let result = execute_script(
        &mut db,
        r#"
MATRIX m = [[1, 2], [3, 4]]
LET bad = PSD m WINDOW 2
"#,
    );
    assert!(result.is_err(), "PSD on a Matrix should be a hard error");
}
