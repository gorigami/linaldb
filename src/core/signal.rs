//! Frequency-domain primitives (FFT/IFFT and the operations built on them --
//! see SIGNAL_PROCESSING_PLAN.md at the repo root for the full checkpoint
//! list and the reasoning behind the design choices below).
//!
//! Deliberately kept separate from `engine/kernels.rs` (existing real-space
//! tensor math) since this is a distinct numerical domain built on a
//! different crate (`realfft`, wrapping `rustfft`).
//!
//! **No new `Value`/`ValueType::Complex` variant.** A complex spectrum is
//! represented as two parallel `Vec<f32>` here (real parts, imaginary
//! parts) -- at the DSL layer (checkpoint 1+) these become the two rows of
//! an ordinary `Value::Matrix(2, N)` (row 0 = real, row 1 = imaginary), so
//! every existing code path that already handles `Matrix` (Display,
//! storage, SELECT, JOIN, persistence, ...) needs zero changes. The
//! tradeoff is that this is a *convention*, not something the type system
//! enforces -- documented here and in `DSL_REFERENCE.md`.
//!
//! `realfft` only computes the non-negative-frequency half of the spectrum
//! for real input (length `N/2 + 1` for an `N`-sample real signal), which is
//! why `fft_inverse` needs the original signal length passed back in --
//! it's not recoverable from the spectrum's length alone (both `N=8` and
//! `N=7` real inputs produce different spectrum lengths, but knowing only
//! "spectrum length 5" doesn't tell you which; `N` must come along for the
//! ride from whoever called `fft_forward`).

use realfft::RealFftPlanner;

/// Forward real-to-complex FFT. Returns `(real_parts, imag_parts)`, each of
/// length `signal.len() / 2 + 1`.
pub fn fft_forward(signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = signal.len();
    assert!(len > 0, "fft_forward: signal must be non-empty");

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(len);

    let mut indata = r2c.make_input_vec();
    indata.copy_from_slice(signal);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut indata, &mut spectrum)
        .expect("realfft forward: mismatched buffer lengths (internal bug)");

    let real: Vec<f32> = spectrum.iter().map(|c| c.re).collect();
    let imag: Vec<f32> = spectrum.iter().map(|c| c.im).collect();
    (real, imag)
}

/// Inverse complex-to-real FFT. `original_len` is the length of the real
/// signal that produced this spectrum (see module docs for why it can't be
/// inferred from `re`/`im`'s length alone). Returns a real signal of that
/// length, correctly normalized (divided by `original_len`) so that
/// `fft_inverse(fft_forward(x)) ≈ x`, unlike `realfft`'s own raw
/// (unnormalized) convention.
pub fn fft_inverse(re: &[f32], im: &[f32], original_len: usize) -> Vec<f32> {
    assert_eq!(
        re.len(),
        im.len(),
        "fft_inverse: real/imaginary parts must be the same length"
    );
    assert_eq!(
        re.len(),
        original_len / 2 + 1,
        "fft_inverse: spectrum length must be original_len / 2 + 1"
    );

    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(original_len);

    let mut spectrum = c2r.make_input_vec();
    for (i, (r, im_part)) in re.iter().zip(im.iter()).enumerate() {
        spectrum[i].re = *r;
        spectrum[i].im = *im_part;
    }
    let mut outdata = c2r.make_output_vec();
    c2r.process(&mut spectrum, &mut outdata)
        .expect("realfft inverse: mismatched buffer lengths (internal bug)");

    let scale = 1.0 / original_len as f32;
    outdata.iter().map(|x| x * scale).collect()
}

/// Magnitude spectrum: `sqrt(re^2 + im^2)` per bin. The convenience most
/// whitening/PSD work actually needs without touching phase.
pub fn magnitude(re: &[f32], im: &[f32]) -> Vec<f32> {
    re.iter()
        .zip(im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_impulse() {
        let mut signal = vec![0.0f32; 16];
        signal[0] = 1.0;
        let (re, im) = fft_forward(&signal);
        let recovered = fft_inverse(&re, &im, signal.len());
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "impulse round-trip mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn round_trip_sine_wave() {
        let n = 64;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / n as f32).sin())
            .collect();
        let (re, im) = fft_forward(&signal);
        let recovered = fft_inverse(&re, &im, n);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "sine round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn round_trip_odd_length() {
        // realfft's spectrum length depends on N's parity in a way that's
        // easy to get wrong (N/2+1 either way, but the inverse plan must
        // still be told the real N) -- explicitly cover an odd-length
        // signal so a parity bug doesn't slip through only-even-length tests.
        let n = 17;
        let signal: Vec<f32> = (0..n).map(|i| i as f32 * 0.37).collect();
        let (re, im) = fft_forward(&signal);
        assert_eq!(re.len(), n / 2 + 1);
        let recovered = fft_inverse(&re, &im, n);
        assert_eq!(recovered.len(), n);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "odd-length round-trip mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn sine_wave_energy_concentrates_at_expected_bin() {
        // A pure sine wave at bin k should show its magnitude spectrum peak
        // at bin k, not smeared elsewhere -- the basic sanity check that
        // fft_forward's output actually means what it claims to.
        let n = 64;
        let target_bin = 5;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * target_bin as f32 * i as f32 / n as f32).sin())
            .collect();
        let (re, im) = fft_forward(&signal);
        let mag = magnitude(&re, &im);
        let (peak_bin, _) = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            peak_bin, target_bin,
            "expected peak magnitude at bin {target_bin}, got {peak_bin}"
        );
    }
}
