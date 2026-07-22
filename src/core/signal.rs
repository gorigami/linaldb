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

/// Power spectral density estimate via averaged periodograms: split
/// `signal` into non-overlapping chunks of `window` samples (any remainder
/// that doesn't fill a full chunk is dropped), FFT each chunk, average the
/// per-bin power (`re^2 + im^2`) across chunks. Returns a real
/// `Vec<f32>` of length `window / 2 + 1`.
///
/// **Simplified vs. textbook Welch's method**: no overlap between chunks
/// (Welch's method typically uses 50% overlap to use more of the data) and
/// no window function applied to each chunk before FFT (Welch's method
/// typically applies a Hann/Hamming window to reduce spectral leakage;
/// this uses an implicit rectangular window). Good enough for the
/// noise-floor estimation `WHITEN` needs; not a research-grade PSD
/// estimator. Documented here and in `DSL_REFERENCE.md` rather than
/// silently claiming full Welch's method.
pub fn psd(signal: &[f32], window: usize) -> Vec<f32> {
    assert!(window > 0, "psd: window must be non-zero");
    assert!(
        signal.len() >= window,
        "psd: signal (len {}) shorter than window ({})",
        signal.len(),
        window
    );

    let num_chunks = signal.len() / window;
    let bins = window / 2 + 1;
    let mut sum_power = vec![0.0f32; bins];

    for chunk in signal.chunks_exact(window).take(num_chunks) {
        let (re, im) = fft_forward(chunk);
        for i in 0..bins {
            sum_power[i] += re[i] * re[i] + im[i] * im[i];
        }
    }

    let n = num_chunks as f32;
    sum_power.iter().map(|p| p / n).collect()
}

/// Whitens `signal` against a noise-floor estimate `psd` (as produced by
/// `psd()`, or any real `Vector` of the right length): divides each bin of
/// `FFT(signal)` by `sqrt(psd[bin])`, then inverse-transforms back to the
/// time domain. Flattens the noise spectrum so no single frequency band
/// dominates -- the standard first step before matched filtering.
///
/// `psd` must have exactly `signal.len() / 2 + 1` entries -- the same
/// spectrum length `FFT(signal)` itself would produce. In practice this
/// means estimating the PSD with `PSD signal WINDOW <signal.len()>` (a
/// single-chunk, unaveraged periodogram -- noisier than a properly
/// averaged multi-chunk PSD, but structurally consistent) or supplying any
/// other real `Vector` of that exact length. Resampling/interpolating a
/// PSD estimated at a *different* window size onto a longer signal (the
/// way a real pipeline would reuse one noise-floor estimate across many
/// segments) is not implemented -- a real limitation, documented rather
/// than silently producing a shape-mismatched or wrong result.
///
/// Divides by `sqrt(psd[bin]) + f32::EPSILON` rather than a bare
/// `sqrt(psd[bin])` to avoid a division-by-zero producing `inf`/`NaN` on
/// a bin with exactly zero estimated power (e.g. the DC bin of a
/// zero-mean synthetic test signal) -- real noise PSDs are never exactly
/// zero, so this floor has no effect on real data.
pub fn whiten(signal: &[f32], psd: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let expected_bins = n / 2 + 1;
    assert_eq!(
        psd.len(),
        expected_bins,
        "whiten: psd must have signal.len()/2+1 = {} entries, got {}",
        expected_bins,
        psd.len()
    );

    let (re, im) = fft_forward(signal);
    let whitened_re: Vec<f32> = re
        .iter()
        .zip(psd.iter())
        .map(|(r, p)| r / (p.sqrt() + f32::EPSILON))
        .collect();
    let whitened_im: Vec<f32> = im
        .iter()
        .zip(psd.iter())
        .map(|(i, p)| i / (p.sqrt() + f32::EPSILON))
        .collect();

    fft_inverse(&whitened_re, &whitened_im, n)
}

/// Brick-wall bandpass filter: zeros every FFT bin whose frequency falls
/// outside `[low_hz, high_hz]`, then inverse-transforms back to the time
/// domain. Bin `i`'s frequency is `i * sample_rate / signal.len()`.
///
/// **Simplified vs. a real filter design** (IIR/FIR with a proper
/// transition band, e.g. Butterworth/Chebyshev): a hard zero/keep cutoff
/// at the bin boundary introduces ringing (Gibbs phenomenon) at sharp
/// edges, unlike a designed filter's smooth rolloff. Documented here and
/// in `DSL_REFERENCE.md` rather than silently claiming a proper filter.
pub fn bandpass(signal: &[f32], low_hz: f64, high_hz: f64, sample_rate: f64) -> Vec<f32> {
    let n = signal.len();
    let (re, im) = fft_forward(signal);
    let bins = re.len();

    let mut filtered_re = re;
    let mut filtered_im = im;
    for i in 0..bins {
        let freq = i as f64 * sample_rate / n as f64;
        if freq < low_hz || freq > high_hz {
            filtered_re[i] = 0.0;
            filtered_im[i] = 0.0;
        }
    }

    fft_inverse(&filtered_re, &filtered_im, n)
}

/// Matched filter via FFT-based cross-correlation: `IFFT(FFT(data) *
/// conj(FFT(template)))`, giving a real correlation-vs-lag series the same
/// length as `data`. The peak of this series (by absolute value) is the
/// standard real-world detection statistic -- the lag where `template`
/// best matches a copy of itself found in `data`.
///
/// `data` and `template` must be the same length.
///
/// **The peak lag is relative to `template`'s own reference point, not an
/// absolute location in `data`.** Concretely: `result[τ] = Σ_n data[n] *
/// template[(n - τ) mod N]`, which peaks when `τ ≈ (data's feature
/// location) - (template's own feature location within its buffer)`. If
/// `template`'s feature (e.g. a burst) sits at index `c` within the
/// template buffer, and `data`'s copy of that feature is truly at index
/// `s`, the correlation peaks at `τ ≈ s - c`, **not** at `τ = s` directly
/// -- recover the real location as `peak_lag + c`. Caught by this
/// project's own ground-truth test (`matched_filter_ground_truth_...`),
/// which initially asserted the wrong relationship (`τ = s`) and failed
/// by exactly the template's own centering offset before this doc comment
/// and the test's assertion were both corrected -- a genuinely easy
/// mistake to make, worth stating explicitly rather than leaving implicit.
///
/// **This computes circular correlation, not linear correlation** -- the
/// FFT-multiply trick inherently wraps around at the buffer edges (a
/// template match straddling the start/end of the buffer would appear to
/// wrap to the other side), unlike a true linear cross-correlation, which
/// needs the inputs zero-padded to `2N` first to avoid that. Not done
/// here: documented as a real simplification rather than silently
/// producing a wrapped-around result near the edges. Fine for finding a
/// peak safely inside the buffer (as in this project's usage), not fine
/// for a template match expected right at the boundary.
pub fn matched_filter(data: &[f32], template: &[f32]) -> Vec<f32> {
    assert_eq!(
        data.len(),
        template.len(),
        "matched_filter: data and template must be the same length"
    );
    let n = data.len();

    let (data_re, data_im) = fft_forward(data);
    let (tmpl_re, tmpl_im) = fft_forward(template);

    // Complex multiply data's spectrum by the conjugate of template's
    // spectrum: (dr + i*di) * (tr - i*ti) = (dr*tr + di*ti) + i*(di*tr - dr*ti).
    let bins = data_re.len();
    let mut result_re = vec![0.0f32; bins];
    let mut result_im = vec![0.0f32; bins];
    for k in 0..bins {
        result_re[k] = data_re[k] * tmpl_re[k] + data_im[k] * tmpl_im[k];
        result_im[k] = data_im[k] * tmpl_re[k] - data_re[k] * tmpl_im[k];
    }

    fft_inverse(&result_re, &result_im, n)
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

    /// Minimal deterministic xorshift PRNG -- avoids adding a `rand`
    /// dependency just for one test's synthetic noise, and a fixed seed
    /// keeps the test reproducible (no flakiness from real randomness).
    fn xorshift_noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to roughly [-1, 1].
                ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn psd_of_single_frequency_signal_peaks_at_expected_bin() {
        let window = 64;
        let target_bin = 5;
        // 8 repeated windows of the same sine wave -- averaging shouldn't
        // change where the peak is for a signal with no noise at all.
        let signal: Vec<f32> = (0..window * 8)
            .map(|i| {
                (2.0 * std::f32::consts::PI * target_bin as f32 * (i % window) as f32
                    / window as f32)
                    .sin()
            })
            .collect();
        let spectrum = psd(&signal, window);
        assert_eq!(spectrum.len(), window / 2 + 1);
        let (peak_bin, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            peak_bin, target_bin,
            "expected PSD peak at bin {target_bin}, got {peak_bin}"
        );
    }

    #[test]
    fn psd_of_white_noise_is_roughly_flat() {
        // White noise's expected PSD is flat -- with enough averaging
        // (many chunks), no single bin should dominate the way a real
        // signal's would. Loose tolerance (this is inherently statistical,
        // not exact) -- just confirms no bin is wildly out of proportion
        // to the mean, distinguishing "flat-ish" from "concentrated at one
        // bin" the way psd_of_single_frequency_signal_peaks_at_expected_bin
        // is concentrated.
        let window = 64;
        let noise = xorshift_noise(window * 200, 0x2026_0721);
        let spectrum = psd(&noise, window);
        let mean: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
        let max = spectrum.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max < mean * 5.0,
            "expected roughly flat PSD for white noise, but max bin ({max}) is >5x the mean ({mean})"
        );
    }

    #[test]
    #[should_panic(expected = "shorter than window")]
    fn psd_panics_on_signal_shorter_than_window() {
        let signal = vec![0.0f32; 10];
        let _ = psd(&signal, 64);
    }

    /// A simple first-order low-pass ("leaky integrator") applied to white
    /// noise, y[n] = 0.9*y[n-1] + x[n], boosts low frequencies relative to
    /// high ones -- deterministic, non-flat "colored" noise with a known
    /// shape (strongly red/low-frequency-heavy), used to verify WHITEN
    /// actually flattens a non-trivial spectral shape rather than just a
    /// trivial no-op on already-flat white noise.
    fn colored_noise(len: usize, seed: u64) -> Vec<f32> {
        let white = xorshift_noise(len, seed);
        let mut y = 0.0f32;
        white
            .iter()
            .map(|&x| {
                y = 0.9 * y + x;
                y
            })
            .collect()
    }

    #[test]
    fn whiten_flattens_colored_noise_spectrum() {
        let n = 4096;
        let colored = colored_noise(n, 0x2026_0721);
        let original_psd = psd(&colored, n); // single-chunk, matches WHITEN's required length
        let whitened = whiten(&colored, &original_psd);
        assert_eq!(whitened.len(), n);

        let whitened_psd = psd(&whitened, n);

        let ratio = |spectrum: &[f32]| -> f32 {
            let mean: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
            let max = spectrum.iter().cloned().fold(0.0f32, f32::max);
            max / mean
        };
        let before = ratio(&original_psd);
        let after = ratio(&whitened_psd);
        assert!(
            after < before * 0.5,
            "expected whitening to substantially flatten the spectrum: \
             max/mean ratio before={before}, after={after}"
        );
    }

    #[test]
    #[should_panic(expected = "psd must have signal.len()/2+1")]
    fn whiten_panics_on_mismatched_psd_length() {
        let signal = vec![0.0f32; 8];
        let wrong_psd = vec![1.0f32; 3]; // should be 8/2+1 = 5
        let _ = whiten(&signal, &wrong_psd);
    }

    #[test]
    fn bandpass_suppresses_out_of_band_component_keeps_in_band() {
        // Two known separate-frequency sine components: 20 Hz (should be
        // suppressed by a 80-150 Hz bandpass) and 100 Hz (should survive).
        // Sample rate chosen so both frequencies land near integer FFT
        // bins for a clean, unambiguous check.
        let sample_rate = 1000.0;
        let n = 200; // bin spacing = sample_rate/n = 5 Hz
        let low_freq = 20.0; // bin 4
        let high_freq = 100.0; // bin 20
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                ((2.0 * std::f64::consts::PI * low_freq * t).sin()
                    + (2.0 * std::f64::consts::PI * high_freq * t).sin()) as f32
            })
            .collect();

        let filtered = bandpass(&signal, 80.0, 150.0, sample_rate);
        assert_eq!(filtered.len(), n);

        let (re, im) = fft_forward(&filtered);
        let mag = magnitude(&re, &im);
        let low_bin = (low_freq * n as f64 / sample_rate).round() as usize;
        let high_bin = (high_freq * n as f64 / sample_rate).round() as usize;

        assert!(
            mag[low_bin] < 1.0,
            "20 Hz component should be suppressed (out of the 80-150 Hz band), got magnitude {}",
            mag[low_bin]
        );
        assert!(
            mag[high_bin] > 50.0,
            "100 Hz component should survive (inside the 80-150 Hz band), got magnitude {}",
            mag[high_bin]
        );
    }

    /// A sine-Gaussian burst: `exp(-(t-center)^2 / (2*sigma^2)) * sin(2*pi*freq*t)`
    /// -- a standard, well-established unmodeled-burst template shape (not
    /// a physically-accurate binary-merger inspiral waveform, which is its
    /// own separate, much larger physics computation, out of scope here;
    /// see SIGNAL_PROCESSING_PLAN.md design decision 5).
    fn sine_gaussian_burst(
        len: usize,
        center: f64,
        sigma: f64,
        freq_cycles_per_sample: f64,
    ) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f64;
                let envelope = (-((t - center).powi(2)) / (2.0 * sigma * sigma)).exp();
                let carrier = (2.0 * std::f64::consts::PI * freq_cycles_per_sample * t).sin();
                (envelope * carrier) as f32
            })
            .collect()
    }

    #[test]
    fn matched_filter_ground_truth_recovers_known_injection_offset() {
        // GROUND TRUTH TEST -- must pass before MATCHED_FILTER is trusted
        // on any real data (see SIGNAL_PROCESSING_PLAN.md checkpoint 6):
        // inject a known synthetic burst at a known sample offset into
        // synthetic noise, template = the same burst shape centered at a
        // known reference point, and confirm the matched-filter peak
        // recovers the true injection offset once the template's own
        // reference point is accounted for (see matched_filter's doc
        // comment -- this test is what caught that relationship needing
        // to be stated explicitly in the first place: an earlier version
        // asserted `peak_lag == true_offset` directly and failed by
        // exactly `template_center`).
        let n = 2048;
        let true_offset = 1300usize;
        let template_center = 200.0;
        let sigma = 15.0;
        let freq = 0.05; // cycles/sample

        // Template: the burst shape on its own, centered at template_center
        // (away from the wraparound edge -- see matched_filter's circular-
        // correlation caveat).
        let template = sine_gaussian_burst(n, template_center, sigma, freq);

        // Data: noise plus the same burst shape injected so its envelope
        // peaks at true_offset.
        let mut data = xorshift_noise(n, 0xC0FF_EE01);
        // Loud injection relative to noise amplitude (~[-1,1]) so the
        // ground-truth check isn't marginal -- this test is about
        // confirming the *lag arithmetic* is right, not tuning detection
        // sensitivity at low SNR.
        for (i, d) in data.iter_mut().enumerate() {
            let envelope =
                (-((i as f64 - true_offset as f64).powi(2)) / (2.0 * sigma * sigma)).exp();
            let carrier = (2.0 * std::f64::consts::PI * freq * i as f64).sin();
            *d += (5.0 * envelope * carrier) as f32;
        }

        let correlation = matched_filter(&data, &template);
        assert_eq!(correlation.len(), n);

        let (peak_lag, _) = correlation
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        let recovered_offset = peak_lag as f64 + template_center;

        assert!(
            (recovered_offset - true_offset as f64).abs() <= 2.0,
            "expected recovered offset (peak_lag {peak_lag} + template_center \
             {template_center}) within 2 samples of the true injection offset \
             {true_offset}, got {recovered_offset}"
        );
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn matched_filter_panics_on_length_mismatch() {
        let data = vec![0.0f32; 8];
        let template = vec![0.0f32; 4];
        let _ = matched_filter(&data, &template);
    }
}
