# Signal Processing / Frequency-Domain Capability — Tracked Plan

Started 2026-07-21, following a deeper discussion of a real finding from
`examples/gw_transient_analysis.lnl` (v0.1.61): raw, non-whitened per-second
strain energy (`L2_NORM`) does not reliably locate the real GW150914 merger
time (loudest raw-energy segment was #11; the real merger is in segment #15).

**Why this happened, in engine terms**: real gravitational-wave detection
needs whitening (flatten the noise spectrum) and matched filtering
(cross-correlate against a template, across all time-shifts) to pull the
signal out of instrument noise. Both are fundamentally frequency-domain
operations. LINAL's numerical DSL today has zero frequency-domain
primitives — no FFT, no complex numbers, no convolution, no digital filter
design — so nothing expressible in the current DSL surface can do this
class of computation. This is a genuine capability gap, not a bug.

**Goal of this plan**: add real FFT-based signal-processing primitives to
the engine, then re-run `examples/gw_transient_analysis.lnl`'s merger-finding
attempt using them, and report honestly whether it now finds the real
merger time (it should — GW150914 is high-SNR and matched filtering is the
standard, well-established technique for exactly this).

---

## Design decisions (confirmed before implementation)

1. **Library**: [`realfft`](https://github.com/HEnquist/realfft) (wraps
   [`rustfft`](https://github.com/ejmahler/RustFFT)) — real-to-complex
   forward transform, complex-to-real inverse. Chosen over a hand-rolled FFT:
   both crates are pure Rust, actively maintained (rustfft: 22.7M downloads,
   updated Sep 2025; realfft: 13.6M downloads, updated Jun 2025, built
   specifically for real-valued input like strain data — roughly 2x more
   efficient than a full complex FFT for this case).
2. **Complex-spectrum representation**: **no new `Value`/`ValueType`
   variant.** A complex spectrum is represented as an existing
   `Matrix(2, N)` — row 0 = real parts, row 1 = imaginary parts. This keeps
   the change contained to new DSL keywords + kernel functions; every
   existing code path that already handles `Matrix` (Display, storage,
   SELECT, JOIN, persistence, ...) needs zero changes. Tradeoff: the row
   convention needs to be documented and consistently used, but that's a
   much smaller risk than threading a genuine complex type through the
   whole `Value` enum, `Tensor` storage, Arrow interop, and every kernel.
3. **DSL surface, this round**: standalone tensor-DSL keywords only (`LET
   x = FFT signal`), matching how `STDEV`/`CORRELATE`/`DISTANCE` are
   already used against whole tensor variables in the existing showcase.
   SQL SELECT-list forms (`FFT(col)` inside a query) are an explicit
   non-goal for this round — can follow later the same way `DISTANCE(a,b)`
   followed the standalone `DISTANCE a TO b` form in v0.1.62, once the
   standalone forms are proven out.
4. **Filter design**: brick-wall (zero out FFT bins outside the target
   band) for `BANDPASS`, not a proper IIR/FIR filter design library. Simple,
   honest, sufficient for this use case; a real Butterworth/Chebyshev filter
   is a separate, later capability if ever needed.
5. **Matched filtering template**: for the first correctness proof, a
   synthetic sine-Gaussian burst template with known parameters injected
   into synthetic noise (ground truth, checked in a test) — *not* a full
   post-Newtonian inspiral waveform (physically accurate binary-merger
   waveform generation is its own substantial physics computation, out of
   scope). Once matched filtering is proven correct against synthetic
   ground truth, apply it to the real GW150914 strain using a simplified
   chirp-like template (rising-frequency sine sweep spanning roughly
   GW150914's known frequency range, ~35–250 Hz over its final ~0.2s) as a
   good-enough real-data proof, reporting the actual result honestly either
   way.

---

## Checkpoints

Each checkpoint is implemented + tested + documented before moving to the
next; check off here as they land. One checkpoint may span more than one
commit but should land as a coherent, working state (build + test green)
before starting the next.

- [x] **0. Dependency + scaffolding** — **Done in v0.1.63**
  - Added `realfft = "3.5.0"` to `Cargo.toml`.
  - New module `src/core/signal.rs`: `fft_forward`, `fft_inverse` (properly
    normalized, unlike `realfft`'s raw convention), `magnitude`. Kept
    separate from `engine/kernels.rs`.
  - 4 unit tests: impulse round-trip, sine-wave round-trip, odd-length
    round-trip (parity edge case), sine-wave energy concentrates at the
    expected FFT bin. All pass.
  - `cargo build --release` / `cargo fmt --check` / `cargo test --release`
    / `cargo clippy --release` all clean.

- [x] **1. `FFT` / `IFFT` keyword forms** — **Done in v0.1.64**
  - `LET spectrum = FFT signal` — real `Vector(N)` → `Matrix(2, N/2+1)`
    (real part row, imaginary part row).
  - `LET signal = IFFT spectrum` — inverse, `Matrix(2, M)` → real
    `Vector(2*(M-1))` (assumes even original length, documented).
  - New `DatabaseInstance::eval_fft`/`eval_ifft` (bypass `ComputeBackend`/
    `UnaryOp` entirely — FFT isn't an elementwise op the SIMD/Rayon-tiered
    backend abstraction fits), wired through `CallExpr::Fft`/`Ifft`, new
    `Fft`/`Ifft` lexer tokens, and a dedicated parser arm.
  - `tests/signal_processing_test.rs` (5 tests, through the full DSL
    layer, not just `core::signal`'s own unit tests): Matrix(2,N/2+1)
    shape check, pure-sine-wave purely-imaginary-at-its-bin correctness
    (verified against theory: unit sine over N=8 gives magnitude N/2=4 at
    the right bin, ~0 elsewhere), full round-trip, and hard-error checks
    for wrong-shaped `FFT`/`IFFT` input.
  - `DSL_REFERENCE.md` §3 "Frequency-Domain Operators" section, with the
    `Matrix(2,N)` convention and `IFFT`'s even-length assumption both
    documented explicitly.

- [x] **2. `MAGNITUDE` (power spectrum)** — **Done in v0.1.65**
  - `LET mag = MAGNITUDE spectrum` — `Matrix(2, M)` → real `Vector(M)`,
    `sqrt(re² + im²)` per bin. Same bypass-`ComputeBackend` pattern as
    `FFT`/`IFFT` (new `eval_magnitude`).
  - Tested against the known analytic case from checkpoint 1: a
    unit-amplitude sine wave over N=8 at bin 2 gives magnitude spectrum
    exactly `[0, 0, 4, 0, 0]` (theory: N/2=4 at that bin) — verified, not
    just "it ran". Plus a hard-error test for wrong-shaped input.
  - `DSL_REFERENCE.md` §3 entry alongside `FFT`/`IFFT`.

- [x] **3. `PSD` (averaged-periodogram noise-floor estimate)** — **Done in v0.1.66**
  - `LET psd = PSD signal WINDOW n` — real `Vector(n/2+1)` output.
  - **Simplified vs. textbook Welch's method** (design call made during
    implementation, documented in `DSL_REFERENCE.md`/`core::signal::psd`):
    non-overlapping chunks (not 50% overlap) and no window function
    applied before each chunk's FFT (implicit rectangular window). Good
    enough for `WHITEN`'s noise-floor estimation need; not a
    research-grade PSD estimator.
  - `core::signal::psd` validated with 3 unit tests: single-frequency
    signal's PSD peaks at the right bin, white noise's PSD is roughly
    flat (no bin >5x the mean), and a `should_panic` test for
    signal-shorter-than-window (an internal-caller-bug case). The DSL
    layer (`eval_psd`) validates rank/length itself and returns a normal
    engine error for bad user input instead of ever reaching that panic.
  - `tests/signal_processing_test.rs`: 3 more tests through the full DSL
    layer (repeated-sine peaks at the right bin with the exact expected
    power N/2², rejects signal shorter than window, rejects non-Vector
    input).

- [x] **4. `WHITEN`** — **Done in v0.1.67**
  - `LET whitened = WHITEN signal WITH psd` — divide `FFT(signal)` by
    `sqrt(psd)` elementwise (both real and imaginary rows against the
    same real PSD row, floored at `f32::EPSILON` to avoid a division by
    exactly zero), `IFFT` back to the time domain. Real `Vector(N)`
    output. `psd` must have exactly `N/2+1` entries (design call:
    resampling a differently-sized PSD onto `signal` is not implemented,
    documented as a real limitation rather than silently mismatched).
  - `core::signal::whiten` validated with a deterministic-colored-noise
    test (first-order low-pass "leaky integrator" over white noise, a
    known non-flat spectral shape): whitening against its own PSD
    substantially flattens the re-estimated PSD (max/mean ratio drops by
    more than half) — the actual test the checkpoint asked for, plus a
    `should_panic` test for mismatched psd length.
  - `tests/signal_processing_test.rs`: 3 more tests through the full DSL
    layer (correct output shape, rejects mismatched psd length, rejects
    non-Vector signal).
  - `docs/DSL_REFERENCE.md` §3 documents `WHITEN` alongside the others.

- [ ] **5. `BANDPASS`**
  - `LET filtered = BANDPASS signal FROM low_hz TO high_hz WITH RATE
    sample_rate` (finalize exact syntax during implementation) —
    brick-wall zeroing of FFT bins outside `[low_hz, high_hz]`, then
    `IFFT`.
  - Test: bandpassing a signal containing two known separate-frequency
    components should suppress the one outside the band and preserve
    the one inside (verified via `MAGNITUDE`/`FFT` on the result, not
    just "it ran").

- [ ] **6. `MATCHED_FILTER` (cross-correlation via FFT)**
  - `LET snr_series = MATCHED_FILTER data WITH template` — `IFFT(FFT(data)
    * conj(FFT(template)))`, giving a correlation-vs-lag time series (the
    real detection statistic). Requires complex multiply + conjugate on
    the `Matrix(2,N)` representation — new small kernel functions, not
    exposed as their own DSL keywords initially unless a real need shows up.
  - **Ground-truth correctness test first**: inject a known synthetic
    sine-Gaussian burst at a known sample offset into synthetic Gaussian
    noise; confirm `MATCHED_FILTER`'s peak lands at that known offset.
    This must pass before touching real data at all.
  - Only after that test passes: apply to real GW150914 H1 strain with a
    simplified chirp-like template (design decision 5, above); report the
    actual result in `examples/gw_transient_analysis.lnl` §4, honestly,
    whichever way it comes out.

- [ ] **7. Showcase integration + honesty pass**
  - Update `examples/gw_transient_analysis.lnl` §4's commentary to reflect
    the actual outcome — replace the "raw energy doesn't find it, and here's
    why" framing with the real matched-filter result, keeping the *original*
    raw-`L2_NORM` finding alongside it as a documented before/after contrast
    (don't delete the honest negative result, extend it).
  - Full `cargo test` + `cargo fmt --check` + a complete script re-run.
  - `CHANGELOG.md` entry, version bump, this plan file deleted once every
    checkpoint above is checked off (matching this repo's existing
    `CONSISTENCY_PLAN.md` convention).

---

## Process for every checkpoint

- Implement + unit test + doc update together, not implementation-then-tests-later.
- `cargo build --release` clean, `cargo fmt --check` clean, `cargo test --release`
  all green before checking off a box.
- Check off the box here with a `**Done in vX.Y.Z**` note once merged.

## Completion

This file is deleted once every checkpoint is checked off and the showcase
integration (checkpoint 7) has landed.
