//! Spectral similarity on peak lists (`SPEC_COSINE`, `SPEC_COSINE_MOD`,
//! `SPEC_MATCHES`), for MS/MS spectra stored as `Matrix(2, *)`: row 0 the
//! m/z values (ascending), row 1 the intensities.
//!
//! Follows matchms' `CosineGreedy` / `ModifiedCosineGreedy` step by step so
//! scores agree with it:
//!
//! 1. Peak pairs are every `(i, j)` with `|mz_a[i] - (mz_b[j] + shift)| <=
//!    tolerance`, found by matchms' sorted sweep (`find_matches`), in
//!    `(i, j)` order. The modified cosine uses `shift = 0` pairs followed
//!    by `shift = precursor_a - precursor_b` pairs; when `|shift| <=
//!    tolerance` it is just the plain cosine.
//! 2. Each pair's weight is `(mz_a^p · I_a^q) · (mz_b^p · I_b^q)` (p =
//!    `mz_power`, q = `intensity_power`; matchms defaults 0 and 1).
//! 3. Pairs are sorted by weight with a stable ascending sort and then
//!    reversed -- so equal weights end up in reverse generation order,
//!    exactly as matchms' `argsort(kind="mergesort")[::-1]`.
//! 4. Greedily, each pair whose two peaks are both still unused is taken.
//! 5. Score = sum of taken weights / (‖a‖ · ‖b‖), with each norm over *all*
//!    of that spectrum's peaks' `mz^p · I^q`.
//!
//! All arithmetic is f64 on the stored f32 values. One deliberate
//! difference: when a spectrum has matching peaks but all-zero
//! intensities, matchms divides 0 by 0 (NaN); this returns 0.

/// A validated peak list.
pub struct Peaks<'a> {
    pub mz: &'a [f32],
    pub intensity: &'a [f32],
}

/// Checks that `m` is a 2-row peak list with finite values and ascending
/// m/z. `what` names the argument in the error.
pub fn peaks<'a>(m: &'a [Vec<f32>], what: &str) -> Result<Peaks<'a>, String> {
    if m.len() != 2 {
        return Err(format!(
            "{} must be a peak list Matrix(2, n) (row 0 m/z, row 1 intensity), got {} rows",
            what,
            m.len()
        ));
    }
    let (mz, intensity) = (&m[0][..], &m[1][..]);
    if let Some(i) = mz.iter().chain(intensity).position(|x| !x.is_finite()) {
        return Err(format!("{} has a non-finite value at position {}", what, i));
    }
    if let Some(i) = mz.windows(2).position(|w| w[1] < w[0]) {
        return Err(format!(
            "{}: m/z values must be sorted ascending ({} at peak {} comes after {})",
            what,
            mz[i + 1],
            i + 1,
            mz[i]
        ));
    }
    Ok(Peaks { mz, intensity })
}

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub tolerance: f64,
    pub mz_power: f64,
    pub intensity_power: f64,
}

impl Params {
    pub fn new(tolerance: f64, mz_power: f64, intensity_power: f64) -> Result<Self, String> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(format!(
                "tolerance must be a finite, non-negative number, got {}",
                tolerance
            ));
        }
        if !mz_power.is_finite() || !intensity_power.is_finite() {
            return Err("mz_power and intensity_power must be finite".to_string());
        }
        Ok(Params {
            tolerance,
            mz_power,
            intensity_power,
        })
    }

    fn weight(&self, mz: f32, intensity: f32) -> f64 {
        (mz as f64).powf(self.mz_power) * (intensity as f64).powf(self.intensity_power)
    }
}

/// matchms' `find_matches`: sweep `b` (shifted) against each peak of `a`.
fn find_matches(a: &[f32], b: &[f32], tolerance: f64, shift: f64, out: &mut Vec<(usize, usize)>) {
    let mut lowest = 0;
    for (i, &mz) in a.iter().enumerate() {
        let mz = mz as f64;
        let (low, high) = (mz - tolerance, mz + tolerance);
        for (j, &mz2) in b.iter().enumerate().skip(lowest) {
            let mz2 = mz2 as f64 + shift;
            if mz2 > high {
                break;
            }
            if mz2 < low {
                lowest = j + 1;
            } else {
                out.push((i, j));
            }
        }
    }
}

/// `(score, matched peaks)`. `shift` = `Some(precursor_a - precursor_b)`
/// for the modified cosine, `None` for the plain one.
pub fn cosine_greedy(a: &Peaks, b: &Peaks, p: Params, shift: Option<f64>) -> (f64, usize) {
    let mut pairs = Vec::new();
    find_matches(a.mz, b.mz, p.tolerance, 0.0, &mut pairs);
    if let Some(shift) = shift {
        if shift.abs() > p.tolerance {
            find_matches(a.mz, b.mz, p.tolerance, shift, &mut pairs);
        }
    }
    if pairs.is_empty() {
        return (0.0, 0);
    }
    let mut weighted: Vec<(usize, usize, f64)> = pairs
        .into_iter()
        .map(|(i, j)| {
            let w = p.weight(a.mz[i], a.intensity[i]) * p.weight(b.mz[j], b.intensity[j]);
            (i, j, w)
        })
        .collect();
    weighted.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));
    weighted.reverse();

    let mut used_a = vec![false; a.mz.len()];
    let mut used_b = vec![false; b.mz.len()];
    let (mut sum, mut matched) = (0.0, 0);
    for (i, j, w) in weighted {
        if !used_a[i] && !used_b[j] {
            sum += w;
            used_a[i] = true;
            used_b[j] = true;
            matched += 1;
        }
    }
    let norm = |s: &Peaks| {
        s.mz.iter()
            .zip(s.intensity)
            .map(|(&m, &i)| p.weight(m, i).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let denom = norm(a) * norm(b);
    let score = if denom == 0.0 { 0.0 } else { sum / denom };
    (score, matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(mz: &[f32], int: &[f32]) -> Vec<Vec<f32>> {
        vec![mz.to_vec(), int.to_vec()]
    }

    #[test]
    fn identical_spectra_score_one_and_greedy_uses_each_peak_once() {
        let a = spec(&[100.0, 150.0, 200.0], &[1.0, 0.5, 0.25]);
        let pa = peaks(&a, "a").unwrap();
        let p = Params::new(0.1, 0.0, 1.0).unwrap();
        let (s, n) = cosine_greedy(&pa, &pa, p, None);
        assert!((s - 1.0).abs() < 1e-12);
        assert_eq!(n, 3);

        // Two b peaks within tolerance of one a peak: only one is used.
        let b = spec(&[100.0, 100.05], &[1.0, 1.0]);
        let pb = peaks(&b, "b").unwrap();
        let (_, n) = cosine_greedy(&pa, &pb, p, None);
        assert_eq!(n, 1);
    }

    #[test]
    fn modified_cosine_matches_shifted_peaks() {
        let a = spec(&[100.0, 200.0], &[1.0, 1.0]);
        let b = spec(&[100.0, 180.0], &[1.0, 1.0]);
        let (pa, pb) = (peaks(&a, "a").unwrap(), peaks(&b, "b").unwrap());
        let p = Params::new(0.1, 0.0, 1.0).unwrap();
        let (plain, _) = cosine_greedy(&pa, &pb, p, None);
        let (modified, n) = cosine_greedy(&pa, &pb, p, Some(20.0));
        assert!((plain - 0.5).abs() < 1e-12);
        assert!((modified - 1.0).abs() < 1e-12);
        assert_eq!(n, 2);
    }

    #[test]
    fn rejects_bad_peak_lists() {
        assert!(peaks(&spec(&[200.0, 100.0], &[1.0, 1.0]), "a").is_err());
        assert!(peaks(&spec(&[100.0], &[f32::NAN]), "a").is_err());
        assert!(peaks(&[vec![1.0]], "a").is_err());
        assert!(Params::new(-0.1, 0.0, 1.0).is_err());
    }
}
