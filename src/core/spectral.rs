//! Spectral similarity and preprocessing on peak lists (`SPEC_COSINE`,
//! `SPEC_COSINE_MOD`, `SPEC_MATCHES`, `SPEC_ENTROPY`, `SPEC_CLEAN`), for
//! MS/MS spectra stored as `Matrix(2, *)`: row 0 the m/z values
//! (ascending), row 1 the intensities.
//!
//! The cosine scores follow matchms' `CosineGreedy` / `ModifiedCosineGreedy` step by step so
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

/// `f(x) = x · log2(x)`, with `f(0) = 0`.
fn xlog2x(x: f64) -> f64 {
    if x > 0.0 {
        x * x.log2()
    } else {
        0.0
    }
}

/// Intensities scaled to sum to 1, then (when `weighted`) given the
/// entropy-based weight of Li et al. (2021): with `H = -Σ p·ln p`, a
/// spectrum with `H < 3` has each `p` raised to `0.25 + 0.25·H` and is
/// scaled to sum to 1 again. `None` when the intensities sum to 0.
fn entropy_intensities(intensity: &[f32], weighted: bool) -> Option<Vec<f64>> {
    let sum: f64 = intensity.iter().map(|&x| x as f64).sum();
    if sum <= 0.0 {
        return None;
    }
    let mut p: Vec<f64> = intensity.iter().map(|&x| x as f64 / sum).collect();
    if weighted {
        let h: f64 = -p
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x * x.ln())
            .sum::<f64>();
        if h < 3.0 {
            let w = 0.25 + 0.25 * h;
            for x in p.iter_mut() {
                *x = if *x > 0.0 { x.powf(w) } else { 0.0 };
            }
            let s: f64 = p.iter().sum();
            for x in p.iter_mut() {
                *x /= s;
            }
        }
    }
    Some(p)
}

/// Entropy similarity (Li et al., *Nature Methods* 18, 1524-1531, 2021),
/// as `ms_entropy.calculate_entropy_similarity(a, b, tolerance,
/// clean_spectra=False)` computes it: intensities scaled to sum to 1 (and
/// weighted, see `entropy_intensities`), peaks paired one-to-one by a
/// two-pointer sweep over the ascending m/z values (a pair when `|mz_a -
/// mz_b| <= tolerance`), and score `Σ [f(pa+pb) - f(pa) - f(pb)] / 2` over
/// the pairs with `f(x) = x·log2(x)`. Result in [0, 1]; 1 for identical
/// spectra. Like `ms_entropy`, the sweep assumes centroided spectra (no two
/// peaks of one spectrum within `2 · tolerance`); `SPEC_CLEAN` with
/// `min_distance` produces them. Intensities must be non-negative; a
/// spectrum with no peaks or all-zero intensities scores 0.
pub fn entropy_similarity(
    a: &Peaks,
    b: &Peaks,
    tolerance: f64,
    weighted: bool,
) -> Result<f64, String> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(format!(
            "tolerance must be a finite, non-negative number, got {}",
            tolerance
        ));
    }
    for (s, what) in [(a, "first spectrum"), (b, "second spectrum")] {
        if let Some(i) = s.intensity.iter().position(|&x| x < 0.0) {
            return Err(format!(
                "{} has a negative intensity ({}) at peak {}",
                what, s.intensity[i], i
            ));
        }
    }
    let (Some(pa), Some(pb)) = (
        entropy_intensities(a.intensity, weighted),
        entropy_intensities(b.intensity, weighted),
    ) else {
        return Ok(0.0);
    };
    let (mut i, mut j) = (0, 0);
    let mut sum = 0.0;
    while i < a.mz.len() && j < b.mz.len() {
        let diff = a.mz[i] as f64 - b.mz[j] as f64;
        if diff < -tolerance {
            i += 1;
        } else if diff > tolerance {
            j += 1;
        } else {
            sum += xlog2x(pa[i] + pb[j]) - xlog2x(pa[i]) - xlog2x(pb[j]);
            i += 1;
            j += 1;
        }
    }
    // Rounding can leave a hair outside [0, 1].
    Ok((sum / 2.0).clamp(0.0, 1.0))
}

/// How `SPEC_CLEAN` scales the kept intensities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalize {
    /// Highest intensity 1.
    Max,
    /// Intensities sum to 1.
    Sum,
    /// Left as they are.
    None,
}

/// `SPEC_CLEAN`'s parameters.
#[derive(Debug, Clone, Copy)]
pub struct CleanParams {
    /// Peaks with m/z above this are dropped.
    pub max_mz: f64,
    /// Peaks below `floor ·` (highest intensity) are dropped.
    pub floor: f64,
    /// Keep only the `max_peaks` most intense peaks; 0 keeps all.
    pub max_peaks: usize,
    /// Intensities are raised to this power after selection (0.5 = sqrt).
    pub power: f64,
    pub normalize: Normalize,
    /// Merge peaks closer than this (Da) into their intensity-weighted
    /// centroid, as `ms_entropy.clean_spectrum` does; 0 skips it.
    pub min_distance: f64,
}

/// `ms_entropy`'s centroiding: from the most intense peak down, merge every
/// peak within `d` of it into one peak at the intensity-weighted mean m/z,
/// carrying the summed intensity; repeat until no two peaks are closer
/// than `d`. Input and output sorted by m/z.
fn centroid(mut peaks: Vec<(f64, f64)>, d: f64) -> Vec<(f64, f64)> {
    let centroided = |p: &[(f64, f64)]| p.windows(2).all(|w| w[1].0 - w[0].0 >= d);
    while !centroided(&peaks) {
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|&x, &y| {
            peaks[x]
                .1
                .partial_cmp(&peaks[y].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut merged = Vec::with_capacity(peaks.len());
        for &idx in order.iter().rev() {
            if peaks[idx].1 <= 0.0 {
                continue;
            }
            let mz = peaks[idx].0;
            let mut left = idx;
            while left > 0 && mz - peaks[left - 1].0 <= d {
                left -= 1;
            }
            let mut right = idx + 1;
            while right < peaks.len() && peaks[right].0 - mz <= d {
                right += 1;
            }
            let total: f64 = peaks[left..right].iter().map(|p| p.1).sum();
            let weighted: f64 = peaks[left..right].iter().map(|p| p.0 * p.1).sum();
            merged.push((weighted / total, total));
            for p in &mut peaks[left..right] {
                p.1 = 0.0;
            }
        }
        merged.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        peaks = merged;
    }
    peaks
}

/// `SPEC_CLEAN`: the preprocessing chain, in this order:
/// 1. drop peaks with m/z <= 0 or intensity <= 0;
/// 2. drop peaks with m/z above `max_mz`;
/// 3. centroid (`min_distance` > 0);
/// 4. drop peaks below `floor ·` the highest intensity;
/// 5. keep the `max_peaks` most intense (ties keep the higher m/z);
/// 6. raise intensities to `power`;
/// 7. normalize.
///
/// Steps 1-5 and `Normalize::Sum` are `ms_entropy.clean_spectrum`'s
/// (`max_mz`, `noise_threshold`, `min_ms2_difference_in_da`,
/// `max_peak_num`). The result is a peak list again, possibly empty.
pub fn clean(p: &Peaks, c: CleanParams) -> Result<(Vec<f32>, Vec<f32>), String> {
    if !(c.floor.is_finite() && (0.0..=1.0).contains(&c.floor)) {
        return Err(format!("floor must be between 0 and 1, got {}", c.floor));
    }
    if !c.power.is_finite() || c.power <= 0.0 {
        return Err(format!("power must be a positive number, got {}", c.power));
    }
    if !c.min_distance.is_finite() || c.min_distance < 0.0 {
        return Err(format!(
            "min_distance must be a non-negative number, got {}",
            c.min_distance
        ));
    }
    let mut peaks: Vec<(f64, f64)> =
        p.mz.iter()
            .zip(p.intensity)
            .map(|(&m, &i)| (m as f64, i as f64))
            .filter(|&(m, i)| m > 0.0 && i > 0.0 && m <= c.max_mz)
            .collect();
    if c.min_distance > 0.0 {
        peaks = centroid(peaks, c.min_distance);
    }
    if let Some(max) = peaks.iter().map(|p| p.1).reduce(f64::max) {
        let threshold = c.floor * max;
        peaks.retain(|p| p.1 >= threshold);
    }
    if c.max_peaks > 0 && peaks.len() > c.max_peaks {
        // Stable ascending sort by intensity, keep the tail: among equal
        // intensities the later (higher m/z) peaks stay.
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|&x, &y| {
            peaks[x]
                .1
                .partial_cmp(&peaks[y].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut keep: Vec<usize> = order[order.len() - c.max_peaks..].to_vec();
        keep.sort_unstable();
        peaks = keep.into_iter().map(|i| peaks[i]).collect();
    }
    if c.power != 1.0 {
        for p in peaks.iter_mut() {
            p.1 = p.1.powf(c.power);
        }
    }
    let scale = match c.normalize {
        Normalize::Max => peaks.iter().map(|p| p.1).reduce(f64::max),
        Normalize::Sum => Some(peaks.iter().map(|p| p.1).sum()),
        Normalize::None => None,
    };
    if let Some(s) = scale.filter(|&s| s > 0.0) {
        for p in peaks.iter_mut() {
            p.1 /= s;
        }
    }
    Ok(peaks.into_iter().map(|(m, i)| (m as f32, i as f32)).unzip())
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

    #[test]
    fn entropy_identical_disjoint_and_symmetric() {
        let a = spec(&[100.0, 150.0, 200.0], &[1.0, 0.5, 0.25]);
        let b = spec(&[100.0, 175.0], &[0.2, 1.0]);
        let c = spec(&[300.0], &[1.0]);
        let (pa, pb, pc) = (
            peaks(&a, "a").unwrap(),
            peaks(&b, "b").unwrap(),
            peaks(&c, "c").unwrap(),
        );
        for w in [true, false] {
            assert!((entropy_similarity(&pa, &pa, 0.01, w).unwrap() - 1.0).abs() < 1e-12);
            assert_eq!(entropy_similarity(&pa, &pc, 0.01, w).unwrap(), 0.0);
            let ab = entropy_similarity(&pa, &pb, 0.01, w).unwrap();
            let ba = entropy_similarity(&pb, &pa, 0.01, w).unwrap();
            assert!((ab - ba).abs() < 1e-15 && ab > 0.0 && ab < 1.0);
        }
        let empty = spec(&[], &[]);
        let pe = peaks(&empty, "e").unwrap();
        assert_eq!(entropy_similarity(&pa, &pe, 0.01, true).unwrap(), 0.0);
        let neg = spec(&[100.0], &[-1.0]);
        assert!(entropy_similarity(&pa, &peaks(&neg, "n").unwrap(), 0.01, true).is_err());
    }

    #[test]
    fn clean_floor_top_n_power_and_normalize() {
        let a = spec(
            &[50.0, 100.0, 150.0, 200.0, 260.0],
            &[0.5, 100.0, 40.0, 0.9, 80.0],
        );
        let pa = peaks(&a, "a").unwrap();
        let base = CleanParams {
            max_mz: 252.0,
            floor: 0.01,
            max_peaks: 0,
            power: 1.0,
            normalize: Normalize::Max,
            min_distance: 0.0,
        };
        // 260 is above max_mz, 50 (0.5 < 1.0) is below the floor.
        let (mz, int) = clean(&pa, base).unwrap();
        assert_eq!(mz, vec![100.0, 150.0]);
        assert_eq!(int, vec![1.0, 0.4]);
        let (mz, _) = clean(
            &pa,
            CleanParams {
                floor: 0.0,
                max_peaks: 2,
                ..base
            },
        )
        .unwrap();
        assert_eq!(mz, vec![100.0, 150.0]);
        let (_, int) = clean(
            &pa,
            CleanParams {
                power: 0.5,
                normalize: Normalize::Sum,
                ..base
            },
        )
        .unwrap();
        assert!((int.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Everything filtered: an empty peak list, not an error.
        let (mz, _) = clean(
            &pa,
            CleanParams {
                max_mz: 10.0,
                ..base
            },
        )
        .unwrap();
        assert!(mz.is_empty());
    }

    #[test]
    fn clean_centroids_close_peaks() {
        let a = spec(&[100.0, 100.01, 100.02, 200.0], &[1.0, 3.0, 1.0, 2.0]);
        let pa = peaks(&a, "a").unwrap();
        let c = CleanParams {
            max_mz: f64::INFINITY,
            floor: 0.0,
            max_peaks: 0,
            power: 1.0,
            normalize: Normalize::None,
            min_distance: 0.05,
        };
        let (mz, int) = clean(&pa, c).unwrap();
        assert_eq!(mz.len(), 2);
        assert!((mz[0] - 100.01).abs() < 1e-4);
        assert_eq!(int, vec![5.0, 2.0]);
    }
}
