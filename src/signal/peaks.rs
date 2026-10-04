//! Peak finding, as `scipy.signal` has it: [`find_peaks`] (local maxima, plateaus included, kept by
//! height, threshold, distance, prominence, width and plateau size), [`peak_prominences`] and
//! [`peak_widths`].

use super::SignalError;
use crate::units::*;

/// An inclusive range a peak property must fall in; either end may be open.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds {
    /// The smallest accepted value (`None`: no minimum).
    pub min: Option<f64>,
    /// The largest accepted value (`None`: no maximum).
    pub max: Option<f64>,
}

impl Bounds {
    /// Values of at least `min`.
    pub fn at_least(min: f64) -> Self {
        Self { min: Some(min), max: None }
    }
    /// Values of at most `max`.
    pub fn at_most(max: f64) -> Self {
        Self { min: None, max: Some(max) }
    }
    /// Values in `[min, max]`.
    pub fn between(min: f64, max: f64) -> Self {
        Self { min: Some(min), max: Some(max) }
    }
    /// Every value (the property is still computed and reported).
    pub fn any() -> Self {
        Self::default()
    }
    fn contains(&self, v: f64) -> bool {
        self.min.is_none_or(|m| v >= m) && self.max.is_none_or(|m| v <= m)
    }
}

/// What [`find_peaks`] keeps (`scipy.signal.find_peaks`'s arguments). Every criterion is optional;
/// a given one also adds its properties to the result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakOptions {
    /// The peak's value.
    pub height: Option<Bounds>,
    /// The vertical distance to both neighbouring samples.
    pub threshold: Option<Bounds>,
    /// The smallest horizontal distance (in samples, at least 1) between peaks; lower peaks within
    /// it of a higher one are dropped.
    pub distance: Option<f64>,
    /// How far the peak stands out from the signal around it ([`peak_prominences`]).
    pub prominence: Option<Bounds>,
    /// The width at `rel_height` of the prominence ([`peak_widths`]).
    pub width: Option<Bounds>,
    /// The window (in samples) prominences are measured in (`None`: the whole signal).
    pub wlen: Option<usize>,
    /// Where widths are measured, as a fraction of the prominence below the peak (0.5: half
    /// prominence).
    pub rel_height: f64,
    /// The number of samples on the flat top.
    pub plateau_size: Option<Bounds>,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self { height: None, threshold: None, distance: None, prominence: None, width: None, wlen: None, rel_height: 0.5, plateau_size: None }
    }
}

/// The flat tops of peaks (a top of one sample is a plateau of size 1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plateaus {
    /// Samples on each peak's top.
    pub sizes: Vec<usize>,
    /// The first sample of each top.
    pub left_edges: Vec<usize>,
    /// The last sample of each top.
    pub right_edges: Vec<usize>,
}

/// Prominences of peaks and the bases they are measured from ([`peak_prominences`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prominences {
    /// Each peak's height above the higher of its two bases.
    pub prominences: Vec<f64>,
    /// The lowest sample between the peak and the nearest higher sample (or the window's edge) to
    /// its left.
    pub left_bases: Vec<usize>,
    /// The same to its right.
    pub right_bases: Vec<usize>,
}

/// Widths of peaks ([`peak_widths`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Widths {
    /// The width in samples (fractional: the crossings are interpolated).
    pub widths: Vec<f64>,
    /// The height the width is measured at.
    pub width_heights: Vec<f64>,
    /// Where the signal crosses that height left of the peak (interpolated position).
    pub left_ips: Vec<f64>,
    /// The same right of it.
    pub right_ips: Vec<f64>,
}

/// The peaks [`find_peaks`] kept, with their properties: the plateau and the requested ones.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Peaks {
    /// The peaks' positions (the middle of a plateau, rounded down).
    pub indices: Vec<usize>,
    /// The peaks' flat tops (with a `plateau_size` criterion).
    pub plateaus: Option<Plateaus>,
    /// The peaks' values (with a `height` criterion).
    pub peak_heights: Option<Vec<f64>>,
    /// Each peak's rise over its left neighbour (with a `threshold` criterion).
    pub left_thresholds: Option<Vec<f64>>,
    /// Each peak's rise over its right neighbour (with a `threshold` criterion).
    pub right_thresholds: Option<Vec<f64>>,
    /// Prominences (with a `prominence` or `width` criterion).
    pub prominences: Option<Prominences>,
    /// Widths (with a `width` criterion).
    pub widths: Option<Widths>,
}

/// The local maxima of `x` and the properties `options` asks about, filtered by its criteria in
/// SciPy's order: plateau size, height, threshold, distance, prominence, width
/// (`scipy.signal.find_peaks`). A local maximum is a sample (or flat run of samples) higher than
/// both neighbours; the ends of the signal are never peaks. NaN samples never are either.
///
/// Errors if `distance < 1`, `wlen < 2` or `rel_height < 0`.
///
/// ```
/// use autodyne::signal::{find_peaks, Bounds, PeakOptions};
///
/// let x = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 4.0, 4.0, 1.0, 0.5, 2.0, 7.0, 3.0];
/// let peaks = find_peaks(&x, &PeakOptions { height: Some(Bounds::at_least(4.0)), ..Default::default() }).unwrap();
/// assert_eq!(peaks.indices, [4, 11]);
/// assert_eq!(peaks.peak_heights.unwrap(), [5.0, 7.0]);
/// ```
pub fn find_peaks<T: Float>(x: &[T], options: &PeakOptions) -> Result<Peaks, SignalError> {
    if options.distance.is_some_and(|d| d < 1.0 || d.is_nan()) {
        return Err(SignalError::invalid("distance must be at least 1"));
    }
    if options.rel_height < 0.0 || options.rel_height.is_nan() {
        return Err(SignalError::invalid("rel_height must be non-negative"));
    }
    let at = |i: usize| x[i].to_f64().unwrap_or(f64::NAN);
    let mut peaks = local_maxima(x, options.plateau_size.is_some());
    // each step keeps the peaks (and the properties gathered so far) that pass
    let retain = |peaks: &mut Peaks, keep: &[bool]| {
        fn pick<V: Copy>(v: &mut Vec<V>, keep: &[bool]) {
            let mut k = keep.iter();
            v.retain(|_| *k.next().expect("one flag per peak"));
        }
        pick(&mut peaks.indices, keep);
        if let Some(p) = &mut peaks.plateaus {
            pick(&mut p.sizes, keep);
            pick(&mut p.left_edges, keep);
            pick(&mut p.right_edges, keep);
        }
        for v in [&mut peaks.peak_heights, &mut peaks.left_thresholds, &mut peaks.right_thresholds].into_iter().flatten() {
            pick(v, keep);
        }
        if let Some(p) = &mut peaks.prominences {
            pick(&mut p.prominences, keep);
            pick(&mut p.left_bases, keep);
            pick(&mut p.right_bases, keep);
        }
        if let Some(w) = &mut peaks.widths {
            pick(&mut w.widths, keep);
            pick(&mut w.width_heights, keep);
            pick(&mut w.left_ips, keep);
            pick(&mut w.right_ips, keep);
        }
    };
    if let Some(b) = options.plateau_size {
        let keep: Vec<bool> = peaks.plateaus.as_ref().expect("recorded for this criterion").sizes.iter().map(|&s| b.contains(s as f64)).collect();
        retain(&mut peaks, &keep);
    }
    if let Some(b) = options.height {
        peaks.peak_heights = Some(peaks.indices.iter().map(|&i| at(i)).collect());
        let keep: Vec<bool> = peaks.indices.iter().map(|&i| b.contains(at(i))).collect();
        retain(&mut peaks, &keep);
    }
    if let Some(b) = options.threshold {
        let left: Vec<f64> = peaks.indices.iter().map(|&i| at(i) - at(i - 1)).collect();
        let right: Vec<f64> = peaks.indices.iter().map(|&i| at(i) - at(i + 1)).collect();
        let keep: Vec<bool> = left.iter().zip(&right).map(|(&l, &r)| b.contains(l.min(r)) && b.contains(l.max(r))).collect();
        (peaks.left_thresholds, peaks.right_thresholds) = (Some(left), Some(right));
        retain(&mut peaks, &keep);
    }
    if let Some(d) = options.distance {
        let heights: Vec<f64> = peaks.indices.iter().map(|&i| at(i)).collect();
        let keep = select_by_distance(&peaks.indices, &heights, d.ceil() as usize);
        retain(&mut peaks, &keep);
    }
    if options.prominence.is_some() || options.width.is_some() {
        peaks.prominences = Some(peak_prominences(x, &peaks.indices, options.wlen)?);
    }
    if let Some(b) = options.prominence {
        let keep: Vec<bool> = peaks.prominences.as_ref().expect("computed above").prominences.iter().map(|&p| b.contains(p)).collect();
        retain(&mut peaks, &keep);
    }
    if let Some(b) = options.width {
        let widths = peak_widths(x, &peaks.indices, options.rel_height, peaks.prominences.as_ref().expect("computed above"))?;
        let keep: Vec<bool> = widths.widths.iter().map(|&w| b.contains(w)).collect();
        peaks.widths = Some(widths);
        retain(&mut peaks, &keep);
    }
    Ok(peaks)
}

/// Every local maximum: samples (or the middle of flat runs) higher than both neighbours, with
/// their plateaus when `plateaus` is set.
///
/// Samples are compared 64 at a time into bit masks (rising into the sample, falling or flat after
/// it), a branch-free loop the compiler vectorizes; only the candidates (rising, then falling or
/// flat) are visited one by one, so noisy signals don't pay a mispredicted branch per sample.
fn local_maxima<T: Float>(x: &[T], plateaus: bool) -> Peaks {
    let mut peaks = Peaks { plateaus: plateaus.then(Plateaus::default), ..Default::default() };
    let n = x.len();
    if n < 3 {
        return peaks;
    }
    // positions 1 ..= n - 2, in blocks of 64
    let mut skip_to = 0;
    let mut block = 1;
    while block + 1 < n {
        let count = (n - 1 - block).min(64);
        let (mut rising, mut falling, mut flat) = (0u64, 0u64, 0u64);
        for j in 0..count {
            let (prev, here, next) = (x[block + j - 1], x[block + j], x[block + j + 1]);
            rising |= ((prev < here) as u64) << j;
            falling |= ((next < here) as u64) << j;
            flat |= ((next == here) as u64) << j;
        }
        let mut candidates = rising & (falling | flat);
        while candidates != 0 {
            let j = candidates.trailing_zeros() as usize;
            candidates &= candidates - 1;
            let i = block + j;
            if i < skip_to {
                continue;
            }
            // a plateau: walk to its end (rare, so scalar)
            let mut ahead = i + 1;
            while ahead + 1 < n && x[ahead] == x[i] {
                ahead += 1;
            }
            if x[ahead] < x[i] {
                let (left, right) = (i, ahead - 1);
                peaks.indices.push((left + right) / 2);
                if let Some(p) = &mut peaks.plateaus {
                    p.sizes.push(right - left + 1);
                    p.left_edges.push(left);
                    p.right_edges.push(right);
                }
                skip_to = ahead;
            }
        }
        block += count;
    }
    peaks
}

/// Which peaks survive when, highest first, each peak removes the lower ones within `distance`
/// samples (`scipy.signal._peak_finding_utils._select_by_peak_distance`).
fn select_by_distance(peaks: &[usize], heights: &[f64], distance: usize) -> Vec<bool> {
    let n = peaks.len();
    let mut keep = vec![true; n];
    let mut order: Vec<usize> = (0..n).collect();
    // highest first; equal heights in reverse position order, as NumPy's stable argsort read backwards
    order.sort_by(|&a, &b| heights[a].total_cmp(&heights[b]).then(a.cmp(&b)));
    for &j in order.iter().rev() {
        if !keep[j] {
            continue;
        }
        let mut k = j;
        while k > 0 && peaks[j] - peaks[k - 1] < distance {
            keep[k - 1] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < n && peaks[k] - peaks[j] < distance {
            keep[k] = false;
            k += 1;
        }
    }
    keep
}

/// How far each peak stands out (`scipy.signal.peak_prominences`): from the peak, the signal is
/// searched each way until a higher sample or the window's edge (`wlen` samples centred on the
/// peak; `None`: the whole signal); the lowest sample on each side is a base, and the prominence
/// is the peak's height above the higher base. Errors if a peak index is out of range or `wlen < 2`.
pub fn peak_prominences<T: Float>(x: &[T], peaks: &[usize], wlen: Option<usize>) -> Result<Prominences, SignalError> {
    if wlen.is_some_and(|w| w < 2) {
        return Err(SignalError::invalid("wlen must be at least 2"));
    }
    let len = x.len();
    let x = |i: usize| x[i].to_f64().unwrap_or(f64::NAN);
    let mut out = Prominences::default();
    for &peak in peaks {
        if peak >= len {
            return Err(SignalError::invalid(format!("peak {peak} is outside the signal (length {len})")));
        }
        let (mut lo, mut hi) = (0, len - 1);
        if let Some(w) = wlen {
            let half = w / 2;
            lo = peak.saturating_sub(half);
            hi = (peak + half).min(len - 1);
        }
        let height = x(peak);
        let (mut left_min, mut left_base) = (height, peak);
        let mut i = peak;
        loop {
            if x(i) > height {
                break;
            }
            if x(i) < left_min {
                (left_min, left_base) = (x(i), i);
            }
            if i == lo {
                break;
            }
            i -= 1;
        }
        let (mut right_min, mut right_base) = (height, peak);
        let mut i = peak;
        while i <= hi && x(i) <= height {
            if x(i) < right_min {
                (right_min, right_base) = (x(i), i);
            }
            i += 1;
        }
        out.prominences.push(height - left_min.max(right_min));
        out.left_bases.push(left_base);
        out.right_bases.push(right_base);
    }
    Ok(out)
}

/// The width of each peak at `rel_height` of its prominence below it (`scipy.signal.peak_widths`):
/// from the peak, the signal is followed down each way (no further than the bases) to where it
/// crosses that height, interpolating between samples. `prominences` comes from
/// [`peak_prominences`] for the same peaks. Errors if `rel_height < 0` or the lengths disagree.
pub fn peak_widths<T: Float>(x: &[T], peaks: &[usize], rel_height: f64, prominences: &Prominences) -> Result<Widths, SignalError> {
    if rel_height < 0.0 || rel_height.is_nan() {
        return Err(SignalError::invalid("rel_height must be non-negative"));
    }
    if prominences.prominences.len() != peaks.len() || prominences.left_bases.len() != peaks.len() || prominences.right_bases.len() != peaks.len() {
        return Err(SignalError::LengthMismatch(peaks.len(), prominences.prominences.len()));
    }
    let len = x.len();
    let x = |i: usize| x[i].to_f64().unwrap_or(f64::NAN);
    let mut out = Widths::default();
    for (p, &peak) in peaks.iter().enumerate() {
        if peak >= len {
            return Err(SignalError::invalid(format!("peak {peak} is outside the signal (length {len})")));
        }
        let (lo, hi) = (prominences.left_bases[p], prominences.right_bases[p]);
        let height = x(peak) - prominences.prominences[p] * rel_height;
        let mut i = peak;
        while lo < i && height < x(i) {
            i -= 1;
        }
        let mut left = i as f64;
        if x(i) < height {
            left += (height - x(i)) / (x(i + 1) - x(i));
        }
        let mut i = peak;
        while i < hi && height < x(i) {
            i += 1;
        }
        let mut right = i as f64;
        if x(i) < height {
            right -= (height - x(i)) / (x(i - 1) - x(i));
        }
        out.widths.push(right - left);
        out.width_heights.push(height);
        out.left_ips.push(left);
        out.right_ips.push(right);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateaus_peak_at_their_middle_and_ends_never_peak() {
        let any_size = PeakOptions { plateau_size: Some(Bounds::any()), ..Default::default() };
        let p = find_peaks(&[0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 3.0], &any_size).unwrap();
        assert_eq!(p.indices, [2, 5]);
        let tops = p.plateaus.unwrap();
        assert_eq!(tops.sizes, [3, 2]);
        assert_eq!((tops.left_edges, tops.right_edges), (vec![1, 5], vec![3, 6]));
        // plateaus are reported only when asked about
        assert!(find_peaks(&[0.0, 1.0, 0.0], &PeakOptions::default()).unwrap().plateaus.is_none());
        // a long noisy signal: the blocked scan agrees with a plain one
        let noise = crate::testing::noise_f64(1000, 5);
        let plain: Vec<usize> = (1..999).filter(|&i| noise[i - 1] < noise[i] && noise[i] > noise[i + 1]).collect();
        assert_eq!(find_peaks(&noise, &PeakOptions::default()).unwrap().indices, plain);
        // a rise into the end is no peak, nor is a plateau that runs into it
        assert!(find_peaks(&[0.0, 1.0, 1.0], &PeakOptions::default()).unwrap().indices.is_empty());
        assert!(find_peaks(&[5.0, 1.0], &PeakOptions::default()).unwrap().indices.is_empty());
        let only_big = PeakOptions { plateau_size: Some(Bounds::at_least(3.0)), ..Default::default() };
        assert_eq!(find_peaks(&[0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0], &only_big).unwrap().indices, [2]);
    }

    #[test]
    fn distance_keeps_the_highest_peak() {
        let x = [0.0, 3.0, 0.0, 5.0, 0.0, 4.0, 0.0, 1.0, 0.0];
        let p = find_peaks(&x, &PeakOptions { distance: Some(3.0), ..Default::default() }).unwrap();
        assert_eq!(p.indices, [3, 7]);
        assert!(find_peaks(&x, &PeakOptions { distance: Some(0.5), ..Default::default() }).is_err());
    }
}
