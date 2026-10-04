//! Signal processing written once over the array traits, so it runs eagerly on `NdArray` /
//! `DynArray` and traces (and differentiates) with `flux`: framing and convolution along the last
//! axis, built from the array primitives alone.

use super::array_math::{ArrayMath, RealArrayMath};
use crate::units::gcd;
/// Overlapping frames along the last axis: `[..., n]` becomes `[..., count, length]`, frame `f`
/// holding samples `f * hop .. f * hop + length`, `count = 1 + (n - length) / hop` (samples after
/// the last whole frame are left out).
///
/// One operation for arrays that have it (`NdArray` copies the windows; a traced program records a
/// single framing node, emitted as slices, reshapes and one concatenation); otherwise built from
/// those primitives. Its transpose is [`ArrayMath::overlap_add`].
///
/// ```
/// use autodyne::signal::{frames, ArrayMath, NdArray};
///
/// let x = NdArray::<f64>::array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[7]);
/// let f = frames(x, 4, 2);
/// assert_eq!(f.shape(), [2, 4]);
/// assert_eq!(f.as_slice(), &[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
pub fn frames<A: ArrayMath>(x: A, length: usize, hop: usize) -> A {
    x.frames(length, hop)
}

/// [rames] from slices, reshapes and one concatenation (no gather): the signal is cut into
/// blocks of `gcd(length, hop)` samples and each frame position takes a strided slice of them.
/// The work grows with `length / gcd(length, hop)`.
pub(crate) fn frames_by_slices<A: ArrayMath>(x: A, length: usize, hop: usize) -> A {
    let shape = x.shape();
    let (&n, lead) = shape.split_last().expect("frames: needs an axis");
    assert!(length >= 1 && hop >= 1, "frames: length and hop must be positive");
    assert!(n >= length, "frames: {n} samples are fewer than one frame of {length}");
    let count = 1 + (n - length) / hop;
    let used = (count - 1) * hop + length;
    let g = gcd(length, hop);
    let (per_frame, per_hop) = (length / g, hop / g);
    let k = lead.len();
    let blocks = x.slice_axis(k, 0, used).reshape(&[lead, &[used / g, g]].concat());
    let parts: Vec<A> = (0..per_frame)
        .map(|r| {
            let (mut start, mut limit, mut stride) = (vec![0; k + 2], [lead, &[used / g, g]].concat(), vec![1; k + 2]);
            (start[k], limit[k], stride[k]) = (r, r + (count - 1) * per_hop + 1, per_hop);
            blocks.clone().slice(&start, &limit, &stride).reshape(&[lead, &[count, 1, g]].concat())
        })
        .collect();
    A::concatenate(&parts, k + 1).reshape(&[lead, &[count, length]].concat())
}

/// [`ArrayMath::overlap_add`] from reshapes, slices, pads and additions (the transpose of
/// [`frames_by_slices`]): each frame position's blocks of `gcd(length, hop)` samples padded into
/// place, summed, then padded to `n`.
pub(crate) fn overlap_add_by_pads<A: ArrayMath>(x: A, n: usize, hop: usize) -> A {
    let shape = x.shape();
    assert!(shape.len() >= 2, "overlap_add: needs [..., count, length]");
    let k = shape.len() - 2;
    let (lead, count, length) = (&shape[..k], shape[k], shape[k + 1]);
    assert!(count >= 1 && hop >= 1, "overlap_add: needs frames and a positive hop");
    let used = (count - 1) * hop + length;
    assert!(n >= used, "overlap_add: {count} frames of {length}, {hop} apart, need {used} samples, not {n}");
    let g = gcd(length, hop);
    let (per_frame, per_hop) = (length / g, hop / g);
    let blocks = x.reshape(&[lead, &[count, per_frame, g]].concat());
    let mut total: Option<A> = None;
    for r in 0..per_frame {
        let (mut start, mut limit) = (vec![0; k + 3], [lead, &[count, per_frame, g]].concat());
        (start[k + 1], limit[k + 1]) = (r, r + 1);
        let part = blocks.clone().slice(&start, &limit, &vec![1; k + 3]).reshape(&[lead, &[count, g]].concat());
        let (mut low, mut high, mut interior) = (vec![0; k + 2], vec![0; k + 2], vec![0; k + 2]);
        (low[k], interior[k]) = (r, per_hop - 1);
        high[k] = used / g - (r + (count - 1) * per_hop + 1);
        let placed = part.pad(&low, &high, &interior);
        total = Some(match total {
            None => placed,
            Some(t) => t + placed,
        });
    }
    let flat = total.expect("at least one frame position").reshape(&[lead, &[used]].concat());
    let mut high = vec![0; k + 1];
    high[k] = n - used;
    flat.pad(&vec![0; k + 1], &high, &vec![0; k + 1])
}

/// Kernels up to this length are convolved directly; longer ones by FFT.
const DIRECT_MAX: usize = 32;

/// Convolution along the last axis, "full" (`n + k - 1` outputs, like `numpy.convolve` on each
/// row): `x` is `[..., n]`, `kernel` is `[..., k]`, their leading axes broadcasting NumPy style (a
/// 1-D kernel filters every row). Causal FIR filtering is the first `n` outputs.
///
/// Kernels of up to 32 taps are applied directly (frames of the zero-padded signal times the
/// reversed kernel, `n k` multiplies); longer ones through zero-padded real FFTs, as
/// `scipy.signal.convolve(method="auto")` would choose.
///
/// ```
/// use autodyne::signal::{convolve, ArrayMath, NdArray};
///
/// let x = NdArray::<f64>::array(&[1.0, 2.0, 3.0], &[3]);
/// let h = NdArray::<f64>::array(&[0.0, 1.0, 0.5], &[3]);
/// assert_eq!(convolve(x, h).as_slice(), &[0.0, 1.0, 2.5, 4.0, 1.5]);
/// ```
pub fn convolve<A: RealArrayMath>(x: A, kernel: A) -> A {
    let (xs, hs) = (x.shape(), kernel.shape());
    let (&n, &k) = (xs.last().expect("convolve: the signal needs an axis"), hs.last().expect("convolve: the kernel needs an axis"));
    assert!(n >= 1 && k >= 1, "convolve: empty signal or kernel");
    let out = n + k - 1;
    let last = |shape: &[usize], v: usize| {
        let mut a = vec![0; shape.len()];
        *a.last_mut().unwrap() = v;
        a
    };
    if k <= DIRECT_MAX {
        // y[i] = Σ_j x[i - j] h[j]: frame i of the padded signal against the reversed kernel
        let padded = x.pad(&last(&xs, k - 1), &last(&xs, k - 1), &vec![0; xs.len()]);
        let framed = frames(padded, k, 1);
        let reversed = kernel.reverse(&[hs.len() - 1]);
        let rows = reversed.reshape(&[&hs[..hs.len() - 1], &[1, k]].concat());
        let product = framed * rows;
        let axis = product.shape().len() - 1;
        product.sum_axes(&[axis])
    } else {
        let size = out.next_power_of_two();
        let spectrum = x.pad(&vec![0; xs.len()], &last(&xs, size - n), &vec![0; xs.len()]).rfft_complex();
        let response = kernel.pad(&vec![0; hs.len()], &last(&hs, size - k), &vec![0; hs.len()]).rfft_complex();
        let y = A::irfft_complex(spectrum * response, size);
        let axis = y.shape().len() - 1;
        y.slice_axis(axis, 0, out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::NdArray;

    fn ramp(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 * 0.37 + seed).sin() * 3.0).fract()).collect()
    }

    #[test]
    fn convolution_matches_the_direct_sum_on_both_paths() {
        for (n, k) in [(1, 1), (7, 3), (50, 32), (50, 33), (40, 90), (300, 64)] {
            let (x, h) = (ramp(n, 0.1), ramp(k, 0.7));
            let want = crate::signal::Signal::convolved(&x[..], &h);
            // rows of a batch against one kernel, and against a kernel per row
            let batch = NdArray::<f64>::array(&[x.clone(), x.iter().map(|v| v * 2.0).collect()].concat(), &[2, n]);
            let got = convolve(batch.clone(), NdArray::array(&h, &[k]));
            assert_eq!(got.shape(), [2, n + k - 1]);
            for (i, w) in want.iter().enumerate() {
                assert!((got.as_slice()[i] - w).abs() < 1e-9 && (got.as_slice()[n + k - 1 + i] - 2.0 * w).abs() < 1e-9, "n {n} k {k} sample {i}");
            }
            let per_row = convolve(batch, NdArray::array(&[h.clone(), h.iter().map(|v| -v).collect()].concat(), &[2, k]));
            for (i, w) in want.iter().enumerate() {
                assert!((per_row.as_slice()[n + k - 1 + i] + 2.0 * w).abs() < 1e-9, "per-row kernel, n {n} k {k}");
            }
        }
    }
}