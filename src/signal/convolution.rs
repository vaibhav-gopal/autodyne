//! Convolution and correlation of 1-D signals, as `scipy.signal` has them: [`convolve_with`] /
//! [`correlate_with`] with a [`ConvMode`] (full, same, valid) and a [`ConvMethod`] (direct sums,
//! one FFT product, or overlap-add for a long signal against a shorter kernel), [`fftconvolve`] and
//! [`oaconvolve`]. [`ConvMethod::Auto`] picks the cheapest method from the two lengths, as
//! `scipy.signal.choose_conv_method` does; `Signal::convolved` and `Signal::correlated` use it.

use crate::alloc_prelude::*;
use crate::fft::{next_fast_len, RealFft};
use crate::units::*;

/// Which part of the full convolution to return (`mode` in `scipy.signal.convolve`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvMode {
    /// Every output where the signals overlap at all: `n + k - 1` samples.
    #[default]
    Full,
    /// The centre of the full output, as long as the first signal (`n` samples).
    Same,
    /// Only the outputs where the signals overlap completely: `max(n, k) - min(n, k) + 1` samples.
    Valid,
}

/// How to compute a convolution (`method` in `scipy.signal.convolve`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvMethod {
    /// The cheapest of the others for the two lengths.
    #[default]
    Auto,
    /// Sums of products: `n k` multiplies, exact for short kernels.
    Direct,
    /// One product of zero-padded real FFTs of a fast length (`O((n + k) log(n + k))`).
    Fft,
    /// The long signal cut into blocks, each convolved by FFT with the short one and the results
    /// overlapped and added (`scipy.signal.oaconvolve`): the cheapest for a long signal and a
    /// kernel of a few hundred taps or more.
    OverlapAdd,
}

/// Convolution of `a` with `b`: the samples `mode` selects from the full convolution
/// `y[i] = sum_j a[i - j] b[j]`, computed by `method`. Empty if either input is.
///
/// ```
/// use autodyne::signal::{convolve_with, ConvMethod, ConvMode};
///
/// let full = convolve_with(&[1.0f64, 2.0, 3.0], &[0.0, 1.0, 0.5], ConvMode::Full, ConvMethod::Auto);
/// assert_eq!(full, [0.0, 1.0, 2.5, 4.0, 1.5]);
/// let same = convolve_with(&[1.0f64, 2.0, 3.0], &[0.0, 1.0, 0.5], ConvMode::Same, ConvMethod::Fft);
/// assert!(same.iter().zip([1.0, 2.5, 4.0]).all(|(y, w)| (y - w).abs() < 1e-12));
/// ```
pub fn convolve_with<T: Float>(a: &[T], b: &[T], mode: ConvMode, method: ConvMethod) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (n, k) = (a.len(), b.len());
    let (start, len) = match mode {
        ConvMode::Full => (0, n + k - 1),
        // scipy's `_centered`: the middle n samples of the full output
        ConvMode::Same => ((k - 1) / 2, n),
        ConvMode::Valid => (n.min(k) - 1, n.max(k) - n.min(k) + 1),
    };
    let full = match resolve(method, n, k) {
        // the direct sums compute only the outputs wanted
        ConvMethod::Direct => return direct(a, b, start, len),
        ConvMethod::Fft => fft(a, b),
        _ => overlap_add(a, b),
    };
    if start == 0 && len == full.len() {
        return full;
    }
    full[start..start + len].to_vec()
}

/// Cross-correlation of `a` with `b`: `c[l] = sum_m a[m + l] b[m]` over the lags `mode` selects;
/// in full mode element `i` is lag `i - (b.len() - 1)` (`scipy.signal.correlate`). The convolution
/// of `a` with `b` reversed.
pub fn correlate_with<T: Float>(a: &[T], b: &[T], mode: ConvMode, method: ConvMethod) -> Vec<T> {
    let reversed: Vec<T> = b.iter().rev().copied().collect();
    convolve_with(a, &reversed, mode, method)
}

/// [`convolve_with`] by one FFT product (`scipy.signal.fftconvolve`).
pub fn fftconvolve<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    convolve_with(a, b, mode, ConvMethod::Fft)
}

/// [`convolve_with`] by overlap-add (`scipy.signal.oaconvolve`).
pub fn oaconvolve<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    convolve_with(a, b, mode, ConvMethod::OverlapAdd)
}

/// The method [`ConvMethod::Auto`] picks for signals of lengths `n` and `k`: whichever of direct
/// sums, one FFT product and overlap-add has the lowest estimated cost
/// (`scipy.signal.choose_conv_method`, with costs measured for these kernels).
pub fn choose_conv_method(n: usize, k: usize) -> ConvMethod {
    resolve(ConvMethod::Auto, n, k)
}

// Relative costs, in multiply-adds of the direct kernel (about 0.05 ns each with AVX2), measured on
// a Zen 4 machine: a real FFT of a fast length m costs about FFT_COST * m log2 m.
const FFT_COST: f64 = 2.6;
/// Below this many multiply-adds the direct sum always wins (FFT set-up dominates).
const DIRECT_ALWAYS: f64 = 4096.0;

fn resolve(method: ConvMethod, n: usize, k: usize) -> ConvMethod {
    if method != ConvMethod::Auto {
        return method;
    }
    let (long, short) = (n.max(k), n.min(k));
    let direct = (long * short) as f64;
    if direct <= DIRECT_ALWAYS || short <= 8 {
        return ConvMethod::Direct;
    }
    let size = next_fast_len(n + k - 1);
    // two forward transforms and one inverse
    let fft = 3.0 * transform_cost(size) + size as f64;
    let (block, blocks) = oa_blocks(long, short);
    let oa = blocks as f64 * (2.0 * transform_cost(block) + block as f64) + transform_cost(block);
    if direct <= fft.min(oa) {
        ConvMethod::Direct
    } else if fft <= oa {
        ConvMethod::Fft
    } else {
        ConvMethod::OverlapAdd
    }
}

fn transform_cost(m: usize) -> f64 {
    FFT_COST * m as f64 * (m as f64).log2().max(1.0)
}

/// Overlap-add blocking for a `long` signal and a `short` kernel: the FFT length minimizing the
/// cost per output sample, and how many blocks of `fft_len - short + 1` input samples cover `long`.
fn oa_blocks(long: usize, short: usize) -> (usize, usize) {
    let mut best = (f64::INFINITY, 0);
    let mut m = (2 * short).next_power_of_two();
    let limit = next_fast_len(long + short - 1);
    loop {
        let size = next_fast_len(m).min(limit);
        let step = size + 1 - short;
        let cost = (2.0 * transform_cost(size) + size as f64) / step as f64;
        if cost < best.0 {
            best = (cost, size);
        }
        if size >= limit {
            break;
        }
        m *= 2;
    }
    let size = best.1;
    (size, long.div_ceil(size + 1 - short))
}

/// Outputs computed together by the direct kernel: four AVX registers of f64, kept in registers
/// while the kernel taps stream past.
const BLOCK: usize = 16;

/// Outputs `start .. start + len` of the full convolution, by direct sums, register-blocked: with
/// `a` zero-padded by `k - 1` on both sides, `y[i] = sum_m b[k - 1 - m] a_padded[i + m]`, and each
/// block of [`BLOCK`] outputs accumulates one tap at a time over contiguous runs of the padded
/// signal (vectorized; AVX2 chosen at run time). Only the part of the padded signal the wanted
/// outputs read is built.
fn direct<T: Float>(a: &[T], b: &[T], start: usize, len: usize) -> Vec<T> {
    // convolution commutes: slide the shorter one
    let (a, b) = if b.len() > a.len() { (b, a) } else { (a, b) };
    let (n, k) = (a.len(), b.len());
    let reversed: Vec<T> = b.iter().rev().copied().collect();
    // padded[start + p] for p in 0..blocks * BLOCK + k - 1: zeros outside the signal
    let blocks = len.div_ceil(BLOCK);
    let mut window = vec![T::_ZERO; blocks * BLOCK + k - 1];
    // padded index q holds a[q - (k - 1)]
    let first = (k - 1).saturating_sub(start);
    let from = (start + first) - (k - 1);
    let count = (n - from).min(window.len() - first);
    window[first..first + count].copy_from_slice(&a[from..from + count]);
    let mut out = vec![T::_ZERO; blocks * BLOCK];
    correlate_blocked(&window, &reversed, &mut out);
    out.truncate(len);
    out
}

/// `out[i] = sum_m taps[m] x[i + m]` (`out.len()` a multiple of [`BLOCK`], `x` reaching
/// `out.len() + taps.len() - 1`), with the AVX2 kernel when the CPU has it.
pub(crate) fn correlate_blocked<T: Float>(x: &[T], taps: &[T], out: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2_available() {
        // SAFETY: the CPU was just checked for AVX2, the only feature `direct_avx2` is compiled with.
        unsafe { direct_avx2(x, taps, out) };
        return;
    }
    direct_kernel(x, taps, out);
}

/// [`correlate_blocked`] for any number of outputs: `x.len() - taps.len() + 1` of them (the
/// "valid" correlation), padding the last block internally. (Savitzky-Golay filtering's kernel.)
#[cfg(feature = "faer")]
pub(crate) fn correlate_valid<T: Float>(x: &[T], taps: &[T]) -> Vec<T> {
    let len = x.len() + 1 - taps.len();
    let blocks = len.div_ceil(BLOCK);
    let mut out = vec![T::_ZERO; blocks * BLOCK];
    let whole = (len / BLOCK) * BLOCK;
    correlate_blocked(x, taps, &mut out[..whole]);
    if whole < len {
        // the last partial block from a zero-padded copy of its inputs
        let mut tail = vec![T::_ZERO; BLOCK + taps.len() - 1];
        tail[..x.len() - whole].copy_from_slice(&x[whole..]);
        correlate_blocked(&tail, taps, &mut out[whole..]);
    }
    out.truncate(len);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn direct_avx2<T: Float>(padded: &[T], reversed: &[T], out: &mut [T]) {
    direct_kernel(padded, reversed, out)
}

/// `out[i] = sum_m reversed[m] padded[i + m]`, [`BLOCK`] outputs at a time (`out.len()` is a
/// multiple of it and `padded` reaches `out.len() + reversed.len() - 1`).
#[inline(always)]
fn direct_kernel<T: Float>(padded: &[T], reversed: &[T], out: &mut [T]) {
    for (b, block) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        let mut acc = [T::_ZERO; BLOCK];
        for (m, &r) in reversed.iter().enumerate() {
            let run = padded[b * BLOCK + m..].first_chunk::<BLOCK>().expect("padded past the last block");
            for t in 0..BLOCK {
                acc[t] = acc[t] + r * run[t];
            }
        }
        block.copy_from_slice(&acc);
    }
}

/// Full convolution through one product of real FFTs of a fast length.
fn fft<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let out = a.len() + b.len() - 1;
    let size = next_fast_len(out);
    let mut plan = RealFft::<T>::new(size);
    // one buffer holds each padded input in turn, then the result
    let mut buf = vec![T::_ZERO; size];
    let (mut product, mut other) = (vec![Complex::zero(); plan.spectrum_len()], vec![Complex::zero(); plan.spectrum_len()]);
    buf[..a.len()].copy_from_slice(a);
    plan.forward(&buf, &mut product);
    buf[..b.len()].copy_from_slice(b);
    buf[b.len()..].iter_mut().for_each(|x| *x = T::_ZERO);
    plan.forward(&buf, &mut other);
    for (p, h) in product.iter_mut().zip(&other) {
        *p *= *h;
    }
    plan.inverse(&product, &mut buf);
    buf.truncate(out);
    buf
}

/// Full convolution by overlap-add: blocks of the longer input, each convolved with the shorter by
/// FFT (its spectrum computed once) and added into place.
fn overlap_add<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let (long, short) = if b.len() > a.len() { (b, a) } else { (a, b) };
    let (size, _) = oa_blocks(long.len(), short.len());
    let step = size + 1 - short.len();
    let mut plan = RealFft::<T>::new(size);
    let bins = plan.spectrum_len();
    let mut buf = vec![T::_ZERO; size];
    buf[..short.len()].copy_from_slice(short);
    let mut response = vec![Complex::zero(); bins];
    plan.forward(&buf, &mut response);
    let mut spectrum = vec![Complex::zero(); bins];
    let mut out = vec![T::_ZERO; long.len() + short.len() - 1];
    for start in (0..long.len()).step_by(step) {
        let block = &long[start..(start + step).min(long.len())];
        buf[..block.len()].copy_from_slice(block);
        buf[block.len()..].iter_mut().for_each(|x| *x = T::_ZERO);
        plan.forward(&buf, &mut spectrum);
        for (s, h) in spectrum.iter_mut().zip(&response) {
            *s *= *h;
        }
        plan.inverse(&spectrum, &mut buf);
        let used = (block.len() + short.len() - 1).min(out.len() - start);
        for (o, &y) in out[start..start + used].iter_mut().zip(&buf) {
            *o = *o + y;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noise_f32, noise_f64};

    /// The full convolution by its definition, in f64.
    fn reference(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &h) in b.iter().enumerate() {
                out[i + j] += x * h;
            }
        }
        out
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn every_method_matches_the_definition() {
        for (n, k) in [(1, 1), (5, 1), (1, 5), (7, 3), (3, 7), (100, 33), (33, 100), (1000, 129), (4097, 300), (2000, 2000)] {
            let (a, b) = (noise_f64(n, n as u64), noise_f64(k, 7 + k as u64));
            let want = reference(&a, &b);
            for method in [ConvMethod::Direct, ConvMethod::Fft, ConvMethod::OverlapAdd, ConvMethod::Auto] {
                let got = convolve_with(&a, &b, ConvMode::Full, method);
                assert!(max_diff(&got, &want) < 1e-10 * (n.min(k) as f64).sqrt().max(1.0), "{method:?} n {n} k {k}");
            }
        }
    }

    #[test]
    fn modes_slice_the_full_output_as_scipy_does() {
        // scipy.signal.convolve([1, 2, 3, 4], [1, 1, 1], mode): full [1 3 6 9 7 4], same [3 6 9 7], valid [6 9]
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0]);
        assert_eq!(convolve_with(&a, &b, ConvMode::Full, ConvMethod::Direct), [1.0, 3.0, 6.0, 9.0, 7.0, 4.0]);
        assert_eq!(convolve_with(&a, &b, ConvMode::Same, ConvMethod::Direct), [3.0, 6.0, 9.0, 7.0]);
        assert_eq!(convolve_with(&a, &b, ConvMode::Valid, ConvMethod::Direct), [6.0, 9.0]);
        // a shorter first input: same keeps its length, valid is symmetric
        assert_eq!(convolve_with(&b, &a, ConvMode::Same, ConvMethod::Direct), [3.0, 6.0, 9.0]);
        assert_eq!(convolve_with(&b, &a, ConvMode::Valid, ConvMethod::Direct), [6.0, 9.0]);
        // even kernel: scipy.signal.convolve([1, 2, 3], [1, 1], "same") == [1, 3, 5]
        assert_eq!(convolve_with(&[1.0, 2.0, 3.0], &[1.0, 1.0], ConvMode::Same, ConvMethod::Direct), [1.0, 3.0, 5.0]);
    }

    #[test]
    fn correlation_lags_match_scipy() {
        // scipy.signal.correlate([1, 2, 3], [0, 1, 0.5]): [0.5, 2, 3.5, 3, 0]
        let c = correlate_with(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5], ConvMode::Full, ConvMethod::Auto);
        assert_eq!(c, [0.5, 2.0, 3.5, 3.0, 0.0]);
        let long = noise_f64(5000, 3);
        let delayed: Vec<f64> = std::iter::repeat_n(0.0, 37).chain(long.iter().copied()).collect();
        let c = correlate_with(&delayed, &long, ConvMode::Full, ConvMethod::Auto);
        let peak = c.iter().enumerate().fold((0, f64::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        assert_eq!(peak as isize - (long.len() as isize - 1), 37);
    }

    #[test]
    fn f32_inputs_stay_accurate() {
        let (a, b) = (noise_f32(3000, 1), noise_f32(400, 2));
        let want = reference(&a.iter().map(|&x| x as f64).collect::<Vec<_>>(), &b.iter().map(|&x| x as f64).collect::<Vec<_>>());
        for method in [ConvMethod::Direct, ConvMethod::Fft, ConvMethod::OverlapAdd] {
            let got: Vec<f64> = convolve_with(&a, &b, ConvMode::Full, method).iter().map(|&x| x as f64).collect();
            assert!(max_diff(&got, &want) < 1e-3, "{method:?}");
        }
    }

    #[test]
    fn auto_picks_direct_for_short_kernels_and_overlap_add_for_long_signals() {
        assert_eq!(choose_conv_method(1000, 8), ConvMethod::Direct);
        assert_eq!(choose_conv_method(48_000, 16), ConvMethod::Direct);
        assert_eq!(choose_conv_method(1_000_000, 1000), ConvMethod::OverlapAdd);
        assert_eq!(choose_conv_method(10_000, 10_000), ConvMethod::Fft);
        assert!(convolve_with::<f64>(&[], &[1.0], ConvMode::Full, ConvMethod::Auto).is_empty());
    }
}
