//! Spectral processing: a reference DFT, FFTs of any length ([`Fft`]; [`RealFft`] for real signals,
//! about twice as fast), and the phase vocoder: [`PitchShifter`] (real time) and [`time_stretch`].
//!
//! Sign convention: forward `X[k] = sum_n x[n] e^(-i 2 pi k n / N)`; inverse divides by N,
//! so `inverse(forward(x)) == x`.

use crate::units::*;

mod estimate;
mod real;
mod vocoder;
mod windows;
pub use estimate::*;
pub use real::*;
pub use vocoder::*;
pub use windows::*;

/// Naive O(n^2) discrete Fourier transform. Exact by definition, so it is the reference the FFT is tested
/// against. Allocates the output.
pub fn dft<T: Float>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input.iter().enumerate().fold(Complex::zero(), |acc, (j, &x)| {
                // reduce k*j mod n first so the angle stays small and accurate for large n
                let angle = -std::f64::consts::TAU * ((k * j) % n) as f64 / n as f64;
                acc + x * Complex::cis(T::_lit(angle))
            })
        })
        .collect()
}

/// Center frequency in Hz of `bin` for an `fft_len`-point transform. Bins above fft_len / 2 are the
/// negative frequencies (bin fft_len - k is -k).
pub fn bin_frequency<T: Float>(bin: usize, fft_len: usize, sample_rate: T) -> T {
    T::_lit(bin as f64) * sample_rate / T::_lit(fft_len as f64)
}

/// FFT of a fixed length (any length >= 1), planned once so transforms run in place without
/// allocating.
///
/// With the default `rustfft` feature, `f32` / `f64` transforms run on
/// [rustfft](https://docs.rs/rustfft)'s kernels (mixed radix, Rader and Bluestein for awkward
/// lengths; AVX / SSE / NEON chosen at runtime). Otherwise (or for other element types) a portable
/// path: an iterative radix-2 Cooley-Tukey kernel for powers of two, and Bluestein's algorithm (the
/// transform as a chirp convolution, done with a power-of-two FFT) for every other length.
/// Twiddle factors, tables and scratch space are made in `new`.
#[derive(Debug, Clone)]
pub struct Fft<T: Float> {
    len: usize,
    kernel: Kernel<T>,
}

#[derive(Debug, Clone)]
enum Kernel<T: Float> {
    /// e^(-i 2 pi k / len) for k in 0..len/2 (computed in f64), and each index with its log2(len)
    /// low bits reversed
    Radix2 { twiddles: Vec<Complex<T>>, bit_reverse: Vec<usize> },
    /// chirp[k] = e^(-i pi k^2 / len); `filter` is the FFT of the conjugate chirp laid out for a
    /// circular convolution of length `inner.len()` (a power of two >= 2 len - 1)
    Bluestein { chirp: Vec<Complex<T>>, filter: Vec<Complex<T>>, inner: Box<Fft<T>>, scratch: Vec<Complex<T>> },
    #[cfg(feature = "rustfft")]
    Fast(fast::Plan),
}

impl<T: Float> Fft<T> {
    /// Panics if `len` is 0.
    pub fn new(len: usize) -> Self {
        assert!(len >= 1, "FFT length must be at least 1");
        #[cfg(feature = "rustfft")]
        if let Some(plan) = fast::Plan::new::<T>(len) {
            return Self { len, kernel: Kernel::Fast(plan) };
        }
        Self::new_portable(len)
    }

    /// The portable kernels whatever features are enabled (radix-2 for powers of two, Bluestein
    /// otherwise): the reference the fast backend is tested against. Panics if `len` is 0.
    pub fn new_portable(len: usize) -> Self {
        assert!(len >= 1, "FFT length must be at least 1");
        if len.is_power_of_two() {
            return Self::new_radix2(len);
        }
        let inner = Fft::new_radix2((2 * len - 1).next_power_of_two());
        let m = inner.len;
        // reduce k^2 modulo 2 len so the angle stays small and accurate for large k
        let chirp: Vec<Complex<T>> = (0..len)
            .map(|k| Complex::cis(T::_lit(-std::f64::consts::PI * ((k as u128 * k as u128) % (2 * len as u128)) as f64 / len as f64)))
            .collect();
        let mut filter = vec![Complex::zero(); m];
        filter[0] = chirp[0].conj();
        for k in 1..len {
            filter[k] = chirp[k].conj();
            filter[m - k] = chirp[k].conj();
        }
        let mut inner = Box::new(inner);
        inner.forward(&mut filter);
        Self { len, kernel: Kernel::Bluestein { chirp, filter, inner, scratch: vec![Complex::zero(); m] } }
    }

    /// The portable radix-2 kernel alone. Panics unless `len` is a power of two (1 included).
    pub fn new_radix2(len: usize) -> Self {
        assert!(len.is_power_of_two(), "radix-2 FFT length must be a power of two, got {len}");
        let bits = len.trailing_zeros();
        let twiddles = (0..len / 2)
            .map(|k| Complex::cis(T::_lit(-std::f64::consts::TAU * k as f64 / len as f64)))
            .collect();
        let bit_reverse = (0..len)
            .map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) })
            .collect();
        Self { len, kernel: Kernel::Radix2 { twiddles, bit_reverse } }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false // the length is at least 1
    }

    /// Forward transform in place. Panics if `buf.len() != self.len()`.
    pub fn forward(&mut self, buf: &mut [Complex<T>]) {
        assert_eq!(buf.len(), self.len, "buffer length must match the FFT length");
        match &mut self.kernel {
            Kernel::Radix2 { twiddles, bit_reverse } => radix2(buf, twiddles, bit_reverse),
            Kernel::Bluestein { chirp, filter, inner, scratch } => {
                // X[k] = chirp[k] * sum_j (x[j] chirp[j]) conj(chirp[k - j]): a convolution
                let n = chirp.len();
                for (s, (x, c)) in scratch.iter_mut().zip(buf.iter().zip(chirp.iter())) {
                    *s = *x * *c;
                }
                scratch[n..].iter_mut().for_each(|s| *s = Complex::zero());
                inner.forward(scratch);
                for (s, f) in scratch.iter_mut().zip(filter.iter()) {
                    *s *= *f;
                }
                inner.inverse(scratch);
                for (x, (s, c)) in buf.iter_mut().zip(scratch.iter().zip(chirp.iter())) {
                    *x = *s * *c;
                }
            }
            #[cfg(feature = "rustfft")]
            Kernel::Fast(plan) => plan.run(buf, false),
        }
    }

    /// Inverse transform in place (scaled by 1/len). Panics if `buf.len() != self.len()`.
    pub fn inverse(&mut self, buf: &mut [Complex<T>]) {
        assert_eq!(buf.len(), self.len, "buffer length must match the FFT length");
        let scale = T::_lit(1.0 / self.len as f64);
        #[cfg(feature = "rustfft")]
        if let Kernel::Fast(plan) = &mut self.kernel {
            plan.run(buf, true);
            buf.iter_mut().for_each(|z| *z *= scale);
            return;
        }
        // ifft(x) = conj(fft(conj(x))) / N
        buf.iter_mut().for_each(|z| *z = z.conj());
        self.forward(buf);
        buf.iter_mut().for_each(|z| *z = z.conj() * scale);
    }

    /// Transforms a real signal: copies `input` into `out` as complex values, then transforms `out`.
    /// Panics unless both lengths equal `self.len()`.
    pub fn forward_real(&mut self, input: &[T], out: &mut [Complex<T>]) {
        assert_eq!(input.len(), self.len, "input length must match the FFT length");
        assert_eq!(out.len(), self.len, "output length must match the FFT length");
        for (o, &x) in out.iter_mut().zip(input) {
            *o = Complex::from(x);
        }
        self.forward(out);
    }
}

/// The iterative radix-2 transform of `buf` (a power-of-two length) in place.
fn radix2<T: Float>(buf: &mut [Complex<T>], twiddles: &[Complex<T>], bit_reverse: &[usize]) {
    let len = buf.len();
    for (i, &j) in bit_reverse.iter().enumerate() {
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut size = 2;
    while size <= len {
        let half = size / 2;
        let stride = len / size; // twiddle index step for this stage
        for block in buf.chunks_exact_mut(size) {
            let (lo, hi) = block.split_at_mut(half);
            for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let t = *b * twiddles[k * stride];
                *b = *a - t;
                *a += t;
            }
        }
        size *= 2;
    }
}

#[cfg(feature = "rustfft")]
mod fast {
    use std::any::TypeId;
    use std::sync::Arc;

    use rustfft::num_complex::Complex as Rc;
    use rustfft::FftPlanner;

    use crate::units::*;

    /// A rustfft plan with its scratch space, for `f32` or `f64`.
    #[derive(Clone)]
    pub(super) enum Plan {
        F32 { forward: Arc<dyn rustfft::Fft<f32>>, inverse: Arc<dyn rustfft::Fft<f32>>, scratch: Vec<Rc<f32>> },
        F64 { forward: Arc<dyn rustfft::Fft<f64>>, inverse: Arc<dyn rustfft::Fft<f64>>, scratch: Vec<Rc<f64>> },
    }

    impl std::fmt::Debug for Plan {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                Plan::F32 { .. } => "rustfft plan (f32)",
                Plan::F64 { .. } => "rustfft plan (f64)",
            })
        }
    }

    macro_rules! plan {
        ($V:ident, $F:ty, $len:expr) => {{
            let mut planner = FftPlanner::<$F>::new();
            let (forward, inverse) = (planner.plan_fft_forward($len), planner.plan_fft_inverse($len));
            let scratch = vec![Rc::new(0.0, 0.0); forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len())];
            Some(Plan::$V { forward, inverse, scratch })
        }};
    }

    impl Plan {
        /// A plan for element type `T`, if rustfft supports it.
        pub(super) fn new<T: Float>(len: usize) -> Option<Self> {
            let t = TypeId::of::<T>();
            if t == TypeId::of::<f32>() {
                plan!(F32, f32, len)
            } else if t == TypeId::of::<f64>() {
                plan!(F64, f64, len)
            } else {
                None
            }
        }

        /// Transforms `buf` (unscaled in both directions), without allocating.
        pub(super) fn run<T: Float>(&mut self, buf: &mut [Complex<T>], inverse: bool) {
            macro_rules! go {
                ($F:ty, $fwd:expr, $inv:expr, $scratch:expr) => {{
                    // a plan's variant matches the element type it was made for
                    assert_eq!(TypeId::of::<T>(), TypeId::of::<$F>(), "FFT plan used with another element type");
                    // SAFETY: T is $F (checked above), and autodyne's and num-complex's Complex are
                    // both repr(C) { re, im }, so the slices have the same layout
                    let buf = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<Rc<$F>>(), buf.len()) };
                    if inverse { $inv.process_with_scratch(buf, $scratch) } else { $fwd.process_with_scratch(buf, $scratch) }
                }};
            }
            match self {
                Plan::F32 { forward, inverse: inv, scratch } => go!(f32, forward, inv, scratch),
                Plan::F64 { forward, inverse: inv, scratch } => go!(f64, forward, inv, scratch),
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Noise, Sine};

    type C = Complex<f64>;

    fn noise_signal(len: usize, seed: u64) -> Vec<C> {
        let mut re = Noise::<f64>::new(seed);
        let mut im = Noise::<f64>::new(seed + 1000);
        (0..len).map(|_| C::new(re.next_sample(), im.next_sample())).collect()
    }

    fn max_error(a: &[C], b: &[C]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x - *y).norm()).fold(0.0, f64::max)
    }

    #[test]
    fn fft_matches_dft() {
        for len in [1, 2, 3, 4, 5, 7, 8, 12, 16, 64, 100, 127, 256, 1000, 1024, 1031] {
            let input = noise_signal(len, len as u64);
            let slow = dft(&input);
            let mut fast = input.clone();
            Fft::new(len).forward(&mut fast);
            let err = max_error(&fast, &slow);
            assert!(err < 1e-9 * len as f64, "len {len}: max error {err}");
            let mut portable = input.clone();
            Fft::new_portable(len).forward(&mut portable);
            let err = max_error(&portable, &slow);
            assert!(err < 1e-9 * len as f64, "portable len {len}: max error {err}");
        }
    }

    #[test]
    fn inverse_roundtrip_any_length() {
        for len in [1, 3, 6, 100, 999] {
            let input = noise_signal(len, 21);
            for mut fft in [Fft::new(len), Fft::new_portable(len)] {
                let mut buf = input.clone();
                fft.forward(&mut buf);
                fft.inverse(&mut buf);
                assert!(max_error(&buf, &input) < 1e-12 * len as f64, "len {len}");
            }
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let mut fft = Fft::new(512);
        let input = noise_signal(512, 3);
        let mut buf = input.clone();
        fft.forward(&mut buf);
        fft.inverse(&mut buf);
        assert!(max_error(&buf, &input) < 1e-12);
    }

    #[test]
    fn parseval_energy_is_preserved() {
        let len = 256;
        let input = noise_signal(len, 9);
        let mut spectrum = input.clone();
        Fft::new(len).forward(&mut spectrum);
        let time: f64 = input.iter().map(|z| z.norm_sqr()).sum();
        let freq: f64 = spectrum.iter().map(|z| z.norm_sqr()).sum::<f64>() / len as f64;
        assert!((time - freq).abs() < 1e-9 * time);
    }

    #[test]
    fn pure_tone_lands_in_its_bin() {
        // 64-point FFT at 6400 Hz -> 100 Hz bins; an 800 Hz tone is exactly bin 8.
        let (len, fs) = (64, 6400.0);
        let mut signal = vec![0.0; len];
        Sine::new(800.0, fs).fill(&mut signal);
        let mut spectrum = vec![C::zero(); len];
        Fft::new(len).forward_real(&signal, &mut spectrum);
        for (k, z) in spectrum.iter().enumerate() {
            // real sine of amplitude 1 -> N/2 at +k and -k, zero everywhere else
            let expected = if k == 8 || k == len - 8 { len as f64 / 2.0 } else { 0.0 };
            assert!((z.norm() - expected).abs() < 1e-9, "bin {k}: {}", z.norm());
        }
        assert_eq!(bin_frequency(8, len, fs), 800.0);
    }

    #[test]
    fn f32_fft_matches_f64() {
        let input64 = noise_signal(1024, 5);
        let input32: Vec<Complex<f32>> = input64.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect();
        let (mut a, mut b) = (input64.clone(), input32.clone());
        Fft::new(1024).forward(&mut a);
        Fft::new(1024).forward(&mut b);
        let err = a.iter().zip(&b).map(|(x, y)| (*x - C::new(y.re as f64, y.im as f64)).norm()).fold(0.0, f64::max);
        assert!(err < 1e-3, "f32 vs f64 max error {err}");
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn rejects_length_zero() {
        Fft::<f64>::new(0);
    }
}
