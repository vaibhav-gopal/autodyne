//! FFTs of real signals ([`RealFft`]): the half spectrum (bins `0..=N/2`) and back, about twice as fast
//! as a complex FFT of the same length.

use crate::alloc_prelude::*;
use super::Fft;
use crate::units::*;

/// FFT of real signals (any length >= 1), about twice as fast as a complex FFT of the same length.
///
/// Only bins `0..=N/2` are produced: the rest mirror them (`X[N-k] = conj(X[k])`). With the default
/// `rustfft` feature, `f32` / `f64` run on [realfft](https://docs.rs/realfft)'s kernels. Otherwise an
/// even-length signal is packed into an N/2-point complex signal (even samples as real parts, odd
/// samples as imaginary parts), transformed with a half-size complex FFT and untangled with one
/// twiddle per bin; an odd-length signal goes through a full-length complex FFT. Allocates only in
/// `new`.
#[derive(Debug, Clone)]
pub struct RealFft<T: Float> {
    len: usize,
    kernel: RealKernel<T>,
}

#[derive(Debug, Clone)]
enum RealKernel<T: Float> {
    /// even lengths: the half-size complex FFT, e^(-i 2 pi k / len) for k in 0..len/2, scratch
    Half { half: Fft<T>, twiddles: Vec<Complex<T>>, scratch: Vec<Complex<T>> },
    /// odd lengths: a full-length complex FFT and its buffer
    Full { fft: Fft<T>, scratch: Vec<Complex<T>> },
    #[cfg(feature = "rustfft")]
    Fast(fast::Plan),
}

impl<T: Float> RealFft<T> {
    /// Panics if `len` is 0.
    pub fn new(len: usize) -> Self {
        assert!(len >= 1, "real FFT length must be at least 1");
        #[cfg(feature = "rustfft")]
        if let Some(plan) = fast::Plan::new::<T>(len) {
            return Self { len, kernel: RealKernel::Fast(plan) };
        }
        Self::new_portable(len)
    }

    /// The portable kernels whatever features are enabled (the reference the fast backend is tested
    /// against). Panics if `len` is 0.
    pub fn new_portable(len: usize) -> Self {
        assert!(len >= 1, "real FFT length must be at least 1");
        let kernel = if len.is_multiple_of(2) {
            let half_len = len / 2;
            let twiddles = (0..half_len).map(|k| Complex::cis(T::_lit(-core::f64::consts::TAU * k as f64 / len as f64))).collect();
            RealKernel::Half { half: Fft::new_portable(half_len), twiddles, scratch: vec![Complex::zero(); half_len] }
        } else {
            RealKernel::Full { fft: Fft::new_portable(len), scratch: vec![Complex::zero(); len] }
        };
        Self { len, kernel }
    }
    /// Length of the real signal.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Always false: a transform has at least one point.
    pub fn is_empty(&self) -> bool {
        false
    }
    /// Number of spectrum bins: `len / 2 + 1`.
    pub fn spectrum_len(&self) -> usize {
        self.len / 2 + 1
    }

    /// Spectrum of `input` into `out` (bins 0..=len/2). Panics on wrong lengths.
    pub fn forward(&mut self, input: &[T], out: &mut [Complex<T>]) {
        assert_eq!(input.len(), self.len, "input length must match the FFT length");
        assert_eq!(out.len(), self.spectrum_len(), "output must hold len / 2 + 1 bins");
        match &mut self.kernel {
            #[cfg(feature = "rustfft")]
            RealKernel::Fast(plan) => plan.forward(input, out),
            RealKernel::Full { fft, scratch } => {
                fft.forward_real(input, scratch);
                out.copy_from_slice(&scratch[..out.len()]);
            }
            RealKernel::Half { half, twiddles, scratch } => {
                let m = self.len / 2;
                for (z, [even, odd]) in scratch.iter_mut().zip(input.as_chunks::<2>().0) {
                    *z = Complex::new(*even, *odd);
                }
                half.forward(scratch);
                let h = T::_lit(0.5);
                // bins 0 and N/2: the even and odd sums are the real and imaginary parts of the first bin
                let z0 = scratch[0];
                out[0] = Complex::new(z0.re + z0.im, T::_ZERO);
                out[m] = Complex::new(z0.re - z0.im, T::_ZERO);
                let minus_i_half = Complex::new(T::_ZERO, -h);
                for (k, o) in out[..m].iter_mut().enumerate().skip(1) {
                    let (z, zc) = (scratch[k], scratch[m - k].conj());
                    // spectra of the even and odd samples, then the length-N butterfly
                    let even = (z + zc) * h;
                    let odd = (z - zc) * minus_i_half;
                    *o = even + twiddles[k] * odd;
                }
            }
        }
    }

    /// Signal from its spectrum (bins 0..=len/2) into `out`, scaled so `inverse(forward(x)) == x`.
    /// The imaginary parts of bin 0 (and of bin len/2 for even lengths) are ignored (they are zero
    /// for real signals). Panics on wrong lengths.
    pub fn inverse(&mut self, spectrum: &[Complex<T>], out: &mut [T]) {
        assert_eq!(spectrum.len(), self.spectrum_len(), "spectrum must hold len / 2 + 1 bins");
        assert_eq!(out.len(), self.len, "output length must match the FFT length");
        let n = self.len;
        match &mut self.kernel {
            #[cfg(feature = "rustfft")]
            RealKernel::Fast(plan) => {
                plan.inverse(spectrum, out, n.is_multiple_of(2));
                let scale = T::_lit(1.0 / n as f64);
                out.iter_mut().for_each(|x| *x = *x * scale);
            }
            RealKernel::Full { fft, scratch } => {
                // the full Hermitian spectrum, then a complex inverse
                scratch[0] = Complex::new(spectrum[0].re, T::_ZERO);
                for k in 1..spectrum.len() {
                    scratch[k] = spectrum[k];
                    scratch[n - k] = spectrum[k].conj();
                }
                fft.inverse(scratch);
                for (o, z) in out.iter_mut().zip(scratch.iter()) {
                    *o = z.re;
                }
            }
            RealKernel::Half { half, twiddles, scratch } => {
                let m = n / 2;
                let h = T::_lit(0.5);
                // bins 0 and N/2 are real; their imaginary parts are ignored
                let bin = |k: usize| if k == 0 || k == m { Complex::new(spectrum[k].re, T::_ZERO) } else { spectrum[k] };
                for (k, z) in scratch.iter_mut().enumerate() {
                    let a = bin(k);
                    let b = bin(m - k).conj();
                    let even = (a + b) * h;
                    let odd = (a - b) * h * twiddles[k].conj();
                    // repack: even samples as real parts, odd samples as imaginary parts
                    *z = even + Complex::i() * odd;
                }
                half.inverse(scratch);
                for ([even, odd], z) in out.as_chunks_mut::<2>().0.iter_mut().zip(scratch.iter()) {
                    *even = z.re;
                    *odd = z.im;
                }
            }
        }
    }
}

#[cfg(feature = "rustfft")]
mod fast {
    use core::any::TypeId;
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use realfft::num_complex::Complex as Rc;
    use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

    use crate::units::*;

    /// realfft plans with their buffers (its transforms use their input as scratch, so the caller's
    /// input is copied in first).
    #[derive(Clone)]
    pub(super) enum Plan {
        F32 { forward: Arc<dyn RealToComplex<f32>>, inverse: Arc<dyn ComplexToReal<f32>>, input: Vec<f32>, spectrum: Vec<Rc<f32>>, scratch: Vec<Rc<f32>> },
        F64 { forward: Arc<dyn RealToComplex<f64>>, inverse: Arc<dyn ComplexToReal<f64>>, input: Vec<f64>, spectrum: Vec<Rc<f64>>, scratch: Vec<Rc<f64>> },
    }

    impl core::fmt::Debug for Plan {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str(match self {
                Plan::F32 { .. } => "realfft plan (f32)",
                Plan::F64 { .. } => "realfft plan (f64)",
            })
        }
    }

    macro_rules! plan {
        ($V:ident, $F:ty, $len:expr) => {{
            // one planner per element type for the whole process: it keeps every plan it made, so
            // a length planned before (twiddles, factorization) costs only the buffers below
            static PLANNER: OnceLock<Mutex<RealFftPlanner<$F>>> = OnceLock::new();
            let mut planner = PLANNER.get_or_init(|| Mutex::new(RealFftPlanner::new())).lock().unwrap_or_else(PoisonError::into_inner);
            let (forward, inverse) = (planner.plan_fft_forward($len), planner.plan_fft_inverse($len));
            let scratch = vec![Rc::new(0.0, 0.0); forward.get_scratch_len().max(inverse.get_scratch_len())];
            Some(Plan::$V { input: forward.make_input_vec(), spectrum: inverse.make_input_vec(), forward, inverse, scratch })
        }};
    }

    impl Plan {
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

        pub(super) fn forward<T: Float>(&mut self, input: &[T], out: &mut [Complex<T>]) {
            macro_rules! go {
                ($F:ty, $plan:expr, $buf:expr, $scratch:expr) => {{
                    assert_eq!(TypeId::of::<T>(), TypeId::of::<$F>(), "FFT plan used with another element type");
                    // SAFETY: T is $F (checked); autodyne's and num-complex's Complex are both
                    // repr(C) { re, im }
                    let (input, out) = unsafe {
                        (core::slice::from_raw_parts(input.as_ptr().cast::<$F>(), input.len()), core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Rc<$F>>(), out.len()))
                    };
                    $buf.copy_from_slice(input);
                    $plan.process_with_scratch($buf, out, $scratch).expect("buffer lengths come from the plan");
                }};
            }
            match self {
                Plan::F32 { forward, input: buf, scratch, .. } => go!(f32, forward, buf, scratch),
                Plan::F64 { forward, input: buf, scratch, .. } => go!(f64, forward, buf, scratch),
            }
        }

        /// Unscaled inverse; the imaginary part of the first bin (and of the last one when `even`,
        /// where it is the Nyquist bin) is taken as zero.
        pub(super) fn inverse<T: Float>(&mut self, spectrum: &[Complex<T>], out: &mut [T], even: bool) {
            macro_rules! go {
                ($F:ty, $plan:expr, $buf:expr, $scratch:expr) => {{
                    assert_eq!(TypeId::of::<T>(), TypeId::of::<$F>(), "FFT plan used with another element type");
                    // SAFETY: as in `forward`
                    let (spectrum, out) = unsafe {
                        (core::slice::from_raw_parts(spectrum.as_ptr().cast::<Rc<$F>>(), spectrum.len()), core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<$F>(), out.len()))
                    };
                    $buf.copy_from_slice(spectrum);
                    let last = $buf.len() - 1;
                    $buf[0].im = 0.0;
                    if even {
                        $buf[last].im = 0.0;
                    }
                    $plan.process_with_scratch($buf, out, $scratch).expect("buffer lengths come from the plan");
                }};
            }
            match self {
                Plan::F32 { inverse, spectrum: buf, scratch, .. } => go!(f32, inverse, buf, scratch),
                Plan::F64 { inverse, spectrum: buf, scratch, .. } => go!(f64, inverse, buf, scratch),
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Noise;

    #[test]
    fn matches_the_complex_fft() {
        for len in [1, 2, 3, 4, 5, 8, 9, 64, 100, 127, 1_000, 1_024] {
            let x: Vec<f64> = Noise::new(len as u64).take(len).collect();
            let mut full = vec![Complex::zero(); len];
            Fft::new(len).forward_real(&x, &mut full);
            for mut rfft in [RealFft::new(len), RealFft::new_portable(len)] {
                let mut half = vec![Complex::zero(); rfft.spectrum_len()];
                rfft.forward(&x, &mut half);
                for (k, (a, b)) in half.iter().zip(&full).enumerate() {
                    assert!((*a - *b).norm() < 1e-9 * len as f64, "len {len} bin {k}: {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn inverse_roundtrip() {
        for len in [1, 2, 3, 7, 16, 100, 999, 4_096] {
            let x: Vec<f64> = Noise::new(7).take(len).collect();
            for mut rfft in [RealFft::new(len), RealFft::new_portable(len)] {
                let mut spectrum = vec![Complex::zero(); rfft.spectrum_len()];
                rfft.forward(&x, &mut spectrum);
                let mut back = vec![0.0; len];
                rfft.inverse(&spectrum, &mut back);
                let err = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                assert!(err < 1e-12 * len as f64, "len {len}: {err}");
            }
        }
    }

    #[test]
    fn odd_lengths_keep_the_last_bins_imaginary_part() {
        // for odd N the last bin is not the Nyquist bin: its imaginary part matters
        let len = 7;
        let x: Vec<f64> = Noise::new(3).take(len).collect();
        for mut rfft in [RealFft::new(len), RealFft::new_portable(len)] {
            let mut spectrum = vec![Complex::zero(); rfft.spectrum_len()];
            rfft.forward(&x, &mut spectrum);
            assert!(spectrum[3].im.abs() > 1e-3);
            let mut back = vec![0.0; len];
            rfft.inverse(&spectrum, &mut back);
            assert!(x.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-12));
        }
    }
}
