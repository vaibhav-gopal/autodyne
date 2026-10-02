use super::Fft;
use crate::units::*;

/// FFT of real signals, about twice as fast as a complex FFT of the same length.
///
/// Only bins `0..=N/2` are produced: the rest mirror them (`X[N-k] = conj(X[k])`). With the default
/// `rustfft` feature, `f32` / `f64` run on [realfft](https://docs.rs/realfft)'s kernels; otherwise a
/// real N-point signal is packed into an N/2-point complex signal (even samples as real parts, odd
/// samples as imaginary parts), transformed with a half-size complex FFT, and untangled with one
/// twiddle per bin. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct RealFft<T: Float> {
    len: usize,
    /// the portable path: half-size complex FFT, untangling twiddles, scratch
    half: Option<Fft<T>>,
    /// e^(-i 2 pi k / len) for k in 0..len/2
    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    #[cfg(feature = "rustfft")]
    fast: Option<fast::Plan>,
}

impl<T: Float> RealFft<T> {
    /// Panics unless `len` is a power of two and at least 2.
    pub fn new(len: usize) -> Self {
        assert!(len >= 2 && len.is_power_of_two(), "real FFT length must be a power of two >= 2, got {len}");
        #[cfg(feature = "rustfft")]
        if let Some(plan) = fast::Plan::new::<T>(len) {
            return Self { len, half: None, twiddles: Vec::new(), scratch: Vec::new(), fast: Some(plan) };
        }
        let half_len = len / 2;
        let twiddles = (0..half_len).map(|k| Complex::cis(T::_lit(-std::f64::consts::TAU * k as f64 / len as f64))).collect();
        Self {
            len,
            half: Some(Fft::new(half_len)),
            twiddles,
            scratch: vec![Complex::zero(); half_len],
            #[cfg(feature = "rustfft")]
            fast: None,
        }
    }
    /// Length of the real signal.
    pub fn len(&self) -> usize {
        self.len
    }
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
        #[cfg(feature = "rustfft")]
        if let Some(plan) = &mut self.fast {
            plan.forward(input, out);
            return;
        }
        let m = self.len / 2;
        for (z, [even, odd]) in self.scratch.iter_mut().zip(input.as_chunks::<2>().0) {
            *z = Complex::new(*even, *odd);
        }
        self.half.as_mut().expect("portable path").forward(&mut self.scratch);
        let half = T::_lit(0.5);
        // bins 0 and N/2: the even and odd sums are the real and imaginary parts of the first bin
        let z0 = self.scratch[0];
        out[0] = Complex::new(z0.re + z0.im, T::_ZERO);
        out[m] = Complex::new(z0.re - z0.im, T::_ZERO);
        let minus_i_half = Complex::new(T::_ZERO, -half);
        for (k, o) in out[..m].iter_mut().enumerate().skip(1) {
            let (z, zc) = (self.scratch[k], self.scratch[m - k].conj());
            // spectra of the even and odd samples, then the length-N butterfly
            let even = (z + zc) * half;
            let odd = (z - zc) * minus_i_half;
            *o = even + self.twiddles[k] * odd;
        }
    }

    /// Signal from its spectrum (bins 0..=len/2) into `out`, scaled so `inverse(forward(x)) == x`.
    /// The imaginary parts of bins 0 and len/2 are ignored (they are zero for real signals).
    /// Panics on wrong lengths.
    pub fn inverse(&mut self, spectrum: &[Complex<T>], out: &mut [T]) {
        assert_eq!(spectrum.len(), self.spectrum_len(), "spectrum must hold len / 2 + 1 bins");
        assert_eq!(out.len(), self.len, "output length must match the FFT length");
        #[cfg(feature = "rustfft")]
        if let Some(plan) = &mut self.fast {
            plan.inverse(spectrum, out);
            let scale = T::_lit(1.0 / self.len as f64);
            out.iter_mut().for_each(|x| *x = *x * scale);
            return;
        }
        let m = self.len / 2;
        let half = T::_lit(0.5);
        for (k, z) in self.scratch.iter_mut().enumerate() {
            let a = spectrum[k];
            let b = spectrum[m - k].conj();
            let even = (a + b) * half;
            let odd = (a - b) * half * self.twiddles[k].conj();
            // repack: even samples as real parts, odd samples as imaginary parts
            *z = even + Complex::i() * odd;
        }
        self.half.as_mut().expect("portable path").inverse(&mut self.scratch);
        for ([even, odd], z) in out.as_chunks_mut::<2>().0.iter_mut().zip(&self.scratch) {
            *even = z.re;
            *odd = z.im;
        }
    }
}

#[cfg(feature = "rustfft")]
mod fast {
    use std::any::TypeId;
    use std::sync::Arc;

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

    impl std::fmt::Debug for Plan {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                Plan::F32 { .. } => "realfft plan (f32)",
                Plan::F64 { .. } => "realfft plan (f64)",
            })
        }
    }

    macro_rules! plan {
        ($V:ident, $F:ty, $len:expr) => {{
            let mut planner = RealFftPlanner::<$F>::new();
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
                        (std::slice::from_raw_parts(input.as_ptr().cast::<$F>(), input.len()), std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Rc<$F>>(), out.len()))
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

        /// Unscaled inverse; the imaginary parts of the first and last bins are taken as zero.
        pub(super) fn inverse<T: Float>(&mut self, spectrum: &[Complex<T>], out: &mut [T]) {
            macro_rules! go {
                ($F:ty, $plan:expr, $buf:expr, $scratch:expr) => {{
                    assert_eq!(TypeId::of::<T>(), TypeId::of::<$F>(), "FFT plan used with another element type");
                    // SAFETY: as in `forward`
                    let (spectrum, out) = unsafe {
                        (std::slice::from_raw_parts(spectrum.as_ptr().cast::<Rc<$F>>(), spectrum.len()), std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<$F>(), out.len()))
                    };
                    $buf.copy_from_slice(spectrum);
                    let last = $buf.len() - 1;
                    $buf[0].im = 0.0;
                    $buf[last].im = 0.0;
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
        for len in [2, 4, 8, 64, 1_024] {
            let x: Vec<f64> = Noise::new(len as u64).take(len).collect();
            let mut full = vec![Complex::zero(); len];
            Fft::new(len).forward_real(&x, &mut full);
            let mut rfft = RealFft::new(len);
            let mut half = vec![Complex::zero(); rfft.spectrum_len()];
            rfft.forward(&x, &mut half);
            for (k, (a, b)) in half.iter().zip(&full).enumerate() {
                assert!((*a - *b).norm() < 1e-9 * len as f64, "len {len} bin {k}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn inverse_roundtrip() {
        for len in [2, 16, 4_096] {
            let x: Vec<f64> = Noise::new(7).take(len).collect();
            let mut rfft = RealFft::new(len);
            let mut spectrum = vec![Complex::zero(); rfft.spectrum_len()];
            rfft.forward(&x, &mut spectrum);
            let mut back = vec![0.0; len];
            rfft.inverse(&spectrum, &mut back);
            let err = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12, "len {len}: {err}");
        }
    }
}
