use super::Fft;
use crate::units::*;

/// FFT of real signals, about twice as fast as a complex FFT of the same length.
///
/// A real N-point signal is packed into an N/2-point complex signal (even samples as real parts, odd
/// samples as imaginary parts), transformed with a half-size complex FFT, and untangled with one
/// twiddle per bin. Only bins `0..=N/2` are produced: the rest mirror them (`X[N-k] = conj(X[k])`).
/// Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct RealFft<T: Float> {
    len: usize,
    half: Fft<T>,
    /// e^(-i 2 pi k / len) for k in 0..len/2
    twiddles: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: Float> RealFft<T> {
    /// Panics unless `len` is a power of two and at least 2.
    pub fn new(len: usize) -> Self {
        assert!(len >= 2 && len.is_power_of_two(), "real FFT length must be a power of two >= 2, got {len}");
        let half_len = len / 2;
        let twiddles = (0..half_len).map(|k| Complex::cis(T::_lit(-std::f64::consts::TAU * k as f64 / len as f64))).collect();
        Self { len, half: Fft::new(half_len), twiddles, scratch: vec![Complex::zero(); half_len] }
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
        let m = self.len / 2;
        for (z, [even, odd]) in self.scratch.iter_mut().zip(input.as_chunks::<2>().0) {
            *z = Complex::new(*even, *odd);
        }
        self.half.forward(&mut self.scratch);
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
        self.half.inverse(&mut self.scratch);
        for ([even, odd], z) in out.as_chunks_mut::<2>().0.iter_mut().zip(&self.scratch) {
            *even = z.re;
            *odd = z.im;
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
