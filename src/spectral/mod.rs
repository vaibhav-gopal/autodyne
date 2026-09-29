//! Spectral analysis: a reference DFT, a radix-2 FFT, and [`RealFft`] for real signals (about twice as fast).
//!
//! Sign convention: forward `X[k] = sum_n x[n] e^(-i 2 pi k n / N)`; inverse divides by N,
//! so `inverse(forward(x)) == x`.

use crate::units::*;

mod real;
pub use real::*;

/// Naive O(n^2) discrete Fourier transform. Exact by definition, so it is the reference the FFT is tested
/// against; also usable for any length (the FFT needs a power of two). Allocates the output.
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

/// Iterative radix-2 Cooley-Tukey FFT of a fixed power-of-two length.
/// Twiddle factors and the bit-reversal table are computed once in `new`, so transforms run
/// in place without allocating.
#[derive(Debug, Clone)]
pub struct Fft<T: Float> {
    len: usize,
    /// e^(-i 2 pi k / len) for k in 0..len/2, computed in f64 for accuracy
    twiddles: Vec<Complex<T>>,
    /// bit_reverse[i] = i with its log2(len) low bits reversed
    bit_reverse: Vec<usize>,
}

impl<T: Float> Fft<T> {
    /// Panics unless `len` is a power of two (1 included).
    pub fn new(len: usize) -> Self {
        assert!(len.is_power_of_two(), "FFT length must be a power of two, got {len}");
        let bits = len.trailing_zeros();
        let twiddles = (0..len / 2)
            .map(|k| Complex::cis(T::_lit(-std::f64::consts::TAU * k as f64 / len as f64)))
            .collect();
        let bit_reverse = (0..len)
            .map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) })
            .collect();
        Self { len, twiddles, bit_reverse }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false // a power of two is never 0
    }

    /// Forward transform in place. Panics if `buf.len() != self.len()`.
    pub fn forward(&self, buf: &mut [Complex<T>]) {
        assert_eq!(buf.len(), self.len, "buffer length must match the FFT length");
        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if i < j {
                buf.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= self.len {
            let half = size / 2;
            let stride = self.len / size; // twiddle index step for this stage
            for block in buf.chunks_exact_mut(size) {
                let (lo, hi) = block.split_at_mut(half);
                for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let t = *b * self.twiddles[k * stride];
                    *b = *a - t;
                    *a += t;
                }
            }
            size *= 2;
        }
    }

    /// Inverse transform in place (scaled by 1/len). Panics if `buf.len() != self.len()`.
    pub fn inverse(&self, buf: &mut [Complex<T>]) {
        // ifft(x) = conj(fft(conj(x))) / N
        buf.iter_mut().for_each(|z| *z = z.conj());
        self.forward(buf);
        let scale = T::_lit(1.0 / self.len as f64);
        buf.iter_mut().for_each(|z| *z = z.conj() * scale);
    }

    /// Transforms a real signal: copies `input` into `out` as complex values, then transforms `out`.
    /// Panics unless both lengths equal `self.len()`.
    pub fn forward_real(&self, input: &[T], out: &mut [Complex<T>]) {
        assert_eq!(input.len(), self.len, "input length must match the FFT length");
        assert_eq!(out.len(), self.len, "output length must match the FFT length");
        for (o, &x) in out.iter_mut().zip(input) {
            *o = Complex::from(x);
        }
        self.forward(out);
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
        for len in [1, 2, 4, 8, 16, 64, 256, 1024] {
            let input = noise_signal(len, len as u64);
            let mut fast = input.clone();
            Fft::new(len).forward(&mut fast);
            let slow = dft(&input);
            let err = max_error(&fast, &slow);
            assert!(err < 1e-9 * len as f64, "len {len}: max error {err}");
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let fft = Fft::new(512);
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
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        Fft::<f64>::new(100);
    }
}
