use super::SignalError;
use crate::units::*;

/// Analysis and transforms for complex signals (IQ baseband, spectra). Implemented for `[Complex<T>]`.
pub trait ComplexSignal {
    type Real: Float;

    fn samples(&self) -> &[Complex<Self::Real>];
    fn samples_mut(&mut self) -> &mut [Complex<Self::Real>];

    /// Sum of |z|^2.
    fn energy(&self) -> Self::Real {
        self.samples().iter().fold(Self::Real::_ZERO, |s, z| s + z.norm_sqr())
    }
    /// Mean of |z|^2.
    fn power(&self) -> Option<Self::Real> {
        let n = self.samples().len();
        (n > 0).then(|| self.energy() / Self::Real::_lit(n as f64))
    }
    fn rms(&self) -> Option<Self::Real> {
        self.power().map(|p| p._sqrt())
    }
    /// Largest magnitude (0 for an empty signal).
    fn peak(&self) -> Self::Real {
        self.samples().iter().fold(Self::Real::_ZERO, |m, z| m._max(z.norm()))
    }
    /// Index of the sample with the largest magnitude, e.g. the strongest FFT bin.
    fn argmax_magnitude(&self) -> Option<usize> {
        let x = self.samples();
        (0..x.len()).reduce(|best, i| if x[i].norm_sqr() > x[best].norm_sqr() { i } else { best })
    }
    /// Hermitian inner product: `sum of a[n] * conj(b[n])`.
    fn inner(&self, other: &[Complex<Self::Real>]) -> Result<Complex<Self::Real>, SignalError> {
        let x = self.samples();
        if x.len() != other.len() {
            return Err(SignalError::LengthMismatch(x.len(), other.len()));
        }
        Ok(x.iter().zip(other).fold(Complex::zero(), |s, (&a, &b)| s + a * b.conj()))
    }
    /// Conjugates every sample in place (mirrors a spectrum, reverses rotation direction).
    fn conj_in_place(&mut self) {
        self.samples_mut().iter_mut().for_each(|z| *z = z.conj());
    }
    /// Multiplies every sample by `gain`.
    fn scale(&mut self, gain: Self::Real) {
        self.samples_mut().iter_mut().for_each(|z| *z *= gain);
    }
    /// |z| per sample into `out` (AM envelope, magnitude spectrum). Errors unless lengths match.
    fn magnitudes_into(&self, out: &mut [Self::Real]) -> Result<(), SignalError> {
        self.map_into(out, |z| z.norm())
    }
    /// arg z per sample into `out`, in (-pi, pi] (PM, phase spectrum).
    fn phases_into(&self, out: &mut [Self::Real]) -> Result<(), SignalError> {
        self.map_into(out, |z| z.arg())
    }
    /// Real parts (the I channel) into `out`.
    fn real_into(&self, out: &mut [Self::Real]) -> Result<(), SignalError> {
        self.map_into(out, |z| z.re)
    }
    /// Imaginary parts (the Q channel) into `out`.
    fn imag_into(&self, out: &mut [Self::Real]) -> Result<(), SignalError> {
        self.map_into(out, |z| z.im)
    }
    /// f(z) per sample into `out`. Errors unless lengths match.
    fn map_into(&self, out: &mut [Self::Real], f: impl Fn(Complex<Self::Real>) -> Self::Real) -> Result<(), SignalError> {
        let x = self.samples();
        if x.len() != out.len() {
            return Err(SignalError::LengthMismatch(x.len(), out.len()));
        }
        for (o, &z) in out.iter_mut().zip(x) {
            *o = f(z);
        }
        Ok(())
    }
}

impl<T: Float> ComplexSignal for [Complex<T>] {
    type Real = T;
    fn samples(&self) -> &[Complex<T>] {
        self
    }
    fn samples_mut(&mut self) -> &mut [Complex<T>] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type C = Complex<f64>;

    #[test]
    fn levels_and_views() {
        let x = [C::new(3.0, 4.0), C::new(0.0, -1.0)];
        assert_eq!(x.energy(), 26.0);
        assert_eq!(x.peak(), 5.0);
        assert_eq!(x.argmax_magnitude(), Some(0));
        let mut mags = [0.0; 2];
        x.magnitudes_into(&mut mags).unwrap();
        assert_eq!(mags, [5.0, 1.0]);
        let mut im = [0.0; 2];
        x.imag_into(&mut im).unwrap();
        assert_eq!(im, [4.0, -1.0]);
        assert!(x.phases_into(&mut [0.0; 3]).is_err());
    }

    #[test]
    fn hermitian_inner_product() {
        let a = [C::new(1.0, 1.0)];
        // <a, a> is |a|^2, real
        assert_eq!(a.inner(&a), Ok(C::new(2.0, 0.0)));
        // (1+i) * conj(i) = (1+i)(-i) = 1 - i
        assert_eq!(a.inner(&[C::i()]), Ok(C::new(1.0, -1.0)));
    }

    #[test]
    fn conj_and_scale() {
        let mut x = [C::new(1.0, 2.0)];
        x.conj_in_place();
        x.scale(2.0);
        assert_eq!(x, [C::new(2.0, -4.0)]);
    }
}
