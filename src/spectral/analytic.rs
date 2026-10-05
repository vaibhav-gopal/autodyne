//! The analytic signal (`scipy.signal.hilbert`): a real signal plus `i` times its Hilbert
//! transform, whose magnitude is the envelope and whose angle is the instantaneous phase.

use crate::alloc_prelude::*;
use super::SpectralError;
use crate::fft::{Fft, RealFft};
use crate::signal::{lanes_f64, NdArray, NdView};
use crate::units::*;

/// The analytic signal of each lane of `x` along `axis` (`scipy.signal.hilbert`): `x + i H[x]`,
/// computed by zeroing the negative frequencies of an `n`-point FFT (`None`: the lane's length;
/// shorter lanes are zero-padded, longer ones truncated) and doubling the positive ones. Its
/// magnitude is the amplitude envelope; the derivative of its angle is the instantaneous frequency.
///
/// ```
/// use autodyne::signal::NdArray;
/// use autodyne::spectral::hilbert;
///
/// // a cosine's analytic signal is e^(i w t): magnitude 1 everywhere
/// let x = NdArray::from_fn(&[64], |i| (core::f64::consts::TAU * 4.0 * i[0] as f64 / 64.0).cos()).unwrap();
/// let z = hilbert(x.view(), 0, None).unwrap();
/// assert!(z.as_slice().iter().all(|z| (z.norm() - 1.0).abs() < 1e-12));
/// ```
pub fn hilbert<T: Float + Default>(x: NdView<'_, T>, axis: usize, n: Option<usize>) -> Result<NdArray<Complex<T>>, SpectralError> {
    let lanes = lanes_f64(&x, axis)?;
    let n = n.unwrap_or(x.shape()[axis]);
    if n == 0 {
        return Err(SpectralError::invalid("the transform length must be positive"));
    }
    let mut shape = x.shape().to_vec();
    shape[axis] = n;
    let mut out = NdArray::<Complex<T>>::zeros(&shape)?;
    let (mut real, mut complex) = (RealFft::<f64>::new(n), Fft::<f64>::new(n));
    let mut padded = vec![0.0; n];
    let mut half = vec![Complex::zero(); real.spectrum_len()];
    let mut full = vec![Complex::<f64>::zero(); n];
    // the positive frequencies doubled; DC (and Nyquist, for even n) kept as they are
    let doubled = n.div_ceil(2);
    for (lane, mut dst) in lanes.iter().zip(out.lanes_mut(axis)?) {
        let used = lane.len().min(n);
        padded[..used].copy_from_slice(&lane[..used]);
        padded[used..].iter_mut().for_each(|v| *v = 0.0);
        real.forward(&padded, &mut half);
        full.iter_mut().for_each(|z| *z = Complex::zero());
        full[0] = half[0];
        for k in 1..doubled {
            full[k] = half[k] * 2.0;
        }
        if n.is_multiple_of(2) {
            full[n / 2] = half[n / 2];
        }
        complex.inverse(&mut full);
        for (d, z) in dst.iter_mut().zip(&full) {
            *d = Complex::new(T::_lit(z.re), T::_lit(z.im));
        }
    }
    Ok(out)
}
