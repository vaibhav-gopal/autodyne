use super::SignalError;
use crate::gain::gain_to_db;
use crate::resample::Resampler;
use crate::units::*;

/// Read-only analysis of a real-valued signal. Implemented for `[T]`, so it works on anything that
/// dereferences to a slice (`Vec<T>`, arrays, `AudioBuffer` channels).
///
/// Measurements that have no meaning for an empty signal (mean, rms, argmax, ...) return `None`;
/// sums and norms of an empty signal are 0. NaN samples are ignored by the ordering methods.
pub trait Signal {
    type Sample: Float;

    /// The underlying samples.
    fn samples(&self) -> &[Self::Sample];

    // Levels and statistics ======================================================================

    /// Sum of all samples (vectorized).
    fn sum(&self) -> Self::Sample {
        crate::simd::sum(self.samples())
    }
    fn mean(&self) -> Option<Self::Sample> {
        let n = self.samples().len();
        (n > 0).then(|| self.sum() / Self::Sample::_lit(n as f64))
    }
    /// Sum of squares (vectorized).
    fn energy(&self) -> Self::Sample {
        crate::simd::dot(self.samples(), self.samples())
    }
    /// Mean of squares.
    fn power(&self) -> Option<Self::Sample> {
        let n = self.samples().len();
        (n > 0).then(|| self.energy() / Self::Sample::_lit(n as f64))
    }
    /// Root mean square level (a full-scale sine has rms 1/sqrt 2).
    fn rms(&self) -> Option<Self::Sample> {
        self.power().map(|p| p._sqrt())
    }
    /// RMS level in dB relative to 1.0 (dBFS for audio).
    fn rms_db(&self) -> Option<Self::Sample> {
        self.rms().map(gain_to_db)
    }
    /// Largest absolute sample (0 for an empty signal).
    fn peak(&self) -> Self::Sample {
        self.samples().iter().fold(Self::Sample::_ZERO, |m, &x| m._max(x._abs()))
    }
    /// Peak level in dB relative to 1.0 (dBFS); negative infinity for silence.
    fn peak_db(&self) -> Self::Sample {
        gain_to_db(self.peak())
    }
    /// Peak / rms: 1 for a square wave, sqrt 2 for a sine, higher for spiky material.
    fn crest_factor(&self) -> Option<Self::Sample> {
        let rms = self.rms()?;
        (rms > Self::Sample::_ZERO).then(|| self.peak() / rms)
    }
    fn norm_l1(&self) -> Self::Sample {
        self.samples().iter().fold(Self::Sample::_ZERO, |s, &x| s + x._abs())
    }
    fn norm_l2(&self) -> Self::Sample {
        self.energy()._sqrt()
    }
    /// Same as `peak`.
    fn norm_inf(&self) -> Self::Sample {
        self.peak()
    }
    /// Population variance.
    fn variance(&self) -> Option<Self::Sample> {
        let mean = self.mean()?;
        let n = Self::Sample::_lit(self.samples().len() as f64);
        let squares = self.samples().iter().fold(Self::Sample::_ZERO, |s, &x| s + (x - mean) * (x - mean));
        Some(squares / n)
    }
    fn std_dev(&self) -> Option<Self::Sample> {
        self.variance().map(|v| v._sqrt())
    }
    /// Number of sign changes between consecutive samples (zeros don't count as a sign).
    fn zero_crossings(&self) -> usize {
        let mut last_sign = None;
        let mut count = 0;
        for &x in self.samples() {
            if x != Self::Sample::_ZERO && !x._is_nan() {
                let positive = x > Self::Sample::_ZERO;
                if last_sign.is_some_and(|s| s != positive) {
                    count += 1;
                }
                last_sign = Some(positive);
            }
        }
        count
    }

    // Ordering ===================================================================================

    fn min(&self) -> Option<Self::Sample> {
        self.argmin().map(|i| self.samples()[i])
    }
    fn max(&self) -> Option<Self::Sample> {
        self.argmax().map(|i| self.samples()[i])
    }
    /// Index of the smallest sample (the first one on ties).
    fn argmin(&self) -> Option<usize> {
        extremum(self.samples(), |candidate, best| candidate < best)
    }
    /// Index of the largest sample (the first one on ties).
    fn argmax(&self) -> Option<usize> {
        extremum(self.samples(), |candidate, best| candidate > best)
    }
    fn min_max(&self) -> Option<(Self::Sample, Self::Sample)> {
        Some((self.min()?, self.max()?))
    }

    // Relations between two signals ==============================================================

    /// Inner (dot) product, vectorized. Errors unless the lengths match.
    fn inner(&self, other: &[Self::Sample]) -> Result<Self::Sample, SignalError> {
        same_len(self.samples(), other)?;
        Ok(crate::simd::dot(self.samples(), other))
    }
    /// Angle between the two signals as vectors, in radians: 0 = same shape, pi/2 = uncorrelated
    /// (orthogonal), pi = inverted.
    fn angle(&self, other: &[Self::Sample]) -> Result<Self::Sample, SignalError> {
        let dot = self.inner(other)?;
        let norms = self.norm_l2() * other.norm_l2();
        if norms == Self::Sample::_ZERO {
            return Err(SignalError::ZeroNorm);
        }
        Ok((dot / norms)._clamp(-Self::Sample::_ONE, Self::Sample::_ONE)._acos())
    }
    /// Euclidean distance: the L2 norm of the difference.
    fn distance(&self, other: &[Self::Sample]) -> Result<Self::Sample, SignalError> {
        same_len(self.samples(), other)?;
        let squares = self.samples().iter().zip(other).fold(Self::Sample::_ZERO, |s, (&a, &b)| s + (a - b) * (a - b));
        Ok(squares._sqrt())
    }

    // Length-changing operations (allocate) ======================================================

    /// Full linear convolution with `kernel`: `len + kernel.len() - 1` samples.
    fn convolve(&self, kernel: &[Self::Sample]) -> Vec<Self::Sample> {
        crate::filter::convolve(self.samples(), kernel)
    }
    /// Full cross-correlation with `other`. Element `k` corresponds to lag `k - (other.len() - 1)`:
    /// the sum over n of self[n + lag] * other[n]. The peak's position shows how far `self` is
    /// delayed relative to `other`.
    fn correlate(&self, other: &[Self::Sample]) -> Vec<Self::Sample> {
        let reversed: Vec<_> = other.iter().rev().copied().collect();
        crate::filter::convolve(self.samples(), &reversed)
    }
    /// The whole signal converted from `from_rate` to `to_rate` with the streaming `Resampler`,
    /// aligned to the input (its filter delay removed) and `ceil(len * to / from)` samples long.
    fn resampled(&self, from_rate: u32, to_rate: u32) -> Vec<Self::Sample> {
        let input = self.samples();
        let mut rs = Resampler::new(from_rate, to_rate);
        let (up, down) = rs.ratio();
        let wanted = (input.len() * up).div_ceil(down);
        let delay = rs.delay() as usize;
        let mut out = Vec::with_capacity(wanted + delay + up);
        let chunk = 4096;
        let mut scratch = vec![Self::Sample::_ZERO; rs.max_output_len(chunk)];
        let zeros = vec![Self::Sample::_ZERO; chunk];
        let mut fed = 0;
        while out.len() < wanted + delay {
            // the input, then silence to flush the filter's delay
            let block = if fed < input.len() { &input[fed..(fed + chunk).min(input.len())] } else { &zeros[..] };
            fed += block.len();
            let n = rs.process(block, &mut scratch);
            out.extend_from_slice(&scratch[..n]);
        }
        out.drain(..delay);
        out.truncate(wanted);
        out
    }
}

impl<T: Float> Signal for [T] {
    type Sample = T;
    fn samples(&self) -> &[T] {
        self
    }
}

fn same_len<T>(a: &[T], b: &[T]) -> Result<(), SignalError> {
    if a.len() == b.len() { Ok(()) } else { Err(SignalError::LengthMismatch(a.len(), b.len())) }
}

/// Index of the sample that wins `better(candidate, best)` against all others, skipping NaNs.
fn extremum<T: Float>(x: &[T], better: impl Fn(T, T) -> bool) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &v) in x.iter().enumerate() {
        if v._is_nan() {
            continue;
        }
        if best.is_none_or(|b| better(v, x[b])) {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Sine;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn levels_and_statistics() {
        let x = [1.0, -2.0, 3.0, -4.0];
        assert_eq!(x.sum(), -2.0);
        assert_eq!(x.mean(), Some(-0.5));
        assert_eq!(x.energy(), 30.0);
        assert_eq!(x.power(), Some(7.5));
        assert_eq!(x.peak(), 4.0);
        assert_eq!(x.norm_l1(), 10.0);
        assert!(close(x.norm_l2(), 30f64.sqrt(), 1e-12));
        assert_eq!(x.variance(), Some(7.25)); // E[x^2] - mean^2 = 7.5 - 0.25
        assert_eq!(x.zero_crossings(), 3);
    }

    #[test]
    fn sine_levels() {
        // phase offset so no sample lands exactly on a zero crossing
        let mut tone = vec![0.0; 48_000];
        Sine::new(1_000.0, 48_000.0).with_amplitude(0.5).with_phase(0.1).fill(&mut tone);
        assert!(close(tone.rms().unwrap(), 0.5 / 2f64.sqrt(), 1e-9));
        // the sample grid misses the true peak slightly (by < 0.05%)
        assert!(close(tone.crest_factor().unwrap(), 2f64.sqrt(), 1e-3));
        assert!(close(tone.peak_db(), -6.0206, 0.01));
        assert_eq!(tone.zero_crossings(), 2_000); // two per cycle, 1000 cycles
    }

    #[test]
    fn empty_signals() {
        let e: [f64; 0] = [];
        assert_eq!((e.sum(), e.energy(), e.peak()), (0.0, 0.0, 0.0));
        assert_eq!((e.mean(), e.rms(), e.argmax(), e.min()), (None, None, None, None));
        assert_eq!(e.peak_db(), f64::NEG_INFINITY);
    }

    #[test]
    fn ordering_skips_nan_and_takes_first_tie() {
        let x = [f64::NAN, 2.0, -1.0, 5.0, 5.0, -1.0];
        assert_eq!((x.argmax(), x.argmin()), (Some(3), Some(2)));
        assert_eq!(x.min_max(), Some((-1.0, 5.0)));
    }

    #[test]
    fn relations_between_signals() {
        let a = [1.0, 0.0];
        let b = [0.0, 2.0];
        assert_eq!(a.inner(&b), Ok(0.0));
        assert!(close(a.angle(&b).unwrap(), std::f64::consts::FRAC_PI_2, 1e-12));
        assert!(close(a.angle(&[-3.0, 0.0]).unwrap(), std::f64::consts::PI, 1e-12));
        assert!(close(a.distance(&b).unwrap(), 5f64.sqrt(), 1e-12));
        assert_eq!(a.inner(&[1.0]), Err(SignalError::LengthMismatch(2, 1)));
        assert_eq!(a.angle(&[0.0, 0.0]), Err(SignalError::ZeroNorm));
    }

    #[test]
    fn correlation_finds_a_delay() {
        let pulse = [0.0, 1.0, 0.5, 0.0, 0.0, 0.0];
        let delayed = [0.0, 0.0, 0.0, 1.0, 0.5, 0.0]; // same shape, 2 samples later
        let corr = delayed.correlate(&pulse);
        let lag = corr.argmax().unwrap() as isize - (pulse.len() as isize - 1);
        assert_eq!(lag, 2);
        assert_eq!([1.0, 2.0].convolve(&[1.0, 1.0]), [1.0, 3.0, 2.0]);
    }

    #[test]
    fn resampled_is_aligned_and_sized() {
        for (from, to) in [(48_000u32, 44_100u32), (44_100, 48_000), (48_000, 16_000), (48_000, 96_000)] {
            let tone: Vec<f64> = Sine::new(1_000.0, from as f64).take(from as usize / 10).collect();
            let out = tone.resampled(from, to);
            assert_eq!(out.len(), (tone.len() * to as usize).div_ceil(from as usize));
            // away from the edges (filter start-up and the flushed tail) it is the same sine at the new rate
            let margin = out.len() / 10;
            for (n, &y) in out.iter().enumerate().take(out.len() - margin).skip(margin) {
                let ideal = (std::f64::consts::TAU * 1_000.0 * n as f64 / to as f64).sin();
                assert!(close(y, ideal, 2e-3), "{from} -> {to} sample {n}: {y} vs {ideal}");
            }
        }
    }
}
