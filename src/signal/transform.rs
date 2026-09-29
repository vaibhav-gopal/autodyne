use super::{Signal, SignalError};
use crate::gain::db_to_gain;
use crate::units::*;

/// How a pointwise operation treats `other` when it is shorter or longer than `self`.
/// `self` always keeps its length; the policy decides what `other` contributes at each index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Broadcast<T> {
    /// Lengths must match exactly, otherwise `SignalError::LengthMismatch`.
    Strict,
    /// Only the overlapping part is combined; the rest of `self` is left unchanged.
    Truncate,
    /// `other` continues as zeros.
    ZeroPad,
    /// `other` continues as this constant.
    Fill(T),
    /// `other` holds its last sample.
    Latch,
    /// `other` repeats from its start.
    Tile,
}

/// In-place transforms of a real-valued signal. Implemented for `[T]`.
pub trait SignalMut: Signal {
    fn samples_mut(&mut self) -> &mut [Self::Sample];

    // Gain and shape =============================================================================

    /// Multiplies every sample by `gain`.
    fn scale(&mut self, gain: Self::Sample) {
        self.samples_mut().iter_mut().for_each(|x| *x = *x * gain);
    }
    /// Changes the level by `db` decibels.
    fn gain_db(&mut self, db: Self::Sample) {
        self.scale(db_to_gain(db));
    }
    /// Adds `value` to every sample.
    fn offset(&mut self, value: Self::Sample) {
        self.samples_mut().iter_mut().for_each(|x| *x = *x + value);
    }
    /// Replaces every sample with `f(sample)`.
    fn apply(&mut self, mut f: impl FnMut(Self::Sample) -> Self::Sample) {
        self.samples_mut().iter_mut().for_each(|x| *x = f(*x));
    }
    /// Hard-clips every sample into [lo, hi].
    fn clip(&mut self, lo: Self::Sample, hi: Self::Sample) {
        self.apply(|x| x._clamp(lo, hi));
    }
    /// Scales so the peak equals `target`; returns the gain applied (1 for silence, which is left as is).
    fn normalize_peak(&mut self, target: Self::Sample) -> Self::Sample {
        let peak = self.peak();
        let gain = if peak > Self::Sample::_ZERO { target / peak } else { Self::Sample::_ONE };
        self.scale(gain);
        gain
    }
    /// Scales so the rms level equals `target`; returns the gain applied (1 for silence).
    fn normalize_rms(&mut self, target: Self::Sample) -> Self::Sample {
        let gain = match self.rms() {
            Some(rms) if rms > Self::Sample::_ZERO => target / rms,
            _ => Self::Sample::_ONE,
        };
        self.scale(gain);
        gain
    }
    /// Subtracts the mean, centering the signal on zero.
    fn remove_dc(&mut self) {
        if let Some(mean) = self.mean() {
            self.offset(-mean);
        }
    }
    /// Linear fade from silence over the first `len` samples (clamped to the signal length).
    fn fade_in(&mut self, len: usize) {
        let x = self.samples_mut();
        let len = len.min(x.len());
        for (i, s) in x[..len].iter_mut().enumerate() {
            *s = *s * Self::Sample::_lit(i as f64 / len as f64);
        }
    }
    /// Linear fade to silence over the last `len` samples (clamped to the signal length).
    fn fade_out(&mut self, len: usize) {
        let x = self.samples_mut();
        let len = len.min(x.len());
        let start = x.len() - len;
        for (i, s) in x[start..].iter_mut().enumerate() {
            *s = *s * Self::Sample::_lit((len - 1 - i) as f64 / len as f64);
        }
    }

    // Running operations =========================================================================

    /// Running sum: y[n] = x[0] + ... + x[n] (discrete integration).
    fn cumsum(&mut self) {
        let mut acc = Self::Sample::_ZERO;
        for x in self.samples_mut() {
            acc = acc + *x;
            *x = acc;
        }
    }
    /// First difference: y[n] = x[n] - x[n-1], with x[-1] = 0 so the length is kept and
    /// `cumsum` undoes it exactly.
    fn diff(&mut self) {
        let mut prev = Self::Sample::_ZERO;
        for x in self.samples_mut() {
            let current = *x;
            *x = current - prev;
            prev = current;
        }
    }
    /// Replaces the signal with its projection onto `basis`: (<self, b> / <b, b>) * b, i.e. the
    /// part of the signal that looks like `basis`.
    fn project_onto(&mut self, basis: &[Self::Sample]) -> Result<(), SignalError> {
        let dot = self.inner(basis)?;
        let norm = basis.energy();
        if norm == Self::Sample::_ZERO {
            return Err(SignalError::ZeroNorm);
        }
        let k = dot / norm;
        for (x, &b) in self.samples_mut().iter_mut().zip(basis) {
            *x = k * b;
        }
        Ok(())
    }

    // Pointwise math between signals =============================================================

    /// self[i] = f(self[i], other'[i]) for every i, where `other'` is `other` extended per `mode`.
    fn zip_apply(
        &mut self,
        other: &[Self::Sample],
        mode: Broadcast<Self::Sample>,
        mut f: impl FnMut(Self::Sample, Self::Sample) -> Self::Sample,
    ) -> Result<(), SignalError> {
        let x = self.samples_mut();
        let (n, m) = (x.len(), other.len());
        if mode == Broadcast::Strict && n != m {
            return Err(SignalError::LengthMismatch(n, m));
        }
        if m == 0 && matches!(mode, Broadcast::Latch | Broadcast::Tile) && n > 0 {
            return Err(SignalError::Empty);
        }
        let overlap = n.min(m);
        for (a, &b) in x[..overlap].iter_mut().zip(other) {
            *a = f(*a, b);
        }
        let rest = &mut x[overlap..];
        match mode {
            Broadcast::Strict | Broadcast::Truncate => {}
            Broadcast::ZeroPad => rest.iter_mut().for_each(|a| *a = f(*a, Self::Sample::_ZERO)),
            Broadcast::Fill(v) => rest.iter_mut().for_each(|a| *a = f(*a, v)),
            Broadcast::Latch => {
                let last = other[m - 1];
                rest.iter_mut().for_each(|a| *a = f(*a, last));
            }
            Broadcast::Tile => {
                for (a, &b) in rest.iter_mut().zip(other.iter().cycle()) {
                    *a = f(*a, b);
                }
            }
        }
        Ok(())
    }
    /// Pointwise sum; lengths must match (use `zip_apply` to broadcast).
    fn add_signal(&mut self, other: &[Self::Sample]) -> Result<(), SignalError> {
        self.zip_apply(other, Broadcast::Strict, |a, b| a + b)
    }
    fn sub_signal(&mut self, other: &[Self::Sample]) -> Result<(), SignalError> {
        self.zip_apply(other, Broadcast::Strict, |a, b| a - b)
    }
    /// Pointwise product (ring modulation, applying an envelope or window).
    fn mul_signal(&mut self, other: &[Self::Sample]) -> Result<(), SignalError> {
        self.zip_apply(other, Broadcast::Strict, |a, b| a * b)
    }
    fn div_signal(&mut self, other: &[Self::Sample]) -> Result<(), SignalError> {
        self.zip_apply(other, Broadcast::Strict, |a, b| a / b)
    }
    /// self += gain * other (mixing a signal in at a level).
    fn mix_in(&mut self, other: &[Self::Sample], gain: Self::Sample) -> Result<(), SignalError> {
        self.zip_apply(other, Broadcast::Strict, |a, b| a + gain * b)
    }
}

impl<T: Float> SignalMut for [T] {
    fn samples_mut(&mut self) -> &mut [T] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_and_shape() {
        let mut x = [1.0f64, -2.0, 4.0];
        x.scale(0.5);
        assert_eq!(x, [0.5, -1.0, 2.0]);
        x.offset(1.0);
        assert_eq!(x, [1.5, 0.0, 3.0]);
        x.clip(0.5, 2.0);
        assert_eq!(x, [1.5, 0.5, 2.0]);
        x.gain_db(20.0 * 2f64.log10()); // +6.02 dB = x2
        assert!((x[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn normalization_and_dc() {
        let mut x = [0.5f64, -0.25, 0.1];
        assert_eq!(x.normalize_peak(1.0), 2.0);
        assert_eq!(x.peak(), 1.0);
        let mut y = [3.0f64, 5.0, 7.0];
        y.remove_dc();
        assert_eq!(y, [-2.0, 0.0, 2.0]);
        y.normalize_rms(1.0);
        assert!((y.rms().unwrap() - 1.0).abs() < 1e-12);
        let mut silence = [0.0; 4];
        assert_eq!(silence.normalize_peak(1.0), 1.0);
    }

    #[test]
    fn fades() {
        let mut x = [1.0; 6];
        x.fade_in(4);
        assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let mut y = [1.0; 6];
        y.fade_out(4);
        assert_eq!(y, [1.0, 1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn diff_and_cumsum_are_inverses() {
        let original = [3.0, 1.0, 4.0, 1.0, 5.0];
        let mut x = original;
        x.diff();
        assert_eq!(x, [3.0, -2.0, 3.0, -3.0, 4.0]);
        x.cumsum();
        assert_eq!(x, original);
    }

    #[test]
    fn projection() {
        let mut x = [3.0, 4.0];
        x.project_onto(&[1.0, 0.0]).unwrap();
        assert_eq!(x, [3.0, 0.0]);
        assert_eq!([1.0, 2.0].project_onto(&[0.0, 0.0]), Err(SignalError::ZeroNorm));
    }

    #[test]
    fn broadcasting_policies() {
        let other = [10.0, 20.0];
        let run = |mode| {
            let mut x = [1.0, 1.0, 1.0, 1.0, 1.0];
            x.zip_apply(&other, mode, |a, b| a + b).map(|_| x)
        };
        assert_eq!(run(Broadcast::Strict), Err(SignalError::LengthMismatch(5, 2)));
        assert_eq!(run(Broadcast::Truncate), Ok([11.0, 21.0, 1.0, 1.0, 1.0]));
        assert_eq!(run(Broadcast::ZeroPad), Ok([11.0, 21.0, 1.0, 1.0, 1.0]));
        assert_eq!(run(Broadcast::Fill(2.0)), Ok([11.0, 21.0, 3.0, 3.0, 3.0]));
        assert_eq!(run(Broadcast::Latch), Ok([11.0, 21.0, 21.0, 21.0, 21.0]));
        assert_eq!(run(Broadcast::Tile), Ok([11.0, 21.0, 11.0, 21.0, 11.0]));
        // ZeroPad vs Truncate differ for multiplication
        let mut m = [2.0, 2.0, 2.0];
        m.zip_apply(&[3.0], Broadcast::ZeroPad, |a, b| a * b).unwrap();
        assert_eq!(m, [6.0, 0.0, 0.0]);
        let mut e = [1.0];
        assert_eq!(e.zip_apply(&[], Broadcast::Tile, |a, b| a + b), Err(SignalError::Empty));
    }

    #[test]
    fn pointwise_shorthands() {
        let mut x = [1.0, 2.0];
        x.add_signal(&[1.0, 1.0]).unwrap();
        x.mul_signal(&[2.0, 3.0]).unwrap();
        x.sub_signal(&[1.0, 1.0]).unwrap();
        x.div_signal(&[3.0, 4.0]).unwrap();
        assert_eq!(x, [1.0, 2.0]);
        x.mix_in(&[10.0, 10.0], 0.5).unwrap();
        assert_eq!(x, [6.0, 7.0]);
        assert!(x.add_signal(&[1.0]).is_err());
    }
}
