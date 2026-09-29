//! Saturation and distortion.
//!
//! A [`Waveshaper`] bends the signal through a fixed curve. That adds harmonics, and any that land
//! above Nyquist fold back as inharmonic aliasing. Wrap it in `resample::Oversampled` to run it at a
//! multiple of the sample rate and filter those harmonics out before coming back down.

use crate::gain::{db_to_gain, SmoothedValue};
use crate::units::*;

/// The transfer curve of a [`Waveshaper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shape {
    /// smooth, tube-like saturation approaching +-1
    Tanh,
    /// cubic soft clip: linear-ish near 0, flat beyond +-1, smooth at the corners
    SoftClip,
    /// hard limit at +-1 (bright, harsh)
    HardClip,
    /// reflects back from +-1 instead of limiting (a wavefolder)
    Fold,
}

impl Shape {
    #[inline]
    pub fn apply<T: Float>(self, x: T) -> T {
        let one = T::_ONE;
        match self {
            Shape::Tanh => {
                // tanh(x) = 1 - 2 / (e^(2x) + 1): saturates cleanly to +-1 even when e^(2x) overflows
                one - T::_lit(2.0) / ((x + x)._exp() + one)
            }
            Shape::SoftClip => {
                if x._abs() >= one {
                    x._signum()
                } else {
                    T::_lit(1.5) * x - T::_lit(0.5) * x * x * x
                }
            }
            Shape::HardClip => x._clamp(-one, one),
            Shape::Fold => {
                // a triangle wave of the input: rises to 1, falls back through -1, ...
                let t = (x + one) / T::_lit(4.0);
                one - T::_lit(4.0) * (t - t._floor() - T::_lit(0.5))._abs()
            }
        }
    }
}

/// Drive -> shape -> output level, blended with the dry signal. Parameters are smoothed.
#[derive(Debug, Clone, Copy)]
pub struct Waveshaper<T: Float> {
    shape: Shape,
    drive: SmoothedValue<T>,
    output: SmoothedValue<T>,
    mix: SmoothedValue<T>,
}

impl<T: Float> Waveshaper<T> {
    /// No drive, unity output, fully wet.
    pub fn new(shape: Shape, sample_rate: T) -> Self {
        let smoothed = |v: T| SmoothedValue::new(v).with_ramp_seconds(T::_lit(0.02), sample_rate);
        Self { shape, drive: smoothed(T::_ONE), output: smoothed(T::_ONE), mix: smoothed(T::_ONE) }
    }
    pub fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// Gain into the curve, in dB: more drive, more saturation.
    pub fn set_drive_db(&mut self, db: T) {
        self.drive.set_target(db_to_gain(db));
    }
    pub fn drive_db(&self) -> T {
        crate::gain::gain_to_db(self.drive.target())
    }
    /// Level after the curve, in dB (to compensate for drive).
    pub fn set_output_db(&mut self, db: T) {
        self.output.set_target(db_to_gain(db));
    }
    pub fn output_db(&self) -> T {
        crate::gain::gain_to_db(self.output.target())
    }
    /// 0 = dry, 1 = fully shaped.
    pub fn set_mix(&mut self, mix: T) {
        self.mix.set_target(mix._clamp(T::_ZERO, T::_ONE));
    }
    pub fn mix(&self) -> T {
        self.mix.target()
    }
    /// Finishes any parameter ramps (a waveshaper has no other state).
    pub fn reset(&mut self) {
        for v in [&mut self.drive, &mut self.output, &mut self.mix] {
            let t = v.target();
            v.set_immediate(t);
        }
    }
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let (drive, output, mix) = (self.drive.next_value(), self.output.next_value(), self.mix.next_value());
        let wet = output * self.shape.apply(drive * x);
        x + mix * (wet - x)
    }
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves() {
        for shape in [Shape::Tanh, Shape::SoftClip, Shape::HardClip, Shape::Fold] {
            assert_eq!(shape.apply(0.0f64), 0.0, "{shape:?} passes silence");
            assert!((shape.apply(0.5f64) + shape.apply(-0.5)).abs() < 1e-12, "{shape:?} is odd-symmetric");
        }
        assert!((Shape::Tanh.apply(0.5f64) - 0.5f64.tanh()).abs() < 1e-12);
        assert_eq!(Shape::Tanh.apply(1_000.0f64), 1.0);
        assert_eq!(Shape::Tanh.apply(-1_000.0f64), -1.0);
        assert_eq!(Shape::SoftClip.apply(1.0f64), 1.0);
        assert_eq!(Shape::SoftClip.apply(3.0f64), 1.0);
        assert_eq!(Shape::HardClip.apply(-3.0f64), -1.0);
        assert!((Shape::Fold.apply(1.0f64) - 1.0).abs() < 1e-12);
        assert!(Shape::Fold.apply(2.0f64).abs() < 1e-12, "folds back down");
        assert!((Shape::Fold.apply(3.0f64) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn drive_output_and_mix() {
        let mut ws = Waveshaper::new(Shape::HardClip, 48_000.0);
        ws.set_drive_db(20.0 * 4f64.log10()); // x4
        ws.set_output_db(20.0 * 0.5f64.log10()); // x0.5
        ws.reset(); // jump straight to the targets
        assert!((ws.process_sample(0.1) - 0.2).abs() < 1e-12);
        assert!((ws.process_sample(1.0) - 0.5).abs() < 1e-12, "clipped at 1, then halved");
        ws.set_mix(0.0);
        ws.reset();
        assert_eq!(ws.process_sample(0.3), 0.3, "fully dry");
    }
}
