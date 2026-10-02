use crate::units::*;

/// One-pole low-pass: `y[n] = y[n-1] + a (x[n] - y[n-1])`, with `a = 1 - exp(-2π fc / fs)`
/// (6 dB/octave, no overshoot; also the usual parameter smoother).
///
/// The maths is written once over [`Real`]: [`tick`](Self::tick) is a pure function of the state,
/// so the same code runs per sample on the audio thread (`f32` / `f64`) and traces into a
/// differentiable, compilable program with `flux` (cargo feature `flux`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePole<T> {
    a: T,
    state: T,
}

impl<T: Real> OnePole<T> {
    /// A low-pass with its -3 dB point near `cutoff` Hz (exact well below Nyquist).
    pub fn lowpass(cutoff: T, sample_rate: T) -> Self {
        Self { a: Self::coefficient(cutoff, sample_rate), state: T::lit(0.0) }
    }

    /// The smoothing coefficient `a = 1 - exp(-2π fc / fs)` for a cutoff.
    pub fn coefficient(cutoff: T, sample_rate: T) -> T {
        T::lit(1.0) - (-T::lit(std::f64::consts::TAU) * cutoff / sample_rate).exp()
    }

    /// One step from `state` with input `x`: returns `(next state, output)`.
    #[inline(always)]
    pub fn tick(&self, state: T, x: T) -> (T, T) {
        let y = state + self.a * (x - state);
        (y, y)
    }
}

impl<T: Float> OnePole<T> {
    pub fn set_cutoff(&mut self, cutoff: T, sample_rate: T) {
        self.a = Self::coefficient(cutoff, sample_rate);
    }

    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let (state, y) = self.tick(self.state, x);
        self.state = state._flush_denormal();
        y
    }

    pub fn process(&mut self, block: &mut [T]) {
        for x in block.iter_mut() {
            *x = self.process_sample(*x);
        }
    }

    pub fn reset(&mut self) {
        self.state = T::_ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_response_reaches_the_input() {
        let mut lp = OnePole::lowpass(1_000.0f32, 48_000.0);
        let mut block = [1.0f32; 2_000];
        lp.process(&mut block);
        assert!(block[0] > 0.0 && block[0] < 0.2);
        assert!((block[1_999] - 1.0).abs() < 1e-5);
        assert!(block.windows(2).all(|w| w[1] >= w[0]), "no overshoot");
    }

    #[test]
    fn cutoff_is_minus_3_db() {
        let (fc, fs) = (500.0f64, 48_000.0);
        let lp = OnePole::lowpass(fc, fs);
        // |H(e^jw)| for y = (1-a) y[-1] + a x
        let w = std::f64::consts::TAU * fc / fs;
        let p = 1.0 - lp.a;
        let mag = lp.a / (1.0 - 2.0 * p * w.cos() + p * p).sqrt();
        assert!((20.0 * mag.log10() + 3.01).abs() < 0.05, "{}", 20.0 * mag.log10());
    }
}
