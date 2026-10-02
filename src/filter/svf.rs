//! State-variable filter in topology-preserving (zero-delay feedback) form.

use crate::units::*;

/// Which response a [`Svf`] outputs. All are available at once through [`Svf::tick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SvfMode {
    Lowpass,
    /// unity gain at the center frequency (constant 0 dB peak)
    Bandpass,
    Highpass,
    Notch,
    /// low-pass minus high-pass: a resonant boost at the cutoff, flat elsewhere
    Peak,
    Allpass,
}

impl SvfMode {
    pub const ALL: [SvfMode; 6] = [SvfMode::Lowpass, SvfMode::Bandpass, SvfMode::Highpass, SvfMode::Notch, SvfMode::Peak, SvfMode::Allpass];
    pub const NAMES: [&'static str; 6] = ["Low-pass", "Band-pass", "High-pass", "Notch", "Peak", "All-pass"];
}

/// The three core outputs of one [`Svf`] step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvfOutputs<T> {
    pub low: T,
    /// band-pass with peak gain `Q` (multiply by `1 / Q` for unity peak)
    pub band: T,
    pub high: T,
}

/// The coefficients of a [`Svf`] for a cutoff and Q, over [`Real`] (so they also trace and
/// differentiate in `flux`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvfCoeffs<T> {
    /// 1 / Q
    pub k: T,
    pub a1: T,
    pub a2: T,
    pub a3: T,
}

impl<T: Real> SvfCoeffs<T> {
    /// The cutoff is kept within 1 Hz .. 0.49 x sample rate, Q at least 0.01.
    pub fn new(cutoff: T, q: T, sample_rate: T) -> Self {
        let one = T::lit(1.0);
        let cutoff = cutoff.clip(one, T::lit(0.49) * sample_rate);
        let q = q.maximum(T::lit(0.01));
        let g = (T::lit(std::f64::consts::PI) * cutoff / sample_rate).tan();
        let k = one / q;
        let a1 = one / (one + g * (g + k));
        let a2 = g * a1;
        Self { k, a1, a2, a3: g * a2 }
    }
    /// One step from the two integrator states: `(next states, outputs)`.
    #[inline(always)]
    pub fn tick(&self, state: [T; 2], x: T) -> ([T; 2], SvfOutputs<T>) {
        let [ic1, ic2] = state;
        let v3 = x - ic2;
        let v1 = self.a1 * ic1 + self.a2 * v3;
        let v2 = ic2 + self.a2 * ic1 + self.a3 * v3;
        let two = T::lit(2.0);
        ([two * v1 - ic1, two * v2 - ic2], SvfOutputs { low: v2, band: v1, high: x - self.k * v1 - v2 })
    }
}

impl SvfMode {
    /// The response of this mode from one step's outputs (`k` = 1 / Q).
    #[inline(always)]
    pub fn output<T: Real>(self, o: SvfOutputs<T>, k: T) -> T {
        match self {
            SvfMode::Lowpass => o.low,
            SvfMode::Bandpass => k * o.band,
            SvfMode::Highpass => o.high,
            SvfMode::Notch => o.low + o.high,
            SvfMode::Peak => o.low - o.high,
            SvfMode::Allpass => o.low + o.high - k * o.band,
        }
    }
}

/// Second-order state-variable filter (Andrew Simper's trapezoidal / TPT form).
///
/// Built for modulation: the cutoff costs one `tan` to change, and the filter stays stable and
/// artifact-free even when cutoff and Q change every sample, unlike a biquad whose recomputed
/// coefficients interact badly with its stored state. Its response is the bilinear transform of
/// the analog state-variable filter, prewarped at the cutoff ([`magnitude_at`](Svf::magnitude_at)).
#[derive(Debug, Clone, Copy)]
pub struct Svf<T: Float> {
    mode: SvfMode,
    cutoff: T,
    q: T,
    sample_rate: T,
    coeffs: SvfCoeffs<T>,
    ic1: T,
    ic2: T,
}

impl<T: Float> Svf<T> {
    pub fn new(mode: SvfMode, cutoff: T, q: T, sample_rate: T) -> Self {
        let mut f = Self {
            mode,
            cutoff,
            q,
            sample_rate,
            coeffs: SvfCoeffs { k: T::_ONE, a1: T::_ZERO, a2: T::_ZERO, a3: T::_ZERO },
            ic1: T::_ZERO,
            ic2: T::_ZERO,
        };
        f.set_cutoff_and_q(cutoff, q);
        f
    }
    pub fn lowpass(cutoff: T, q: T, sample_rate: T) -> Self {
        Self::new(SvfMode::Lowpass, cutoff, q, sample_rate)
    }
    pub fn highpass(cutoff: T, q: T, sample_rate: T) -> Self {
        Self::new(SvfMode::Highpass, cutoff, q, sample_rate)
    }
    pub fn bandpass(center: T, q: T, sample_rate: T) -> Self {
        Self::new(SvfMode::Bandpass, center, q, sample_rate)
    }
    pub fn set_mode(&mut self, mode: SvfMode) {
        self.mode = mode;
    }
    pub fn mode(&self) -> SvfMode {
        self.mode
    }
    /// Cutoff in Hz, kept within 1 Hz .. 0.49 x sample rate.
    pub fn set_cutoff(&mut self, hz: T) {
        self.set_cutoff_and_q(hz, self.q);
    }
    /// Resonance (Q >= 0.01; 0.707 = Butterworth for the low- and high-pass).
    pub fn set_q(&mut self, q: T) {
        self.set_cutoff_and_q(self.cutoff, q);
    }
    /// Both at once (one coefficient update): the cheapest call for per-sample modulation.
    #[inline]
    pub fn set_cutoff_and_q(&mut self, hz: T, q: T) {
        self.cutoff = hz._max(T::_ONE)._min(T::_lit(0.49) * self.sample_rate);
        self.q = q._max(T::_lit(0.01));
        self.coeffs = SvfCoeffs::new(self.cutoff, self.q, self.sample_rate);
    }
    pub fn cutoff(&self) -> T {
        self.cutoff
    }
    pub fn q(&self) -> T {
        self.q
    }
    pub fn sample_rate(&self) -> T {
        self.sample_rate
    }
    pub fn reset(&mut self) {
        self.ic1 = T::_ZERO;
        self.ic2 = T::_ZERO;
    }

    /// One sample in, all three core outputs out.
    #[inline]
    pub fn tick(&mut self, x: T) -> SvfOutputs<T> {
        let ([ic1, ic2], o) = self.coeffs.tick([self.ic1, self.ic2], x);
        (self.ic1, self.ic2) = (ic1, ic2);
        o
    }
    /// The coefficients (for driving the same filter from generic or traced code).
    pub fn coeffs(&self) -> SvfCoeffs<T> {
        self.coeffs
    }

    /// One sample through the selected mode.
    #[inline]
    pub fn process_sample(&mut self, x: T) -> T {
        let o = self.tick(x);
        self.mode.output(o, self.coeffs.k)
    }

    /// Filters `block` in place.
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            *s = self.process_sample(*s);
        }
        // once per block: a decaying state reaches zero instead of slow subnormal numbers
        self.ic1 = self.ic1._flush_denormal();
        self.ic2 = self.ic2._flush_denormal();
    }

    /// |H| of the selected mode at `frequency` (exact: the analog prototype at the prewarped
    /// frequency).
    pub fn magnitude_at(&self, frequency: T) -> T {
        let w = (T::_PI * frequency / self.sample_rate)._tan() / (T::_PI * self.cutoff / self.sample_rate)._tan();
        let k = self.coeffs.k;
        let (re, im) = (T::_ONE - w * w, k * w); // denominator s^2 + k s + 1 at s = jw
        let den = re._hypot(im);
        let num = match self.mode {
            SvfMode::Lowpass => T::_ONE,
            SvfMode::Bandpass => k * w,
            SvfMode::Highpass => w * w,
            SvfMode::Notch => (T::_ONE - w * w)._abs(),
            SvfMode::Peak => T::_ONE + w * w,
            SvfMode::Allpass => den,
        };
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    /// Steady-state gain of `filter` for a sine at `freq` (RMS out / RMS in after settling).
    fn measured_gain(mut filter: Svf<f64>, freq: f64) -> f64 {
        let n = 48_000;
        let x: Vec<f64> = (0..n).map(|i| (std::f64::consts::TAU * freq * i as f64 / FS).sin()).collect();
        let y: Vec<f64> = x.iter().map(|&s| filter.process_sample(s)).collect();
        let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        rms(&y[n / 2..]) / rms(&x[n / 2..])
    }

    #[test]
    fn every_mode_matches_its_analytic_response() {
        for mode in SvfMode::ALL {
            for (cutoff, q) in [(1_000.0, std::f64::consts::FRAC_1_SQRT_2), (3_000.0, 4.0)] {
                let filter = Svf::new(mode, cutoff, q, FS);
                for freq in [100.0, 700.0, 1_000.0, 2_500.0, 9_000.0] {
                    let (measured, analytic) = (measured_gain(filter, freq), filter.magnitude_at(freq));
                    assert!(
                        (measured - analytic).abs() < 2e-3 * analytic.max(1e-2),
                        "{mode:?} fc={cutoff} q={q} at {freq} Hz: {measured} vs {analytic}"
                    );
                }
            }
        }
        let lp = Svf::lowpass(1_000.0, std::f64::consts::FRAC_1_SQRT_2, FS);
        assert!((20.0 * lp.magnitude_at(1_000.0).log10() + 3.0103).abs() < 1e-3, "-3 dB at the cutoff");
        assert!((Svf::bandpass(1_000.0, 5.0, FS).magnitude_at(1_000.0) - 1.0).abs() < 1e-12, "unity band-pass peak");
    }

    #[test]
    fn stays_stable_under_audio_rate_modulation() {
        // cutoff swept 30 Hz .. 18 kHz at 200 Hz with high resonance: the trapezoidal SVF stays bounded
        let mut f = Svf::lowpass(1_000.0, 20.0, FS);
        let mut peak = 0.0f64;
        for i in 0..48_000 {
            let t = i as f64 / FS;
            let sweep = 0.5 + 0.5 * (std::f64::consts::TAU * 200.0 * t).sin();
            f.set_cutoff(30.0 * (18_000.0f64 / 30.0).powf(sweep));
            let y = f.process_sample((std::f64::consts::TAU * 110.0 * t).sin());
            peak = peak.max(y.abs());
        }
        assert!(peak.is_finite() && peak < 50.0, "bounded: peak {peak}");
    }

    #[test]
    fn outputs_are_simultaneous_and_consistent() {
        let mut f = Svf::lowpass(2_000.0, 1.0, FS);
        let x = 0.37;
        let o = f.tick(x);
        // high = x - k * band - low, by construction of the state-variable topology
        assert!((o.high - (x - o.band / f.q() - o.low)).abs() < 1e-15);
    }
}
