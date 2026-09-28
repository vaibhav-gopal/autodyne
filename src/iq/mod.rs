//! IQ (quadrature) modulation and demodulation, plus the AM / PM / FM views of complex baseband.
//!
//! A complex baseband sample z = I + iQ rides on a real carrier as
//!     s[n] = Re{ z[n] e^(i w n) } = I cos(w n) - Q sin(w n)
//! and the demodulator recovers z by mixing back down and low-pass filtering. AM, PM and FM are all
//! just choices of z: AM varies |z| (read it with `envelope`), PM varies arg z (`phase`), and FM varies
//! the rate of change of arg z (`FmModulator` / `FmDiscriminator`).

use crate::filter::Biquad;
use crate::osc::Phasor;
use crate::units::*;

// CARRIER =========================================================================================

/// Puts complex baseband onto a real carrier.
#[derive(Debug, Clone, Copy)]
pub struct IqModulator<T: Float> {
    lo: Phasor<T>,
}

impl<T: Float> IqModulator<T> {
    pub fn new(carrier: T, sample_rate: T) -> Self {
        Self { lo: Phasor::new(carrier, sample_rate) }
    }
    pub fn reset(&mut self) {
        self.lo.reset();
    }
    #[inline]
    pub fn modulate_sample(&mut self, z: Complex<T>) -> T {
        (z * self.lo.next_sample()).re
    }
    /// Panics if the lengths differ.
    pub fn process(&mut self, baseband: &[Complex<T>], out: &mut [T]) {
        assert_eq!(baseband.len(), out.len(), "baseband and output lengths must match");
        for (o, &z) in out.iter_mut().zip(baseband) {
            *o = self.modulate_sample(z);
        }
    }
}

/// Recovers complex baseband from a real carrier: multiplies by 2 e^(-i w n), which shifts the wanted
/// signal to 0 Hz and leaves an image at twice the carrier, then removes the image with a 4th-order
/// Butterworth low-pass on each of I and Q.
#[derive(Debug, Clone, Copy)]
pub struct IqDemodulator<T: Float> {
    lo: Phasor<T>,
    lp_i: [Biquad<T>; 2],
    lp_q: [Biquad<T>; 2],
}

impl<T: Float> IqDemodulator<T> {
    /// `bandwidth` is the one-sided baseband bandwidth kept (the low-pass cutoff): it must cover the
    /// signal (e.g. for FM, Carson's rule: deviation + highest message frequency) and sit well below
    /// 2 * carrier so the image is rejected.
    pub fn new(carrier: T, bandwidth: T, sample_rate: T) -> Self {
        // A 4th-order Butterworth is two biquads with Q = 1 / (2 cos(pi/8)) and 1 / (2 cos(3pi/8)).
        let q1 = T::_lit(1.0 / (2.0 * (std::f64::consts::PI / 8.0).cos()));
        let q2 = T::_lit(1.0 / (2.0 * (3.0 * std::f64::consts::PI / 8.0).cos()));
        let stages = [Biquad::lowpass(bandwidth, sample_rate, q1), Biquad::lowpass(bandwidth, sample_rate, q2)];
        Self { lo: Phasor::new(carrier, sample_rate), lp_i: stages, lp_q: stages }
    }
    pub fn reset(&mut self) {
        self.lo.reset();
        self.lp_i.iter_mut().chain(self.lp_q.iter_mut()).for_each(Biquad::reset);
    }
    #[inline]
    pub fn demodulate_sample(&mut self, x: T) -> Complex<T> {
        let mixed = self.lo.next_sample().conj() * (x * T::_lit(2.0));
        let i = self.lp_i.iter_mut().fold(mixed.re, |s, f| f.process_sample(s));
        let q = self.lp_q.iter_mut().fold(mixed.im, |s, f| f.process_sample(s));
        Complex::new(i, q)
    }
    /// Panics if the lengths differ.
    pub fn process(&mut self, input: &[T], out: &mut [Complex<T>]) {
        assert_eq!(input.len(), out.len(), "input and output lengths must match");
        for (o, &x) in out.iter_mut().zip(input) {
            *o = self.demodulate_sample(x);
        }
    }
}

// BASEBAND VIEWS ==================================================================================

/// AM demodulation: |z| per sample. Panics if the lengths differ.
pub fn envelope<T: Float>(baseband: &[Complex<T>], out: &mut [T]) {
    assert_eq!(baseband.len(), out.len(), "baseband and output lengths must match");
    for (o, z) in out.iter_mut().zip(baseband) {
        *o = z.norm();
    }
}

/// PM demodulation: arg z per sample, in (-pi, pi]. Panics if the lengths differ.
pub fn phase<T: Float>(baseband: &[Complex<T>], out: &mut [T]) {
    assert_eq!(baseband.len(), out.len(), "baseband and output lengths must match");
    for (o, z) in out.iter_mut().zip(baseband) {
        *o = z.arg();
    }
}

/// FM modulation to baseband: a message value of 1.0 shifts the instantaneous frequency by `deviation` Hz.
#[derive(Debug, Clone, Copy)]
pub struct FmModulator<T: Float> {
    /// accumulated phase in radians, kept in [-pi, pi)
    phase: T,
    /// radians of phase advance per sample per unit of message
    radians_per_unit: T,
}

impl<T: Float> FmModulator<T> {
    pub fn new(deviation: T, sample_rate: T) -> Self {
        Self { phase: T::_ZERO, radians_per_unit: T::_TAU * deviation / sample_rate }
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn modulate_sample(&mut self, message: T) -> Complex<T> {
        let out = Complex::cis(self.phase);
        self.phase = self.phase + self.radians_per_unit * message;
        if self.phase >= T::_PI {
            self.phase = self.phase - T::_TAU;
        } else if self.phase < -T::_PI {
            self.phase = self.phase + T::_TAU;
        }
        out
    }
    /// Panics if the lengths differ.
    pub fn process(&mut self, message: &[T], out: &mut [Complex<T>]) {
        assert_eq!(message.len(), out.len(), "message and output lengths must match");
        for (o, &m) in out.iter_mut().zip(message) {
            *o = self.modulate_sample(m);
        }
    }
}

/// FM demodulation from baseband: the phase step between consecutive samples, arg(z[n] * conj(z[n-1])),
/// is the instantaneous frequency; dividing by the modulator's `deviation` returns the message.
/// Using the product rather than subtracting two args avoids phase-unwrapping problems.
#[derive(Debug, Clone, Copy)]
pub struct FmDiscriminator<T: Float> {
    prev: Complex<T>,
    /// converts a per-sample phase step (radians) into message units
    units_per_radian: T,
}

impl<T: Float> FmDiscriminator<T> {
    pub fn new(deviation: T, sample_rate: T) -> Self {
        Self { prev: Complex::one(), units_per_radian: sample_rate / (T::_TAU * deviation) }
    }
    pub fn reset(&mut self) {
        self.prev = Complex::one();
    }
    #[inline]
    pub fn demodulate_sample(&mut self, z: Complex<T>) -> T {
        let step = (z * self.prev.conj()).arg();
        self.prev = z;
        step * self.units_per_radian
    }
    /// Panics if the lengths differ.
    pub fn process(&mut self, baseband: &[Complex<T>], out: &mut [T]) {
        assert_eq!(baseband.len(), out.len(), "baseband and output lengths must match");
        for (o, &z) in out.iter_mut().zip(baseband) {
            *o = self.demodulate_sample(z);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::Sine;

    const FS: f64 = 48_000.0;
    const CARRIER: f64 = 12_000.0;
    const SETTLE: usize = 4_800; // skip filter transients (0.1 s)

    /// baseband -> carrier -> baseband through the full real-valued channel
    fn through_channel(baseband: &[Complex<f64>], bandwidth: f64) -> Vec<Complex<f64>> {
        let mut rf = vec![0.0; baseband.len()];
        IqModulator::new(CARRIER, FS).process(baseband, &mut rf);
        let mut recovered = vec![Complex::zero(); baseband.len()];
        IqDemodulator::new(CARRIER, bandwidth, FS).process(&rf, &mut recovered);
        recovered
    }

    /// RMS over a window holding a whole number of message periods.
    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn modulator_is_i_cos_minus_q_sin() {
        let mut m = IqModulator::new(1000.0, FS);
        let z = Complex::new(0.6, -0.3);
        for n in 0..100 {
            let w = std::f64::consts::TAU * 1000.0 * n as f64 / FS;
            let expected = 0.6 * w.cos() + 0.3 * w.sin();
            assert!((m.modulate_sample(z) - expected).abs() < 1e-12, "sample {n}");
        }
    }

    #[test]
    fn iq_roundtrip_recovers_constant_baseband() {
        let z = Complex::new(0.6, -0.3);
        let recovered = through_channel(&vec![z; 24_000], 1_000.0);
        for (n, r) in recovered.iter().enumerate().skip(SETTLE) {
            assert!((*r - z).norm() < 1e-3, "sample {n}: {r:?}");
        }
    }

    #[test]
    fn am_envelope_follows_the_message() {
        // 50% modulation depth with a 50 Hz tone: envelope swings 0.5 .. 1.5
        let message: Vec<f64> = Sine::new(50.0, FS).with_amplitude(0.5).take(48_000).collect();
        let baseband: Vec<Complex<f64>> = message.iter().map(|&m| Complex::from(1.0 + m)).collect();
        let recovered = through_channel(&baseband, 500.0);
        let mut env = vec![0.0; recovered.len()];
        envelope(&recovered, &mut env);
        let tail = &env[SETTLE..];
        let (lo, hi) = tail.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &e| (lo.min(e), hi.max(e)));
        assert!((lo - 0.5).abs() < 0.01 && (hi - 1.5).abs() < 0.01, "envelope range {lo}..{hi}");
    }

    #[test]
    fn pm_phase_is_recovered() {
        let recovered = through_channel(&vec![Complex::cis(0.7); 24_000], 1_000.0);
        let mut ph = vec![0.0; recovered.len()];
        phase(&recovered, &mut ph);
        assert!(ph[SETTLE..].iter().all(|p| (p - 0.7).abs() < 1e-3));
    }

    #[test]
    fn fm_baseband_roundtrip_is_exact() {
        // Without the carrier and filters, discriminator(modulator(m)) returns m one sample late.
        let message: Vec<f64> = Sine::new(100.0, FS).with_amplitude(0.8).take(4_800).collect();
        let mut baseband = vec![Complex::zero(); message.len()];
        FmModulator::new(1_000.0, FS).process(&message, &mut baseband);
        let mut out = vec![0.0; message.len()];
        FmDiscriminator::new(1_000.0, FS).process(&baseband, &mut out);
        for n in 1..message.len() {
            assert!((out[n] - message[n - 1]).abs() < 1e-9, "sample {n}");
        }
    }

    #[test]
    fn fm_over_the_air() {
        // 1 kHz deviation, 100 Hz tone: Carson bandwidth ~ 1.1 kHz one-sided, so keep 3 kHz.
        let (deviation, bandwidth) = (1_000.0, 3_000.0);
        let run = |message: &[f64]| {
            let mut baseband = vec![Complex::zero(); message.len()];
            FmModulator::new(deviation, FS).process(message, &mut baseband);
            let recovered = through_channel(&baseband, bandwidth);
            let mut out = vec![0.0; message.len()];
            FmDiscriminator::new(deviation, FS).process(&recovered, &mut out);
            out
        };

        // constant message -> constant frequency offset
        let out = run(&vec![0.5; 24_000]);
        let mean = out[SETTLE..].iter().sum::<f64>() / (out.len() - SETTLE) as f64;
        assert!((mean - 0.5).abs() < 1e-3, "mean {mean}");

        // tone message -> same tone out (RMS over 90 whole periods of 100 Hz)
        let message: Vec<f64> = Sine::new(100.0, FS).with_amplitude(0.8).take(48_000).collect();
        let out = run(&message);
        let (got, want) = (rms(&out[SETTLE..]), rms(&message[SETTLE..]));
        assert!((got - want).abs() < 0.01 * want, "rms {got} vs {want}");
    }
}
