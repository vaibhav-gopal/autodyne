//! Signal generators: oscillators, noise and impulses.
//!
//! Every generator can be used two ways:
//! - as an infinite `Iterator` (lazy, composes with the std adapters: `sine.take(n).zip(noise)...`)
//! - as a block processor: `fill` overwrites a buffer, `add_to` mixes into one.
//!   Neither allocates, so both are safe to call from a real-time audio callback.
//! - as a `signal::Source`, to compose lazily (`mix`, `scaled`, `through`) or stream (`stream_for`).

use crate::signal::Source;
use crate::units::*;

/// Implements the block-processing methods for a generator that has `fn next_sample(&mut self) -> T`.
macro_rules! impl_generator_blocks {
    ($Gen:ident) => {
        impl<T: Float> $Gen<T> {
            /// Overwrites `out` with the next `out.len()` samples.
            pub fn fill(&mut self, out: &mut [T]) {
                for s in out {
                    *s = self.next_sample();
                }
            }
            /// Adds the next `out.len()` samples onto `out` (mixing).
            pub fn add_to(&mut self, out: &mut [T]) {
                for s in out {
                    *s = *s + self.next_sample();
                }
            }
        }

        impl<T: Float> Source for $Gen<T> {
            type Sample = T;
            fn next_sample(&mut self) -> T {
                $Gen::next_sample(self)
            }
            fn fill(&mut self, out: &mut [T]) {
                $Gen::fill(self, out)
            }
        }

        impl<T: Float> Iterator for $Gen<T> {
            type Item = T;
            fn next(&mut self) -> Option<T> {
                Some(self.next_sample())
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, None)
            }
        }
    };
}

// SINE ============================================================================================

/// Sine oscillator using a phase accumulator.
/// Phase is kept in cycles and wrapped to [0, 1), so precision doesn't degrade over long runs.
#[derive(Debug, Clone, Copy)]
pub struct Sine<T: Float> {
    /// current phase in cycles, [0, 1)
    phase: T,
    /// phase increment per sample: frequency / sample_rate
    increment: T,
    amplitude: T,
}

impl<T: Float> Sine<T> {
    pub fn new(frequency: T, sample_rate: T) -> Self {
        Self { phase: T::_ZERO, increment: frequency / sample_rate, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    /// Starting phase in cycles (0.25 = start at the peak, i.e. a cosine).
    pub fn with_phase(mut self, phase: T) -> Self {
        self.phase = phase._fract();
        self
    }
    /// Changes frequency without resetting phase, so there is no click.
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.increment = frequency / sample_rate;
    }
    pub fn set_amplitude(&mut self, amplitude: T) {
        self.amplitude = amplitude;
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        let out = self.amplitude * (T::_TAU * self.phase)._sin();
        self.phase = self.phase + self.increment;
        if self.phase >= T::_ONE {
            self.phase = self.phase - T::_ONE;
        }
        out
    }
}

impl_generator_blocks!(Sine);

// BAND-LIMITED OSCILLATOR =========================================================================

/// The shape an [`Oscillator`] produces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform<T> {
    Sine,
    /// rises from -1 to 1 over each cycle
    Saw,
    /// +1 for the first `pulse_width` of each cycle, -1 for the rest (0.5 = square)
    Pulse { pulse_width: T },
    Triangle,
}

/// Correction for a unit step at phase 0, spread over the neighbouring samples (PolyBLEP).
#[inline]
fn poly_blep<T: Float>(t: T, dt: T) -> T {
    if t < dt {
        let x = t / dt;
        x + x - x * x - T::_ONE
    } else if t > T::_ONE - dt {
        let x = (t - T::_ONE) / dt;
        x * x + x + x + T::_ONE
    } else {
        T::_ZERO
    }
}

/// Band-limited oscillator: saw and pulse waves have their jumps smoothed (PolyBLEP), so high notes
/// don't alias into inharmonic noise. The triangle is generated exactly: it has no jumps and its
/// harmonics fall at 12 dB per octave, so its aliasing is already low (a 2-point PolyBLAMP corner
/// correction measured only ~15% closer to the ideal band-limited triangle, so it isn't applied).
/// The waveform can be switched while running without resetting phase.
#[derive(Debug, Clone, Copy)]
pub struct Oscillator<T: Float> {
    waveform: Waveform<T>,
    /// phase in cycles, [0, 1)
    phase: T,
    /// cycles per sample
    increment: T,
    amplitude: T,
}

impl<T: Float> Oscillator<T> {
    pub fn new(waveform: Waveform<T>, frequency: T, sample_rate: T) -> Self {
        Self { waveform, phase: T::_ZERO, increment: frequency / sample_rate, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    /// Changes frequency without resetting phase (no click).
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.increment = frequency / sample_rate;
    }
    pub fn set_waveform(&mut self, waveform: Waveform<T>) {
        self.waveform = waveform;
    }
    pub fn waveform(&self) -> Waveform<T> {
        self.waveform
    }
    pub fn set_amplitude(&mut self, amplitude: T) {
        self.amplitude = amplitude;
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        let (t, dt) = (self.phase, self.increment);
        let (one, two, half) = (T::_ONE, T::_lit(2.0), T::_lit(0.5));
        let value = match self.waveform {
            Waveform::Sine => (T::_TAU * t)._sin(),
            Waveform::Saw => two * t - one - poly_blep(t, dt),
            Waveform::Pulse { pulse_width } => {
                let pw = pulse_width._clamp(dt, one - dt); // keep both edges at least a sample apart
                let naive = if t < pw { one } else { -one };
                naive + poly_blep(t, dt) - poly_blep((t - pw + one)._fract(), dt)
            }
            Waveform::Triangle => {
                let four = T::_lit(4.0);
                if t < half { four * t - one } else { T::_lit(3.0) - four * t }
            }
        };
        self.phase = self.phase + dt;
        if self.phase >= one {
            self.phase = self.phase - one;
        }
        self.amplitude * value
    }
}

impl_generator_blocks!(Oscillator);

// PHASOR ==========================================================================================

/// Complex oscillator: yields e^(i*2*pi*f*n/fs), i.e. (cos, sin) together from one phase accumulator.
/// This is the local oscillator for IQ modulation / demodulation (and a numerically controlled oscillator).
#[derive(Debug, Clone, Copy)]
pub struct Phasor<T: Float> {
    phase: T,
    increment: T,
}

impl<T: Float> Phasor<T> {
    pub fn new(frequency: T, sample_rate: T) -> Self {
        Self { phase: T::_ZERO, increment: frequency / sample_rate }
    }
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.increment = frequency / sample_rate;
    }
    pub fn reset(&mut self) {
        self.phase = T::_ZERO;
    }
    #[inline]
    pub fn next_sample(&mut self) -> Complex<T> {
        let out = Complex::cis(T::_TAU * self.phase);
        self.phase = self.phase + self.increment;
        if self.phase >= T::_ONE {
            self.phase = self.phase - T::_ONE;
        } else if self.phase < T::_ZERO {
            // negative frequencies are valid for a complex oscillator
            self.phase = self.phase + T::_ONE;
        }
        out
    }
    pub fn fill(&mut self, out: &mut [Complex<T>]) {
        for s in out {
            *s = self.next_sample();
        }
    }
}

impl<T: Float> Source for Phasor<T> {
    type Sample = Complex<T>;
    fn next_sample(&mut self) -> Complex<T> {
        Phasor::next_sample(self)
    }
    fn fill(&mut self, out: &mut [Complex<T>]) {
        Phasor::fill(self, out)
    }
}

impl<T: Float> Iterator for Phasor<T> {
    type Item = Complex<T>;
    fn next(&mut self) -> Option<Complex<T>> {
        Some(self.next_sample())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

// NOISE ===========================================================================================

/// Uniform white noise in [-amplitude, amplitude).
/// Uses xorshift64* seeded through splitmix64: fast, allocation-free and deterministic per seed,
/// which keeps tests reproducible. Not cryptographic.
#[derive(Debug, Clone, Copy)]
pub struct Noise<T: Float> {
    state: u64,
    amplitude: T,
}

impl<T: Float> Noise<T> {
    pub fn new(seed: u64) -> Self {
        // splitmix64 spreads similar seeds apart and never yields the all-zero state xorshift can't leave.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self { state: if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    pub fn set_amplitude(&mut self, amplitude: T) {
        self.amplitude = amplitude;
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        // top 53 bits -> uniform f64 in [0, 1) -> [-1, 1)
        let unit = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        self.amplitude * T::_lit(unit * 2.0 - 1.0)
    }
}

impl_generator_blocks!(Noise);

// IMPULSE =========================================================================================

/// A single sample of `amplitude` followed by silence (the Kronecker delta).
/// Feeding it through a filter gives that filter's impulse response.
#[derive(Debug, Clone, Copy)]
pub struct Impulse<T: Float> {
    fired: bool,
    amplitude: T,
}

impl<T: Float> Impulse<T> {
    pub fn new() -> Self {
        Self { fired: false, amplitude: T::_ONE }
    }
    pub fn with_amplitude(mut self, amplitude: T) -> Self {
        self.amplitude = amplitude;
        self
    }
    /// Re-arms the impulse so the next sample fires again.
    pub fn reset(&mut self) {
        self.fired = false;
    }
    #[inline]
    pub fn next_sample(&mut self) -> T {
        if self.fired {
            T::_ZERO
        } else {
            self.fired = true;
            self.amplitude
        }
    }
}

impl<T: Float> Default for Impulse<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl_generator_blocks!(Impulse);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_matches_closed_form() {
        let (f, fs) = (1000.0, 48_000.0);
        let mut osc = Sine::new(f, fs);
        for n in 0..4800 {
            let expected = (f64::_TAU * f * n as f64 / fs).sin();
            assert!((osc.next_sample() - expected).abs() < 1e-9, "sample {n}");
        }
    }

    #[test]
    fn sine_stays_accurate_over_long_runs() {
        // 10 s at 48 kHz: the wrapped phase accumulator must not drift audibly.
        let (f, fs) = (440.0, 48_000.0);
        let mut osc = Sine::new(f, fs);
        let n = 480_000;
        let last = osc.by_ref().nth(n).unwrap();
        let expected = (f64::_TAU * (f * n as f64 / fs).fract()).sin();
        assert!((last - expected).abs() < 1e-6);
    }

    #[test]
    fn sine_amplitude_phase_and_f32() {
        let mut osc = Sine::<f32>::new(1.0, 4.0).with_amplitude(0.5).with_phase(0.25);
        let got: Vec<f32> = osc.by_ref().take(4).collect();
        let expected = [0.5, 0.0, -0.5, 0.0];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn phasor_is_cos_plus_i_sin_and_supports_negative_frequency() {
        let (f, fs) = (1000.0, 48_000.0);
        let mut pos = Phasor::new(f, fs);
        let mut neg = Phasor::new(-f, fs);
        for n in 0..1000 {
            let w = f64::_TAU * f * n as f64 / fs;
            let (p, q) = (pos.next_sample(), neg.next_sample());
            assert!((p - Complex::new(w.cos(), w.sin())).norm() < 1e-9, "sample {n}");
            assert!((q - p.conj()).norm() < 1e-9, "sample {n}");
        }
    }

    /// Ideal band-limited waveform at phase `t`: every harmonic below Nyquist (Fourier series).
    fn band_limited(waveform: Waveform<f64>, t: f64, harmonics: usize) -> f64 {
        use std::f64::consts::PI;
        let w = std::f64::consts::TAU * t;
        (1..=harmonics)
            .map(|k| {
                let k_f = k as f64;
                match waveform {
                    Waveform::Saw => -(2.0 / PI) * (k_f * w).sin() / k_f,
                    Waveform::Pulse { .. } if k % 2 == 1 => (4.0 / PI) * (k_f * w).sin() / k_f,
                    Waveform::Triangle if k % 2 == 1 => -(8.0 / (PI * PI)) * (k_f * w).cos() / (k_f * k_f),
                    _ => 0.0,
                }
            })
            .sum()
    }

    /// RMS difference from the ideal band-limited waveform, for the anti-aliased and a naive version.
    fn errors(waveform: Waveform<f64>, freq: f64) -> (f64, f64) {
        let fs = 48_000.0;
        let harmonics = ((fs / 2.0) / freq) as usize;
        let n = 4_800;
        let mut osc = Oscillator::new(waveform, freq, fs);
        let (mut e_blep, mut e_naive) = (0.0, 0.0);
        for i in 0..n {
            let t = (i as f64 * freq / fs).fract();
            let ideal = band_limited(waveform, t, harmonics);
            let naive = match waveform {
                Waveform::Saw => 2.0 * t - 1.0,
                Waveform::Pulse { .. } => if t < 0.5 { 1.0 } else { -1.0 },
                Waveform::Triangle => if t < 0.5 { 4.0 * t - 1.0 } else { 3.0 - 4.0 * t },
                Waveform::Sine => (std::f64::consts::TAU * t).sin(),
            };
            e_blep += (osc.next_sample() - ideal).powi(2);
            e_naive += (naive - ideal).powi(2);
        }
        ((e_blep / n as f64).sqrt(), (e_naive / n as f64).sqrt())
    }

    #[test]
    fn polyblep_is_much_closer_to_band_limited_than_naive() {
        // a high note (few harmonics below Nyquist), where aliasing is worst
        for waveform in [Waveform::Saw, Waveform::Pulse { pulse_width: 0.5 }] {
            let (blep, naive) = errors(waveform, 3_456.7);
            assert!(blep < 0.6 * naive, "{waveform:?}: polyblep error {blep:.4} vs naive {naive:.4}");
        }
    }

    #[test]
    fn low_notes_match_the_naive_shape() {
        // at 50 Hz the corrections only touch the samples next to each edge
        let mut saw = Oscillator::new(Waveform::Saw, 50.0, 48_000.0);
        let samples: Vec<f64> = saw.by_ref().take(960).collect();
        for (n, &s) in samples.iter().enumerate().skip(2).take(470) {
            let t = n as f64 * 50.0 / 48_000.0;
            assert!((s - (2.0 * t - 1.0)).abs() < 1e-9, "sample {n}");
        }
        let mut tri = Oscillator::<f64>::new(Waveform::Triangle, 50.0, 48_000.0);
        let peak = tri.by_ref().take(960).fold(0.0f64, |m, s| m.max(s.abs()));
        assert!((peak - 1.0).abs() < 1e-3);
    }

    #[test]
    fn pulse_width_sets_the_duty_cycle() {
        let mut p = Oscillator::new(Waveform::Pulse { pulse_width: 0.25 }, 100.0, 48_000.0);
        let v: Vec<f64> = p.by_ref().take(480).collect(); // one cycle
        let high = v.iter().filter(|&&s| s > 0.0).count();
        assert!((high as f64 / 480.0 - 0.25).abs() < 0.01, "{high} of 480 samples high");
        // switching waveform keeps running from the same phase
        p.set_waveform(Waveform::Sine);
        assert!(p.next_sample().abs() < 1e-9); // back at phase 0 after exactly one cycle
    }


    #[test]
    fn noise_is_bounded_centered_and_deterministic() {
        let samples: Vec<f64> = Noise::new(42).take(100_000).collect();
        assert!(samples.iter().all(|s| (-1.0..1.0).contains(s)));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");

        let again: Vec<f64> = Noise::new(42).take(1000).collect();
        assert_eq!(&samples[..1000], &again[..]);
        let other: Vec<f64> = Noise::new(43).take(1000).collect();
        assert_ne!(again, other);
    }

    #[test]
    fn impulse_fires_once() {
        let mut imp = Impulse::<f64>::new();
        let got: Vec<f64> = imp.by_ref().take(4).collect();
        assert_eq!(got, [1.0, 0.0, 0.0, 0.0]);
        imp.reset();
        assert_eq!(imp.next_sample(), 1.0);
    }

    #[test]
    fn fill_and_add_to() {
        let mut buf = [0.0f64; 4];
        Impulse::new().with_amplitude(2.0).fill(&mut buf);
        assert_eq!(buf, [2.0, 0.0, 0.0, 0.0]);
        Impulse::new().add_to(&mut buf);
        assert_eq!(buf, [3.0, 0.0, 0.0, 0.0]);
    }
}
