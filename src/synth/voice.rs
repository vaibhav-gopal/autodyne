use super::{midi_to_hz, Voice};
use crate::envelope::Adsr;
use crate::filter::{Biquad, BiquadDesign, BiquadKind};
use crate::osc::{Oscillator, Waveform};
use crate::units::*;

/// Samples between filter-cutoff updates while the filter envelope moves (coefficients cost a sin/cos).
const MOD_INTERVAL: usize = 16;

/// A subtractive synth voice: band-limited oscillator -> resonant low-pass whose cutoff is swept by its
/// own envelope -> amplitude ADSR, scaled by velocity.
#[derive(Debug, Clone)]
pub struct SynthVoice<T: Float> {
    sample_rate: T,
    osc: Oscillator<T>,
    filter: Biquad<T>,
    amp_env: Adsr<T>,
    filter_env: Adsr<T>,
    note: u8,
    velocity: T,
    bend: T,
    cutoff: T,
    resonance: T,
    env_amount: T,
}

impl<T: Float> SynthVoice<T> {
    /// A saw voice with a plucky filter envelope: cutoff 800 Hz swept up 3 octaves.
    pub fn new(sample_rate: T) -> Self {
        let lit = T::_lit;
        let cutoff = lit(800.0);
        Self {
            sample_rate,
            osc: Oscillator::new(Waveform::Saw, lit(440.0), sample_rate),
            // (render keeps the cutoff below Nyquist; the initial design must be valid at low rates too)
            filter: Biquad::from_design(BiquadDesign {
                kind: BiquadKind::Lowpass,
                frequency: cutoff._min(lit(0.45) * sample_rate),
                q: lit(1.2),
                gain_db: T::_ZERO,
                sample_rate,
            }),
            amp_env: Adsr::new(lit(0.005), lit(0.3), lit(0.6), lit(0.3), sample_rate),
            filter_env: Adsr::new(lit(0.002), lit(0.25), lit(0.2), lit(0.3), sample_rate),
            note: 69,
            velocity: T::_ZERO,
            bend: T::_ZERO,
            cutoff,
            resonance: lit(1.2),
            env_amount: lit(3.0),
        }
    }
    pub fn set_waveform(&mut self, waveform: Waveform<T>) {
        self.osc.set_waveform(waveform);
    }
    pub fn waveform(&self) -> Waveform<T> {
        self.osc.waveform()
    }
    /// Base filter cutoff in Hz (before the envelope).
    pub fn set_cutoff(&mut self, hz: T) {
        self.cutoff = hz;
    }
    pub fn cutoff(&self) -> T {
        self.cutoff
    }
    /// Filter Q (resonance).
    pub fn set_resonance(&mut self, q: T) {
        self.resonance = q;
    }
    pub fn resonance(&self) -> T {
        self.resonance
    }
    /// How far the filter envelope opens the cutoff at its peak, in octaves.
    pub fn set_env_amount(&mut self, octaves: T) {
        self.env_amount = octaves;
    }
    pub fn env_amount(&self) -> T {
        self.env_amount
    }
    pub fn amp_env(&self) -> &Adsr<T> {
        &self.amp_env
    }
    pub fn amp_env_mut(&mut self) -> &mut Adsr<T> {
        &mut self.amp_env
    }
    pub fn filter_env(&self) -> &Adsr<T> {
        &self.filter_env
    }
    pub fn filter_env_mut(&mut self) -> &mut Adsr<T> {
        &mut self.filter_env
    }

    fn update_pitch(&mut self) {
        let note = self.note as f64 + self.bend.to_f64().unwrap_or(0.0);
        self.osc.set_frequency(T::_lit(midi_to_hz(note)), self.sample_rate);
    }

    /// Cutoff for the filter envelope's current level: at least 20 Hz, but always below Nyquist
    /// (which wins at very low sample rates).
    fn current_cutoff(&self) -> T {
        let hz = self.cutoff * T::_lit(2.0)._pow(self.env_amount * self.filter_env.level());
        hz._max(T::_lit(20.0))._min(T::_lit(0.45) * self.sample_rate)
    }
}

impl<T: Float> Voice for SynthVoice<T> {
    type Sample = T;

    fn note_on(&mut self, note: u8, velocity: T) {
        self.note = note;
        self.velocity = velocity._clamp(T::_ZERO, T::_ONE);
        self.update_pitch();
        self.amp_env.note_on();
        self.filter_env.note_on();
    }
    fn note_off(&mut self) {
        self.amp_env.note_off();
        self.filter_env.note_off();
    }
    fn is_active(&self) -> bool {
        self.amp_env.is_active()
    }
    fn render(&mut self, out: &mut [T]) {
        if !self.is_active() {
            out.iter_mut().for_each(|s| *s = T::_ZERO);
            return;
        }
        self.osc.fill(out);
        for chunk in out.chunks_mut(MOD_INTERVAL) {
            let design = BiquadDesign { frequency: self.current_cutoff(), q: self.resonance, ..*self.filter.design().expect("built from a design") };
            self.filter.set_design(design);
            self.filter.process(chunk);
            for _ in 0..chunk.len() {
                self.filter_env.next_value();
            }
        }
        self.amp_env.process(out);
        out.iter_mut().for_each(|s| *s = *s * self.velocity);
    }
    fn set_pitch_bend(&mut self, semitones: T) {
        self.bend = semitones;
        self.update_pitch();
    }
    fn reset(&mut self) {
        self.amp_env.reset();
        self.filter_env.reset();
        self.filter.reset();
        self.osc.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{ComplexSignal, Signal};
    use crate::spectral::{bin_frequency, Fft};
    use crate::units::Complex;

    const FS: f64 = 48_000.0;

    fn render(voice: &mut SynthVoice<f64>, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n];
        voice.render(&mut out);
        out
    }

    /// Frequency of the strongest FFT bin once the voice has settled.
    fn peak_hz(voice: &mut SynthVoice<f64>) -> f64 {
        let out = render(voice, 16_384);
        let fft = Fft::new(8_192);
        let mut spectrum = vec![Complex::zero(); 8_192];
        fft.forward_real(&out[8_192..], &mut spectrum);
        bin_frequency(spectrum[..4_096].argmax_magnitude().unwrap(), 8_192, FS)
    }

    fn sine_voice() -> SynthVoice<f64> {
        let mut v = SynthVoice::new(FS);
        v.set_waveform(Waveform::Sine);
        v.set_cutoff(15_000.0);
        v
    }

    #[test]
    fn plays_the_right_pitch_and_bends() {
        let bin = FS / 8_192.0;
        let mut v = sine_voice();
        v.note_on(69, 1.0); // A4
        assert!((peak_hz(&mut v) - 440.0).abs() < bin);
        v.set_pitch_bend(12.0); // up an octave
        assert!((peak_hz(&mut v) - 880.0).abs() < bin);
    }

    #[test]
    fn velocity_scales_and_release_ends_the_voice() {
        let (mut loud, mut soft) = (SynthVoice::new(FS), SynthVoice::new(FS));
        loud.note_on(60, 1.0);
        soft.note_on(60, 0.5);
        let (a, b) = (render(&mut loud, 4_800), render(&mut soft, 4_800));
        assert!((b.rms().unwrap() / a.rms().unwrap() - 0.5).abs() < 1e-9);

        loud.note_off();
        render(&mut loud, (0.31 * FS) as usize); // release is 0.3 s
        assert!(!loud.is_active());
        assert!(render(&mut loud, 64).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn works_at_any_sample_rate() {
        // the default 800 Hz cutoff is above Nyquist at 1 kHz; 20 Hz is above it at 40 Hz
        for fs in [40.0f64, 1_000.0, 8_000.0, 768_000.0] {
            let mut v = SynthVoice::new(fs);
            v.note_on(96, 1.0);
            let mut out = vec![0.0; 512];
            v.render(&mut out);
            assert!(out.iter().all(|s| s.is_finite()), "{fs} Hz");
        }
    }
}
