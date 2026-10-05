//! [`SynthVoice`]: the subtractive voice (unison oscillators, a [`VoiceFilter`], envelopes) for
//! [`Poly`](super::Poly).

#[cfg(not(any(feature = "std", test)))]
use crate::alloc_prelude::*;
use super::{midi_to_hz, Voice};
use crate::envelope::Adsr;
use crate::filter::{Ladder, Svf, SvfMode};
use alloc::sync::Arc;

use crate::osc::{Oscillator, Waveform, Wavetable, WavetableOsc};
use crate::units::*;

/// Samples between pitch and filter-coefficient updates (glide, envelopes, expression).
const CONTROL: usize = 8;
/// Most oscillators one voice stacks in unison.
pub const MAX_UNISON: usize = 7;

/// The filter of a [`SynthVoice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceFilter {
    /// 12 dB/octave state-variable low-pass
    Lowpass12,
    /// 12 dB/octave state-variable band-pass
    Bandpass12,
    /// 12 dB/octave state-variable high-pass
    Highpass12,
    /// 24 dB/octave Moog-style ladder (self-oscillates at full resonance)
    Ladder24,
}

impl VoiceFilter {
    /// Every filter, in parameter order.
    pub const ALL: [VoiceFilter; 4] = [VoiceFilter::Lowpass12, VoiceFilter::Bandpass12, VoiceFilter::Highpass12, VoiceFilter::Ladder24];
    /// Display names, in the order of [`ALL`](Self::ALL).
    pub const NAMES: [&'static str; 4] = ["Low-pass 12", "Band-pass 12", "High-pass 12", "Ladder 24"];
}

/// A subtractive synth voice: up to [`MAX_UNISON`] detuned band-limited oscillators (classic
/// waveforms or a morphing [`Wavetable`]) -> a resonant filter whose cutoff is swept by its own
/// envelope, by pressure and by timbre -> amplitude ADSR, scaled by velocity. Glides between notes
/// when a glide time is set.
#[derive(Debug, Clone)]
pub struct SynthVoice<T: Float> {
    sample_rate: T,
    oscs: [Oscillator<T>; MAX_UNISON],
    /// the wavetable source (used instead of `oscs` when `wavetable` is set)
    wt_oscs: [WavetableOsc<T>; MAX_UNISON],
    wavetable: bool,
    wt_position: T,
    /// each oscillator's starting phase (spread, so unison doesn't start phase-aligned)
    phases: [T; MAX_UNISON],
    unison: usize,
    detune_cents: T,
    /// kept while another waveform plays, so switching back to the pulse restores it
    pulse_width: T,
    filter_type: VoiceFilter,
    svf: Svf<T>,
    ladder: Ladder<T>,
    cutoff: T,
    /// 0..1: SVF Q 0.5..25 (exponentially), ladder feedback 0..4
    resonance: T,
    env_amount: T,
    amp_env: Adsr<T>,
    filter_env: Adsr<T>,
    note: u8,
    velocity: T,
    bend: T,
    /// current and target pitch in semitones (MIDI note numbers), and the glide speed
    pitch: T,
    target_pitch: T,
    glide_seconds: T,
    glide_per_sample: T,
    pitch_dirty: bool,
    pressure: T,
    timbre: T,
    pressure_octaves: T,
    timbre_octaves: T,
    /// cutoff the filter coefficients were last computed for (NaN = recompute)
    applied_cutoff: T,
}

impl<T: Float> SynthVoice<T> {
    /// A single saw through the 12 dB low-pass with a plucky filter envelope: cutoff 800 Hz swept
    /// up 3 octaves.
    pub fn new(sample_rate: T) -> Self {
        let lit = T::_lit;
        // golden-ratio spacing keeps unison oscillators' starting phases apart for any count
        let phases: [T; MAX_UNISON] = core::array::from_fn(|i| lit((i as f64 * 0.618_034).fract()));
        let cutoff = lit(800.0);
        let top = lit(0.45) * sample_rate;
        let mut voice = Self {
            sample_rate,
            oscs: core::array::from_fn(|i| Oscillator::new(Waveform::Saw, lit(440.0), sample_rate).with_phase(phases[i])),
            wt_oscs: {
                let table = Wavetable::shared_classic();
                core::array::from_fn(|i| WavetableOsc::new(table.clone(), lit(440.0), sample_rate).with_phase(phases[i]))
            },
            wavetable: false,
            wt_position: T::_ZERO,
            phases,
            unison: 1,
            detune_cents: lit(15.0),
            pulse_width: lit(0.5),
            filter_type: VoiceFilter::Lowpass12,
            svf: Svf::lowpass(cutoff._min(top), lit(1.2), sample_rate),
            ladder: Ladder::new(cutoff._min(top), T::_ZERO, sample_rate),
            cutoff,
            resonance: lit(0.22),
            env_amount: lit(3.0),
            amp_env: Adsr::new(lit(0.005), lit(0.3), lit(0.6), lit(0.3), sample_rate),
            filter_env: Adsr::new(lit(0.002), lit(0.25), lit(0.2), lit(0.3), sample_rate),
            note: 69,
            velocity: T::_ZERO,
            bend: T::_ZERO,
            pitch: lit(69.0),
            target_pitch: lit(69.0),
            glide_seconds: T::_ZERO,
            glide_per_sample: T::_ZERO,
            pitch_dirty: true,
            pressure: T::_ZERO,
            timbre: lit(0.5),
            pressure_octaves: T::_ONE,
            timbre_octaves: T::_ONE,
            applied_cutoff: T::_NAN,
        };
        voice.set_resonance(lit(0.22));
        voice
    }
    /// The classic waveform of every oscillator (a pulse's width is remembered).
    pub fn set_waveform(&mut self, waveform: Waveform<T>) {
        if let Waveform::Pulse { pulse_width } = waveform {
            self.pulse_width = pulse_width;
        }
        self.oscs.iter_mut().for_each(|o| o.set_waveform(waveform));
    }
    /// The classic waveform.
    pub fn waveform(&self) -> Waveform<T> {
        self.oscs[0].waveform()
    }
    /// The source names of the `waveform` parameter: the classic waveforms, then the wavetable.
    pub const SOURCE_NAMES: [&'static str; 5] = ["Sine", "Saw", "Pulse", "Triangle", "Wavetable"];
    /// Plays the wavetable instead of the classic waveform (the table and position are kept).
    pub fn set_wavetable_source(&mut self, on: bool) {
        if on != self.wavetable {
            self.wavetable = on;
            self.pitch_dirty = true;
        }
    }
    /// Whether the wavetable plays instead of the classic waveform.
    pub fn wavetable_source(&self) -> bool {
        self.wavetable
    }
    /// The table the wavetable source plays (default: [`Wavetable::classic`], shared). Allocation-free.
    pub fn set_wavetable(&mut self, table: Arc<Wavetable<T>>) {
        self.wt_oscs.iter_mut().for_each(|o| o.set_table(table.clone()));
    }
    /// Morph position across the table's frames, 0..1.
    pub fn set_wavetable_position(&mut self, position: T) {
        self.wt_position = position._clamp(T::_ZERO, T::_ONE);
        self.wt_oscs.iter_mut().for_each(|o| o.set_position(self.wt_position));
    }
    /// Wavetable morph position, 0..1.
    pub fn wavetable_position(&self) -> T {
        self.wt_position
    }
    /// Duty cycle of the pulse wave, 0.5 = square (remembered while another waveform is selected).
    pub fn set_pulse_width(&mut self, pulse_width: T) {
        self.pulse_width = pulse_width._clamp(T::_lit(0.01), T::_lit(0.99));
        if let Waveform::Pulse { .. } = self.waveform() {
            self.set_waveform(Waveform::Pulse { pulse_width: self.pulse_width });
        }
    }
    /// Pulse duty cycle.
    pub fn pulse_width(&self) -> T {
        self.pulse_width
    }
    /// Oscillators stacked per note, 1..=[`MAX_UNISON`] (clamped).
    pub fn set_unison(&mut self, count: usize) {
        self.unison = count.clamp(1, MAX_UNISON);
        self.pitch_dirty = true;
    }
    /// Oscillators per note.
    pub fn unison(&self) -> usize {
        self.unison
    }
    /// Spread of the unison oscillators in cents, outermost to the center.
    pub fn set_detune(&mut self, cents: T) {
        self.detune_cents = cents._max(T::_ZERO);
        self.pitch_dirty = true;
    }
    /// Unison spread in cents.
    pub fn detune(&self) -> T {
        self.detune_cents
    }
    /// The filter type (switching keeps the oscillators running).
    pub fn set_filter(&mut self, filter: VoiceFilter) {
        if filter != self.filter_type {
            self.filter_type = filter;
            self.svf.set_mode(match filter {
                VoiceFilter::Bandpass12 => SvfMode::Bandpass,
                VoiceFilter::Highpass12 => SvfMode::Highpass,
                _ => SvfMode::Lowpass,
            });
            self.applied_cutoff = T::_NAN;
        }
    }
    /// The filter type.
    pub fn filter(&self) -> VoiceFilter {
        self.filter_type
    }
    /// Base filter cutoff in Hz (before the envelope and expression).
    pub fn set_cutoff(&mut self, hz: T) {
        self.cutoff = hz;
        self.applied_cutoff = T::_NAN;
    }
    /// Base cutoff in Hz.
    pub fn cutoff(&self) -> T {
        self.cutoff
    }
    /// Resonance, 0 (none) .. 1 (the ladder self-oscillates; the SVF reaches Q 25).
    pub fn set_resonance(&mut self, resonance: T) {
        self.resonance = resonance._clamp(T::_ZERO, T::_ONE);
        self.ladder.set_resonance(self.resonance);
        self.applied_cutoff = T::_NAN;
    }
    /// Resonance, 0..1.
    pub fn resonance(&self) -> T {
        self.resonance
    }
    /// The ladder's drive in dB (0 = clean at moderate levels).
    pub fn set_drive_db(&mut self, db: T) {
        self.ladder.set_drive(T::_lit(10.0)._pow(db / T::_lit(20.0)));
    }
    /// The ladder's drive in dB.
    pub fn drive_db(&self) -> T {
        T::_lit(20.0) * self.ladder.drive()._log10()
    }
    /// How far the filter envelope opens the cutoff at its peak, in octaves.
    pub fn set_env_amount(&mut self, octaves: T) {
        self.env_amount = octaves;
    }
    /// Filter envelope depth in octaves.
    pub fn env_amount(&self) -> T {
        self.env_amount
    }
    /// Time to glide from one note to the next (0 = jump). Glides whenever the voice is still
    /// sounding when the next note arrives (portamento), and in legato playing.
    pub fn set_glide(&mut self, seconds: T) {
        self.glide_seconds = seconds._max(T::_ZERO);
    }
    /// Glide time in seconds.
    pub fn glide(&self) -> T {
        self.glide_seconds
    }
    /// Octaves the cutoff rises at full pressure.
    pub fn set_pressure_amount(&mut self, octaves: T) {
        self.pressure_octaves = octaves;
    }
    /// Cutoff octaves at full pressure.
    pub fn pressure_amount(&self) -> T {
        self.pressure_octaves
    }
    /// Octaves the cutoff moves at the timbre extremes (0 and 1; 0.5 is neutral).
    pub fn set_timbre_amount(&mut self, octaves: T) {
        self.timbre_octaves = octaves;
    }
    /// Cutoff octaves at the timbre extremes.
    pub fn timbre_amount(&self) -> T {
        self.timbre_octaves
    }
    /// The amplitude envelope.
    pub fn amp_env(&self) -> &Adsr<T> {
        &self.amp_env
    }
    /// The amplitude envelope, to change its settings.
    pub fn amp_env_mut(&mut self) -> &mut Adsr<T> {
        &mut self.amp_env
    }
    /// The filter envelope.
    pub fn filter_env(&self) -> &Adsr<T> {
        &self.filter_env
    }
    /// The filter envelope, to change its settings.
    pub fn filter_env_mut(&mut self) -> &mut Adsr<T> {
        &mut self.filter_env
    }
    /// The pitch sounding now, in semitones (a MIDI note number, fractional while gliding or bent).
    pub fn current_pitch(&self) -> T {
        self.pitch + self.bend
    }

    /// Heads for `note`, gliding there when `glide` and a glide time are set.
    fn retarget(&mut self, note: u8, glide: bool) {
        self.note = note;
        self.target_pitch = T::_lit(note as f64);
        if glide && self.glide_seconds > T::_ZERO {
            self.glide_per_sample = (self.target_pitch - self.pitch)._abs() / (self.glide_seconds * self.sample_rate);
        } else {
            self.pitch = self.target_pitch;
        }
        self.pitch_dirty = true;
    }

    /// Advances glide by `samples` and updates oscillator frequencies and filter coefficients.
    fn update_controls(&mut self, samples: usize) {
        if self.pitch != self.target_pitch {
            let step = self.glide_per_sample * T::_lit(samples as f64);
            let distance = self.target_pitch - self.pitch;
            self.pitch = if distance._abs() <= step { self.target_pitch } else { self.pitch + step * distance._signum() };
            self.pitch_dirty = true;
        }
        if self.pitch_dirty {
            self.pitch_dirty = false;
            let center = (self.pitch + self.bend).to_f64().unwrap_or(69.0);
            let spread = self.detune_cents.to_f64().unwrap_or(0.0) / 100.0;
            let n = self.unison;
            for i in 0..n {
                let offset = if n > 1 { spread * (2.0 * i as f64 / (n - 1) as f64 - 1.0) } else { 0.0 };
                let hz = T::_lit(midi_to_hz(center + offset));
                if self.wavetable {
                    self.wt_oscs[i].set_frequency(hz, self.sample_rate);
                } else {
                    self.oscs[i].set_frequency(hz, self.sample_rate);
                }
            }
        }
        let octaves = self.env_amount * self.filter_env.level()
            + self.pressure_octaves * self.pressure
            + self.timbre_octaves * (T::_lit(2.0) * self.timbre - T::_ONE);
        let hz = (self.cutoff * T::_lit(2.0)._pow(octaves))._max(T::_lit(20.0))._min(T::_lit(0.45) * self.sample_rate);
        if hz != self.applied_cutoff {
            self.applied_cutoff = hz;
            match self.filter_type {
                VoiceFilter::Ladder24 => self.ladder.set_cutoff(hz),
                _ => self.svf.set_cutoff_and_q(hz, T::_lit(0.5) * T::_lit(50.0)._pow(self.resonance)),
            }
        }
    }
}

impl<T: Float> Voice for SynthVoice<T> {
    type Sample = T;

    fn note_on(&mut self, note: u8, velocity: T) {
        // a voice still sounding glides to its next note (portamento); a silent one starts there
        let sounding = self.is_active();
        self.velocity = velocity._clamp(T::_ZERO, T::_ONE);
        self.retarget(note, sounding);
        self.amp_env.note_on();
        self.filter_env.note_on();
    }
    fn legato(&mut self, note: u8, _velocity: T) {
        self.retarget(note, true);
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
        let n = self.unison;
        // equal loudness for any unison count (the oscillators are uncorrelated)
        let gain = T::_ONE / T::_lit(n as f64)._sqrt();
        for chunk in out.chunks_mut(CONTROL) {
            self.update_controls(chunk.len());
            for s in chunk.iter_mut() {
                let mut x = T::_ZERO;
                if self.wavetable {
                    for osc in &mut self.wt_oscs[..n] {
                        x = x + osc.next_sample();
                    }
                } else {
                    for osc in &mut self.oscs[..n] {
                        x = x + osc.next_sample();
                    }
                }
                *s = x * gain;
            }
            match self.filter_type {
                VoiceFilter::Ladder24 => self.ladder.process(chunk),
                _ => self.svf.process(chunk),
            }
            for _ in 0..chunk.len() {
                self.filter_env.next_value();
            }
        }
        self.amp_env.process(out);
        out.iter_mut().for_each(|s| *s = *s * self.velocity);
    }
    fn set_pitch_bend(&mut self, semitones: T) {
        if semitones != self.bend {
            self.bend = semitones;
            self.pitch_dirty = true;
        }
    }
    fn set_pressure(&mut self, pressure: T) {
        self.pressure = pressure._clamp(T::_ZERO, T::_ONE);
    }
    fn set_timbre(&mut self, timbre: T) {
        self.timbre = timbre._clamp(T::_ZERO, T::_ONE);
    }
    fn reset(&mut self) {
        self.amp_env.reset();
        self.filter_env.reset();
        self.svf.reset();
        self.ladder.reset();
        for ((osc, wt), &phase) in self.oscs.iter_mut().zip(&mut self.wt_oscs).zip(&self.phases) {
            *osc = osc.with_phase(phase);
            wt.set_phase(phase);
        }
        self.pitch = self.target_pitch;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{ComplexSignal, Signal};
    use crate::fft::{bin_frequency, Fft};
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
        let mut fft = Fft::new(8_192);
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

    /// Magnitude spectrum (first half) of 32768 settled samples.
    fn spectrum(voice: &mut SynthVoice<f64>) -> Vec<f64> {
        render(voice, 9_600); // settle past the envelope attack
        let out = render(voice, 32_768);
        let mut fft = Fft::new(32_768);
        let mut z = vec![Complex::zero(); 32_768];
        fft.forward_real(&out, &mut z);
        z[..16_384].iter().map(|c| c.norm_sqr().sqrt()).collect()
    }

    #[test]
    fn unison_detunes_symmetrically_in_cents() {
        let mut v = sine_voice();
        v.set_unison(3);
        v.set_detune(50.0);
        v.note_on(69, 1.0);
        let mags = spectrum(&mut v);
        let bin = FS / 32_768.0;
        let top = mags.iter().cloned().fold(0.0, f64::max);
        for cents in [-50.0f64, 0.0, 50.0] {
            let expected = 440.0 * 2f64.powf(cents / 1_200.0);
            let k = (expected / bin).round() as usize;
            let (best, mag) = (k - 3..=k + 3).map(|i| (i, mags[i])).fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            assert!(((best as f64 * bin) - expected).abs() < 2.0 * bin, "an oscillator at {expected:.1} Hz");
            assert!(mag > 0.3 * top, "{expected:.1} Hz is a real peak");
        }
        // equal loudness whatever the count
        let level = |n: usize| {
            let mut v = sine_voice();
            v.set_unison(n);
            v.note_on(69, 1.0);
            render(&mut v, 4_800);
            render(&mut v, 48_000).rms().unwrap()
        };
        assert!((20.0 * (level(7) / level(1)).log10()).abs() < 2.0);
    }

    #[test]
    fn glide_moves_the_pitch_in_the_set_time() {
        let mut v = sine_voice();
        v.set_glide(0.1);
        v.note_on(60, 1.0);
        assert_eq!(v.current_pitch(), 60.0, "a silent voice starts on its note");
        render(&mut v, 4_800);
        v.note_on(72, 1.0); // still sounding: glides
        render(&mut v, 2_400); // 50 ms
        assert!((v.current_pitch() - 66.0).abs() < 0.2, "halfway after half the time: {}", v.current_pitch());
        render(&mut v, 2_480);
        assert_eq!(v.current_pitch(), 72.0);
    }

    #[test]
    fn legato_changes_pitch_without_retriggering() {
        let mut v = sine_voice();
        v.note_on(60, 1.0);
        render(&mut v, 48_000); // well into the sustain (0.6)
        let sustain = v.amp_env().level();
        v.legato(67, 1.0);
        render(&mut v, 480);
        assert!((v.amp_env().level() - sustain).abs() < 1e-9, "the envelope carries on");
        assert_eq!(v.current_pitch(), 67.0, "no glide time: the pitch jumps");
        v.note_on(72, 1.0);
        render(&mut v, 480);
        assert!(v.amp_env().level() > sustain + 0.1, "a retrigger restarts the attack");
    }

    /// RMS of the first difference: grows with high-frequency content.
    fn brightness(x: &[f64]) -> f64 {
        let d: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        d.rms().unwrap()
    }

    #[test]
    fn filter_types_and_expression_shape_the_tone() {
        let tone = |filter: VoiceFilter, pressure: f64, timbre: f64| {
            let mut v = SynthVoice::new(FS);
            v.set_filter(filter);
            v.set_cutoff(300.0);
            v.set_env_amount(0.0);
            v.set_pressure_amount(4.0);
            v.set_timbre_amount(2.0);
            v.set_pressure(pressure);
            v.set_timbre(timbre);
            v.note_on(45, 1.0); // a 110 Hz saw
            render(&mut v, 4_800);
            render(&mut v, 24_000)
        };
        let low = tone(VoiceFilter::Lowpass12, 0.0, 0.5);
        let high = tone(VoiceFilter::Highpass12, 0.0, 0.5);
        assert!(brightness(&high) > 3.0 * brightness(&low), "the high-pass keeps the edges, the low-pass the body");
        let ladder = tone(VoiceFilter::Ladder24, 0.0, 0.5);
        assert!(brightness(&ladder) < brightness(&low), "24 dB/octave is darker than 12");
        assert!(brightness(&tone(VoiceFilter::Lowpass12, 1.0, 0.5)) > 2.0 * brightness(&low), "pressure opens the filter");
        assert!(brightness(&tone(VoiceFilter::Lowpass12, 0.0, 1.0)) > 1.5 * brightness(&low), "timbre opens it");
        assert!(brightness(&tone(VoiceFilter::Lowpass12, 0.0, 0.0)) < brightness(&low), "and closes it");
        assert!(low.iter().chain(&high).chain(&ladder).all(|s| s.is_finite()));
    }

    #[test]
    fn the_wavetable_source_plays_and_morphs() {
        let mut v = sine_voice();
        v.set_wavetable_source(true);
        v.set_wavetable_position(0.0); // the classic table's first frame: a sine
        v.note_on(69, 1.0);
        assert!((peak_hz(&mut v) - 440.0).abs() < FS / 8_192.0, "the right pitch");
        let pure = spectrum(&mut v);
        let fundamental = pure[(440.0 * 32_768.0 / FS) as usize - 2..(440.0 * 32_768.0 / FS) as usize + 3].iter().cloned().fold(0.0, f64::max);
        let second = pure[(880.0 * 32_768.0 / FS) as usize - 2..(880.0 * 32_768.0 / FS) as usize + 3].iter().cloned().fold(0.0, f64::max);
        // (unwindowed FFT: leakage alone reaches about -60 dB here, a saw is at -6 dB)
        assert!(second < 1e-2 * fundamental, "position 0 is a sine");
        v.set_wavetable_position(2.0 / 3.0); // the saw frame
        let saw = spectrum(&mut v);
        let second = saw[(880.0 * 32_768.0 / FS) as usize - 2..(880.0 * 32_768.0 / FS) as usize + 3].iter().cloned().fold(0.0, f64::max);
        assert!(second > 0.3 * fundamental, "position 2/3 has a saw's harmonics");
        // unison works on the wavetable too
        v.set_unison(3);
        v.set_detune(50.0);
        assert!(render(&mut v, 4_800).iter().all(|s| s.is_finite()));
    }
}