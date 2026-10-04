use crate::channels::{AudioBuffer, MultiProcessor};
use crate::delay::DelayLine;
use crate::gain::SmoothedValue;
use crate::units::*;

/// Delay lines in the network.
const LINES: usize = 8;
/// Line lengths in samples at 44.1 kHz and size 1 (the classic Freeverb comb lengths: spread out and
/// sharing no common factors, so their echoes don't line up into audible patterns).
const LENGTHS_44K: [f64; LINES] = [1_116.0, 1_188.0, 1_277.0, 1_356.0, 1_422.0, 1_491.0, 1_557.0, 1_617.0];
/// Input diffusers (all-pass lengths at 44.1 kHz): smear transients into a dense wash before the network.
const DIFFUSERS_44K: [f64; 4] = [225.0, 341.0, 441.0, 556.0];
const MAX_SIZE: f64 = 2.0;
/// Largest delay sweep, in samples at 48 kHz (at modulation 1).
const MAX_MOD_48K: f64 = 12.0;
/// Sweep rates per line (Hz): slow, and not simple ratios of each other, so the lines never move in step.
const MOD_RATES: [f64; LINES] = [0.23, 0.29, 0.37, 0.41, 0.47, 0.53, 0.59, 0.67];
/// Samples between updates of the swept delays: at these rates a delay moves ~0.01 samples per update.
const MOD_INTERVAL: usize = 16;
const MAX_PREDELAY: f64 = 0.25;

/// Signs mixing the input into the lines and the lines into the outputs: three distinct rows of an
/// 8x8 Hadamard matrix, so input, left and right see the lines in mutually orthogonal combinations
/// (which is what makes left and right sound decorrelated).
const INPUT_SIGNS: [f64; LINES] = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
const LEFT_SIGNS: [f64; LINES] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
const RIGHT_SIGNS: [f64; LINES] = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

/// In-place fast Walsh-Hadamard transform, normalized to be orthogonal (it preserves energy, so the
/// feedback loop can only lose energy through the decay gains and damping: always stable).
fn hadamard<T: Float>(x: &mut [T; LINES]) {
    let mut h = 1;
    while h < LINES {
        for i in (0..LINES).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = T::_lit(1.0 / (LINES as f64).sqrt());
    x.iter_mut().for_each(|v| *v = *v * norm);
}

/// Schroeder all-pass diffuser: flat magnitude, smeared phase.
#[derive(Debug, Clone)]
struct Diffuser<T: Float> {
    line: DelayLine<T>,
    delay: usize,
}

impl<T: Float> Diffuser<T> {
    #[inline]
    fn process(&mut self, x: T) -> T {
        let g = T::_lit(0.5);
        let delayed = self.line.read(self.delay - 1);
        let v = (x + g * delayed)._flush_denormal();
        self.line.push(v);
        delayed - g * v
    }
}

/// Algorithmic stereo reverb: an 8-line feedback delay network.
///
/// Input (summed to mono) -> pre-delay -> four all-pass diffusers -> 8 delay lines whose outputs are
/// damped (one-pole low-pass), scaled so the tail falls 60 dB in `decay` seconds, mixed by an
/// orthogonal Hadamard matrix and fed back. Left and right are different orthogonal mixes of the
/// lines. `size` scales every delay (bigger room, sparser early echoes); `damping` sets how fast the
/// highs die away; `modulation` slowly sweeps each line's delay so long tails don't ring metallically.
/// Needs a stereo buffer. Allocates only in `new` (sized for the largest `size`).
#[derive(Debug, Clone)]
pub struct Reverb<T: Float> {
    sample_rate: T,
    lines: Vec<DelayLine<T>>,
    lengths: [usize; LINES],
    gains: [T; LINES],
    lowpass: [T; LINES],
    damping_coeff: T,
    /// per-line triangle LFO phase (cycles) and increment (cycles per sample)
    mod_phase: [T; LINES],
    mod_increment: [T; LINES],
    /// sweep depth in samples
    mod_depth: T,
    modulation: T,
    /// previous output of each line's all-pass interpolator
    allpass_state: [T; LINES],
    /// per-line integer delay and all-pass coefficient, refreshed every MOD_INTERVAL samples
    mod_whole: [usize; LINES],
    mod_eta: [T; LINES],
    mod_countdown: usize,
    diffusers: Vec<Diffuser<T>>,
    predelay_line: DelayLine<T>,
    predelay: usize,
    size: T,
    decay: T,
    damping: T,
    predelay_seconds: T,
    mix: SmoothedValue<T>,
    width: SmoothedValue<T>,
}

impl<T: Float> Reverb<T> {
    /// A medium room: size 1, 1.8 s decay, damping at 6 kHz, 20 ms pre-delay, modulation 0.5, 30% mix,
    /// full width.
    pub fn new(sample_rate: T) -> Self {
        let fs = sample_rate.to_f64().unwrap_or(48_000.0);
        let scale = fs / 44_100.0;
        let smoothed = |v: f64| SmoothedValue::new(T::_lit(v)).with_ramp_seconds(T::_lit(0.02), sample_rate);
        let mut reverb = Self {
            sample_rate,
            lines: LENGTHS_44K
                .iter()
                .map(|&l| DelayLine::new((l * scale * MAX_SIZE + MAX_MOD_48K * fs / 48_000.0).ceil() as usize + 2))
                .collect(),
            lengths: [1; LINES],
            gains: [T::_ZERO; LINES],
            lowpass: [T::_ZERO; LINES],
            damping_coeff: T::_ONE,
            mod_phase: std::array::from_fn(|i| T::_lit(i as f64 / LINES as f64)),
            mod_increment: std::array::from_fn(|i| T::_lit(MOD_RATES[i] / fs)),
            mod_depth: T::_ZERO,
            modulation: T::_lit(0.5),
            allpass_state: [T::_ZERO; LINES],
            mod_whole: [0; LINES],
            mod_eta: [T::_ZERO; LINES],
            mod_countdown: 0,
            diffusers: DIFFUSERS_44K
                .iter()
                .map(|&l| {
                    let delay = ((l * scale).round() as usize).max(1);
                    Diffuser { line: DelayLine::new(delay), delay }
                })
                .collect(),
            predelay_line: DelayLine::new((MAX_PREDELAY * fs).ceil() as usize + 1),
            predelay: 0,
            size: T::_ONE,
            decay: T::_lit(1.8),
            damping: T::_lit(6_000.0),
            predelay_seconds: T::_lit(0.02),
            mix: smoothed(0.3),
            width: smoothed(1.0),
        };
        reverb.update();
        reverb
    }

    /// Recomputes line lengths, decay gains, damping and pre-delay from the settings.
    fn update(&mut self) {
        let fs = self.sample_rate.to_f64().unwrap_or(48_000.0);
        let size = self.size.to_f64().unwrap_or(1.0);
        let decay = self.decay.to_f64().unwrap_or(1.0);
        for (i, &base) in LENGTHS_44K.iter().enumerate() {
            let len = ((base * fs / 44_100.0 * size).round() as usize).clamp(1, self.lines[i].max_delay());
            self.lengths[i] = len;
            // a line of `len` samples is traversed decay*fs/len times per `decay` seconds; together
            // those passes must attenuate by 60 dB
            self.gains[i] = T::_lit(10f64.powf(-3.0 * len as f64 / (decay * fs)));
        }
        let fc = self.damping.to_f64().unwrap_or(20_000.0);
        // at the top of the range the low-pass is bypassed (coefficient 1) rather than left slightly
        // closed: even a mild loss per pass adds up over the many passes of a long decay
        self.damping_coeff = if fc >= 0.45 * fs { T::_ONE } else { T::_lit(1.0 - (-std::f64::consts::TAU * fc / fs).exp()) };
        self.mod_depth = self.modulation * T::_lit(MAX_MOD_48K * fs / 48_000.0);
        let predelay = (self.predelay_seconds.to_f64().unwrap_or(0.0) * fs).round() as usize;
        self.predelay = predelay.min(self.predelay_line.max_delay());
    }

    /// Room size, 0.25-2 (scales every delay line). Takes effect immediately.
    pub fn set_size(&mut self, size: T) {
        self.size = size._clamp(T::_lit(0.25), T::_lit(MAX_SIZE));
        self.update();
    }
    /// Room size.
    pub fn size(&self) -> T {
        self.size
    }
    /// RT60: seconds for the tail to fall by 60 dB (at low frequencies; damping shortens the highs).
    pub fn set_decay(&mut self, seconds: T) {
        self.decay = seconds._max(T::_lit(0.05));
        self.update();
    }
    /// RT60 in seconds.
    pub fn decay(&self) -> T {
        self.decay
    }
    /// Cutoff (Hz) of the low-pass in the feedback loop: lower = darker, faster-dying highs. At or above
    /// 45% of the sample rate damping is off and every frequency decays at the `decay` rate.
    pub fn set_damping(&mut self, hz: T) {
        self.damping = hz._max(T::_lit(100.0));
        self.update();
    }
    /// Damping cutoff in Hz.
    pub fn damping(&self) -> T {
        self.damping
    }
    /// Delay before the reverb starts, in seconds (up to 0.25).
    pub fn set_predelay(&mut self, seconds: T) {
        self.predelay_seconds = seconds._clamp(T::_ZERO, T::_lit(MAX_PREDELAY));
        self.update();
    }
    /// Pre-delay in seconds.
    pub fn predelay(&self) -> T {
        self.predelay_seconds
    }
    /// How much the delay lines sweep (0..1, up to ~12 samples at 48 kHz), which smears the network's
    /// resonances so long tails don't ring metallically. 0 = static delays.
    pub fn set_modulation(&mut self, amount: T) {
        self.modulation = amount._clamp(T::_ZERO, T::_ONE);
        self.update();
    }
    /// Delay modulation amount, 0..1.
    pub fn modulation(&self) -> T {
        self.modulation
    }
    /// 0 = dry only, 1 = reverb only.
    pub fn set_mix(&mut self, mix: T) {
        self.mix.set_target(mix._clamp(T::_ZERO, T::_ONE));
    }
    /// Target dry / wet mix.
    pub fn mix(&self) -> T {
        self.mix.target()
    }
    /// Stereo width of the reverb: 0 = mono, 1 = fully decorrelated.
    pub fn set_width(&mut self, width: T) {
        self.width.set_target(width._clamp(T::_ZERO, T::_ONE));
    }
    /// Target stereo width.
    pub fn width(&self) -> T {
        self.width.target()
    }
    /// Silences the tail.
    pub fn reset(&mut self) {
        self.lines.iter_mut().for_each(DelayLine::reset);
        self.diffusers.iter_mut().for_each(|d| d.line.reset());
        self.predelay_line.reset();
        self.lowpass = [T::_ZERO; LINES];
        self.mod_phase = std::array::from_fn(|i| T::_lit(i as f64 / LINES as f64));
        self.allpass_state = [T::_ZERO; LINES];
        self.mod_countdown = 0;
    }

    /// Advances the triangle LFOs by MOD_INTERVAL samples and recomputes each line's swept delay as an
    /// integer part plus an all-pass coefficient. The integer part is chosen so the fraction stays in
    /// [0.5, 1.5), which keeps the coefficient within (-0.2, 0.34], far from the unstable edge at 1.
    fn update_modulation(&mut self) {
        for i in 0..LINES {
            let p = self.mod_phase[i];
            let tri = T::_lit(4.0) * (p - T::_lit(0.5))._abs() - T::_ONE;
            self.mod_phase[i] = (p + self.mod_increment[i] * T::_lit(MOD_INTERVAL as f64))._fract();
            let delay = T::_lit(self.lengths[i] as f64 - 1.0) + self.mod_depth * tri;
            let whole = (delay - T::_lit(0.5))._floor();
            let frac = delay - whole;
            self.mod_eta[i] = (T::_ONE - frac) / (T::_ONE + frac);
            self.mod_whole[i] = whole.to_usize().unwrap_or(0);
        }
    }

    /// One stereo frame: returns the wet (left, right) for a mono input.
    #[inline]
    fn tick(&mut self, input: T) -> (T, T) {
        if self.mod_countdown == 0 {
            self.update_modulation();
            self.mod_countdown = MOD_INTERVAL;
        }
        self.mod_countdown -= 1;
        // subnormal numbers are slow on many CPUs (and hosts reject them in the output), so every
        // recursion is flushed each sample: what enters a delay line (where a tail recirculates for
        // seconds), the damping and interpolation states, the input in case the host sends
        // subnormals, and the outputs (`process_stereo`). A tail ends in exact zeros.
        self.predelay_line.push(input._flush_denormal());
        let mut x = self.predelay_line.read(self.predelay);
        for d in &mut self.diffusers {
            x = d.process(x);
        }
        let mut state = [T::_ZERO; LINES];
        let (mut left, mut right) = (T::_ZERO, T::_ZERO);
        for i in 0..LINES {
            let out = if self.mod_depth == T::_ZERO {
                self.lines[i].read(self.lengths[i] - 1)
            } else {
                // the fractional part goes through a first-order all-pass interpolator, whose magnitude
                // response is exactly flat, so the loop loses no energy to interpolation and the decay
                // time stays exact (linear or cubic interpolation dull the highs a little every pass)
                let (x0, x1) = (self.lines[i].read(self.mod_whole[i]), self.lines[i].read(self.mod_whole[i] + 1));
                let y = (self.mod_eta[i] * (x0 - self.allpass_state[i]) + x1)._flush_denormal();
                self.allpass_state[i] = y;
                y
            };
            left = left + T::_lit(LEFT_SIGNS[i]) * out;
            right = right + T::_lit(RIGHT_SIGNS[i]) * out;
            self.lowpass[i] = (self.lowpass[i] + self.damping_coeff * (out - self.lowpass[i]))._flush_denormal();
            state[i] = self.lowpass[i] * self.gains[i];
        }
        hadamard(&mut state);
        let input_gain = T::_lit(1.0 / (LINES as f64).sqrt());
        // computed as a whole array first (vectorizes) rather than interleaved with the line writes
        let writes: [T; LINES] = std::array::from_fn(|i| (state[i] + x * T::_lit(INPUT_SIGNS[i]) * input_gain)._flush_denormal());
        for (line, w) in self.lines.iter_mut().zip(writes) {
            line.push(w);
        }
        let out_gain = T::_lit(1.0 / (LINES as f64).sqrt());
        (left * out_gain, right * out_gain)
    }

    /// Processes a stereo pair in place.
    pub fn process_stereo(&mut self, left: &mut [T], right: &mut [T]) {
        let half = T::_lit(0.5);
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (wl, wr) = self.tick((*l + *r) * half);
            // width: scale the side (difference) of the wet signal
            let (mid, side) = ((wl + wr) * half, (wl - wr) * half * self.width.next_value());
            let mix = self.mix.next_value();
            *l = (*l + mix * (mid + side - *l))._flush_denormal();
            *r = (*r + mix * (mid - side - *r))._flush_denormal();
        }
    }
}

/// Stereo in, stereo out. Panics unless the buffer has exactly 2 channels.
impl<T: Float> MultiProcessor<T> for Reverb<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        let (left, right) = buffer.stereo_mut();
        self.process_stereo(left, right);
    }
    fn reset(&mut self) {
        Reverb::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;
    use crate::fft::RealFft;

    const FS: f64 = 48_000.0;

    /// Wet-only stereo impulse response, `seconds` long.
    fn impulse_response(reverb: &mut Reverb<f64>, seconds: f64) -> (Vec<f64>, Vec<f64>) {
        reverb.set_mix(1.0);
        reverb.mix.set_immediate(1.0);
        let n = (seconds * FS) as usize;
        let (mut l, mut r) = (vec![0.0; n], vec![0.0; n]);
        l[0] = 1.0;
        r[0] = 1.0;
        reverb.process_stereo(&mut l, &mut r);
        (l, r)
    }

    /// Decay time from the Schroeder energy-decay curve: time from -5 to -25 dB, times 3 (T20).
    fn measured_rt60(ir: &[f64]) -> f64 {
        let mut edc: Vec<f64> = ir.iter().map(|x| x * x).collect();
        for i in (0..edc.len() - 1).rev() {
            edc[i] += edc[i + 1];
        }
        let db = |i: usize| 10.0 * (edc[i] / edc[0]).log10();
        let t5 = (0..edc.len()).find(|&i| db(i) <= -5.0).unwrap();
        let t25 = (0..edc.len()).find(|&i| db(i) <= -25.0).unwrap();
        3.0 * (t25 - t5) as f64 / FS
    }

    /// How sharply the tail's spectrum peaks: the mean, over the strongest bins, of each bin's power
    /// relative to the median power around it. Static delay networks ring at sharp modal peaks.
    fn modal_peakiness(modulation: f64) -> f64 {
        let mut reverb = Reverb::new(FS);
        reverb.set_decay(8.0);
        reverb.set_damping(24_000.0);
        reverb.set_modulation(modulation);
        let (l, _) = impulse_response(&mut reverb, 2.5);
        let n = 65_536;
        let tail = &l[(1.0 * FS) as usize..][..n];
        let mut fft = RealFft::new(n);
        let mut spectrum = vec![Complex::zero(); fft.spectrum_len()];
        fft.forward(tail, &mut spectrum);
        let power: Vec<f64> = spectrum.iter().map(|z| z.norm_sqr()).collect();
        // 100 Hz - 4 kHz, where the modes are most audible
        let (lo, hi) = (100 * n / FS as usize, 4_000 * n / FS as usize);
        let mut ratios: Vec<f64> = (lo..hi)
            .map(|k| {
                let mut around: Vec<f64> = power[k - 64..k + 64].to_vec();
                around.sort_by(f64::total_cmp);
                power[k] / around[64]
            })
            .collect();
        ratios.sort_by(|a, b| b.total_cmp(a));
        ratios[..50].iter().sum::<f64>() / 50.0
    }

    #[test]
    fn modulation_smears_modal_peaks() {
        let (static_delays, modulated) = (modal_peakiness(0.0), modal_peakiness(1.0));
        println!("modal peakiness: {static_delays:.1} static, {modulated:.1} modulated");
        assert!(modulated < 0.5 * static_delays, "static {static_delays:.1} vs modulated {modulated:.1}");
    }

    #[test]
    fn hadamard_is_orthogonal() {
        let mut x = [1.0, 2.0, -3.0, 0.5, 0.0, 4.0, -1.0, 2.5];
        let energy: f64 = x.iter().map(|v| v * v).sum();
        hadamard(&mut x);
        let after: f64 = x.iter().map(|v| v * v).sum();
        assert!((energy - after).abs() < 1e-12);
        hadamard(&mut x); // its own inverse
        assert!((x[2] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn decay_time_matches_the_setting() {
        for rt60 in [0.5, 1.0, 2.5] {
            let mut reverb = Reverb::new(FS);
            reverb.set_decay(rt60);
            reverb.set_damping(24_000.0); // no damping: the whole spectrum decays at the set rate
            reverb.set_predelay(0.0);
            let (l, _) = impulse_response(&mut reverb, rt60 * 1.5);
            let measured = measured_rt60(&l);
            println!("RT60 set {rt60} s, measured {measured:.3} s");
            assert!((measured / rt60 - 1.0).abs() < 0.05, "set {rt60} s, measured {measured:.3} s");
        }
    }

    #[test]
    fn damping_darkens_the_tail() {
        let centroid = |damping: f64| {
            let mut reverb = Reverb::new(FS);
            reverb.set_damping(damping);
            let (l, _) = impulse_response(&mut reverb, 0.6);
            let tail = &l[(0.3 * FS) as usize..][..8_192];
            let mut fft = RealFft::new(8_192);
            let mut spectrum = vec![Complex::zero(); fft.spectrum_len()];
            fft.forward(tail, &mut spectrum);
            let (weighted, total) = spectrum.iter().enumerate().fold((0.0, 0.0), |(w, t), (k, z)| (w + k as f64 * z.norm_sqr(), t + z.norm_sqr()));
            weighted / total * FS / 8_192.0
        };
        let (bright, dark) = (centroid(20_000.0), centroid(2_000.0));
        assert!(dark < 0.6 * bright, "spectral centroid {dark:.0} Hz with damping vs {bright:.0} Hz without");
    }

    #[test]
    fn tail_ends_in_exact_silence_without_subnormals() {
        // subnormal numbers are 10-100x slower on many CPUs: the tail must skip them, even when the
        // host sends subnormal input, without relying on the CPU's flush-to-zero mode
        let fs = 48_000.0f32;
        let mut reverb = Reverb::<f32>::new(fs);
        reverb.set_decay(0.2); // the 1e-30 flush level (-600 dB) is reached after about 2 s
        reverb.set_mix(1.0);
        let n = 4 * fs as usize;
        let (mut l, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
        (l[0], r[0]) = (1.0, 1.0);
        l[1..1_000].fill(f32::MIN_POSITIVE / 2.0);
        reverb.process_stereo(&mut l, &mut r);
        // not a single subnormal sample, anywhere: not from the tail, nor passed through from the input
        assert!(l.iter().chain(&r).all(|s| !s.is_subnormal()), "no subnormal output");
        let last = n - fs as usize / 10;
        assert!(l[last..].iter().chain(&r[last..]).all(|&s| s == 0.0), "the tail ends in exact silence");
    }

    #[test]
    fn stereo_is_decorrelated_and_width_zero_is_mono() {
        let mut reverb = Reverb::new(FS);
        let (l, r) = impulse_response(&mut reverb, 0.5);
        let (tl, tr) = (&l[4_800..], &r[4_800..]);
        let correlation = tl.inner(tr).unwrap() / (tl.norm_l2() * tr.norm_l2());
        assert!(correlation.abs() < 0.3, "left/right correlation {correlation:.3}");

        let mut mono = Reverb::new(FS);
        mono.set_width(0.0);
        mono.width.set_immediate(0.0);
        let (l, r) = impulse_response(&mut mono, 0.2);
        assert!(l.iter().zip(&r).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn long_decays_stay_stable_and_mix_zero_is_dry() {
        let mut reverb = Reverb::new(FS);
        reverb.set_decay(30.0);
        let input: Vec<f64> = crate::osc::Noise::new(2).take(240_000).collect(); // 5 s of noise
        let (mut l, mut r) = (input.clone(), input.clone());
        reverb.process_stereo(&mut l, &mut r);
        assert!(l.iter().chain(&r).all(|s| s.is_finite()) && l.peak() < 20.0, "peak {}", l.peak());

        let mut dry = Reverb::new(FS);
        dry.set_mix(0.0);
        dry.mix.set_immediate(0.0);
        let (mut l, mut r) = (input[..1_000].to_vec(), input[..1_000].to_vec());
        dry.process_stereo(&mut l, &mut r);
        assert_eq!(l, input[..1_000]);
    }
}
