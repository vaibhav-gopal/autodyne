//! Click-free parameter changes and sample-accurate parameter events.
//!
//! - [`Smoothed`] wraps any [`Parameterized`] processor: changes to its continuous parameters
//!   ([`Smoothing::Ramp`]) glide to the new value instead of jumping, in normalized space, so a log
//!   frequency glides evenly through the octaves. Parameters the processor already smooths itself
//!   ([`Smoothing::Internal`]) and discrete ones ([`Smoothing::Instant`]) pass straight through.
//! - [`ParamEvent`] and [`process_events`] / [`process_buffer_events`] apply parameter changes at
//!   exact sample offsets within a block, the way `synth::TimedEvent` does for notes.
//!
//! Ramps advance at control rate: the block is split into steps of [`RAMP_STEP`] samples and each
//! ramping parameter is updated before each step. Coefficient-based processors (filters,
//! compressors) recompute only while something is moving; settled parameters cost nothing.

use std::ops::Range;

use super::{ParamError, ParamInfo, Parameterized, Smoothing};
use crate::channels::{AudioBuffer, MultiProcessor};
use crate::processor::Processor;
use crate::units::Float;

/// Samples between parameter updates while a ramp is in progress.
pub const RAMP_STEP: usize = 32;

#[derive(Debug, Clone, Copy)]
struct Ramp {
    info: ParamInfo,
    /// normalized (0..1) position now and at the end of the ramp
    current: f64,
    target: f64,
    /// the value set, in plain units: applied exactly when the ramp ends
    target_value: f64,
    /// normalized change per sample
    step: f64,
    remaining: usize,
}

/// A processor whose continuous parameters ramp to new values instead of jumping.
///
/// Wrap a processor (or a whole chain) once, then change parameters through the wrapper's
/// `Parameterized` interface as usual. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct Smoothed<P> {
    inner: P,
    ramps: Vec<Ramp>,
    ramp_samples: usize,
    /// number of ramps in progress
    active: usize,
}

impl<P: Parameterized> Smoothed<P> {
    /// Ramps take `ramp_seconds` (20 ms is a common click-free choice; 0 applies changes
    /// immediately).
    pub fn new(inner: P, ramp_seconds: f64, sample_rate: f64) -> Self {
        let ramps = (0..inner.param_count())
            .map(|index| {
                let info = inner.param_info(index).expect("index below param_count");
                let value = inner.get_param(index).unwrap_or(info.default);
                let n = info.to_normalized(value);
                Ramp { info, current: n, target: n, target_value: value, step: 0.0, remaining: 0 }
            })
            .collect();
        let ramp_samples = (ramp_seconds * sample_rate).round().max(0.0) as usize;
        Self { inner, ramps, ramp_samples, active: 0 }
    }
    pub fn inner(&self) -> &P {
        &self.inner
    }
    /// The wrapped processor. Parameters changed on it directly bypass the ramps: call
    /// [`sync`](Self::sync) afterwards.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
    pub fn into_inner(self) -> P {
        self.inner
    }
    /// Whether any parameter is still gliding.
    pub fn is_ramping(&self) -> bool {
        self.active > 0
    }
    /// Re-reads every value from the wrapped processor (after changing it directly), cancelling ramps.
    pub fn sync(&mut self) {
        for (index, r) in self.ramps.iter_mut().enumerate() {
            let value = self.inner.get_param(index).unwrap_or(r.target_value);
            r.current = r.info.to_normalized(value);
            r.target = r.current;
            r.target_value = value;
            r.remaining = 0;
        }
        self.active = 0;
    }
    /// Ends every ramp at its target now.
    pub fn settle(&mut self) {
        for (index, r) in self.ramps.iter_mut().enumerate() {
            if r.remaining > 0 {
                r.remaining = 0;
                r.current = r.target;
                let _ = self.inner.set_param(index, r.target_value);
            }
        }
        self.active = 0;
    }

    /// Drives a block of `frames` samples through `render`, which receives the wrapped processor and
    /// the range of the block to produce. While ramps are in progress the block is split into
    /// [`RAMP_STEP`]-sample pieces with parameters updated before each; otherwise `render` is
    /// called once for the whole block. This is how `process` is implemented, and the hook for
    /// anything that isn't a plain `Processor` (synth voices, custom render loops).
    pub fn run(&mut self, frames: usize, mut render: impl FnMut(&mut P, Range<usize>)) {
        let mut pos = 0;
        while pos < frames {
            if self.active == 0 {
                render(&mut self.inner, pos..frames);
                return;
            }
            let len = RAMP_STEP.min(frames - pos);
            self.advance(len);
            render(&mut self.inner, pos..pos + len);
            pos += len;
        }
    }

    /// Moves every active ramp `samples` forward and pushes the new values into the processor.
    fn advance(&mut self, samples: usize) {
        let inner = &mut self.inner;
        for (index, r) in self.ramps.iter_mut().enumerate() {
            if r.remaining == 0 {
                continue;
            }
            let k = samples.min(r.remaining);
            r.remaining -= k;
            let value = if r.remaining == 0 {
                r.current = r.target;
                self.active -= 1;
                r.target_value
            } else {
                r.current += r.step * k as f64;
                r.info.from_normalized(r.current)
            };
            // values stay inside the range by construction, so this can't fail
            let _ = inner.set_param(index, value);
        }
    }
}

impl<P: Parameterized> Parameterized for Smoothed<P> {
    fn param_count(&self) -> usize {
        self.inner.param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.inner.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.inner.param_group(index)
    }
    /// The value last set (a ramp's target), not the value mid-glide.
    fn get_param(&self, index: usize) -> Option<f64> {
        match self.ramps.get(index) {
            Some(r) if r.info.smoothing == Smoothing::Ramp => Some(r.target_value),
            _ => self.inner.get_param(index),
        }
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let Some(r) = self.ramps.get_mut(index) else {
            // a parameter the wrapped processor gained after wrapping (a growing `Vec` chain):
            // no ramp state for it, so it applies directly
            return self.inner.set_param(index, value);
        };
        let applied = r.info.validate(value)?;
        let n = r.info.to_normalized(applied);
        if r.info.smoothing != Smoothing::Ramp || self.ramp_samples == 0 {
            if r.remaining > 0 {
                r.remaining = 0;
                self.active -= 1;
            }
            (r.current, r.target, r.target_value) = (n, n, applied);
            return self.inner.set_param(index, applied);
        }
        if applied == r.target_value {
            // already there or on the way: re-sending a value (as some hosts do every block) must
            // not restart the glide, or it would never settle
            return Ok(applied);
        }
        if r.remaining == 0 {
            self.active += 1;
        }
        // (re)start from wherever the glide is now: interrupting a ramp never jumps
        r.target = n;
        r.target_value = applied;
        r.remaining = self.ramp_samples;
        r.step = (n - r.current) / self.ramp_samples as f64;
        Ok(applied)
    }
}

impl<T: Float, P: Processor<T> + Parameterized> Processor<T> for Smoothed<P> {
    fn process(&mut self, block: &mut [T]) {
        self.run(block.len(), |p, range| p.process(&mut block[range]));
    }
    fn reset(&mut self) {
        self.settle();
        self.inner.reset();
    }
}

impl<T: Float, P: MultiProcessor<T> + Parameterized> MultiProcessor<T> for Smoothed<P> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.run(buffer.frames(), |p, range| buffer.with_window(range.start, range.len(), |b| p.process(b)));
    }
    fn reset(&mut self) {
        self.settle();
        self.inner.reset();
    }
}

// SAMPLE-ACCURATE EVENTS ==========================================================================

/// A parameter change at a sample offset within the next block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamEvent {
    pub offset: usize,
    pub index: usize,
    pub value: f64,
}

/// Processes `block`, applying each event at its sample offset. Events must be sorted by offset;
/// offsets past the end apply at the end. With a [`Smoothed`] processor each event starts its ramp
/// on that exact sample. The whole block is processed even if an event is invalid; the first
/// error is returned.
pub fn process_events<T: Float, P: Processor<T> + Parameterized + ?Sized>(
    processor: &mut P,
    block: &mut [T],
    events: &[ParamEvent],
) -> Result<(), ParamError> {
    let mut result = Ok(());
    let mut pos = 0;
    for event in events {
        let at = event.offset.clamp(pos, block.len());
        if at > pos {
            processor.process(&mut block[pos..at]);
            pos = at;
        }
        if let Err(e) = processor.set_param(event.index, event.value) {
            result = result.and(Err(e));
        }
    }
    if pos < block.len() {
        processor.process(&mut block[pos..]);
    }
    result
}

/// [`process_events`] for multichannel processors: the buffer is split at event offsets without
/// copying (`AudioBuffer::with_window`).
pub fn process_buffer_events<T: Float, P: MultiProcessor<T> + Parameterized + ?Sized>(
    processor: &mut P,
    buffer: &mut AudioBuffer<T>,
    events: &[ParamEvent],
) -> Result<(), ParamError> {
    let mut result = Ok(());
    let frames = buffer.frames();
    let mut pos = 0;
    for event in events {
        let at = event.offset.clamp(pos, frames);
        if at > pos {
            buffer.with_window(pos, at - pos, |b| processor.process(b));
            pos = at;
        }
        if let Err(e) = processor.set_param(event.index, event.value) {
            result = result.and(Err(e));
        }
    }
    if pos < frames {
        buffer.with_window(pos, frames - pos, |b| processor.process(b));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::PerChannel;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::gain::Gain;
    use crate::osc::Waveform;
    use crate::synth::SynthVoice;

    const FS: f64 = 48_000.0;

    #[test]
    fn log_parameters_glide_geometrically_and_land_exactly() {
        let mut lp = Smoothed::new(Biquad::<f64>::lowpass(100.0, FS, BUTTERWORTH_Q), 0.01, FS); // 480 samples
        let cutoff = lp.param_index("frequency_hz").unwrap();
        assert_eq!(lp.set_param(cutoff, 10_000.0), Ok(10_000.0));
        assert_eq!(lp.get_param(cutoff), Some(10_000.0), "reads back the target at once");
        assert_eq!(lp.inner().design().unwrap().frequency, 100.0, "but nothing has moved yet");
        let mut block = vec![0.0; 240];
        lp.process(&mut block);
        let halfway = lp.inner().design().unwrap().frequency;
        assert!((halfway / 1_000.0 - 1.0).abs() < 1e-9, "halfway is the geometric mean: {halfway}");
        lp.process(&mut block);
        assert_eq!(lp.inner().design().unwrap().frequency, 10_000.0, "lands exactly");
        assert!(!lp.is_ramping());
    }

    #[test]
    fn interrupting_a_ramp_continues_from_where_it_is() {
        let mut lp = Smoothed::new(Biquad::<f64>::lowpass(100.0, FS, BUTTERWORTH_Q), 0.01, FS);
        let cutoff = lp.param_index("frequency_hz").unwrap();
        lp.set_param(cutoff, 10_000.0).unwrap();
        lp.process(&mut [0.0; 240]);
        lp.set_param(cutoff, 100.0).unwrap(); // reverse halfway
        lp.process(&mut [0.0; 32]);
        let f = lp.inner().design().unwrap().frequency;
        assert!(f < 1_000.0 && f > 800.0, "continues smoothly back down from ~1 kHz: {f}");
    }

    #[test]
    fn resending_the_target_does_not_restart_the_glide() {
        let mut lp = Smoothed::new(Biquad::<f64>::lowpass(100.0, FS, BUTTERWORTH_Q), 0.01, FS); // 480 samples
        let cutoff = lp.param_index("frequency_hz").unwrap();
        lp.set_param(cutoff, 10_000.0).unwrap();
        for _ in 0..15 {
            // 15 x 32 = 480 samples, with the host re-sending the same value every block
            lp.set_param(cutoff, 10_000.0).unwrap();
            lp.process(&mut [0.0; 32]);
        }
        assert!(!lp.is_ramping(), "settles on schedule");
        assert_eq!(lp.inner().design().unwrap().frequency, 10_000.0);
    }

    #[test]
    fn parameters_added_after_wrapping_pass_through() {
        let mut chain = Smoothed::new(vec![Gain::<f64>::new(1.0, 0.0, FS)], 0.02, FS);
        chain.inner_mut().push(Gain::new(1.0, 0.0, FS));
        assert_eq!(chain.param_count(), 2);
        assert_eq!(chain.set_param(1, -6.0), Ok(-6.0), "no ramp state for it, so it applies directly");
        assert_eq!(chain.get_param(1).map(f64::round), Some(-6.0));
        assert_eq!(chain.set_param(2, 0.0), Err(ParamError::UnknownIndex(2)));
    }

    #[test]
    fn discrete_and_internally_smoothed_parameters_pass_through() {
        let mut voice = Smoothed::new(SynthVoice::<f32>::new(48_000.0), 0.02, FS);
        voice.set_param_by_id("waveform", 2.0).unwrap();
        assert!(matches!(voice.inner().waveform(), Waveform::Pulse { .. }), "a choice switches at once");
        let mut gain = Smoothed::new(Gain::<f32>::new(1.0, 0.02, 48_000.0), 0.02, FS);
        gain.set_param(0, -6.0).unwrap();
        assert!(!gain.is_ramping(), "Gain smooths itself: no second ramp on top");
        assert_eq!(gain.inner().get_param(0).map(|v| v.round()), Some(-6.0));
    }

    /// Largest jump between consecutive samples: where a parameter step clicks.
    fn max_step(x: &[f32]) -> f32 {
        x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn ramps_remove_zipper_steps() {
        // a 200 Hz sine through a high-pass whose cutoff jumps from 5 kHz (almost silent) to 20 Hz
        // (wide open) mid-stream: unsmoothed, the output steps straight to full level
        let sine: Vec<f32> = (0..4_800).map(|n| (std::f32::consts::TAU * 200.0 * n as f32 / 48_000.0).sin()).collect();
        let render = |ramp_seconds: f64| {
            let mut hp = Smoothed::new(Biquad::<f32>::highpass(5_000.0, 48_000.0, BUTTERWORTH_Q as f32), ramp_seconds, FS);
            let cutoff = hp.param_index("frequency_hz").unwrap();
            let mut out = sine.clone();
            process_events(&mut hp, &mut out, &[ParamEvent { offset: 1_000, index: cutoff, value: 20.0 }]).unwrap();
            out
        };
        let (jumpy, smooth) = (max_step(&render(0.0)), max_step(&render(0.05)));
        // the sine's own largest step is 2 pi f / fs: smoothed, the cutoff change adds nothing to it
        let natural = std::f32::consts::TAU * 200.0 / 48_000.0;
        assert!(smooth < 1.05 * natural, "max sample step {smooth} smoothed (sine alone: {natural})");
        assert!(jumpy > 10.0 * natural, "unsmoothed, the jump shows: {jumpy}");
    }
    #[test]
    fn events_land_on_their_sample() {
        let mut gain = Gain::<f32>::new(1.0, 0.0, 48_000.0); // no ramp: changes are immediate
        let mut block = [1.0f32; 8];
        let half = -20.0 * 2f64.log10();
        let events = [ParamEvent { offset: 3, index: 0, value: half }, ParamEvent { offset: 99, index: 0, value: 0.0 }];
        process_events(&mut gain, &mut block, &events).unwrap();
        assert_eq!(&block[..3], &[1.0; 3]);
        assert!(block[3..].iter().all(|&s| (s - 0.5).abs() < 1e-6), "{block:?}");
        assert_eq!(gain.get_param(0), Some(0.0), "an offset past the end applies at the end");

        let mut block = [1.0f32; 4];
        let bad = [ParamEvent { offset: 1, index: 7, value: 0.0 }];
        assert_eq!(process_events(&mut gain, &mut block, &bad), Err(ParamError::UnknownIndex(7)));
        assert_eq!(block, [1.0; 4], "the block is still processed");
    }

    #[test]
    fn buffer_events_split_every_channel_at_the_same_sample() {
        let mut stereo = PerChannel::new(2, |_| Gain::<f32>::new(1.0, 0.0, 48_000.0));
        let mut buffer = AudioBuffer::new(2, 6);
        buffer.fill(1.0);
        let events = [ParamEvent { offset: 2, index: 0, value: -20.0 * 2f64.log10() }];
        process_buffer_events(&mut stereo, &mut buffer, &events).unwrap();
        for ch in 0..2 {
            assert_eq!(&buffer.channel(ch)[..2], &[1.0, 1.0]);
            assert!(buffer.channel(ch)[2..].iter().all(|&s| (s - 0.5).abs() < 1e-6));
        }
    }
}
