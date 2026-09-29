//! Multichannel audio: a planar buffer, per-channel and linked processing, and stereo tools.
//!
//! Processors in this crate are mono. For more channels:
//! - `PerChannel` runs an independent copy of a processor on each channel (EQ, echo, chorus with
//!   offset LFOs, ...).
//! - Linked processors look at all channels together so the image doesn't shift: a `Compressor`
//!   used as a `MultiProcessor` reacts to the loudest channel and turns every channel down equally.
//! - `StereoWidth` and `Panner` work on the stereo pair itself.
//!
//! Audio APIs usually deliver interleaved frames (L R L R ...); `AudioBuffer::copy_from_interleaved`
//! and `copy_to_interleaved` convert without allocating.

use crate::dynamics::Compressor;
use crate::gain::SmoothedValue;
use crate::processor::Processor;
use crate::units::*;

// BUFFER ==========================================================================================

/// Planar multichannel audio: each channel is a contiguous slice. Storage is allocated once for
/// `max_frames`; `set_frames` chooses how many are in use for the current block.
#[derive(Debug, Clone)]
pub struct AudioBuffer<T: Float> {
    data: Vec<T>,
    channels: usize,
    max_frames: usize,
    frames: usize,
}

impl<T: Float> AudioBuffer<T> {
    /// Panics if `channels` is 0.
    pub fn new(channels: usize, max_frames: usize) -> Self {
        assert!(channels > 0, "need at least one channel");
        Self { data: vec![T::_ZERO; channels * max_frames], channels, max_frames, frames: max_frames }
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
    /// Panics if `frames > max_frames()`.
    pub fn set_frames(&mut self, frames: usize) {
        assert!(frames <= self.max_frames, "{frames} frames exceeds capacity {}", self.max_frames);
        self.frames = frames;
    }
    pub fn channel(&self, ch: usize) -> &[T] {
        &self.data[ch * self.max_frames..][..self.frames]
    }
    pub fn channel_mut(&mut self, ch: usize) -> &mut [T] {
        let (start, len) = (ch * self.max_frames, self.frames);
        &mut self.data[start..][..len]
    }
    /// Every channel as a mutable slice, in order.
    pub fn channels_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let frames = self.frames;
        self.data.chunks_exact_mut(self.max_frames).map(move |c| &mut c[..frames])
    }
    /// (left, right) of a stereo buffer. Panics unless there are exactly 2 channels.
    pub fn stereo_mut(&mut self) -> (&mut [T], &mut [T]) {
        assert_eq!(self.channels, 2, "stereo_mut needs a 2-channel buffer");
        let frames = self.frames;
        let (l, r) = self.data.split_at_mut(self.max_frames);
        (&mut l[..frames], &mut r[..frames])
    }
    pub fn fill(&mut self, value: T) {
        self.channels_mut().for_each(|c| c.iter_mut().for_each(|s| *s = value));
    }
    /// Loads interleaved frames (L R L R ...) and sets `frames` to match.
    /// Panics if the length isn't a whole number of frames or exceeds capacity.
    pub fn copy_from_interleaved(&mut self, interleaved: &[T]) {
        assert_eq!(interleaved.len() % self.channels, 0, "interleaved length must be a whole number of frames");
        self.set_frames(interleaved.len() / self.channels);
        let (n, max) = (self.channels, self.max_frames);
        for (f, frame) in interleaved.chunks_exact(n).enumerate() {
            for (ch, &s) in frame.iter().enumerate() {
                self.data[ch * max + f] = s;
            }
        }
    }
    /// Writes the current frames interleaved. Panics unless `out.len() == frames * channels`.
    pub fn copy_to_interleaved(&self, out: &mut [T]) {
        assert_eq!(out.len(), self.frames * self.channels, "output must hold exactly frames * channels samples");
        let (n, max) = (self.channels, self.max_frames);
        for (f, frame) in out.chunks_exact_mut(n).enumerate() {
            for (ch, s) in frame.iter_mut().enumerate() {
                *s = self.data[ch * max + f];
            }
        }
    }
}

// MULTICHANNEL PROCESSING =========================================================================

/// The multichannel counterpart of `Processor`: transforms a whole `AudioBuffer` in place.
pub trait MultiProcessor<T: Float> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>);
    fn reset(&mut self);
}

/// One independent mono processor per channel.
#[derive(Debug, Clone)]
pub struct PerChannel<P> {
    processors: Vec<P>,
}

impl<P> PerChannel<P> {
    /// Builds one processor per channel; `make(channel_index)` lets channels differ
    /// (e.g. LFO phase 0 on the left, 0.5 on the right).
    pub fn new(channels: usize, make: impl FnMut(usize) -> P) -> Self {
        Self { processors: (0..channels).map(make).collect() }
    }
    pub fn channel(&mut self, ch: usize) -> &mut P {
        &mut self.processors[ch]
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.processors.iter_mut()
    }
}

impl<T: Float, P: Processor<T>> MultiProcessor<T> for PerChannel<P> {
    /// Panics if the buffer's channel count differs from the number of processors.
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        assert_eq!(buffer.channels(), self.processors.len(), "channel count mismatch");
        for (p, ch) in self.processors.iter_mut().zip(buffer.channels_mut()) {
            p.process(ch);
        }
    }
    fn reset(&mut self) {
        self.processors.iter_mut().for_each(Processor::reset);
    }
}

/// Linked compression: the loudest channel drives one gain applied to all channels.
impl<T: Float> MultiProcessor<T> for Compressor<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        let (n, max) = (buffer.channels, buffer.max_frames);
        for f in 0..buffer.frames {
            let level = (0..n).fold(T::_ZERO, |m, ch| m._max(buffer.data[ch * max + f]._abs()));
            let g = self.gain_for_level(level);
            for ch in 0..n {
                buffer.data[ch * max + f] = buffer.data[ch * max + f] * g;
            }
        }
    }
    fn reset(&mut self) {
        Compressor::reset(self)
    }
}

impl<T: Float, M: MultiProcessor<T> + ?Sized> MultiProcessor<T> for Box<M> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        (**self).process(buffer)
    }
    fn reset(&mut self) {
        (**self).reset()
    }
}

impl<T: Float, M: MultiProcessor<T>> MultiProcessor<T> for Vec<M> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        for m in self.iter_mut() {
            m.process(buffer);
        }
    }
    fn reset(&mut self) {
        self.iter_mut().for_each(MultiProcessor::reset);
    }
}

macro_rules! impl_multi_chain {
    ($($M:ident . $i:tt),+) => {
        impl<T: Float, $($M: MultiProcessor<T>),+> MultiProcessor<T> for ($($M,)+) {
            fn process(&mut self, buffer: &mut AudioBuffer<T>) {
                $(self.$i.process(buffer);)+
            }
            fn reset(&mut self) {
                $(self.$i.reset();)+
            }
        }
    };
}

impl_multi_chain!(A.0);
impl_multi_chain!(A.0, B.1);
impl_multi_chain!(A.0, B.1, C.2);
impl_multi_chain!(A.0, B.1, C.2, D.3);
impl_multi_chain!(A.0, B.1, C.2, D.3, E.4);
impl_multi_chain!(A.0, B.1, C.2, D.3, E.4, F.5);
impl_multi_chain!(A.0, B.1, C.2, D.3, E.4, F.5, G.6);
impl_multi_chain!(A.0, B.1, C.2, D.3, E.4, F.5, G.6, H.7);

// STEREO ==========================================================================================

/// Mid/side stereo width: 0 = mono, 1 = unchanged, above 1 = wider. Needs a 2-channel buffer.
#[derive(Debug, Clone, Copy)]
pub struct StereoWidth<T: Float> {
    width: SmoothedValue<T>,
}

impl<T: Float> StereoWidth<T> {
    pub fn new(width: T, sample_rate: T) -> Self {
        Self { width: SmoothedValue::new(width).with_ramp_seconds(T::_lit(0.02), sample_rate) }
    }
    pub fn set_width(&mut self, width: T) {
        self.width.set_target(width._max(T::_ZERO));
    }
}

impl<T: Float> MultiProcessor<T> for StereoWidth<T> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        let half = T::_lit(0.5);
        let (left, right) = buffer.stereo_mut();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (mid, side) = ((*l + *r) * half, (*l - *r) * half * self.width.next_value());
            *l = mid + side;
            *r = mid - side;
        }
    }
    fn reset(&mut self) {
        let w = self.width.target();
        self.width.set_immediate(w);
    }
}

/// Places a mono signal in a stereo field with the constant-power pan law: -1 = hard left, 0 = center
/// (each side at -3 dB), 1 = hard right. The total power stays the same wherever it is panned.
#[derive(Debug, Clone, Copy)]
pub struct Panner<T: Float> {
    position: SmoothedValue<T>,
}

impl<T: Float> Panner<T> {
    pub fn new(position: T, sample_rate: T) -> Self {
        Self { position: SmoothedValue::new(position._clamp(-T::_ONE, T::_ONE)).with_ramp_seconds(T::_lit(0.02), sample_rate) }
    }
    pub fn set_position(&mut self, position: T) {
        self.position.set_target(position._clamp(-T::_ONE, T::_ONE));
    }
    /// (left gain, right gain) for a pan position.
    pub fn gains(position: T) -> (T, T) {
        let angle = (position + T::_ONE) * T::_PI / T::_lit(4.0);
        let (s, c) = angle._sin_cos();
        (c, s)
    }
    /// Writes `mono` panned into a stereo buffer (overwriting it; frames set to `mono.len()`).
    pub fn process(&mut self, mono: &[T], out: &mut AudioBuffer<T>) {
        out.set_frames(mono.len());
        let (left, right) = out.stereo_mut();
        for ((l, r), &x) in left.iter_mut().zip(right.iter_mut()).zip(mono) {
            let (gl, gr) = Self::gains(self.position.next_value());
            *l = x * gl;
            *r = x * gr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Biquad;
    use crate::modulation::ModulatedDelay;
    use crate::osc::Noise;

    const FS: f64 = 48_000.0;

    fn stereo_noise(frames: usize) -> AudioBuffer<f64> {
        let mut buf = AudioBuffer::new(2, frames);
        Noise::new(1).fill(buf.channel_mut(0));
        Noise::new(2).fill(buf.channel_mut(1));
        buf
    }

    #[test]
    fn interleave_roundtrip_and_frame_count() {
        let interleaved = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let mut buf = AudioBuffer::new(2, 8);
        buf.copy_from_interleaved(&interleaved);
        assert_eq!(buf.frames(), 3);
        assert_eq!(buf.channel(0), [1.0, 2.0, 3.0]);
        assert_eq!(buf.channel(1), [-1.0, -2.0, -3.0]);
        let mut back = [0.0; 6];
        buf.copy_to_interleaved(&mut back);
        assert_eq!(back, interleaved);
    }

    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn rejects_oversized_blocks() {
        AudioBuffer::<f32>::new(2, 4).copy_from_interleaved(&[0.0; 10]);
    }

    #[test]
    fn per_channel_matches_processing_each_channel_separately() {
        let mut buf = stereo_noise(1_000);
        let (mut l, mut r) = (buf.channel(0).to_vec(), buf.channel(1).to_vec());
        let mut fx = PerChannel::new(2, |ch| ModulatedDelay::chorus(FS).with_lfo_phase(ch as f64 * 0.5));
        fx.process(&mut buf);
        ModulatedDelay::chorus(FS).process(&mut l);
        ModulatedDelay::chorus(FS).with_lfo_phase(0.5).process(&mut r);
        assert_eq!(buf.channel(0), &l[..]);
        assert_eq!(buf.channel(1), &r[..]);
    }

    #[test]
    fn linked_compressor_turns_both_channels_down_equally() {
        let mut buf = AudioBuffer::new(2, 4_800);
        buf.channel_mut(0).iter_mut().for_each(|s| *s = 0.9); // loud left
        buf.channel_mut(1).iter_mut().for_each(|s| *s = 0.05); // quiet right
        let mut c = Compressor::new(FS);
        MultiProcessor::process(&mut c, &mut buf);
        for f in 0..buf.frames() {
            assert!((buf.channel(0)[f] / 0.9 - buf.channel(1)[f] / 0.05).abs() < 1e-12, "frame {f}");
        }
        assert!(buf.channel(1)[4_799] < 0.05 * 0.8, "quiet side is pulled down with the loud one");
    }

    #[test]
    fn width_zero_is_mono_and_one_is_transparent() {
        let original = stereo_noise(256);
        let mut same = original.clone();
        StereoWidth::new(1.0, FS).process(&mut same);
        assert_eq!(same.channel(0), original.channel(0));
        assert_eq!(same.channel(1), original.channel(1));

        let mut mono = original.clone();
        StereoWidth::new(0.0, FS).process(&mut mono);
        for f in 0..256 {
            let mid = (original.channel(0)[f] + original.channel(1)[f]) / 2.0;
            assert!((mono.channel(0)[f] - mid).abs() < 1e-15 && (mono.channel(1)[f] - mid).abs() < 1e-15);
        }
    }

    #[test]
    fn constant_power_pan_law() {
        for p in [-1.0, -0.5, 0.0, 0.3, 1.0] {
            let (l, r) = Panner::gains(p);
            assert!((l * l + r * r - 1.0f64).abs() < 1e-12, "power at {p}");
        }
        let (l, r) = Panner::gains(0.0f64);
        assert!((l - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12 && (l - r).abs() < 1e-12, "center is -3 dB each side");
        let (l, r) = Panner::gains(-1.0f64);
        assert!((l - 1.0).abs() < 1e-12 && r.abs() < 1e-12, "hard left");

        let mut out = AudioBuffer::new(2, 4);
        Panner::new(1.0, FS).process(&[1.0; 4], &mut out);
        assert!(out.channel(0).iter().all(|s| s.abs() < 1e-12) && out.channel(1).iter().all(|&s| (s - 1.0).abs() < 1e-12));
    }

    #[test]
    fn multichannel_tuple_chains_run_in_order() {
        let original = stereo_noise(512);
        let mut chained = original.clone();
        let mut chain = (PerChannel::new(2, |_| Biquad::lowpass(1_000.0, FS, 0.707)), StereoWidth::new(0.0, FS));
        chain.process(&mut chained);

        let mut manual = original;
        PerChannel::new(2, |_| Biquad::lowpass(1_000.0, FS, 0.707)).process(&mut manual);
        StereoWidth::new(0.0, FS).process(&mut manual);
        assert_eq!(chained.channel(0), manual.channel(0));
        assert_eq!(chained.channel(1), manual.channel(1));
    }
}
