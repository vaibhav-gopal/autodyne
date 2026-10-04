use std::ops::{Add, Mul};

use super::SourceStream;
use crate::processor::Processor;
use crate::units::*;

/// A procedural or streamed signal: produces samples on demand, forever or until the caller stops.
///
/// Sources compose lazily; nothing is computed until samples are pulled:
/// `Sine::new(220.0, fs).mix(Noise::new(1).scaled(0.1)).through(Biquad::lowpass(1_000.0, q, fs))`.
/// `fill` works a block at a time, so a source feeding processors via `through` runs each processor
/// on whole blocks, just like a hand-written processing loop.
pub trait Source {
    type Sample: Copy;

    /// Produces the next sample.
    fn next_sample(&mut self) -> Self::Sample;

    /// Overwrites `out` with the next `out.len()` samples.
    fn fill(&mut self, out: &mut [Self::Sample]) {
        for s in out {
            *s = self.next_sample();
        }
    }

    /// Adds the next `out.len()` samples onto `out` (mixing into an existing buffer).
    fn add_to(&mut self, out: &mut [Self::Sample])
    where
        Self::Sample: Add<Output = Self::Sample>,
    {
        for s in out {
            *s = *s + self.next_sample();
        }
    }

    /// Multiplies every sample by `gain`.
    fn scaled<G: Copy>(self, gain: G) -> Scaled<Self, G>
    where
        Self: Sized,
        Self::Sample: Mul<G, Output = Self::Sample>,
    {
        Scaled { source: self, gain }
    }

    /// Sums this source with another sample by sample.
    fn mix<S: Source<Sample = Self::Sample>>(self, other: S) -> Mixed<Self, S>
    where
        Self: Sized,
        Self::Sample: Add<Output = Self::Sample>,
    {
        Mixed { a: self, b: other }
    }

    /// Runs the source's output through a processor (or a tuple chain of them).
    fn through<P>(self, processor: P) -> Through<Self, P>
    where
        Self: Sized,
        Self::Sample: Float,
        P: Processor<Self::Sample>,
    {
        Through { source: self, processor }
    }

    /// An infinite `Iterator` over the samples, for use with the std adapters (`take`, `zip`, ...).
    fn samples(self) -> Samples<Self>
    where
        Self: Sized,
    {
        Samples(self)
    }

    /// This source as an endless `SignalRead` stream.
    fn stream(self) -> SourceStream<Self>
    where
        Self: Sized,
    {
        SourceStream::new(self, None)
    }

    /// This source as a `SignalRead` stream that ends after `samples` samples.
    fn stream_for(self, samples: u64) -> SourceStream<Self>
    where
        Self: Sized,
    {
        SourceStream::new(self, Some(samples))
    }
}

/// A source computed by a closure, e.g. `from_fn(move || { phase += step; phase.sin() })`.
pub fn from_fn<S: Copy, F: FnMut() -> S>(f: F) -> FromFn<F> {
    FromFn(f)
}

#[derive(Debug, Clone, Copy)]
pub struct FromFn<F>(F);

impl<S: Copy, F: FnMut() -> S> Source for FromFn<F> {
    type Sample = S;
    fn next_sample(&mut self) -> S {
        (self.0)()
    }
}

/// See [`Source::scaled`].
#[derive(Debug, Clone, Copy)]
pub struct Scaled<S, G> {
    source: S,
    gain: G,
}

impl<S: Source, G: Copy> Source for Scaled<S, G>
where
    S::Sample: Mul<G, Output = S::Sample>,
{
    type Sample = S::Sample;
    fn next_sample(&mut self) -> S::Sample {
        self.source.next_sample() * self.gain
    }
    fn fill(&mut self, out: &mut [S::Sample]) {
        self.source.fill(out);
        out.iter_mut().for_each(|s| *s = *s * self.gain);
    }
}

/// See [`Source::mix`].
#[derive(Debug, Clone, Copy)]
pub struct Mixed<A, B> {
    a: A,
    b: B,
}

impl<A: Source, B: Source<Sample = A::Sample>> Source for Mixed<A, B>
where
    A::Sample: Add<Output = A::Sample>,
{
    type Sample = A::Sample;
    fn next_sample(&mut self) -> A::Sample {
        self.a.next_sample() + self.b.next_sample()
    }
    fn fill(&mut self, out: &mut [A::Sample]) {
        self.a.fill(out);
        self.b.add_to(out);
    }
}

/// See [`Source::through`].
#[derive(Debug, Clone)]
pub struct Through<S, P> {
    source: S,
    processor: P,
}

impl<S, P> Through<S, P> {
    /// The processor, e.g. to change parameters while the source is running.
    pub fn processor_mut(&mut self) -> &mut P {
        &mut self.processor
    }
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

impl<S: Source, P: Processor<S::Sample>> Source for Through<S, P>
where
    S::Sample: Float,
{
    type Sample = S::Sample;
    /// Works, but processes a one-sample block; prefer `fill` with real block sizes.
    fn next_sample(&mut self) -> S::Sample {
        let mut one = [self.source.next_sample()];
        self.processor.process(&mut one);
        one[0]
    }
    fn fill(&mut self, out: &mut [S::Sample]) {
        self.source.fill(out);
        self.processor.process(out);
    }
}

/// See [`Source::samples`].
#[derive(Debug, Clone, Copy)]
pub struct Samples<S>(S);

impl<S: Source> Iterator for Samples<S> {
    type Item = S::Sample;
    fn next(&mut self) -> Option<S::Sample> {
        Some(self.0.next_sample())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::{Impulse, Noise, Phasor, Sine};
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::signal::Signal;

    const FS: f64 = 48_000.0;

    #[test]
    fn scaled_and_mixed_sources() {
        let mut s = from_fn(|| 1.0).scaled(0.5).mix(from_fn(|| 2.0));
        let mut buf = [0.0; 3];
        s.fill(&mut buf);
        assert_eq!(buf, [2.5; 3]);
        assert_eq!(s.next_sample(), 2.5);
    }

    #[test]
    fn through_equals_filling_then_processing() {
        let mut lazy = Sine::new(3_000.0, FS).mix(Noise::new(1).scaled(0.1)).through(Biquad::lowpass(1_000.0, BUTTERWORTH_Q, FS));
        let mut a = vec![0.0; 1_000];
        lazy.fill(&mut a[..600]);
        lazy.fill(&mut a[600..]); // block boundaries don't matter

        let mut b = vec![0.0; 1_000];
        Sine::new(3_000.0, FS).fill(&mut b);
        Noise::new(1).scaled(0.1).add_to(&mut b);
        Biquad::lowpass(1_000.0, BUTTERWORTH_Q, FS).process(&mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
        // and the low-pass did its job on the 3 kHz tone
        assert!(a[500..].rms().unwrap() < 0.1);
    }

    #[test]
    fn processors_can_be_retuned_while_running() {
        let mut src = from_fn(|| 1.0).through(crate::gain::Gain::new(1.0, 0.0, FS));
        assert_eq!(src.next_sample(), 1.0);
        src.processor_mut().set_gain(0.25);
        assert_eq!(src.next_sample(), 0.25);
    }

    #[test]
    fn sources_are_iterators_on_request() {
        let v: Vec<f64> = Impulse::new().samples().take(3).collect();
        assert_eq!(v, [1.0, 0.0, 0.0]);
        let z: Vec<Complex<f64>> = Phasor::new(0.0, FS).samples().take(2).collect();
        assert_eq!(z, [Complex::one(), Complex::one()]);
    }
}
