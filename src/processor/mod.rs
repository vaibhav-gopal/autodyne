//! The shared interface of in-place block processors, and ways to chain them.
//!
//! Every effect/filter (`Biquad`, `Fir`, `Gain`, `Echo`, `Compressor`, ...) implements `Processor`.
//! They keep their inherent `process` / `reset` methods too, so the trait only needs importing for
//! generic code and chains.
//!
//! - Static chain: a tuple, e.g. `(low_shelf, compressor, echo).process(block)` runs them in order,
//!   fully inlined, no allocation.
//! - Dynamic chain: `Vec<Box<dyn Processor<T>>>`, for chains assembled at runtime.

use crate::units::*;

/// A stateful processor that transforms a block of samples in place.
/// Implementations must not allocate in `process`, so they can run inside an audio callback.
pub trait Processor<T: Float> {
    /// Processes `block` in place.
    fn process(&mut self, block: &mut [T]);
    /// Clears internal state (filter memory, delay contents, envelopes) without changing parameters.
    fn reset(&mut self);
}

impl<T: Float, P: Processor<T> + ?Sized> Processor<T> for Box<P> {
    fn process(&mut self, block: &mut [T]) {
        (**self).process(block)
    }
    fn reset(&mut self) {
        (**self).reset()
    }
}

impl<T: Float, P: Processor<T> + ?Sized> Processor<T> for &mut P {
    fn process(&mut self, block: &mut [T]) {
        (**self).process(block)
    }
    fn reset(&mut self) {
        (**self).reset()
    }
}

/// Runs each processor in order (a dynamic chain when `P = Box<dyn Processor<T>>`).
impl<T: Float, P: Processor<T>> Processor<T> for Vec<P> {
    fn process(&mut self, block: &mut [T]) {
        for p in self.iter_mut() {
            p.process(block);
        }
    }
    fn reset(&mut self) {
        self.iter_mut().for_each(Processor::reset);
    }
}

/// Tuples of processors are chains: element 0 runs first.
macro_rules! impl_chain {
    ($($P:ident . $i:tt),+) => {
        impl<T: Float, $($P: Processor<T>),+> Processor<T> for ($($P,)+) {
            fn process(&mut self, block: &mut [T]) {
                $(self.$i.process(block);)+
            }
            fn reset(&mut self) {
                $(self.$i.reset();)+
            }
        }
    };
}

impl_chain!(A.0);
impl_chain!(A.0, B.1);
impl_chain!(A.0, B.1, C.2);
impl_chain!(A.0, B.1, C.2, D.3);
impl_chain!(A.0, B.1, C.2, D.3, E.4);
impl_chain!(A.0, B.1, C.2, D.3, E.4, F.5);
impl_chain!(A.0, B.1, C.2, D.3, E.4, F.5, G.6);
impl_chain!(A.0, B.1, C.2, D.3, E.4, F.5, G.6, H.7);

/// Implements `Processor` by forwarding to a type's inherent `process` and `reset` (invoked next
/// to each processor type, so this module depends on nothing but `units`).
macro_rules! forward_processor {
    ($($Ty:ident),+) => {$(
        impl<T: $crate::units::Float> $crate::processor::Processor<T> for $Ty<T> {
            fn process(&mut self, block: &mut [T]) {
                $Ty::process(self, block)
            }
            fn reset(&mut self) {
                $Ty::reset(self)
            }
        }
    )+};
}
pub(crate) use forward_processor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delay::Echo;
    use crate::filter::{Biquad, Fir, BUTTERWORTH_Q};
    use crate::gain::Gain;
    use crate::osc::Noise;

    fn noise(len: usize) -> Vec<f64> {
        Noise::new(1).take(len).collect()
    }

    #[test]
    fn tuple_chain_equals_running_each_in_order() {
        let fs = 48_000.0;
        let make = || (Biquad::lowpass(2_000.0, BUTTERWORTH_Q, fs), Gain::new(0.5, 0.0, fs), Biquad::highpass(100.0, BUTTERWORTH_Q, fs));
        let input = noise(1_000);

        let mut chained = input.clone();
        make().process(&mut chained);

        let (mut a, mut b, mut c) = make();
        let mut manual = input;
        a.process(&mut manual);
        b.process(&mut manual);
        c.process(&mut manual);
        assert_eq!(chained, manual);
    }

    #[test]
    fn dynamic_chain_equals_static_chain() {
        let fs = 48_000.0;
        let input = noise(1_000);

        let mut stat = input.clone();
        (Biquad::peaking(1_000.0, 1.0, 6.0, fs), Fir::lowpass(5_000.0, 31, fs)).process(&mut stat);

        let mut chain: Vec<Box<dyn Processor<f64>>> =
            vec![Box::new(Biquad::peaking(1_000.0, 1.0, 6.0, fs)), Box::new(Fir::lowpass(5_000.0, 31, fs))];
        let mut dynamic = input;
        chain.process(&mut dynamic);
        assert_eq!(stat, dynamic);
    }

    #[test]
    fn reset_reaches_every_stage() {
        let fs = 48_000.0;
        let mut chain = (Biquad::lowpass(1_000.0, BUTTERWORTH_Q, fs), Echo::new(0.1, fs));
        let mut first = noise(2_000);
        let input = first.clone();
        chain.process(&mut first);
        chain.reset();
        let mut second = input;
        chain.process(&mut second);
        assert_eq!(first, second, "after reset the chain must behave like a fresh one");
    }

    #[test]
    fn generic_code_accepts_any_processor() {
        fn run<P: Processor<f64>>(mut p: P, block: &mut [f64]) {
            p.process(block);
        }
        let mut block = [1.0, 2.0];
        run(Gain::new(3.0, 0.0, 48_000.0), &mut block);
        assert_eq!(block, [3.0, 6.0]);
        let mut g = Gain::new(0.5, 0.0, 48_000.0);
        run(&mut g, &mut block); // borrowed processors work too
        assert_eq!(block, [1.5, 3.0]);
    }
}
