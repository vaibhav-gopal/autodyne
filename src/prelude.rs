//! Everything most code needs in one import:
//!
//! ```
//! use autodyne::prelude::*;
//! ```
//!
//! Brings the traits into scope (so their methods work on slices, vectors, arrays, processors and
//! streams) plus the most common types. Importing them all at once is safe: every type implements at
//! most one of `Processor` / `MultiProcessor` (multichannel linking goes through `Linked`), and the
//! signal traits apply to disjoint types, so method calls are never ambiguous.

pub use crate::channels::{AudioBuffer, Linked, MultiProcessor, PerChannel};
pub use crate::dynamic::{DynArray, DynProcessor};
pub use crate::params::{ParamInfo, Parameterized};
pub use crate::processor::Processor;
pub use crate::resample::Resample;
pub use crate::signal::{
    Axis, Broadcast, ComplexSignal, NdArray, SigOwnedOps, SigResizeOps, Signal, SignalMut, SignalOwned, SignalRead,
    SignalResizable, SignalSeek, SignalStream, SignalWrite, Source,
};
pub use crate::units::{Complex, DType, Float, Reflection};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::Compressor;
    use crate::osc::Sine;

    /// Every call here would fail to compile if two prelude traits claimed the same method.
    #[test]
    fn prelude_methods_are_unambiguous() {
        let fs = 48_000.0;

        let mut v: Vec<f64> = Sine::new(1_000.0, fs).samples().take(480).collect();
        v.normalize_peak(1.0);
        assert!(v.rms().unwrap() > 0.7 && v.samples().len() == 480);
        let _ = v.scaled_by(0.5);
        v.pad(0, 2, 0.0);

        let z = [Complex::new(3.0f64, 4.0)];
        assert_eq!(z.energy(), 25.0);
        assert_eq!(z.peak(), 5.0);

        // a chain of mono compressors is only a Processor ...
        let mut chain = (Compressor::new(fs), Compressor::new(fs));
        chain.process(&mut v);
        chain.reset();
        // ... and linked compression is only a MultiProcessor
        let mut buf = AudioBuffer::new(2, 64);
        let mut linked = (Linked(Compressor::new(fs)), PerChannel::new(2, |_| Compressor::new(fs)));
        linked.process(&mut buf);
        linked.reset();
        assert_eq!(linked.param_count(), 12);

        let mut arr = NdArray::from_vec(vec![1.0f32, -2.0], &[2]).unwrap();
        assert_eq!(arr.peak(), 2.0);
        arr.scale(0.5);
        assert_eq!(<f32 as Reflection>::DTYPE, DType::F32);
    }
}
