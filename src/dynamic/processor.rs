//! Processors whose sample type is chosen at runtime.
//!
//! A host that learns the sample format at runtime (from a device, a file header, an ML runtime)
//! builds processors with [`build_dyn`] from a [`ProcessorFactory`] and drives them through
//! [`DynProcessor`]: object-safe, parameterized, and checked against the data's `DType` instead of
//! panicking on a mismatch.

use std::marker::PhantomData;

use super::{DynArray, DynElement, DynError};
use crate::params::{ParamError, ParamInfo, Parameterized};
use crate::processor::Processor;
use crate::units::*;

/// A block of samples in a float type chosen at runtime.
#[derive(Debug)]
pub enum DynBlock<'a> {
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
}

impl DynBlock<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            DynBlock::F32(_) => DType::F32,
            DynBlock::F64(_) => DType::F64,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            DynBlock::F32(b) => b.len(),
            DynBlock::F64(b) => b.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DynArray {
    /// All elements as one block, for float arrays.
    pub fn as_block_mut(&mut self) -> Result<DynBlock<'_>, DynError> {
        match self {
            DynArray::F32(a) => Ok(DynBlock::F32(a.as_mut_slice())),
            DynArray::F64(a) => Ok(DynBlock::F64(a.as_mut_slice())),
            other => Err(DynError::Unsupported(other.dtype())),
        }
    }
}

/// The float types a `DynProcessor` can run on.
pub trait FloatElement: Float + DynElement {
    #[doc(hidden)]
    fn from_block(block: DynBlock<'_>) -> Result<&mut [Self], DType>;
    #[doc(hidden)]
    fn into_block(samples: &mut [Self]) -> DynBlock<'_>;
}

impl FloatElement for f32 {
    fn from_block(block: DynBlock<'_>) -> Result<&mut [f32], DType> {
        match block {
            DynBlock::F32(b) => Ok(b),
            other => Err(other.dtype()),
        }
    }
    fn into_block(samples: &mut [f32]) -> DynBlock<'_> {
        DynBlock::F32(samples)
    }
}

impl FloatElement for f64 {
    fn from_block(block: DynBlock<'_>) -> Result<&mut [f64], DType> {
        match block {
            DynBlock::F64(b) => Ok(b),
            other => Err(other.dtype()),
        }
    }
    fn into_block(samples: &mut [f64]) -> DynBlock<'_> {
        DynBlock::F64(samples)
    }
}

/// An object-safe processor with a runtime sample type and runtime-accessible parameters.
pub trait DynProcessor: Parameterized + Send {
    /// The sample type this processor runs on.
    fn dtype(&self) -> DType;
    /// Processes a block in place; errors if its type isn't `dtype()`.
    fn process_dyn(&mut self, block: DynBlock<'_>) -> Result<(), DynError>;
    fn reset(&mut self);

    /// Processes every lane along `axis` of a runtime-typed array (e.g. each clip of a
    /// `[batch, time]` tensor along time) as a separate, complete signal: state is reset before each
    /// lane, so lanes don't leak into each other. For streaming multichannel audio, where state must
    /// carry over between blocks, use one processor per channel (`channels::PerChannel`) instead.
    fn process_lanes(&mut self, array: &mut DynArray, axis: usize) -> Result<(), DynError> {
        let mut result = Ok(());
        let mut run = |block: DynBlock<'_>| {
            if result.is_ok() {
                self.reset();
                result = self.process_dyn(block);
            }
        };
        match array {
            DynArray::F32(a) => a.for_each_lane(axis, |lane| run(DynBlock::F32(lane)))?,
            DynArray::F64(a) => a.for_each_lane(axis, |lane| run(DynBlock::F64(lane)))?,
            other => return Err(DynError::Unsupported(other.dtype())),
        }
        result
    }
}

/// Wraps a concrete processor for sample type `T` as a `DynProcessor`.
#[derive(Debug, Clone)]
pub struct Typed<P, T> {
    processor: P,
    // fn() -> T: carries the type without affecting Send / Sync
    _sample: PhantomData<fn() -> T>,
}

impl<P, T> Typed<P, T> {
    pub fn new(processor: P) -> Self {
        Self { processor, _sample: PhantomData }
    }
    pub fn inner(&self) -> &P {
        &self.processor
    }
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.processor
    }
    pub fn into_inner(self) -> P {
        self.processor
    }
}

impl<T: FloatElement, P: Processor<T> + Parameterized + Send> DynProcessor for Typed<P, T> {
    fn dtype(&self) -> DType {
        T::DTYPE
    }
    fn process_dyn(&mut self, block: DynBlock<'_>) -> Result<(), DynError> {
        let samples = T::from_block(block).map_err(|found| DynError::DTypeMismatch { expected: T::DTYPE, found })?;
        self.processor.process(samples);
        Ok(())
    }
    fn reset(&mut self) {
        Processor::reset(&mut self.processor);
    }
}

impl<P: Parameterized, T> Parameterized for Typed<P, T> {
    fn param_count(&self) -> usize {
        self.processor.param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.processor.param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.processor.param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.processor.get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        self.processor.set_param(index, value)
    }
}

/// A runtime-built chain: every stage must have the chain's sample type.
impl DynProcessor for Vec<Box<dyn DynProcessor>> {
    /// The first stage's type (`F32` for an empty chain).
    fn dtype(&self) -> DType {
        self.first().map_or(DType::F32, |p| p.dtype())
    }
    fn process_dyn(&mut self, block: DynBlock<'_>) -> Result<(), DynError> {
        // re-borrow the block for each stage by matching once on its type
        match block {
            DynBlock::F32(b) => self.iter_mut().try_for_each(|p| p.process_dyn(DynBlock::F32(b))),
            DynBlock::F64(b) => self.iter_mut().try_for_each(|p| p.process_dyn(DynBlock::F64(b))),
        }
    }
    fn reset(&mut self) {
        self.iter_mut().for_each(|p| p.reset());
    }
}

/// Describes how to build a processor for any float sample type, so the type can be picked at
/// runtime with [`build_dyn`].
///
/// ```
/// use autodyne::dynamic::{build_dyn, FloatElement, ProcessorFactory};
/// use autodyne::dynamics::Compressor;
/// use autodyne::params::Parameterized;
/// use autodyne::units::DType;
///
/// struct Comp;
/// impl ProcessorFactory for Comp {
///     type Output<T: FloatElement> = Compressor<T>;
///     fn build<T: FloatElement>(&self, sample_rate: T) -> Compressor<T> {
///         Compressor::new(sample_rate)
///     }
/// }
///
/// let mut p = build_dyn(&Comp, DType::F64, 48_000.0).unwrap();
/// p.set_param_by_id("slope", 0.125).unwrap(); // 8:1
/// assert_eq!(p.dtype(), DType::F64);
/// ```
pub trait ProcessorFactory {
    type Output<T: FloatElement>: Processor<T> + Parameterized + Send + 'static;
    fn build<T: FloatElement>(&self, sample_rate: T) -> Self::Output<T>;
}

/// Builds the factory's processor for a sample type chosen at runtime (`F32` or `F64`).
pub fn build_dyn<F: ProcessorFactory>(factory: &F, dtype: DType, sample_rate: f64) -> Result<Box<dyn DynProcessor>, DynError> {
    match dtype {
        DType::F32 => Ok(Box::new(Typed::<_, f32>::new(factory.build(sample_rate as f32)))),
        DType::F64 => Ok(Box::new(Typed::<_, f64>::new(factory.build(sample_rate)))),
        other => Err(DynError::Unsupported(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::Compressor;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::osc::Noise;
    use crate::signal::NdArray;

    struct LowpassThenCompress;

    impl ProcessorFactory for LowpassThenCompress {
        type Output<T: FloatElement> = (Biquad<T>, Compressor<T>);
        fn build<T: FloatElement>(&self, sample_rate: T) -> Self::Output<T> {
            (Biquad::lowpass(T::_lit(2_000.0), T::_lit(BUTTERWORTH_Q), sample_rate), Compressor::new(sample_rate))
        }
    }

    #[test]
    fn runtime_chosen_type_matches_static_processing() {
        let input: Vec<f64> = Noise::new(3).take(2_000).collect();
        let mut expected = input.clone();
        LowpassThenCompress.build(48_000.0).process(&mut expected);

        let mut dynamic = build_dyn(&LowpassThenCompress, DType::F64, 48_000.0).unwrap();
        let mut got = input.clone();
        dynamic.process_dyn(DynBlock::F64(&mut got)).unwrap();
        assert_eq!(got, expected);

        // the same factory, f32 chosen at runtime
        let mut single = build_dyn(&LowpassThenCompress, DType::F32, 48_000.0).unwrap();
        let mut got32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        single.process_dyn(DynBlock::F32(&mut got32)).unwrap();
        assert!(got32.iter().zip(&expected).all(|(a, b)| (*a as f64 - b).abs() < 1e-4));
    }

    #[test]
    fn type_mismatch_is_an_error() {
        let mut p = build_dyn(&LowpassThenCompress, DType::F32, 48_000.0).unwrap();
        let mut block = [0.0f64; 4];
        assert_eq!(p.process_dyn(DynBlock::F64(&mut block)), Err(DynError::DTypeMismatch { expected: DType::F32, found: DType::F64 }));
        assert!(build_dyn(&LowpassThenCompress, DType::I16, 48_000.0).is_err());
    }

    #[test]
    fn parameters_through_the_trait_object() {
        let mut p = build_dyn(&LowpassThenCompress, DType::F32, 48_000.0).unwrap();
        assert_eq!(p.param_count(), 3 + 6);
        assert_eq!(p.param_group(0), Some("Biquad"));
        p.set_param_by_id("threshold_db", -30.0).unwrap();
        assert_eq!(p.get_param_by_id("threshold_db"), Some(-30.0));
    }

    #[test]
    fn lanes_of_a_runtime_typed_array_and_runtime_chains() {
        let mut array = DynArray::from_array(NdArray::from_fn(&[2, 1_000], |i| if i[1] == 0 { 1.0f32 } else { 0.0 }).unwrap());
        let mut chain: Vec<Box<dyn DynProcessor>> = vec![
            build_dyn(&LowpassThenCompress, DType::F32, 48_000.0).unwrap(),
            build_dyn(&LowpassThenCompress, DType::F32, 48_000.0).unwrap(),
        ];
        chain.process_lanes(&mut array, 1).unwrap();
        let a = array.as_array::<f32>().unwrap();
        // both channels got the same impulse, so the same response
        assert_eq!(a.view().index_axis(0, 0).unwrap().to_vec(), a.view().index_axis(0, 1).unwrap().to_vec());
        assert_eq!(chain.param_count(), 18);

        let mut ints = DynArray::zeros(DType::I16, &[4]).unwrap();
        assert_eq!(chain.process_lanes(&mut ints, 0), Err(DynError::Unsupported(DType::I16)));
    }
}
