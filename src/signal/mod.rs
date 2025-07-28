/// again using extension traits here
/// I think I want to base buffer manipulation around the adapter pattern and iterator pattern
/// I also think I want to represent signals as infinite iterators ( does not implement ExactSizeIterator)
/// todo:
/// - find a way to represent signals (procedural iterator):
///     - map on Iterator already emulates functions/signals ( maybe make an alias for this or something )
///     - and Successors & Cycle already represents procedural/infinite values ; but would only return 1 value
///     - maybe introduce zip to pack the independent and dependant arguments (rather than just dependant) in the signal as an alternative extended type
/// - find a way to represent basic DSP functions (as iterator adapters and traits? or as plain functions?)
/// - find a way to represent chunkable buffers (for real-time applications), Iterators have next_chunk as a nightly feature and there also iterators with the specific feature of chunking in the iter_tools crate
///     - should I just use the iter_tools for chunking? (only problem is that it provides so many blanket impl that I don't need)
///     - maybe we don't even need this... since chunking is only really only useful for streaming data into some buffer continuously via another thread while reading it

use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::fmt::{Debug};
use std::io::{BufRead, Read, Seek, Write};
use crate::units::*;
use thiserror::Error;

pub mod adapters;
pub mod ndarray;
pub mod buffer;

// GENERAL =========================================================================================

/// Signal trait : buffer backed data
/// - must guarantee immutable access to the underlying data via as_ref
/// - must guarantee immutable access to the underlying container via as_container
/// - must guarantee an iterator can be made on the underlying data
/// - must guarantee that there is an underlying buffer from which we can take a slice
///     - consequence : all streamed or procedural signals must be buffer-backed
pub trait Signal:
    Debug +
    IntoIterator<Item = Self::Sample> + 
    FromIterator<Self::Sample> +
    AsRef<[Self::Sample]> + 
    Deref<Target = [Self::Sample]> +
    From<&[Self::Sample]> + From<Self::Container>
{
    type Sample: Unit;
    type Container: FromIterator<Self::Sample>;
    fn new(data: Self::Container) -> Self {
        Self::from(data)
    }
    fn new_copy<T: AsRef<[Self::Sample]>>(data: T) -> Self {
        Self::from(data.as_ref())
    }
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    fn min(&self) -> Self::Sample;
    fn max(&self) -> Self::Sample;
    fn argmax(&self) -> usize;
    fn argmin(&self) -> usize;
}

/// Mutable Signal Trait
/// - must guarantee mutable (interior) access to the underlying data via a slice
pub trait SignalMut:
    Signal + 
    AsMut<[Self::Sample]> +
    DerefMut<Target = [Self::Sample]>
{
}

/// Container Mutable Signal Trait
/// - must guarantee that access to the underlying buffer is permitted
///     - underlying buffer may or may not have interior mutability, so we choose worst case (SignalMut bound)
/// - must guarantee there is ownership of the underlying buffer
pub trait SignalOwned: SignalMut {
    fn into_container(self) -> Self::Container;
    fn as_container(&self) -> &Self::Container;
    fn as_container_mut(&mut self) -> &mut Self::Container;
}

/// Resizeable Container Signal Trait
pub trait SignalResizable: SignalOwned {
    fn resize(&mut self, new_len: usize, fill_value: Self::Sample);
    fn clear(&mut self);
    fn append(&mut self, value: Self::Sample);
}

/// Signal to Scalar Operations Trait
/// - guarantees scalar arithmetic via op overload
/// - always &self since output is scalar
pub trait ScalarOps: Signal + Add<Self::Sample> + Mul<Self::Sample> + Div<Self::Sample> + Sub<Self::Sample> + Neg {
    fn inner(&self, other: impl AsRef<[Self::Sample]>) -> Result<Self::Sample, SigOpsError>;
    fn angle(&self, other: impl AsRef<[Self::Sample]>) -> Result<Self::Sample, SigOpsError>;
    fn norm_l2(&self) -> Self::Sample;
    fn norm_l1(&self) -> Self::Sample;
    fn norm_inf(&self) -> Self::Sample;
    fn mean(&self) -> Self::Sample;
}

/// Signal to Signal Operations Trait
/// - guarantees point-wise buffer arithmetic via trait methods
/// - always &mut self due to output being a signal as well (if copy wanted, clone beforehand)
pub trait SignalOps: SignalMut + AddAssign + SubAssign + MulAssign + DivAssign {
    fn sig_add<T: AsRef<[Self::Sample]>>(&mut self, rhs: T) -> Result<(), SigOpsError>;
    fn sig_sub<T: AsRef<[Self::Sample]>>(&mut self, rhs: T) -> Result<(), SigOpsError>;
    fn sig_mul<T: AsRef<[Self::Sample]>>(&mut self, rhs: T) -> Result<(), SigOpsError>;
    fn sig_div<T: AsRef<[Self::Sample]>>(&mut self, rhs: T) -> Result<(), SigOpsError>;
    fn project(&mut self, other: &Self) -> Result<(), SigOpsError>;
    fn convolve(&mut self, other: &Self) -> Result<(), SigOpsError>;
    fn resample(&mut self, factor: Self::Sample) -> Result<(), SigOpsError>;
    fn diff(&mut self);
    fn sum(&mut self);
}

pub trait SigOwnedOps: SignalOwned {}
pub trait SigResizeOps: SignalResizable {}

pub enum SigBroadcast {
    Scalar(),
    ZeroPad(),
    Fill(),
    Latch(),
    Tile(),
}

#[derive(Error, Debug)]
pub enum SigOpsError {
    #[error("Buffers are not the same size! {0} != {1}")]
    BufferMismatch(usize, usize),
}

/// Procedurally Generated or Streamed Signal
pub trait SignalStream: Signal + BufRead + Read + Seek + Write {}