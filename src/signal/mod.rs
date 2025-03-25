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

use std::ops::{Add, Deref, DerefMut, Div, Index, Mul, Neg, Sub};
use std::fmt::{Debug};
use std::io::{BufRead, Read, Seek, Write};
use crate::units::*;

pub mod adapters;
pub mod ndarray;
pub mod buffer;

// GENERAL =========================================================================================

/// Signal trait : buffer backed data
/// - must guarantee immutable access to the underlying data via a slice
///     - must guarantee an iterator can be made on the underlying data
/// - must guarantee that there is an underlying buffer from which we can take a slice
///     - consequence : all streamed or procedural signals must be buffer-backed
pub trait Signal:
    Debug +
    IntoIterator<Item = Self::Sample> + 
    AsRef<[Self::Sample]> + 
    Deref<Target = [Self::Sample]>
{
    type Sample: Unit;
    fn view(&self) -> &[Self::Sample];
    fn len(&self) -> usize {
        self.view().len()
    }
}

/// Mutable Signal Trait
/// - must guarantee mutable (interior) access to the underlying data via a slice
pub trait SignalMut:
    Signal + 
    AsMut<[Self::Sample]> +
    DerefMut<Target = [Self::Sample]>
{
    fn view_mut(&mut self) -> &mut [Self::Sample];
}

/// Container Mutable Signal Trait
/// - must guarantee that access to the underlying buffer is permitted
///     - underlying buffer may or may not have interior mutability, so we choose worst case (SignalMut bound)
/// - must guarantee there is ownership of the underlying buffer
pub trait SignalOwned: SignalMut {
    type Container;
    fn as_container(&self) -> &Self::Container;
    fn as_container_mut(&mut self) -> &mut Self::Container;
    fn into_container(self) -> Self::Container;
}

/// Resizeable Container Signal Trait
pub trait SignalResizable: SignalOwned {
    fn resize(&mut self, new_len: usize, fill_value: Self::Sample);
    fn clear(&mut self);
    fn append(&mut self, value: Self::Sample);
}

/// Main Signal Operations Trait
/// - guarantees element-wise arithmetic via op overload
/// - guarantees single-value broadcast via op overload
pub trait SignalOps: Signal + Add + Mul + Div + Sub + Neg {
    fn try_add();
    fn try_mul();
    fn try_div();
    fn try_sub();
    fn try_neg();
}

/// Procedurally Generated or Streamed Signal
pub trait SignalStream: Signal + BufRead + Read + Seek + Write {}