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

use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::fmt::{Debug};
use crate::units::*;

mod extension;
pub mod adapters;
pub mod ops;

// GENERAL =========================================================================================

#[repr(transparent)]
pub struct Buffer<T> {
    data: Vec<T>
}

#[repr(transparent)]
pub struct BufferView<'a, T> {
    slice: &'a [T],
}

#[repr(transparent)]
pub struct BufferViewMut<'a, T> {
    slice: &'a mut [T],
}

/// Trait for general buffer applications : stored in program memory
pub trait IntoBuffer: PartialEq + Debug + IntoIterator + Index<usize> + AsRef<[Self::Item]> {
    type Item;
}

pub trait FromBuffer: IntoBuffer {}

/// Trait that provides static buffer operations
pub trait BufferOps<T: Buffer> {
    // associated adapter methods
    fn conv(input: &T, impulse: &T, output: &mut T, state: &mut T);
}

/// Signal traits : not stored in program memory
/// - use std::io as much as possible for this or computations
///     - specifically in std::io use BufRead (and by ext Read), Seek, and Write traits
///     - this enables "signals" to be used as reader or writers in io applications
/// Procedural signal (available immediately ; through computation)
pub trait ProcSignal {}

/// Streamed signal (not available immediately ; through i/o, wait for new data)
pub trait StreamSignal {}