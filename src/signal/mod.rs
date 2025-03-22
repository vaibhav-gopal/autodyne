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
use std::io::{BufRead, Read, Seek, Write};
use crate::units::*;

pub mod adapters;
pub mod buffer;

// GENERAL =========================================================================================

/// Signal trait
/// - must guarantee non-mutable access to an underlying buffer via a slice
pub trait Signal: 
    IntoIterator<Item = Self::Sample> + 
    Deref<Target = [Self::Sample]> +
    Debug
{
    type Sample: PrimitiveUnit;
    fn view(&self) -> &[Self::Sample];
}

/// Mutable Signal Trait
/// - guarantees mutable access to an underlying buffer via a slice
pub trait SignalMut:
    Signal + 
    DerefMut<Target = [Self::Sample]>
{
    fn view_mut(&mut self) -> &mut [Self::Sample];
}

/// 
pub trait SignalContainer:
    Signal
{
    
    fn as_container(&self) -> 
}

/// Main Signal Operations Trait
pub trait SignalOps: Signal {}

/// Signal input/output trait for io bound signal operations
pub trait SignalIO: Signal + BufRead + Read + Seek + Write {}

/// Signal conversion trait
pub trait IntoSignal<T: Signal>: IntoIterator {
    fn into_signal(self) -> T;
}
pub trait FromSignal<T: Signal> {
    fn from_signal(signal: T) -> Self;
}