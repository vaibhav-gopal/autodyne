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
/// - use std::io as much as possible for this or computations
///     - specifically in std::io use BufRead (and by ext Read), Seek, and Write traits
///     - this enables "signals" to be used as reader or writers in io applications
/// - maybe have Deref and DerefMut as supertraits as well
/// - or AsRef and AsMut
/// - separate arithmetic operations into a separate SignalOps trait
pub trait Signal: IntoIterator {
    type View;
    fn view(&self) -> Self::View;
    fn len(&self) -> usize {
        self.view().len()
    }
}

/// Mutable Signal Trait
pub trait SignalMut: Signal {
    type ViewMut;
    fn view_mut(&mut self) -> Self::ViewMut;
}

/// Main Signal Operations Trait
pub trait SignalOps: Signal {}

/// Signal
pub trait SignalIO: Signal + BufRead + Read + Seek + Write {}

/// Signal conversion trait
pub trait IntoSignal<T: Signal>: IntoIterator {
    fn into_signal(self) -> T;
}
pub trait FromSignal<T: Signal> {
    fn from_signal(signal: T) -> Self;
}