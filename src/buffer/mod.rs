/// again using extension traits here
/// I think I want to base buffer manipulation around the adapter pattern and iterator pattern
/// I also think I want to represent signals as infinite iterators ( does not implement ExactSizeIterator)
/// todo:
/// - find a way to represent signals (procedural iterator):
///     - map on Iterator already emulates signals ( maybe make an alias for this or something )
///     - and Successors & Cycle already represents procedural/infinite values ; but would only return 1 value
///     - maybe introduce zip to pack the independent and dependant arguments (rather than just dependant) in the signal as an alternative extended type
/// - find a way to represent basic DSP functions (as iterator adapters and traits? or as plain functions?)


use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt::{Debug};
use crate::units::*;

mod extension;
pub mod adapters;
pub mod signals;

// GENERAL =========================================================================================
/// Trait for general buffer applications
pub trait Buffer: PartialEq + Debug + IntoIterator {}

/// Trait for identifying which types can split into chunks for block processing (real-time)   
pub trait ChunkableBuffer: Buffer {}