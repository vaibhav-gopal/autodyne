/// again using extension traits here
/// I think I want to base buffer manipulation around the adapter pattern and iterator pattern
/// I also think I want to represent signals as infinite iterators ( does not implement ExactSizeIterator)

use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt::{Debug};

use crate::units::*;

pub mod extension;
pub mod adapters;

// GENERAL =========================================================================================
/// Trait for general buffer applications
pub trait Buffer: PartialEq + Debug + IntoIterator {}

/// Trait for identifying which types can split into chunks for block processing (real-time)   
pub trait ChunkableBuffer: Buffer {}