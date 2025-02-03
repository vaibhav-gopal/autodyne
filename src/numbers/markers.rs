use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

// Private marker traits for QOL
pub(crate) trait Sz16 {}
pub(crate) trait Sz32 {}
pub(crate) trait Sz64 {}
pub(crate) trait FloatType: NumberType {}
pub(crate) trait FixedType: NumberType {}
pub(crate) trait NumberType: Add + Sub + Mul + Div + Neg + Copy + Debug + PartialEq + PartialOrd {}

impl NumberType for f32 {}
impl NumberType for f64 {}
impl NumberType for i16 {}
impl NumberType for i32 {}
impl NumberType for i64 {}
impl Sz32 for f32 {}
impl Sz64 for f64 {}
impl Sz16 for i16 {}
impl Sz32 for i32 {}
impl Sz64 for i64 {}
impl FloatType for f32 {}
impl FloatType for f64 {}
impl FixedType for i16 {}
impl FixedType for i32 {}
impl FixedType for i64 {}