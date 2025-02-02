use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

// Private marker traits for QOL
pub trait Sz16 {}
pub trait Sz32 {}
pub trait Sz64 {}
pub trait Sz128 {}
pub trait FloatType: NumberType {}
pub trait FixedType: NumberType {}
pub trait NumberType: Add + Sub + Mul + Div + Neg + Copy + Debug + PartialEq + PartialOrd {}

impl NumberType for f32 {}
impl NumberType for f64 {}
impl NumberType for i16 {}
impl NumberType for i32 {}
impl NumberType for i64 {}
impl NumberType for i128 {}
impl Sz32 for f32 {}
impl Sz64 for f64 {}
impl Sz16 for i16 {}
impl Sz32 for i32 {}
impl Sz64 for i64 {}
impl Sz128 for i128 {}
impl FloatType for f32 {}
impl FloatType for f64 {}
impl FixedType for i16 {}
impl FixedType for i32 {}
impl FixedType for i64 {}
impl FixedType for i128 {}