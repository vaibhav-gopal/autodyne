use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub};
use super::*;
use super::markers::*;

/// Real (floating-point) number type
#[repr(transparent)]
pub struct RealF<T: FloatType>(T);
pub type RealF32 = RealF<f32>;
pub type RealF64 = RealF<f64>;
impl<T: FloatType> RealF<T> {
    pub fn new(val: T) -> Self {
        Self(val)
    }
}
/// Generic Number implementation for float types
/// TODO: instead of using explicit float primitives, define trait bounds to require From and Into using f32 for example instead
impl<T: FloatType> Number for RealF<T> {
    fn zero() -> Self {
        RealF::new(0.0)?
    }

    fn one() -> Self {
        RealF::new(1.0)?
    }
}
impl<T: FloatType> Clone for RealF<T> {
    fn clone(&self) -> Self {
        self.0.clone()
    }
}

impl<T: FloatType> Copy for RealF<T> {}

impl<T: FloatType> Debug for RealF<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// FLOAT OPERATOR OVERLOADS

impl<T: FloatType> Add for RealF<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.0 + rhs.0
    }
}
impl<T: FloatType> Sub for RealF<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}
impl<T: FloatType> Mul for RealF<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.0 * rhs.0
    }
}
impl<T: FloatType> Div for RealF<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.0 / rhs.0
    }
}
impl<T: FloatType> Neg for RealF<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -self.0
    }
}
impl<T: FloatType> PartialEq for RealF<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(other.0)
    }
}
impl<T: FloatType> PartialOrd for RealF<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(other.0)
    }
}
