use std::fmt::Formatter;
use super::*;

/// Real (fixed-point) number type
#[repr(transparent)]
pub struct Real<T: FloatType>(T);
pub type RealF32 = Real<f32>;
pub type RealF64 = Real<f64>;

// GENERAL IMPLEMENTATIONS =========================================================================

impl<T: FloatType> Real<T> {
    pub fn new(val: T) -> Self {
        Self(val)
    }
}
impl<T: FloatType> Clone for Real<T> {
    fn clone(&self) -> Self {
        Real(self.0.clone())
    }
}
impl<T: FloatType> Copy for Real<T> {}
impl<T: FloatType> Debug for Real<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// UNIT IMPLEMENTATIONS ============================================================================

// impl<T: FloatType> Unit for RealF<T> {}
// impl<T: FloatType> PrimitiveUnit for RealF<T> {}
// 
// impl<T: FloatType> IntUnit for RealF<T> {
//     fn zero() -> Self {
//         RealF::new(0)
//     }
//     fn one() -> Self {
//         RealF::new(1)
//     }
//     fn pow(self, n: &impl IntUnit) -> Self {
//         todo!()
//     }
//     fn signum(self) -> Self {
//         todo!()
//     }
//     fn abs(self) -> Self {
//         todo!()
//     }
//     fn max(self, other: &impl IntUnit) -> Self {
//         todo!()
//     }
//     fn min(self, other: &impl IntUnit) -> Self {
//         todo!()
//     }
//     fn clamp(self, min: &impl IntUnit, max: &impl IntUnit) -> Self {
//         todo!()
//     }
// }
// 
// impl<T: FloatType + Sz32> RealUnit for RealF<T> {
//     const MANTISSA_DIGITS: u32 = 8;
//     const EPSILON: Self = RealF(0);
//     const INFINITY: Self = RealF(0);
//     const NAN: Self = RealF(0);
//     const NEG_INFINITY: Self = RealF(0);
// }
// impl<T: FixedType + Sz64> RealUnit for RealF<T> {
//     const MANTISSA_DIGITS: u32 = 16;
//     const EPSILON: Self = RealF(0);
//     const INFINITY: Self = RealF(0);
//     const NAN: Self = RealF(0);
//     const NEG_INFINITY: Self = RealF(0);
// }

// FLOAT OPERATOR OVERLOADS

// impl<T: FloatType> Add for RealF<T> {
//     type Output = Self;
//     fn add(self, rhs: Self) -> Self::Output {
//         self.0 + rhs.0
//     }
// }
// impl<T: FloatType> Sub for RealF<T> {
//     type Output = Self;
//     fn sub(self, rhs: Self) -> Self::Output {
//         self.0 - rhs.0
//     }
// }
// impl<T: FloatType> Mul for RealF<T> {
//     type Output = Self;
//     fn mul(self, rhs: Self) -> Self::Output {
//         self.0 * rhs.0
//     }
// }
// impl<T: FloatType> Div for RealF<T> {
//     type Output = Self;
//     fn div(self, rhs: Self) -> Self::Output {
//         self.0 / rhs.0
//     }
// }
// impl<T: FloatType> Neg for RealF<T> {
//     type Output = Self;
//     fn neg(self) -> Self::Output {
//         -self.0
//     }
// }
// impl<T: FloatType> PartialEq for RealF<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.0.eq(other.0)
//     }
// }
// impl<T: FloatType> PartialOrd for RealF<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         self.0.partial_cmp(other.0)
//     }
// }
