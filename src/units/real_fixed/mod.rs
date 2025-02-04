use std::fmt::Formatter;
/// Using extension traits compared to newtypes ; extending the primitive types
/// Although for fixed-point I could go the wrapper type way similar to complex

use super::*;
use super::markers::*;

use delegate::delegate;

// #[repr(transparent)]
// pub struct RealF<T: FixedType>(T);
// pub type Real16 = crate::units::real_fixed::real_fixed_new::RealF<i16>;
// pub type Real32 = crate::units::real_fixed::real_fixed_new::RealF<i32>;
// pub type Real64 = crate::units::real_fixed::real_fixed_new::RealF<i64>;

// GENERAL IMPLEMENTATIONS =========================================================================

// impl<T: FixedType> crate::units::real_fixed::real_fixed_new::RealF<T> {
//     pub fn new(val: T) -> Self {
//         Self(val)
//     }
// }
// impl<T: FixedType> Clone for crate::units::real_fixed::real_fixed_new::RealF<T> {
//     fn clone(&self) -> Self {
//         crate::units::real_fixed::real_fixed_new::RealF(self.0.clone())
//     }
// }
// impl<T: FixedType> Copy for crate::units::real_fixed::real_fixed_new::RealF<T> {}
// impl<T: FixedType> Debug for crate::units::real_fixed::real_fixed_new::RealF<T> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//     }
// }

// UNIT IMPLEMENTATIONS ============================================================================

// impl<T: FixedType> Unit for Real<T> {}
// impl<T: FixedType> PrimitiveUnit for Real<T> {}
// 
// impl<T: FixedType> IntUnit for Real<T> {
//     fn zero() -> Self {
//         Real::new(0)
//     }
//     fn one() -> Self {
//         Real::new(1)
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

// THE ISSUE (DELEGATION)
// solution: impl Deref trait on all newtypes --> will automatically deref until method call is found
// solution: macro to delegate functions

// THE ISSUE (SPECIALIZATION)
// need the following some shared state between types of fixed point reals
// specifically, assigning how many fractional digits per fixed-point type

// can hard code every function to use specific fractional digits per operation via macros

// can have an associated constant per fixed-point type (most sound approach)
// however, do not want to reimplement every RealUnit method every time I want to specialize into a concrete fixed-point type
// (i.e. need some sort of blanket implementation and then specific implementations per type for the associated constants)
// solutions:
// - macro that copies code over all possible specializations

// - delegate specializations to inner fields...?? (delegate the constants to inner field) (best solution, if it works)
// - maybe a proc macro that uses the above macro to achieve specialization???

// impl<T: FixedType> RealUnit for Real<T> {
//     
//     delegate! {
//         to self.0 {
//             const NAN: Self;
//             const INFINITY: Self;
//             const NEG_INFINITY: Self;
//             const EPSILON: Self;
//             const MANTISSA_DIGITS: u32;
//         }
//     }
//     const NAN: Self;
//     const INFINITY: Self;
//     const NEG_INFINITY: Self;
//     const EPSILON: Self;
//     const MANTISSA_DIGITS: u32;
// }
// 
// // FIXED OPERATOR OVERLOADS (WITH OTHERS)
// // use macros please!
// // add checks to make sure no overflow occurs!
// 
// macro_rules! fixed_add {
//     ( $([$from:ident, $to:ident, $scale_from:ident, $scale_to:expr]),* ) => {
//         $(
//         impl<T: FixedType + $from, A: FixedType + $to> Add<Real<A>> for Real<T> {
//             type Output = Self;
//             fn add(self, rhs: Real<A>) -> Self::Output { self.0 + rhs.0 }
//         }
//         )*
//     };
// }
// macro_rules! fixed_sub {
//     ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
//         $(
//         impl<T: FixedType + $from, A: FixedType + $to> Sub<Real<A>> for Real<T> {
//             type Output = Self;
//             fn sub(self, rhs: Real<A>) -> Self::Output { self.0 - rhs.0 }
//         }
//         )*
//     };
// }
// macro_rules! fixed_mul {
//     ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
//         $(
//         impl<T: FixedType + $from, A: FixedType + $to> Mul<Real<A>> for Real<T> {
//             type Output = Self;
//             fn mul(self, rhs: Real<A>) -> Self::Output { self.0 * rhs.0 }
//         }
//         )*
//     };
// }
// macro_rules! fixed_div {
//     ( $([$from:ident, $to:ident, $scale_from:expr, $scale_to:expr]),* ) => {
//         $(
//         impl<T: FixedType + $from, A: FixedType + $to> Div<Real<A>> for Real<T> {
//             type Output = Self;
//             fn div(self, rhs: Real<A>) -> Self::Output { self.0 / rhs.0 }
//         }
//         )*
//     };
// }
// 
// macro_rules! apply_fixed_ops {
//     ($($in_macro:ident),*) => {
//         $(
//         $in_macro!(
//             [Sz128, Sz64, SCALE_FACTOR_128, SCALE_FACTOR_64],
//             [Sz128, Sz32, SCALE_FACTOR_128, SCALE_FACTOR_32],
//             [Sz128, Sz16, SCALE_FACTOR_128, SCALE_FACTOR_16],
//             [Sz64, Sz128, SCALE_FACTOR_64, SCALE_FACTOR_128],
//             [Sz64, Sz32, SCALE_FACTOR_64, SCALE_FACTOR_32],
//             [Sz64, Sz16, SCALE_FACTOR_64, SCALE_FACTOR_16],
//             [Sz32, Sz128, SCALE_FACTOR_32, SCALE_FACTOR_128],
//             [Sz32, Sz64, SCALE_FACTOR_32, SCALE_FACTOR_64],
//             [Sz32, Sz16, SCALE_FACTOR_32, SCALE_FACTOR_16],
//             [Sz16, Sz128, SCALE_FACTOR_16, SCALE_FACTOR_128],
//             [Sz16, Sz64, SCALE_FACTOR_16, SCALE_FACTOR_64],
//             [Sz16, Sz32, SCALE_FACTOR_16, SCALE_FACTOR_32]
//         );
//         )*
//     };
// }
// apply_fixed_ops!(fixed_add, fixed_sub, fixed_mul, fixed_div);
// 
// // FIXED OPERATOR OVERLOADS (WITH ITSELF)
// // minimum requirement for number trait
// 
// impl<T: FixedType> Add for Real<T> {
//     type Output = Self;
//     fn add(self, rhs: Self) -> Self::Output {
//         self.0 + rhs.0
//     }
// }
// impl<T: FixedType> Sub for Real<T> {
//     type Output = Self;
//     fn sub(self, rhs: Self) -> Self::Output {
//         self.0 - rhs.0
//     }
// }
// impl<T: FixedType> Mul for Real<T> {
//     type Output = Self;
//     fn mul(self, rhs: Self) -> Self::Output {
//         self.0 * rhs.0
//     }
// }
// impl<T: FixedType> Div for Real<T> {
//     type Output = Self;
//     fn div(self, rhs: Self) -> Self::Output {
//         self.0 / rhs.0
//     }
// }
// impl<T: FixedType> Neg for Real<T> {
//     type Output = Self;
//     fn neg(self) -> Self::Output {
//         -self.0
//     }
// }
// impl<T: FixedType> PartialEq for Real<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.0.eq(other.0)
//     }
// }
// impl<T: FixedType> PartialOrd for Real<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         self.0.partial_cmp(other.0)
//     }
// }
