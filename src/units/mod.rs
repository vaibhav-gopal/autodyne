//! Numbers: the traits sample types implement (from [`Unit`] up to [`Float`] and [`Integer`], and
//! [`Elementwise`] / [`Real`], the traceable maths processors are written over), [`Complex`],
//! fixed point ([`Fixed`]), arbitrary precision ([`BigInt`], [`BigFloat`]), checked casts between
//! primitive types, decibel conversions, and runtime element types ([`DType`], [`Reflection`]).
//!
//! tend: Core / units

pub(crate) mod fastmath;
mod cast;
pub use cast::*;

mod properties;
pub use properties::*;

mod ops;
pub use ops::*;

mod unit;
pub use unit::*;

mod integer;
pub use integer::*;
pub(crate) use integer::gcd;

mod float;
pub use float::*;

mod real;
pub use real::*;

mod fixed;
pub use fixed::*;

mod bignum;
pub use bignum::*;

mod reflection;
pub use reflection::*;

mod complex;
pub use complex::*;

mod decibel;
pub use decibel::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_constants_agree_across_precisions() {
        assert_eq!(f32::_PI * 2.0, f32::_TAU);
        assert_eq!(f64::_PI * 2.0, f64::_TAU);
        assert_eq!(f32::_PI, f64::_PI as f32);
        assert_eq!(f32::_E, f64::_E as f32);
    }
}
