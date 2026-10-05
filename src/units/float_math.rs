//! Float maths on both paths: the `f32` / `f64` methods `core` lacks (`sqrt`, `sin`, `powi` ...),
//! under std's names, computed by std's inherent methods with the `std` feature and by `libm`
//! without.
//!
//! Without `std`, `crate::alloc_prelude` brings [`FloatMath`] into scope, so `x.sqrt()` reads the
//! same on both paths. Its methods take `&self`: where [`Elementwise`](super::Elementwise) or
//! [`RealValued`](super::RealValued) is in scope, their by-value methods of the same name are found
//! first (for `f32`, `exp` / `ln` / `tanh` are then the vectorizable kernels in `fastmath`, within
//! 2 ulp of std's), and `FloatMath` supplies the rest. The units' own trait impls call it by name
//! (`FloatMath::sqrt(&x)`), which is unambiguous on both paths and never recurses into the impl.
//!
//! tend: no_std

/// The `std`-only float methods, for `f32` and `f64`.
// std's whole set, so call sites never miss one; each build uses only some
#[allow(dead_code)]
pub(crate) trait FloatMath: Sized {
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
    fn round(&self) -> Self;
    fn round_ties_even(&self) -> Self;
    fn trunc(&self) -> Self;
    fn fract(&self) -> Self;
    fn mul_add(&self, a: Self, b: Self) -> Self;
    fn rem_euclid(&self, rhs: Self) -> Self;
    fn div_euclid(&self, rhs: Self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, n: Self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn exp_m1(&self) -> Self;
    fn ln(&self) -> Self;
    fn log(&self, base: Self) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
    fn ln_1p(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn atan2(&self, other: Self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn hypot(&self, other: Self) -> Self;
}

/// With `std`: the inherent methods (which path resolution prefers to the trait's).
#[cfg(feature = "std")]
macro_rules! impl_float_math {
    ($($T:ty),+) => {$(
        impl FloatMath for $T {
            impl_float_math!(@unary $T; sqrt cbrt floor ceil round round_ties_even trunc fract exp exp2 exp_m1 ln
                log2 log10 ln_1p sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh);
            impl_float_math!(@binary $T; rem_euclid div_euclid powf log atan2 hypot);
            #[inline(always)]
            fn mul_add(&self, a: Self, b: Self) -> Self {
                <$T>::mul_add(*self, a, b)
            }
            #[inline(always)]
            fn powi(&self, n: i32) -> Self {
                <$T>::powi(*self, n)
            }
            #[inline(always)]
            fn sin_cos(&self) -> (Self, Self) {
                <$T>::sin_cos(*self)
            }
        }
    )+};
    (@unary $T:ty; $($f:ident)+) => {$(
        #[inline(always)]
        fn $f(&self) -> Self {
            <$T>::$f(*self)
        }
    )+};
    (@binary $T:ty; $($f:ident)+) => {$(
        #[inline(always)]
        fn $f(&self, other: Self) -> Self {
            <$T>::$f(*self, other)
        }
    )+};
}

#[cfg(feature = "std")]
impl_float_math!(f32, f64);

/// Without `std`: `libm`, given each function's name for this type.
#[cfg(not(feature = "std"))]
macro_rules! impl_float_math {
    ($T:ty, $sqrt:ident, $cbrt:ident, $floor:ident, $ceil:ident, $round:ident, $roundeven:ident, $trunc:ident,
     $fma:ident, $pow:ident, $exp:ident, $exp2:ident, $expm1:ident, $log:ident, $log2:ident, $log10:ident,
     $log1p:ident, $sin:ident, $cos:ident, $tan:ident, $sincos:ident, $asin:ident, $acos:ident, $atan:ident,
     $atan2:ident, $sinh:ident, $cosh:ident, $tanh:ident, $asinh:ident, $acosh:ident, $atanh:ident,
     $hypot:ident) => {
        impl FloatMath for $T {
            fn sqrt(&self) -> Self { libm::$sqrt(*self) }
            fn cbrt(&self) -> Self { libm::$cbrt(*self) }
            fn floor(&self) -> Self { libm::$floor(*self) }
            fn ceil(&self) -> Self { libm::$ceil(*self) }
            fn round(&self) -> Self { libm::$round(*self) }
            fn round_ties_even(&self) -> Self { libm::$roundeven(*self) }
            fn trunc(&self) -> Self { libm::$trunc(*self) }
            fn fract(&self) -> Self { *self - libm::$trunc(*self) }
            fn mul_add(&self, a: Self, b: Self) -> Self { libm::$fma(*self, a, b) }
            // as std defines them
            fn rem_euclid(&self, rhs: Self) -> Self {
                let r = *self % rhs;
                if r < 0.0 { r + rhs.abs() } else { r }
            }
            fn div_euclid(&self, rhs: Self) -> Self {
                let q = libm::$trunc(*self / rhs);
                if *self % rhs < 0.0 {
                    return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
                }
                q
            }
            fn powi(&self, n: i32) -> Self { libm::$pow(*self, n as $T) }
            fn powf(&self, n: Self) -> Self { libm::$pow(*self, n) }
            fn exp(&self) -> Self { libm::$exp(*self) }
            fn exp2(&self) -> Self { libm::$exp2(*self) }
            fn exp_m1(&self) -> Self { libm::$expm1(*self) }
            fn ln(&self) -> Self { libm::$log(*self) }
            fn log(&self, base: Self) -> Self { libm::$log(*self) / libm::$log(base) }
            fn log2(&self) -> Self { libm::$log2(*self) }
            fn log10(&self) -> Self { libm::$log10(*self) }
            fn ln_1p(&self) -> Self { libm::$log1p(*self) }
            fn sin(&self) -> Self { libm::$sin(*self) }
            fn cos(&self) -> Self { libm::$cos(*self) }
            fn tan(&self) -> Self { libm::$tan(*self) }
            fn sin_cos(&self) -> (Self, Self) { libm::$sincos(*self) }
            fn asin(&self) -> Self { libm::$asin(*self) }
            fn acos(&self) -> Self { libm::$acos(*self) }
            fn atan(&self) -> Self { libm::$atan(*self) }
            fn atan2(&self, other: Self) -> Self { libm::$atan2(*self, other) }
            fn sinh(&self) -> Self { libm::$sinh(*self) }
            fn cosh(&self) -> Self { libm::$cosh(*self) }
            fn tanh(&self) -> Self { libm::$tanh(*self) }
            fn asinh(&self) -> Self { libm::$asinh(*self) }
            fn acosh(&self) -> Self { libm::$acosh(*self) }
            fn atanh(&self) -> Self { libm::$atanh(*self) }
            fn hypot(&self, other: Self) -> Self { libm::$hypot(*self, other) }
        }
    };
}

#[cfg(not(feature = "std"))]
impl_float_math!(
    f32, sqrtf, cbrtf, floorf, ceilf, roundf, roundevenf, truncf, fmaf, powf, expf, exp2f, expm1f, logf, log2f,
    log10f, log1pf, sinf, cosf, tanf, sincosf, asinf, acosf, atanf, atan2f, sinhf, coshf, tanhf, asinhf, acoshf,
    atanhf, hypotf
);
#[cfg(not(feature = "std"))]
impl_float_math!(
    f64, sqrt, cbrt, floor, ceil, round, roundeven, trunc, fma, pow, exp, exp2, expm1, log, log2, log10, log1p,
    sin, cos, tan, sincos, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, hypot
);
