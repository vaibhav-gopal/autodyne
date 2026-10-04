//! Runtime-typed math: arithmetic, element-wise functions, reductions and casts on [`DynView`] /
//! [`DynArray`], whose element types are only known at runtime.
//!
//! Each operation picks its result type once (see [`result_type`]), then runs typed code:
//! operands already of that type are read in place (zero-copy, the typed [`Zip`] path); with mixed
//! types the first operand is converted into the result and the second combined into it, in two
//! stride-planned passes, so no operand is ever copied.
//!
//! How operand types combine is described on [`Promotion`].

use super::{DynArray, DynElement, DynError, DynView};
use crate::signal::{broadcast_shapes, NdArray, NdView, Zip};
use crate::units::*;

/// How two operand types combine.
///
/// [`Promotion::Standard`] (the default) follows NumPy's table for mixing two arrays, with two
/// differences that make silent information loss impossible:
/// - every converted value is checked: a value that doesn't convert exactly (an integer beyond 2^53
///   going to `f64`, a `u64` beyond `i64::MAX` going to `i64`) is an [`DynError::Inexact`] error
///   (cast explicitly to accept the loss);
/// - a signed integer meeting `u64` gives `i64` (checked), not NumPy's `f64`: integer arithmetic
///   stays integer.
///
/// In NumPy's table, the wider type of the same kind wins; signed meets unsigned in the smallest
/// signed type holding both (`i8 + u8 -> i16`); integers of up to 16 bits keep `f32` as `f32`, wider
/// ones make it `f64`; complex follows its real precision. Integer division gives `f64`.
///
/// [`Promotion::KeepFloat`] (opt-in, as in PyTorch / JAX) lets a float keep its width when it meets
/// an integer (`i32 + f32 -> f32`), rounding integers beyond its precision instead of checking.
///
/// Integer `+ - *` wrap on overflow, as in NumPy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Promotion {
    /// NumPy's table, every conversion checked, `i* + u64 -> i64`.
    #[default]
    Standard,
    /// Like `Standard`, but a float keeps its width against integers (`i32 + f32 -> f32`), and
    /// conversions round instead of being checked.
    KeepFloat,
}

/// How [`DynView::cast`] treats values that don't fit the target type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastMode {
    /// An error unless every value converts exactly.
    Checked,
    /// Clamp to the target's range (floats to integers truncate toward zero; NaN becomes 0).
    Saturating,
    /// Integers keep their low bits (two's complement), like Rust's `as`.
    Wrapping,
}

/// Binary element-wise operations on runtime-typed arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// true division: integers give `f64`
    Div,
    /// element-wise minimum (NaN if either is NaN)
    Min,
    /// element-wise maximum (NaN if either is NaN)
    Max,
}

/// Unary element-wise operations on runtime-typed arrays (see [`DynView::unary`] for the result
/// types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// same type (unsigned integers wrap)
    Neg,
    /// same type for real numbers; complex numbers give their magnitude (real)
    Abs,
    /// Square root.
    Sqrt,
    /// e to the power.
    Exp,
    /// Natural logarithm.
    Ln,
    /// Base-10 logarithm.
    Log10,
    /// Sine (radians).
    Sin,
    /// Cosine (radians).
    Cos,
    /// Tangent (radians).
    Tan,
}

/// A single value of a runtime type: a reduction's result, or a scalar operand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynScalar {
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit float.
    F64(f64),
    /// An 8-bit signed integer.
    I8(i8),
    /// A 16-bit signed integer.
    I16(i16),
    /// A 32-bit signed integer.
    I32(i32),
    /// A 64-bit signed integer.
    I64(i64),
    /// An 8-bit unsigned integer.
    U8(u8),
    /// A 16-bit unsigned integer.
    U16(u16),
    /// A 32-bit unsigned integer.
    U32(u32),
    /// A 64-bit unsigned integer.
    U64(u64),
    /// A complex number of 32-bit floats.
    ComplexF32(Complex<f32>),
    /// A complex number of 64-bit floats.
    ComplexF64(Complex<f64>),
}

impl DynScalar {
    /// The value's element type.
    pub fn dtype(&self) -> DType {
        match self {
            DynScalar::F32(_) => DType::F32,
            DynScalar::F64(_) => DType::F64,
            DynScalar::I8(_) => DType::I8,
            DynScalar::I16(_) => DType::I16,
            DynScalar::I32(_) => DType::I32,
            DynScalar::I64(_) => DType::I64,
            DynScalar::U8(_) => DType::U8,
            DynScalar::U16(_) => DType::U16,
            DynScalar::U32(_) => DType::U32,
            DynScalar::U64(_) => DType::U64,
            DynScalar::ComplexF32(_) => DType::ComplexF32,
            DynScalar::ComplexF64(_) => DType::ComplexF64,
        }
    }
    /// The value as `f64` (rounded for large integers); `None` for complex values.
    pub fn to_f64(&self) -> Option<f64> {
        Some(match *self {
            DynScalar::F32(v) => v as f64,
            DynScalar::F64(v) => v,
            DynScalar::I8(v) => v as f64,
            DynScalar::I16(v) => v as f64,
            DynScalar::I32(v) => v as f64,
            DynScalar::I64(v) => v as f64,
            DynScalar::U8(v) => v as f64,
            DynScalar::U16(v) => v as f64,
            DynScalar::U32(v) => v as f64,
            DynScalar::U64(v) => v as f64,
            DynScalar::ComplexF32(_) | DynScalar::ComplexF64(_) => return None,
        })
    }
}

/// A scalar operand with NumPy's "weak" typing (NEP 50): it adapts to the array instead of
/// widening it (see [`DynView::binary_scalar`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// An integer: takes the array's type (an error if it doesn't fit), except that dividing an
    /// integer array gives `f64`.
    Int(i64),
    /// A float: takes the array's type for float arrays, `f64` for integer arrays.
    Float(f64),
    /// A complex number `(re, im)`: complex of the array's precision.
    Complex(f64, f64),
}

// TYPES ===========================================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    UInt,
    Float,
    Complex,
}

fn kind(d: DType) -> Kind {
    match d {
        DType::F32 | DType::F64 => Kind::Float,
        DType::ComplexF32 | DType::ComplexF64 => Kind::Complex,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Kind::UInt,
        _ => Kind::Int,
    }
}

fn bits(d: DType) -> u32 {
    match d {
        DType::ComplexF32 => 32,
        DType::ComplexF64 => 64,
        d => d.size_bytes() as u32 * 8,
    }
}

fn int_type(signed: bool, bits: u32) -> DType {
    match (signed, bits) {
        (true, 8) => DType::I8,
        (true, 16) => DType::I16,
        (true, 32) => DType::I32,
        (true, _) => DType::I64,
        (false, 8) => DType::U8,
        (false, 16) => DType::U16,
        (false, 32) => DType::U32,
        (false, _) => DType::U64,
    }
}

/// The float type an integer becomes in mixed arithmetic: `f32` for up to 16 bits, else `f64`.
fn float_precision_of(d: DType) -> u32 {
    match kind(d) {
        Kind::Float | Kind::Complex => bits(d),
        _ if bits(d) <= 16 => 32,
        _ => 64,
    }
}

/// The type `a op b` computes in (`Div` of two integers is `f64`).
pub fn result_type(a: DType, b: DType, op: BinaryOp, policy: Promotion) -> Result<DType, DynError> {
    for d in [a, b] {
        if d == DType::I24 {
            return Err(DynError::Unsupported(d));
        }
    }
    let (ka, kb) = (kind(a), kind(b));
    let real = |k: Kind| matches!(k, Kind::Int | Kind::UInt);
    if real(ka) && real(kb) {
        if op == BinaryOp::Div {
            return Ok(DType::F64);
        }
        if ka == kb {
            return Ok(int_type(ka == Kind::Int, bits(a).max(bits(b))));
        }
        let (signed, unsigned) = if ka == Kind::Int { (a, b) } else { (b, a) };
        let (bs, bu) = (bits(signed), bits(unsigned));
        // the smallest signed type holding both; past 64 bits, i64 (checked) rather than f64
        return Ok(int_type(true, if bu < bs { bs } else { (2 * bu).min(64).max(bs) }));
    }
    let complex = ka == Kind::Complex || kb == Kind::Complex;
    let precision = match policy {
        Promotion::Standard => float_precision_of(a).max(float_precision_of(b)),
        // integers don't widen a float
        Promotion::KeepFloat => [a, b].into_iter().filter(|&d| !real(kind(d))).map(bits).max().unwrap_or(64),
    };
    Ok(match (complex, precision) {
        (true, 32) => DType::ComplexF32,
        (true, _) => DType::ComplexF64,
        (false, 32) => DType::F32,
        (false, _) => DType::F64,
    })
}

/// The result type of a float function (`sqrt`, `exp`...): `f32` for floats and integers of up to
/// 16 bits, `f64` for `f64` and wider integers.
fn float_function_type(d: DType) -> Result<DType, DynError> {
    match kind(d) {
        Kind::Complex => Err(DynError::Unsupported(d)),
        _ if float_precision_of(d) == 32 => Ok(DType::F32),
        _ => Ok(DType::F64),
    }
}

// CONVERSIONS =====================================================================================

/// A value in a domain where every element type converts exactly, to compare before and after a
/// conversion (64-bit integers keep their own variant, so no wider arithmetic is needed).
#[derive(Debug, Clone, Copy)]
enum Exact {
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex(f64, f64),
}

/// 2^63 and 2^64 as floats: the first values beyond `i64` / `u64`.
const TWO_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

impl PartialEq for Exact {
    fn eq(&self, other: &Exact) -> bool {
        let same = |x: f64, y: f64| x == y || (x.is_nan() && y.is_nan());
        // an integer equals a float only if the float is exactly that integer
        let int_float = |a: i64, x: f64| x.fract() == 0.0 && (-TWO_63..TWO_63).contains(&x) && x as i64 == a;
        let uint_float = |a: u64, x: f64| x.fract() == 0.0 && (0.0..TWO_64).contains(&x) && x as u64 == a;
        match (*self, *other) {
            (Exact::Int(a), Exact::Int(b)) => a == b,
            (Exact::UInt(a), Exact::UInt(b)) => a == b,
            (Exact::Int(a), Exact::UInt(b)) | (Exact::UInt(b), Exact::Int(a)) => a >= 0 && a as u64 == b,
            (Exact::Int(a), Exact::Float(x)) | (Exact::Float(x), Exact::Int(a)) => int_float(a, x),
            (Exact::UInt(a), Exact::Float(x)) | (Exact::Float(x), Exact::UInt(a)) => uint_float(a, x),
            (Exact::Int(a), Exact::Complex(re, im)) | (Exact::Complex(re, im), Exact::Int(a)) => im == 0.0 && int_float(a, re),
            (Exact::UInt(a), Exact::Complex(re, im)) | (Exact::Complex(re, im), Exact::UInt(a)) => im == 0.0 && uint_float(a, re),
            (Exact::Float(x), Exact::Float(y)) => same(x, y),
            (Exact::Float(x), Exact::Complex(re, im)) | (Exact::Complex(re, im), Exact::Float(x)) => im == 0.0 && same(x, re),
            (Exact::Complex(a, b), Exact::Complex(c, d)) => same(a, c) && same(b, d),
        }
    }
}

/// Whether every value of `from` converts exactly to `to` (so a checked conversion can skip the
/// check): widening integers, integers that fit the float's mantissa, wider floats.
fn lossless(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }
    let mantissa = |d: DType| if bits(d) == 32 { 24 } else { 53 };
    match (kind(from), kind(to)) {
        (Kind::Int, Kind::Int) | (Kind::UInt, Kind::UInt) => bits(from) <= bits(to),
        (Kind::UInt, Kind::Int) => bits(from) < bits(to),
        (Kind::Int | Kind::UInt, Kind::Float | Kind::Complex) => bits(from) <= mantissa(to),
        (Kind::Float, Kind::Float | Kind::Complex) | (Kind::Complex, Kind::Complex) => bits(from) <= bits(to),
        _ => false,
    }
}
/// Element types as seen by the conversion and arithmetic code.
trait Num: DynElement {
    fn exact(self) -> Exact;
    /// From any value, clamping to this type's range (floats to integers truncate; NaN is 0).
    fn saturating(e: Exact) -> Self;
    /// From any value, integers keeping their low bits.
    fn wrapping(e: Exact) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    /// `None` for complex numbers (no order).
    fn min_of(self, other: Self) -> Option<Self>;
    fn max_of(self, other: Self) -> Option<Self>;
    fn neg(self) -> Self;
}

macro_rules! int_num {
    ($($T:ty => $V:ident as $W:ty),+) => {$(
        impl Num for $T {
            #[inline]
            fn exact(self) -> Exact {
                Exact::$V(self as $W)
            }
            #[inline]
            fn saturating(e: Exact) -> Self {
                match e {
                    Exact::Int(v) => {
                        if v < <$T>::MIN as i64 { <$T>::MIN } else if (v as i128) > (<$T>::MAX as i128) { <$T>::MAX } else { v as $T }
                    }
                    Exact::UInt(v) => if (v as u128) > (<$T>::MAX as u128) { <$T>::MAX } else { v as $T },
                    Exact::Float(x) | Exact::Complex(x, _) => x as $T,
                }
            }
            #[inline]
            fn wrapping(e: Exact) -> Self {
                match e {
                    Exact::Int(v) => v as $T,
                    Exact::UInt(v) => v as $T,
                    Exact::Float(x) | Exact::Complex(x, _) => x as $T,
                }
            }
            #[inline]
            fn add(self, o: Self) -> Self {
                self.wrapping_add(o)
            }
            #[inline]
            fn sub(self, o: Self) -> Self {
                self.wrapping_sub(o)
            }
            #[inline]
            fn mul(self, o: Self) -> Self {
                self.wrapping_mul(o)
            }
            #[inline]
            fn div(self, o: Self) -> Self {
                self.checked_div(o).unwrap_or(0)
            }
            #[inline]
            fn min_of(self, o: Self) -> Option<Self> {
                Some(Ord::min(self, o))
            }
            #[inline]
            fn max_of(self, o: Self) -> Option<Self> {
                Some(Ord::max(self, o))
            }
            #[inline]
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
        }
    )+};
}

int_num!(i8 => Int as i64, i16 => Int as i64, i32 => Int as i64, i64 => Int as i64, u8 => UInt as u64, u16 => UInt as u64, u32 => UInt as u64, u64 => UInt as u64);

macro_rules! float_num {
    ($($T:ty),+) => {$(
        impl Num for $T {
            #[inline]
            fn exact(self) -> Exact {
                Exact::Float(self as f64)
            }
            #[inline]
            fn saturating(e: Exact) -> Self {
                match e {
                    Exact::Int(v) => v as $T,
                    Exact::UInt(v) => v as $T,
                    Exact::Float(x) | Exact::Complex(x, _) => x as $T,
                }
            }
            #[inline]
            fn wrapping(e: Exact) -> Self {
                Self::saturating(e)
            }
            #[inline]
            fn add(self, o: Self) -> Self {
                self + o
            }
            #[inline]
            fn sub(self, o: Self) -> Self {
                self - o
            }
            #[inline]
            fn mul(self, o: Self) -> Self {
                self * o
            }
            #[inline]
            fn div(self, o: Self) -> Self {
                self / o
            }
            #[inline]
            fn min_of(self, o: Self) -> Option<Self> {
                Some(if self.is_nan() || o.is_nan() { <$T>::NAN } else { self.min(o) })
            }
            #[inline]
            fn max_of(self, o: Self) -> Option<Self> {
                Some(if self.is_nan() || o.is_nan() { <$T>::NAN } else { self.max(o) })
            }
            #[inline]
            fn neg(self) -> Self {
                -self
            }
        }
    )+};
}

float_num!(f32, f64);

macro_rules! complex_num {
    ($($F:ty),+) => {$(
        impl Num for Complex<$F> {
            #[inline]
            fn exact(self) -> Exact {
                Exact::Complex(self.re as f64, self.im as f64)
            }
            #[inline]
            fn saturating(e: Exact) -> Self {
                match e {
                    Exact::Int(v) => Complex::new(v as $F, 0.0),
                    Exact::UInt(v) => Complex::new(v as $F, 0.0),
                    Exact::Float(x) => Complex::new(x as $F, 0.0),
                    Exact::Complex(re, im) => Complex::new(re as $F, im as $F),
                }
            }
            #[inline]
            fn wrapping(e: Exact) -> Self {
                Self::saturating(e)
            }
            #[inline]
            fn add(self, o: Self) -> Self {
                self + o
            }
            #[inline]
            fn sub(self, o: Self) -> Self {
                self - o
            }
            #[inline]
            fn mul(self, o: Self) -> Self {
                self * o
            }
            #[inline]
            fn div(self, o: Self) -> Self {
                self / o
            }
            fn min_of(self, _: Self) -> Option<Self> {
                None
            }
            fn max_of(self, _: Self) -> Option<Self> {
                None
            }
            #[inline]
            fn neg(self) -> Self {
                Complex::new(-self.re, -self.im)
            }
        }
    )+};
}

complex_num!(f32, f64);

/// Runs `$body` with `$T` bound to the Rust type of a runtime dtype (every dtype but `I24`).
macro_rules! with_num {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            DType::F32 => { type $T = f32; $body }
            DType::F64 => { type $T = f64; $body }
            DType::I8 => { type $T = i8; $body }
            DType::I16 => { type $T = i16; $body }
            DType::I32 => { type $T = i32; $body }
            DType::I64 => { type $T = i64; $body }
            DType::U8 => { type $T = u8; $body }
            DType::U16 => { type $T = u16; $body }
            DType::U32 => { type $T = u32; $body }
            DType::U64 => { type $T = u64; $body }
            DType::ComplexF32 => { type $T = Complex<f32>; $body }
            DType::ComplexF64 => { type $T = Complex<f64>; $body }
            DType::I24 => Err(DynError::Unsupported(DType::I24)),
        }
    };
}

// BINARY ==========================================================================================

/// Both operands already of the result type: zero-copy, in memory order.
fn binary_same<R: Num>(a: NdView<'_, R>, b: NdView<'_, R>, shape: &[usize], op: BinaryOp) -> Result<NdArray<R>, DynError> {
    let zip = Zip::from(a.broadcast_to(shape)?).and(b.broadcast_to(shape)?)?;
    Ok(match op {
        BinaryOp::Add => zip.map_collect(|&x, &y| Num::add(x, y)),
        BinaryOp::Sub => zip.map_collect(|&x, &y| Num::sub(x, y)),
        BinaryOp::Mul => zip.map_collect(|&x, &y| Num::mul(x, y)),
        BinaryOp::Div => zip.map_collect(|&x, &y| Num::div(x, y)),
        BinaryOp::Min => zip.map_collect(|&x, &y| Num::min_of(x, y).expect("ordered type")),
        BinaryOp::Max => zip.map_collect(|&x, &y| Num::max_of(x, y).expect("ordered type")),
    })
}

/// `out = convert(src)`, with `src` broadcast to `out`'s shape: one planned pass, no copy of `src`.
fn convert_into<S: Num, R: Num>(out: &mut NdArray<R>, src: NdView<'_, S>, mode: CastMode) -> Result<(), DynError> {
    let src = src.broadcast_to(out.shape())?;
    let mode = if mode == CastMode::Checked && lossless(S::DTYPE, R::DTYPE) { CastMode::Saturating } else { mode };
    let mut failed = false;
    match mode {
        CastMode::Saturating => Zip::from(out.view_mut()).and(src)?.for_each(|o, &s| *o = R::saturating(s.exact())),
        CastMode::Wrapping => Zip::from(out.view_mut()).and(src)?.for_each(|o, &s| *o = R::wrapping(s.exact())),
        CastMode::Checked => Zip::from(out.view_mut()).and(src)?.for_each(|o, &s| {
            let e = s.exact();
            *o = R::saturating(e);
            failed |= o.exact() != e;
        }),
    }
    if failed { Err(DynError::Inexact { from: S::DTYPE, to: R::DTYPE }) } else { Ok(()) }
}

/// `out = convert(src)` for a source of any runtime type.
fn convert_dyn_into<R: Num>(out: &mut NdArray<R>, src: &DynView<'_>, mode: CastMode) -> Result<(), DynError> {
    with_num!(src.dtype(), S => convert_into::<S, R>(out, src.typed::<S>()?, mode))
}

/// `out = op(out, convert(src))`, with `src` broadcast to `out`'s shape.
fn combine_into<U: Num, R: Num>(out: &mut NdArray<R>, src: NdView<'_, U>, op: BinaryOp, mode: CastMode) -> Result<(), DynError> {
    let src = src.broadcast_to(out.shape())?;
    let f: fn(R, R) -> R = match op {
        BinaryOp::Add => R::add,
        BinaryOp::Sub => R::sub,
        BinaryOp::Mul => R::mul,
        BinaryOp::Div => R::div,
        BinaryOp::Min => |x, y| R::min_of(x, y).expect("ordered type"),
        BinaryOp::Max => |x, y| R::max_of(x, y).expect("ordered type"),
    };
    let mut failed = false;
    let checked = mode == CastMode::Checked && !lossless(U::DTYPE, R::DTYPE);
    Zip::from(out.view_mut()).and(src)?.for_each(|o, &u| {
        let e = u.exact();
        let y = R::saturating(e);
        if checked {
            failed |= y.exact() != e;
        }
        *o = f(*o, y);
    });
    if failed { Err(DynError::Inexact { from: U::DTYPE, to: R::DTYPE }) } else { Ok(()) }
}

/// `out = out op src` (or `src op out` when `reversed`), `src` already of the result type: the
/// operation is chosen once, so the loop inlines it.
fn combine_same<R: Num>(out: &mut NdArray<R>, src: NdView<'_, R>, op: BinaryOp, reversed: bool) -> Result<(), DynError> {
    let src = src.broadcast_to(out.shape())?;
    let zip = Zip::from(out.view_mut()).and(src)?;
    macro_rules! run {
        ($f:expr) => {
            if reversed { zip.for_each(|o, &s| *o = $f(s, *o)) } else { zip.for_each(|o, &s| *o = $f(*o, s)) }
        };
    }
    match op {
        BinaryOp::Add => run!(R::add),
        BinaryOp::Sub => run!(R::sub),
        BinaryOp::Mul => run!(R::mul),
        BinaryOp::Div => run!(R::div),
        BinaryOp::Min => run!(|x, y| R::min_of(x, y).expect("ordered type")),
        BinaryOp::Max => run!(|x, y| R::max_of(x, y).expect("ordered type")),
    }
    Ok(())
}

/// Mixed types: convert one operand into the result, then combine the other into it. Two planned
/// passes (contiguous inner loops wherever the layouts allow); neither operand is copied. An operand
/// already of the result type is the one combined, on the typed path.
fn binary_mixed<R: Num>(a: &DynView<'_>, b: &DynView<'_>, shape: &[usize], op: BinaryOp, mode: CastMode) -> Result<NdArray<R>, DynError> {
    let mut out = NdArray::<R>::full(shape, R::default())?;
    if b.dtype() == R::DTYPE {
        convert_dyn_into(&mut out, a, mode)?;
        combine_same(&mut out, b.typed::<R>()?, op, false)?;
    } else if a.dtype() == R::DTYPE {
        convert_dyn_into(&mut out, b, mode)?;
        combine_same(&mut out, a.typed::<R>()?, op, true)?;
    } else {
        convert_dyn_into(&mut out, a, mode)?;
        with_num!(b.dtype(), U => combine_into::<U, R>(&mut out, b.typed::<U>()?, op, mode))?;
    }
    Ok(out)
}

fn binary_as(a: &DynView<'_>, b: &DynView<'_>, op: BinaryOp, result: DType, mode: CastMode) -> Result<DynArray, DynError> {
    if matches!(op, BinaryOp::Min | BinaryOp::Max) && kind(result) == Kind::Complex {
        return Err(DynError::Unsupported(result));
    }
    let (shape, n) = broadcast_shapes(a.shape(), b.shape())?;
    let shape = &shape[..n];
    with_num!(result, R => {
        if a.dtype() == R::DTYPE && b.dtype() == R::DTYPE {
            if let (Ok(va), Ok(vb)) = (a.typed::<R>(), b.typed::<R>()) {
                return Ok(R::wrap(binary_same(va, vb, shape, op)?));
            }
        }
        Ok(R::wrap(binary_mixed::<R>(a, b, shape, op, mode)?))
    })
}

fn mode_of(policy: Promotion) -> CastMode {
    match policy {
        Promotion::Standard => CastMode::Checked,
        Promotion::KeepFloat => CastMode::Saturating,
    }
}

// VIEW API ========================================================================================

impl<'a> DynView<'a> {
    /// `self op other`, broadcast to their common shape, in the type `policy` picks (see
    /// [`Promotion`]). Errors on shapes that don't broadcast, conversions that would lose
    /// information (`Standard`), `Min`/`Max` of complex numbers, and unaligned memory.
    pub fn binary(&self, other: &DynView<'_>, op: BinaryOp, policy: Promotion) -> Result<DynArray, DynError> {
        let result = result_type(self.dtype(), other.dtype(), op, policy)?;
        binary_as(self, other, op, result, mode_of(policy))
    }
    /// `self op value` with NumPy's weak scalar typing: an integer keeps the array's type (an error
    /// if it doesn't fit) unless the array is integer and the operation is a division; a float keeps
    /// a float or complex array's type and makes an integer array `f64`; a complex value makes the
    /// array complex at its own precision.
    pub fn binary_scalar(&self, value: Scalar, op: BinaryOp) -> Result<DynArray, DynError> {
        let d = self.dtype();
        let k = kind(d);
        let int_array = matches!(k, Kind::Int | Kind::UInt);
        let result = match value {
            _ if op == BinaryOp::Div && int_array => DType::F64,
            Scalar::Int(_) => d,
            Scalar::Float(_) if int_array => DType::F64,
            Scalar::Float(_) => d,
            Scalar::Complex(..) => match (k, float_precision_of(d)) {
                (Kind::Complex, _) => d,
                (_, 32) if !int_array => DType::ComplexF32,
                _ => DType::ComplexF64,
            },
        };
        let e = match value {
            Scalar::Int(v) => Exact::Int(v),
            Scalar::Float(x) => Exact::Float(x),
            Scalar::Complex(re, im) => Exact::Complex(re, im),
        };
        let scalar: DynArray = with_num!(result, R => {
            let r = R::saturating(e);
            if matches!(value, Scalar::Int(_)) && r.exact() != e {
                return Err(DynError::Inexact { from: DType::I64, to: result });
            }
            Ok(R::wrap(NdArray::from_vec(vec![r], &[])?))
        })?;
        binary_as(self, &scalar.view(), op, result, CastMode::Saturating)
    }
    /// An element-wise function (see [`UnaryOp`]): float functions give `f32` for floats and small
    /// integers, `f64` for `f64` and wide integers; complex numbers support `Neg` and `Abs` only.
    pub fn unary(&self, op: UnaryOp) -> Result<DynArray, DynError> {
        let d = self.dtype();
        match op {
            UnaryOp::Neg => with_num!(d, T => Ok(T::wrap(self.typed::<T>()?.map(|&x| Num::neg(x))))),
            UnaryOp::Abs => match d {
                DType::ComplexF32 => Ok(DynArray::from_array(self.typed::<Complex<f32>>()?.map(|z| z.norm()))),
                DType::ComplexF64 => Ok(DynArray::from_array(self.typed::<Complex<f64>>()?.map(|z| z.norm()))),
                DType::U8 | DType::U16 | DType::U32 | DType::U64 => with_num!(d, T => Ok(T::wrap(self.typed::<T>()?.to_owned()))),
                _ => with_num!(d, T => Ok(T::wrap(self.typed::<T>()?.map(|&x| if Num::min_of(x, T::default()) == Some(x) { Num::neg(x) } else { x })))),
            },
            _ => match float_function_type(d)? {
                DType::F32 => Ok(DynArray::from_array(float_function::<f32>(self, op)?)),
                _ => Ok(DynArray::from_array(float_function::<f64>(self, op)?)),
            },
        }
    }
    /// Sum of every element: integers in `i64` / `u64` (wrapping, as NumPy), floats pairwise in
    /// their own type, complex numbers in theirs.
    pub fn sum(&self) -> Result<DynScalar, DynError> {
        Ok(match self.dtype() {
            DType::F32 => DynScalar::F32(self.typed::<f32>()?.sum()),
            DType::F64 => DynScalar::F64(self.typed::<f64>()?.sum()),
            DType::I8 => DynScalar::I64(self.typed::<i8>()?.fold(0i64, |a, &x| a.wrapping_add(x as i64))),
            DType::I16 => DynScalar::I64(self.typed::<i16>()?.fold(0i64, |a, &x| a.wrapping_add(x as i64))),
            DType::I32 => DynScalar::I64(self.typed::<i32>()?.fold(0i64, |a, &x| a.wrapping_add(x as i64))),
            DType::I64 => DynScalar::I64(self.typed::<i64>()?.fold(0i64, |a, &x| a.wrapping_add(x))),
            DType::U8 => DynScalar::U64(self.typed::<u8>()?.fold(0u64, |a, &x| a.wrapping_add(x as u64))),
            DType::U16 => DynScalar::U64(self.typed::<u16>()?.fold(0u64, |a, &x| a.wrapping_add(x as u64))),
            DType::U32 => DynScalar::U64(self.typed::<u32>()?.fold(0u64, |a, &x| a.wrapping_add(x as u64))),
            DType::U64 => DynScalar::U64(self.typed::<u64>()?.fold(0u64, |a, &x| a.wrapping_add(x))),
            DType::ComplexF32 => DynScalar::ComplexF32(self.typed::<Complex<f32>>()?.fold(Complex::zero(), |a, &x| a + x)),
            DType::ComplexF64 => DynScalar::ComplexF64(self.typed::<Complex<f64>>()?.fold(Complex::zero(), |a, &x| a + x)),
            DType::I24 => return Err(DynError::Unsupported(DType::I24)),
        })
    }
    /// Mean: `f64` for integers, the element type otherwise; `None` when empty.
    pub fn mean(&self) -> Result<Option<DynScalar>, DynError> {
        if self.shape().contains(&0) {
            return Ok(None);
        }
        let n = self.shape().iter().product::<usize>() as f64;
        Ok(Some(match self.sum()? {
            DynScalar::F32(s) => DynScalar::F32(s / n as f32),
            DynScalar::F64(s) => DynScalar::F64(s / n),
            DynScalar::ComplexF32(s) => DynScalar::ComplexF32(s * (1.0 / n as f32)),
            DynScalar::ComplexF64(s) => DynScalar::ComplexF64(s * (1.0 / n)),
            other => DynScalar::F64(other.to_f64().unwrap_or(f64::NAN) / n),
        }))
    }
    /// Smallest element (NaNs skipped), in the element type; `None` when empty.
    pub fn min(&self) -> Result<Option<DynScalar>, DynError> {
        self.extreme(false)
    }
    /// Largest element (NaNs skipped), in the element type; `None` when empty.
    pub fn max(&self) -> Result<Option<DynScalar>, DynError> {
        self.extreme(true)
    }
    fn extreme(&self, largest: bool) -> Result<Option<DynScalar>, DynError> {
        macro_rules! pick {
            ($V:ident, $T:ty) => {{
                let v = self.typed::<$T>()?;
                (if largest { v.max() } else { v.min() }).map(DynScalar::$V)
            }};
        }
        Ok(match self.dtype() {
            DType::F32 => pick!(F32, f32),
            DType::F64 => pick!(F64, f64),
            DType::I8 => pick!(I8, i8),
            DType::I16 => pick!(I16, i16),
            DType::I32 => pick!(I32, i32),
            DType::I64 => pick!(I64, i64),
            DType::U8 => pick!(U8, u8),
            DType::U16 => pick!(U16, u16),
            DType::U32 => pick!(U32, u32),
            DType::U64 => pick!(U64, u64),
            d => return Err(DynError::Unsupported(d)),
        })
    }
    /// A copy converted to `dtype` (see [`CastMode`]); complex to real is an error.
    pub fn cast(&self, dtype: DType, mode: CastMode) -> Result<DynArray, DynError> {
        if kind(self.dtype()) == Kind::Complex && kind(dtype) != Kind::Complex {
            return Err(DynError::ComplexToReal(dtype));
        }
        let shape: Vec<usize> = self.shape().to_vec();
        with_num!(dtype, R => {
            let mut out = NdArray::<R>::zeros(&shape)?;
            convert_dyn_into(&mut out, self, mode)?;
            Ok(R::wrap(out))
        })
    }
}

fn float_function<R: Num + Float>(view: &DynView<'_>, op: UnaryOp) -> Result<NdArray<R>, DynError> {
    let f: fn(R) -> R = match op {
        UnaryOp::Sqrt => |x: R| x._sqrt(),
        UnaryOp::Exp => |x: R| x._exp(),
        UnaryOp::Ln => |x: R| x._ln(),
        UnaryOp::Log10 => |x: R| x._log10(),
        UnaryOp::Sin => |x: R| x._sin(),
        UnaryOp::Cos => |x: R| x._cos(),
        UnaryOp::Tan => |x: R| x._tan(),
        UnaryOp::Neg | UnaryOp::Abs => unreachable!("handled by unary"),
    };
    if view.dtype() == R::DTYPE {
        if let Ok(v) = view.typed::<R>() {
            return Ok(v.map(|&x| f(x)));
        }
    }
    let shape: Vec<usize> = view.shape().to_vec();
    let mut out = NdArray::<R>::zeros(&shape)?;
    // integers convert exactly into the float type chosen for them (16 bits into f32, wider into f64)
    convert_dyn_into(&mut out, view, CastMode::Saturating)?;
    out.map_inplace(|x| *x = f(*x));
    Ok(out)
}

impl DynArray {
    /// A zero-copy runtime-typed view of the array.
    pub fn view(&self) -> DynView<'_> {
        DynView::new(self.as_bytes(), self.dtype(), self.shape()).expect("an array's bytes match its shape")
    }
    /// See [`DynView::binary`].
    pub fn binary(&self, other: &DynArray, op: BinaryOp, policy: Promotion) -> Result<DynArray, DynError> {
        self.view().binary(&other.view(), op, policy)
    }
    /// `self + other`, broadcast, with the standard promotion.
    pub fn add(&self, other: &DynArray) -> Result<DynArray, DynError> {
        self.binary(other, BinaryOp::Add, Promotion::Standard)
    }
    /// `self - other`, broadcast, with the standard promotion.
    pub fn sub(&self, other: &DynArray) -> Result<DynArray, DynError> {
        self.binary(other, BinaryOp::Sub, Promotion::Standard)
    }
    /// `self * other`, broadcast, with the standard promotion.
    pub fn mul(&self, other: &DynArray) -> Result<DynArray, DynError> {
        self.binary(other, BinaryOp::Mul, Promotion::Standard)
    }
    /// `self / other` (true division), broadcast, with the standard promotion.
    pub fn div(&self, other: &DynArray) -> Result<DynArray, DynError> {
        self.binary(other, BinaryOp::Div, Promotion::Standard)
    }
    /// See [`DynView::binary_scalar`].
    pub fn binary_scalar(&self, value: Scalar, op: BinaryOp) -> Result<DynArray, DynError> {
        self.view().binary_scalar(value, op)
    }
    /// See [`DynView::unary`].
    pub fn unary(&self, op: UnaryOp) -> Result<DynArray, DynError> {
        self.view().unary(op)
    }
    /// See [`DynView::sum`].
    pub fn sum(&self) -> Result<DynScalar, DynError> {
        self.view().sum()
    }
    /// See [`DynView::mean`].
    pub fn mean(&self) -> Result<Option<DynScalar>, DynError> {
        self.view().mean()
    }
    /// See [`DynView::min`].
    pub fn min(&self) -> Result<Option<DynScalar>, DynError> {
        self.view().min()
    }
    /// See [`DynView::max`].
    pub fn max(&self) -> Result<Option<DynScalar>, DynError> {
        self.view().max()
    }
    /// A copy converted to `dtype` with the given handling of values that don't fit.
    pub fn cast_with(&self, dtype: DType, mode: CastMode) -> Result<DynArray, DynError> {
        self.view().cast(dtype, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDER: [DType; 12] = [
        DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F32, DType::F64, DType::ComplexF32, DType::ComplexF64,
    ];

    /// NumPy 2.5's np.result_type for these types, row by row in ORDER (i8..u64, f32, f64, c64, c128).
    const NUMPY: &str = "
        i8 i16 i32 i64 i16 i32 i64 f64 f32 f64 c64 c128
        i16 i16 i32 i64 i16 i32 i64 f64 f32 f64 c64 c128
        i32 i32 i32 i64 i32 i32 i64 f64 f64 f64 c128 c128
        i64 i64 i64 i64 i64 i64 i64 f64 f64 f64 c128 c128
        i16 i16 i32 i64 u8 u16 u32 u64 f32 f64 c64 c128
        i32 i32 i32 i64 u16 u16 u32 u64 f32 f64 c64 c128
        i64 i64 i64 i64 u32 u32 u32 u64 f64 f64 c128 c128
        f64 f64 f64 f64 u64 u64 u64 u64 f64 f64 c128 c128
        f32 f32 f64 f64 f32 f32 f64 f64 f32 f64 c64 c128
        f64 f64 f64 f64 f64 f64 f64 f64 f64 f64 c128 c128
        c64 c64 c128 c128 c64 c64 c128 c128 c64 c128 c64 c128
        c128 c128 c128 c128 c128 c128 c128 c128 c128 c128 c128 c128";

    fn numpy_name(d: DType) -> &'static str {
        match d {
            DType::ComplexF32 => "c64",
            DType::ComplexF64 => "c128",
            d => d.name(),
        }
    }

    #[test]
    fn standard_promotion_is_numpys_table_except_u64_with_signed() {
        let table: Vec<&str> = NUMPY.split_whitespace().collect();
        for (i, &a) in ORDER.iter().enumerate() {
            for (j, &b) in ORDER.iter().enumerate() {
                let numpy = table[i * 12 + j];
                let ours = result_type(a, b, BinaryOp::Add, Promotion::Standard).unwrap();
                let signed_with_u64 = (kind(a) == Kind::Int && b == DType::U64) || (kind(b) == Kind::Int && a == DType::U64);
                if signed_with_u64 {
                    assert_eq!((numpy, ours), ("f64", DType::I64), "{a} + {b}: the deliberate difference");
                } else {
                    assert_eq!(numpy_name(ours), numpy, "{a} + {b}");
                }
                assert_eq!(result_type(b, a, BinaryOp::Add, Promotion::Standard).unwrap(), ours, "symmetric");
            }
        }
        assert_eq!(result_type(DType::I8, DType::U8, BinaryOp::Div, Promotion::Standard).unwrap(), DType::F64);
        assert_eq!(result_type(DType::I16, DType::F32, BinaryOp::Div, Promotion::Standard).unwrap(), DType::F32);
        // KeepFloat: integers never widen a float
        assert_eq!(result_type(DType::I32, DType::F32, BinaryOp::Add, Promotion::KeepFloat).unwrap(), DType::F32);
        assert_eq!(result_type(DType::I64, DType::ComplexF32, BinaryOp::Mul, Promotion::KeepFloat).unwrap(), DType::ComplexF32);
        assert_eq!(result_type(DType::I32, DType::U16, BinaryOp::Add, Promotion::KeepFloat).unwrap(), DType::I32);
    }

    fn arr<T: DynElement>(data: Vec<T>, shape: &[usize]) -> DynArray {
        DynArray::from_array(NdArray::from_vec(data, shape).unwrap())
    }

    #[test]
    fn mixed_arithmetic_with_broadcasting() {
        let a = arr(vec![1i16, 2, 3, 4, 5, 6], &[2, 3]);
        let b = arr(vec![0.5f32, 1.0, 1.5], &[3]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.as_array::<f32>().unwrap().as_slice(), [0.5, 2.0, 4.5, 2.0, 5.0, 9.0]);
        // signed meets unsigned in a wider signed type
        let u = arr(vec![200u8, 100], &[2]);
        let s = arr(vec![-100i8, 50], &[2]);
        let sum = u.add(&s).unwrap();
        assert_eq!((sum.dtype(), sum.as_array::<i16>().unwrap().as_slice()), (DType::I16, &[100i16, 150][..]));
        // integer division is true division
        let q = arr(vec![7i32, -7], &[2]).div(&arr(vec![2i32, 2], &[2])).unwrap();
        assert_eq!(q.as_array::<f64>().unwrap().as_slice(), [3.5, -3.5]);
        // integers wrap, as in NumPy
        let w = arr(vec![120i8], &[1]).add(&arr(vec![10i8], &[1])).unwrap();
        assert_eq!(w.as_array::<i8>().unwrap().as_slice(), [-126]);
        // min / max propagate NaN; complex has no order
        let m = arr(vec![1.0f64, f64::NAN], &[2]).binary(&arr(vec![0.5f64, 0.0], &[2]), BinaryOp::Min, Promotion::Standard).unwrap();
        let m = m.as_array::<f64>().unwrap().as_slice().to_vec();
        assert!(m[0] == 0.5 && m[1].is_nan());
        let z = arr(vec![Complex::new(1.0f32, 0.0)], &[1]);
        assert_eq!(z.binary(&z, BinaryOp::Max, Promotion::Standard).err(), Some(DynError::Unsupported(DType::ComplexF32)));
    }

    #[test]
    fn standard_promotion_refuses_to_lose_information() {
        let big = arr(vec![(1i64 << 53) + 1], &[1]);
        let half = arr(vec![0.5f64], &[1]);
        assert_eq!(big.add(&half).err(), Some(DynError::Inexact { from: DType::I64, to: DType::F64 }));
        let small = arr(vec![1i64 << 40], &[1]);
        assert_eq!(small.add(&half).unwrap().as_array::<f64>().unwrap().as_slice(), [(1u64 << 40) as f64 + 0.5]);
        // u64 with a signed type stays integer, checked
        let u = arr(vec![5u64, 7], &[2]);
        let i = arr(vec![-1i8, 1], &[2]);
        assert_eq!(u.add(&i).unwrap().as_array::<i64>().unwrap().as_slice(), [4, 8]);
        assert_eq!(arr(vec![u64::MAX], &[1]).add(&arr(vec![0i8], &[1])).err(), Some(DynError::Inexact { from: DType::U64, to: DType::I64 }));
        // KeepFloat rounds instead
        let pcm = arr(vec![(1i32 << 24) + 1], &[1]);
        let gain = arr(vec![1.0f32], &[1]);
        let kept = pcm.binary(&gain, BinaryOp::Mul, Promotion::KeepFloat).unwrap();
        assert_eq!((kept.dtype(), kept.as_array::<f32>().unwrap().as_slice()), (DType::F32, &[16_777_216.0f32][..]));
        // the standard rule widens to f64 and stays exact
        assert_eq!(pcm.mul(&gain).unwrap().as_array::<f64>().unwrap().as_slice(), [16_777_217.0]);
    }

    #[test]
    fn same_type_runs_in_place_on_strided_views() {
        // the zero-copy path and the chunked path agree, on a transposed view
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let a = arr(data.clone(), &[3, 4]);
        let bytes = a.as_bytes();
        let t = DynView::with_strides(bytes, DType::F64, &[4, 3], Some(&[1, 4]), 0).unwrap();
        let same = t.binary(&t, BinaryOp::Add, Promotion::Standard).unwrap();
        let ints = arr((0..12).collect::<Vec<i32>>(), &[3, 4]);
        let ti = DynView::with_strides(ints.as_bytes(), DType::I32, &[4, 3], Some(&[1, 4]), 0).unwrap();
        let mixed = t.binary(&ti, BinaryOp::Add, Promotion::Standard).unwrap();
        assert_eq!(same, mixed);
        assert_eq!(same.as_array::<f64>().unwrap().as_slice()[..4], [0.0, 8.0, 16.0, 2.0]);
    }

    #[test]
    fn weak_scalars_adapt_to_the_array() {
        let f = arr(vec![1.0f32, 2.0], &[2]);
        assert_eq!(f.binary_scalar(Scalar::Float(2.5), BinaryOp::Mul).unwrap().dtype(), DType::F32);
        assert_eq!(f.binary_scalar(Scalar::Complex(0.0, 1.0), BinaryOp::Mul).unwrap().dtype(), DType::ComplexF32);
        let i = arr(vec![1i8, 2], &[2]);
        let plus = i.binary_scalar(Scalar::Int(3), BinaryOp::Add).unwrap();
        assert_eq!((plus.dtype(), plus.as_array::<i8>().unwrap().as_slice()), (DType::I8, &[4i8, 5][..]));
        assert_eq!(i.binary_scalar(Scalar::Float(2.5), BinaryOp::Mul).unwrap().dtype(), DType::F64);
        assert_eq!(i.binary_scalar(Scalar::Int(2), BinaryOp::Div).unwrap().as_array::<f64>().unwrap().as_slice(), [0.5, 1.0]);
        assert_eq!(i.binary_scalar(Scalar::Int(300), BinaryOp::Add).err(), Some(DynError::Inexact { from: DType::I64, to: DType::I8 }));
    }

    #[test]
    fn functions_reductions_and_casts() {
        let i = arr(vec![4i16, 9, 16], &[3]);
        let r = i.unary(UnaryOp::Sqrt).unwrap();
        assert_eq!((r.dtype(), r.as_array::<f32>().unwrap().as_slice()), (DType::F32, &[2.0f32, 3.0, 4.0][..]));
        assert_eq!(arr(vec![4i64], &[1]).unwrap_dtype(UnaryOp::Ln), DType::F64);
        let z = arr(vec![Complex::new(3.0f32, 4.0)], &[1]).unary(UnaryOp::Abs).unwrap();
        assert_eq!(z.as_array::<f32>().unwrap().as_slice(), [5.0]);
        assert_eq!(arr(vec![-3i8, 3], &[2]).unary(UnaryOp::Abs).unwrap().as_array::<i8>().unwrap().as_slice(), [3, 3]);
        // integer sums can't overflow their element type
        let bytes = arr(vec![200u8; 10], &[10]);
        assert_eq!(bytes.sum().unwrap(), DynScalar::U64(2_000));
        assert_eq!(arr(vec![1i32, 2], &[2]).mean().unwrap(), Some(DynScalar::F64(1.5)));
        assert_eq!(arr(vec![3i16, -7, 5], &[3]).min().unwrap(), Some(DynScalar::I16(-7)));
        assert_eq!(arr(vec![0.5f32, f32::NAN, 2.0], &[3]).max().unwrap(), Some(DynScalar::F32(2.0)));
        // casts: checked, saturating, wrapping
        let v = arr(vec![300i32, -1, 7], &[3]);
        assert_eq!(v.cast_with(DType::U8, CastMode::Checked).err(), Some(DynError::Inexact { from: DType::I32, to: DType::U8 }));
        assert_eq!(v.cast_with(DType::U8, CastMode::Saturating).unwrap().as_array::<u8>().unwrap().as_slice(), [255, 0, 7]);
        assert_eq!(v.cast_with(DType::U8, CastMode::Wrapping).unwrap().as_array::<u8>().unwrap().as_slice(), [44, 255, 7]);
        // 64-bit integers cast exactly (no detour through f64)
        let big = arr(vec![i64::MAX - 1], &[1]);
        assert_eq!(big.cast_with(DType::U64, CastMode::Checked).unwrap().as_array::<u64>().unwrap().as_slice(), [(i64::MAX - 1) as u64]);
    }

    impl DynArray {
        fn unwrap_dtype(&self, op: UnaryOp) -> DType {
            self.unary(op).unwrap().dtype()
        }
    }
}
