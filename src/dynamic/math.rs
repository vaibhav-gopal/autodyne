//! The generic maths traits on runtime-typed arrays: [`Elementwise`], [`RealValued`],
//! [`ArrayMath`], [`RealArrayMath`] and [`ComplexArrayMath`] for [`DynArray`], so code written once
//! over those traits also runs on data whose element type is only known at run time.
//!
//! Arithmetic follows NumPy's promotion ([`Promotion::Standard`]); a 0-d operand (such as
//! [`Elementwise::lit`]) is a weak scalar that adopts the other operand's type unless it is of a
//! higher kind (a float with integers, a complex with reals), as with NumPy's Python scalars; of
//! two 0-d operands of one kind, the narrower type wins.
//! Elementary functions and FFTs compute in the matching float type (`f32` for `f32` and small
//! integers, `f64` otherwise, complex for complex). Integer sums and products wrap, as NumPy's do.
//! Operators panic on combinations the checked methods report as errors.

use crate::alloc_prelude::*;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{BinaryOp, DynArray, DynElement, Promotion, UnaryOp};
use crate::signal::{
    broadcast_in_dim, broadcast_to_any, concatenate_any, pad_any, reduce_axes_with, reshape_any, reverse_any, select_any, slice_any, sum_axes_with, take_any, transpose_any,
    ArrayMath, ComplexArrayMath,
    NdArray, RealArrayMath,
};
use crate::units::*;

fn rank(d: DType) -> u8 {
    match d {
        DType::ComplexF32 | DType::ComplexF64 => 2,
        DType::F32 | DType::F64 => 1,
        _ => 0,
    }
}

/// `scalar` (0-d) adopting `other`'s type when it is of the same or a lower kind.
fn weak(scalar: &DynArray, other: DType) -> DynArray {
    if rank(scalar.dtype()) <= rank(other) {
        scalar.cast(other).unwrap_or_else(|e| panic!("cannot convert a scalar to {other:?}: {e}"))
    } else {
        scalar.clone()
    }
}

fn binary(a: DynArray, b: DynArray, op: BinaryOp) -> DynArray {
    let (a, b) = match (a.shape().is_empty(), b.shape().is_empty()) {
        // two scalars of one kind: both weak, the narrower type wins (a literal keeps an f32 sum f32)
        (true, true) if rank(a.dtype()) == rank(b.dtype()) && a.dtype() != b.dtype() => {
            let target = if a.dtype().size_bytes() <= b.dtype().size_bytes() { a.dtype() } else { b.dtype() };
            (a.cast(target).expect("converts"), b.cast(target).expect("converts"))
        }
        (true, false) => (weak(&a, b.dtype()), b),
        (false, true) => {
            let b = weak(&b, a.dtype());
            (a, b)
        }
        _ => (a, b),
    };
    a.binary(&b, op, Promotion::Standard).unwrap_or_else(|e| panic!("{op:?} of {:?} and {:?}: {e}", a.dtype(), b.dtype()))
}

macro_rules! operator {
    ($($Trait:ident $method:ident $Op:ident),+) => {$(
        impl $Trait for DynArray {
            type Output = DynArray;
            fn $method(self, rhs: DynArray) -> DynArray {
                binary(self, rhs, BinaryOp::$Op)
            }
        }
    )+};
}

operator!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div);

impl Neg for DynArray {
    type Output = DynArray;
    fn neg(self) -> DynArray {
        self.unary(UnaryOp::Neg).unwrap_or_else(|e| panic!("negating {:?}: {e}", self.dtype()))
    }
}

/// The float type elementary functions of `d` compute in.
fn float_type(d: DType) -> DType {
    match d {
        DType::F32 | DType::ComplexF32 | DType::ComplexF64 | DType::F64 => d,
        DType::I8 | DType::U8 | DType::I16 | DType::U16 => DType::F32,
        _ => DType::F64,
    }
}

impl DynArray {
    /// This array in the float type its elementary functions compute in.
    fn to_float(&self) -> DynArray {
        let target = float_type(self.dtype());
        if target == self.dtype() {
            self.clone()
        } else {
            self.cast(target).expect("integers convert to floats")
        }
    }

    /// Applies `$f` to the typed float array (real or complex).
    fn float_map(&self, f: impl FnOnce(FloatArray) -> FloatArray) -> DynArray {
        f(FloatArray::from(self.to_float())).into()
    }
}

/// A `DynArray` known to hold floats or complex numbers.
enum FloatArray {
    F32(NdArray<f32>),
    F64(NdArray<f64>),
    C32(NdArray<Complex<f32>>),
    C64(NdArray<Complex<f64>>),
}

impl From<DynArray> for FloatArray {
    fn from(a: DynArray) -> Self {
        match a {
            DynArray::F32(x) => FloatArray::F32(x),
            DynArray::F64(x) => FloatArray::F64(x),
            DynArray::ComplexF32(x) => FloatArray::C32(x),
            DynArray::ComplexF64(x) => FloatArray::C64(x),
            other => panic!("expected a float array, got {:?}", other.dtype()),
        }
    }
}

impl From<FloatArray> for DynArray {
    fn from(a: FloatArray) -> Self {
        match a {
            FloatArray::F32(x) => DynArray::F32(x),
            FloatArray::F64(x) => DynArray::F64(x),
            FloatArray::C32(x) => DynArray::ComplexF32(x),
            FloatArray::C64(x) => DynArray::ComplexF64(x),
        }
    }
}

/// Runs an `Elementwise` method on whichever float array this is.
macro_rules! float_apply {
    ($value:expr, $a:ident => $body:expr) => {
        $value.float_map(|f| match f {
            FloatArray::F32($a) => FloatArray::F32($body),
            FloatArray::F64($a) => FloatArray::F64($body),
            FloatArray::C32($a) => FloatArray::C32($body),
            FloatArray::C64($a) => FloatArray::C64($body),
        })
    };
}

impl Elementwise for DynArray {
    /// A 0-d `f64`, weak in arithmetic: it takes the other operand's float type.
    fn lit(v: f64) -> Self {
        DynArray::F64(NdArray::from_vec(vec![v], &[]).expect("a scalar"))
    }
    fn exp(self) -> Self {
        float_apply!(self, a => a.exp())
    }
    fn ln(self) -> Self {
        float_apply!(self, a => a.ln())
    }
    fn sin(self) -> Self {
        float_apply!(self, a => a.sin())
    }
    fn cos(self) -> Self {
        float_apply!(self, a => a.cos())
    }
    fn tanh(self) -> Self {
        float_apply!(self, a => a.tanh())
    }
    fn sqrt(self) -> Self {
        float_apply!(self, a => a.sqrt())
    }
    fn powf(self, e: Self) -> Self {
        // both in the float type of their promotion
        let probe = binary(self.clone(), e.clone(), BinaryOp::Mul);
        let target = float_type(probe.dtype());
        let (a, e) = (self.cast(target).expect("converts"), e.cast(target).expect("converts"));
        match (FloatArray::from(a), FloatArray::from(e)) {
            (FloatArray::F32(a), FloatArray::F32(e)) => DynArray::F32(a.powf(e)),
            (FloatArray::F64(a), FloatArray::F64(e)) => DynArray::F64(a.powf(e)),
            (FloatArray::C32(a), FloatArray::C32(e)) => DynArray::ComplexF32(a.powf(e)),
            (FloatArray::C64(a), FloatArray::C64(e)) => DynArray::ComplexF64(a.powf(e)),
            _ => unreachable!("both were cast to one type"),
        }
    }
}

/// Real values compared in `f64` (exact for integers up to 2^53); complex arrays panic.
fn real_f64(a: &DynArray) -> NdArray<f64> {
    assert!(rank(a.dtype()) < 2, "complex values have no order");
    match a.cast(DType::F64).expect("reals convert to f64") {
        DynArray::F64(x) => x,
        _ => unreachable!("cast to f64"),
    }
}

impl RealValued for DynArray {
    type Mask = NdArray<bool>;
    /// Complex arrays give their magnitudes.
    fn abs(self) -> Self {
        self.unary(UnaryOp::Abs).unwrap_or_else(|e| panic!("abs of {:?}: {e}", self.dtype()))
    }
    fn minimum(self, other: Self) -> Self {
        binary(self, other, BinaryOp::Min)
    }
    fn maximum(self, other: Self) -> Self {
        binary(self, other, BinaryOp::Max)
    }
    fn less(self, other: Self) -> NdArray<bool> {
        real_f64(&self).less(real_f64(&other))
    }
    fn greater(self, other: Self) -> NdArray<bool> {
        real_f64(&self).greater(real_f64(&other))
    }
    fn select(mask: NdArray<bool>, if_true: Self, if_false: Self) -> Self {
        // the promoted type of the two branches
        let target = binary(if_true.clone(), if_false.clone(), BinaryOp::Add).dtype();
        let (a, b) = (if_true.cast(target).expect("converts"), if_false.cast(target).expect("converts"));
        match (a, b) {
            (DynArray::F32(a), DynArray::F32(b)) => DynArray::F32(select_any(&mask, &a, &b)),
            (DynArray::F64(a), DynArray::F64(b)) => DynArray::F64(select_any(&mask, &a, &b)),
            (DynArray::I8(a), DynArray::I8(b)) => DynArray::I8(select_any(&mask, &a, &b)),
            (DynArray::I16(a), DynArray::I16(b)) => DynArray::I16(select_any(&mask, &a, &b)),
            (DynArray::I32(a), DynArray::I32(b)) => DynArray::I32(select_any(&mask, &a, &b)),
            (DynArray::I64(a), DynArray::I64(b)) => DynArray::I64(select_any(&mask, &a, &b)),
            (DynArray::U8(a), DynArray::U8(b)) => DynArray::U8(select_any(&mask, &a, &b)),
            (DynArray::U16(a), DynArray::U16(b)) => DynArray::U16(select_any(&mask, &a, &b)),
            (DynArray::U32(a), DynArray::U32(b)) => DynArray::U32(select_any(&mask, &a, &b)),
            (DynArray::U64(a), DynArray::U64(b)) => DynArray::U64(select_any(&mask, &a, &b)),
            (DynArray::ComplexF32(a), DynArray::ComplexF32(b)) => DynArray::ComplexF32(select_any(&mask, &a, &b)),
            (DynArray::ComplexF64(a), DynArray::ComplexF64(b)) => DynArray::ComplexF64(select_any(&mask, &a, &b)),
            _ => unreachable!("both were cast to one type"),
        }
    }
    /// Integers are already whole; floats round down.
    fn floor(self) -> Self {
        match self {
            DynArray::F32(a) => DynArray::F32(a.floor()),
            DynArray::F64(a) => DynArray::F64(a.floor()),
            DynArray::ComplexF32(_) | DynArray::ComplexF64(_) => panic!("complex values have no floor"),
            ints => ints,
        }
    }
}

/// Integer arrays as `i64` (signed) or `u64` (unsigned), for wrapping sums and products.
fn wide_int(a: &DynArray) -> DynArray {
    match a.dtype() {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => a.cast(DType::I64).expect("converts"),
        _ => a.cast(DType::U64).expect("converts"),
    }
}

impl ArrayMath for DynArray {
    /// An `f64` array.
    fn array(values: &[f64], shape: &[usize]) -> Self {
        DynArray::F64(NdArray::array(values, shape))
    }
    fn shape(&self) -> Vec<usize> {
        DynArray::shape(self).to_vec()
    }
    fn broadcast_to(self, shape: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(broadcast_to_any(a, shape)))
    }
    fn broadcast_in_dim(self, shape: &[usize], dims: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(broadcast_in_dim(&a, shape, dims)))
    }
    fn reshape(self, shape: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(reshape_any(a, shape)))
    }
    fn transpose(self, perm: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(transpose_any(a, perm)))
    }
    fn sum_axes(self, axes: &[usize]) -> Self {
        match self {
            DynArray::F32(a) => DynArray::F32(a.sum_axes(axes)),
            DynArray::F64(a) => DynArray::F64(a.sum_axes(axes)),
            DynArray::ComplexF32(a) => DynArray::ComplexF32(a.sum_axes(axes)),
            DynArray::ComplexF64(a) => DynArray::ComplexF64(a.sum_axes(axes)),
            ints => match wide_int(&ints) {
                DynArray::I64(a) => DynArray::I64(sum_axes_with(a, axes, i64::wrapping_add)),
                DynArray::U64(a) => DynArray::U64(sum_axes_with(a, axes, u64::wrapping_add)),
                _ => unreachable!("widened to i64 or u64"),
            },
        }
    }
    fn dot_general(self, rhs: Self, ca: &[usize], cb: &[usize]) -> Self {
        let target = binary(self.clone(), rhs.clone(), BinaryOp::Mul).dtype();
        let (a, b) = (self.cast(target).expect("converts"), rhs.cast(target).expect("converts"));
        match (a, b) {
            (DynArray::F32(a), DynArray::F32(b)) => DynArray::F32(a.dot_general(b, ca, cb)),
            (DynArray::F64(a), DynArray::F64(b)) => DynArray::F64(a.dot_general(b, ca, cb)),
            (DynArray::ComplexF32(a), DynArray::ComplexF32(b)) => DynArray::ComplexF32(a.dot_general(b, ca, cb)),
            (DynArray::ComplexF64(a), DynArray::ComplexF64(b)) => DynArray::ComplexF64(a.dot_general(b, ca, cb)),
            (a, b) => {
                // integers: wrapping products and sums in i64 / u64, then back to the promoted type
                let (a, b) = (wide_int(&a), wide_int(&b).cast(wide_int(&a).dtype()).expect("converts"));
                let wide = match (a, b) {
                    (DynArray::I64(a), DynArray::I64(b)) => {
                        let (am, bm, m, k, n, shape) = crate::signal::dot_operands(&a, &b, ca, cb);
                        DynArray::I64(NdArray::from_vec(int_matmul(&am, &bm, m, k, n, i64::wrapping_add, i64::wrapping_mul), &shape).expect("shape"))
                    }
                    (DynArray::U64(a), DynArray::U64(b)) => {
                        let (am, bm, m, k, n, shape) = crate::signal::dot_operands(&a, &b, ca, cb);
                        DynArray::U64(NdArray::from_vec(int_matmul(&am, &bm, m, k, n, u64::wrapping_add, u64::wrapping_mul), &shape).expect("shape"))
                    }
                    _ => unreachable!("widened to one integer type"),
                };
                wide.cast(target).expect("converts back")
            }
        }
    }
    fn slice(self, start: &[usize], limit: &[usize], stride: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(slice_any(&a, start, limit, stride)))
    }
    fn pad(self, low: &[usize], high: &[usize], interior: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(pad_any(&a, low, high, interior)))
    }
    /// The parts are promoted to a common type first.
    fn concatenate(parts: &[Self], axis: usize) -> Self {
        let first = parts.first().expect("concatenate: needs at least one array");
        let target = parts[1..].iter().fold(first.dtype(), |t, p| {
            super::result_type(t, p.dtype(), BinaryOp::Add, Promotion::Standard).unwrap_or_else(|e| panic!("concatenate: {t:?} with {:?}: {e}", p.dtype()))
        });
        let parts: Vec<DynArray> = parts.iter().map(|p| p.cast(target).expect("converts")).collect();
        dyn_match!(&parts[0], a => concatenate_as(a, &parts, axis))
    }
    /// Integer products wrap, in `i64` / `u64` (like their sums).
    fn prod_axes(self, axes: &[usize]) -> Self {
        match self {
            DynArray::F32(a) => DynArray::F32(a.prod_axes(axes)),
            DynArray::F64(a) => DynArray::F64(a.prod_axes(axes)),
            DynArray::ComplexF32(a) => DynArray::ComplexF32(a.prod_axes(axes)),
            DynArray::ComplexF64(a) => DynArray::ComplexF64(a.prod_axes(axes)),
            ints => match wide_int(&ints) {
                DynArray::I64(a) => DynArray::I64(reduce_axes_with(a, axes, 1, i64::wrapping_mul)),
                DynArray::U64(a) => DynArray::U64(reduce_axes_with(a, axes, 1, u64::wrapping_mul)),
                _ => unreachable!("widened to i64 or u64"),
            },
        }
    }
    fn reverse(self, axes: &[usize]) -> Self {
        dyn_match!(self, a => DynArray::from_array(reverse_any(a, axes)))
    }
}

/// `f` over `axes` for the real types (floats propagate NaN; complex arrays have no order).
fn extreme(a: DynArray, axes: &[usize], max: bool) -> DynArray {
    macro_rules! ints {
        ($($V:ident $T:ty),+) => {
            match a {
                DynArray::F32(x) => DynArray::F32(if max { x.max_axes(axes) } else { x.min_axes(axes) }),
                DynArray::F64(x) => DynArray::F64(if max { x.max_axes(axes) } else { x.min_axes(axes) }),
                $(DynArray::$V(x) => DynArray::$V(if max { reduce_axes_with(x, axes, <$T>::MIN, <$T>::max) } else { reduce_axes_with(x, axes, <$T>::MAX, <$T>::min) }),)+
                DynArray::ComplexF32(_) | DynArray::ComplexF64(_) => panic!("complex values have no order"),
            }
        };
    }
    ints!(I8 i8, I16 i16, I32 i32, I64 i64, U8 u8, U16 u16, U32 u32, U64 u64)
}

/// Concatenates `parts`, all of `T` (the first argument only names the type).
fn concatenate_as<T: DynElement>(_: &NdArray<T>, parts: &[DynArray], axis: usize) -> DynArray {
    let arrays: Vec<NdArray<T>> = parts.iter().map(|p| p.as_array::<T>().expect("one type").clone()).collect();
    DynArray::from_array(concatenate_any(&arrays, axis))
}

fn int_matmul<T: Copy + Default>(a: &[T], b: &[T], m: usize, k: usize, n: usize, add: fn(T, T) -> T, mul: fn(T, T) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push((0..k).fold(T::default(), |s, p| add(s, mul(a[i * k + p], b[p * n + j]))));
        }
    }
    out
}

impl RealArrayMath for DynArray {
    fn max_axes(self, axes: &[usize]) -> Self {
        extreme(self, axes, true)
    }
    fn min_axes(self, axes: &[usize]) -> Self {
        extreme(self, axes, false)
    }
    /// Indices of any real type.
    fn take(self, indices: Self) -> Self {
        let shape = DynArray::shape(&indices).to_vec();
        let FloatArray::F64(at) = FloatArray::from(indices.cast(DType::F64).unwrap_or_else(|e| panic!("take: indices must be real: {e}"))) else {
            unreachable!("cast to f64")
        };
        dyn_match!(self, a => DynArray::from_array(take_any(&a, at.as_slice().iter().copied(), &shape)))
    }
    type Complex = DynArray;
    /// Real arrays only (complex arrays have no real FFT; use `fft`).
    fn rfft_complex(self) -> Self {
        match FloatArray::from(self.to_float()) {
            FloatArray::F32(a) => DynArray::ComplexF32(a.rfft_complex()),
            FloatArray::F64(a) => DynArray::ComplexF64(a.rfft_complex()),
            _ => panic!("rfft needs real values"),
        }
    }
    fn irfft_complex(spectrum: Self, n: usize) -> Self {
        match RealArrayMath::to_complex(spectrum) {
            DynArray::ComplexF32(z) => DynArray::F32(NdArray::irfft_complex(z, n)),
            DynArray::ComplexF64(z) => DynArray::F64(NdArray::irfft_complex(z, n)),
            _ => unreachable!("made complex"),
        }
    }
    /// Complex at the matching precision (complex arrays are returned as they are).
    fn to_complex(self) -> Self {
        match self {
            z @ (DynArray::ComplexF32(_) | DynArray::ComplexF64(_)) => z,
            real => match FloatArray::from(real.to_float()) {
                FloatArray::F32(a) => DynArray::ComplexF32(a.to_complex()),
                FloatArray::F64(a) => DynArray::ComplexF64(a.to_complex()),
                _ => unreachable!("real"),
            },
        }
    }
    /// The parts in their common float type.
    fn complex(re: Self, im: Self) -> Self {
        let target = float_type(binary(re.clone(), im.clone(), BinaryOp::Add).dtype());
        match (FloatArray::from(re.cast(target).expect("converts")), FloatArray::from(im.cast(target).expect("converts"))) {
            (FloatArray::F32(r), FloatArray::F32(i)) => DynArray::ComplexF32(NdArray::complex(r, i)),
            (FloatArray::F64(r), FloatArray::F64(i)) => DynArray::ComplexF64(NdArray::complex(r, i)),
            _ => panic!("complex needs real parts"),
        }
    }
    fn real_part(z: Self) -> Self {
        match z {
            DynArray::ComplexF32(c) => DynArray::F32(NdArray::real_part(c)),
            DynArray::ComplexF64(c) => DynArray::F64(NdArray::real_part(c)),
            real => real,
        }
    }
    fn imag_part(z: Self) -> Self {
        match z {
            DynArray::ComplexF32(c) => DynArray::F32(NdArray::imag_part(c)),
            DynArray::ComplexF64(c) => DynArray::F64(NdArray::imag_part(c)),
            real => {
                let zero = DynArray::from_array(NdArray::<f64>::array(&[0.0], &[])).cast(float_type(real.dtype())).expect("converts");
                let shape = DynArray::shape(&real).to_vec();
                zero.broadcast_to(&shape)
            }
        }
    }
}

/// Complex FFTs; real arrays are promoted to complex first.
impl ComplexArrayMath for DynArray {
    fn fft(self) -> Self {
        complex(self).fft_dyn(false)
    }
    fn ifft(self) -> Self {
        complex(self).fft_dyn(true)
    }
    fn conj(self) -> Self {
        match complex(self) {
            DynArray::ComplexF32(a) => DynArray::ComplexF32(a.conj()),
            DynArray::ComplexF64(a) => DynArray::ComplexF64(a.conj()),
            _ => unreachable!("promoted to complex"),
        }
    }
}

/// `a` as complex numbers (`ComplexF32` from `f32` and small integers, `ComplexF64` otherwise).
fn complex(a: DynArray) -> DynArray {
    match float_type(a.dtype()) {
        DType::F32 | DType::ComplexF32 => a.cast(DType::ComplexF32).expect("converts"),
        _ => a.cast(DType::ComplexF64).expect("converts"),
    }
}

impl DynArray {
    fn fft_dyn(self, inverse: bool) -> DynArray {
        match self {
            DynArray::ComplexF32(a) => DynArray::ComplexF32(if inverse { a.ifft() } else { a.fft() }),
            DynArray::ComplexF64(a) => DynArray::ComplexF64(if inverse { a.ifft() } else { a.fft() }),
            _ => unreachable!("promoted to complex"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One generic function, three element types.
    fn energy<A: ArrayMath>(x: A) -> A {
        (x.clone() * x).sum_all() / A::lit(2.0)
    }

    #[test]
    fn generic_code_runs_on_runtime_types() {
        let f = DynArray::from_array(NdArray::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap());
        let e = energy(f);
        assert_eq!(e.dtype(), DType::F32, "the literal is weak: f32 stays f32");
        assert_eq!(e.as_array::<f32>().unwrap().as_slice(), &[7.0]);
        let i = DynArray::from_array(NdArray::from_vec(vec![1i16, 2, 3], &[3]).unwrap());
        let e = energy(i);
        assert_eq!(e.dtype(), DType::F64, "an integer sum divided by a float literal is a float");
        assert_eq!(e.as_array::<f64>().unwrap().as_slice(), &[7.0]);
        let z = DynArray::from_array(NdArray::from_vec(vec![Complex::new(0.0f64, 1.0)], &[1]).unwrap());
        assert_eq!(energy(z).as_array::<Complex<f64>>().unwrap().as_slice(), &[Complex::new(-0.5, 0.0)]);
    }

    #[test]
    fn promotion_functions_and_shapes() {
        let a = DynArray::from_array(NdArray::from_vec(vec![1i32, 2, 3, 4], &[2, 2]).unwrap());
        let b = DynArray::from_array(NdArray::from_vec(vec![0.5f32, 0.25], &[2]).unwrap());
        let c = a.clone() * b;
        assert_eq!(c.dtype(), DType::F64, "int32 with float32 promotes to float64, as NumPy");
        assert_eq!(c.as_array::<f64>().unwrap().as_slice(), &[0.5, 0.5, 1.5, 1.0]);
        // integer matrix product stays integer
        let p = a.clone().dot(a.clone());
        assert_eq!(p.dtype(), DType::I32);
        assert_eq!(p.as_array::<i32>().unwrap().as_slice(), &[7, 10, 15, 22]);
        assert_eq!(a.clone().sum_axes(&[0]).as_array::<i64>().unwrap().as_slice(), &[4, 6]);
        assert_eq!(a.clone().transpose(&[1, 0]).as_array::<i32>().unwrap().as_slice(), &[1, 3, 2, 4]);
        let s = a.clone().sqrt();
        assert_eq!(s.dtype(), DType::F64);
        let m = DynArray::select(a.clone().greater(DynArray::lit(2.0)), a.clone(), -a.clone());
        assert_eq!(m.as_array::<i32>().unwrap().as_slice(), &[-1, -2, 3, 4]);
        let (re, _) = DynArray::from_array(NdArray::from_vec(vec![1u8, 0, 0, 0], &[4]).unwrap()).rfft();
        assert_eq!(re.dtype(), DType::F32);
        assert_eq!(re.as_array::<f32>().unwrap().as_slice(), &[1.0, 1.0, 1.0]);
        let spectrum = DynArray::from_array(NdArray::from_vec(vec![1.0f64, 0.0], &[2]).unwrap()).fft();
        assert_eq!(spectrum.dtype(), DType::ComplexF64);
    }
}
