//! N-dimensional computation on views: [`Zip`] (element-wise over several views, with
//! broadcasting), arithmetic, reductions, and allocation-free processing along lanes.
//!
//! Everything here works on views of any layout (strided, reversed, transposed, broadcast) without
//! copying. Each operation plans its own loop: axes are ordered by memory stride and axes that form
//! one contiguous run are merged, so typical arrays reduce to a few long inner loops; when every
//! operand's inner loop is contiguous, it runs on consecutive elements (which the compiler
//! vectorizes).
//!
//! Writes only go through mutable views, which are injective. Combining several values into one
//! (the write side of broadcasting) is a reduction with an explicit rule: [`NdView::fold_into`],
//! [`NdView::sum_into`] and friends write into a smaller array whose shape broadcasts to the input.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::ndarray::{Layout, NdArray, NdError, NdView, NdViewMut, MAX_DIMS};
use crate::processor::Processor;
use crate::units::*;

// LOOP PLANNING ===================================================================================

/// A loop nest over operands that share a shape: the merged axes (outer first) and each operand's
/// stride along them.
struct Plan<const N: usize> {
    ndim: usize,
    dims: [usize; MAX_DIMS],
    strides: [[isize; MAX_DIMS]; N],
}

/// Orders the axes by the stride of operand `driver` (largest outer, so the inner loop has the
/// smallest), drops length-1 axes, and merges neighbours that every operand walks as one run.
fn plan<const N: usize>(shape: &[usize], strides: &[[isize; MAX_DIMS]; N], driver: usize) -> Plan<N> {
    let mut axes = [0usize; MAX_DIMS];
    let mut n = 0;
    for (a, &d) in shape.iter().enumerate() {
        if d > 1 {
            axes[n] = a;
            n += 1;
        }
    }
    // insertion sort, stable: descending stride size of the driver
    for i in 1..n {
        let mut j = i;
        while j > 0 && strides[driver][axes[j - 1]].unsigned_abs() < strides[driver][axes[j]].unsigned_abs() {
            axes.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut p = Plan { ndim: 0, dims: [1; MAX_DIMS], strides: [[0; MAX_DIMS]; N] };
    for &a in &axes[..n] {
        let d = shape[a];
        if p.ndim > 0 {
            // merge into the previous (outer) axis if it steps exactly over this one, everywhere
            let outer = p.ndim - 1;
            if (0..N).all(|k| p.strides[k][outer] == strides[k][a] * d as isize) {
                p.dims[outer] *= d;
                for (ps, s) in p.strides.iter_mut().zip(strides) {
                    ps[outer] = s[a];
                }
                continue;
            }
        }
        p.dims[p.ndim] = d;
        for (ps, s) in p.strides.iter_mut().zip(strides) {
            ps[p.ndim] = s[a];
        }
        p.ndim += 1;
    }
    if p.ndim == 0 {
        // a single element: one loop of length 1
        p.ndim = 1;
    }
    p
}

/// Runs the plan: `body(offsets, length, inner_strides)` for each inner run, where `offsets` are the
/// operands' element offsets at the run's start.
#[inline]
fn execute<const N: usize>(p: &Plan<N>, mut body: impl FnMut([isize; N], usize, [isize; N])) {
    let inner = p.ndim - 1;
    let inner_strides: [isize; N] = std::array::from_fn(|k| p.strides[k][inner]);
    let mut index = [0usize; MAX_DIMS];
    let mut offsets = [0isize; N];
    loop {
        body(offsets, p.dims[inner], inner_strides);
        // odometer over the outer axes
        let mut axis = inner;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            for (o, s) in offsets.iter_mut().zip(&p.strides) {
                *o += s[axis];
            }
            if index[axis] < p.dims[axis] {
                break;
            }
            for (o, s) in offsets.iter_mut().zip(&p.strides) {
                *o -= s[axis] * p.dims[axis] as isize;
            }
            index[axis] = 0;
        }
    }
}

// ZIP =============================================================================================

mod sealed {
    pub trait Sealed {}
}

/// An operand of a [`Zip`]: a view (yielding `&T`) or a mutable view (yielding `&mut T`).
pub trait NdProducer: sealed::Sealed {
    type Item;
    #[doc(hidden)]
    type Ptr: Copy;
    #[doc(hidden)]
    fn layout_parts(&self) -> (&[usize], [isize; MAX_DIMS]);
    #[doc(hidden)]
    fn base(&self) -> Self::Ptr;
    /// # Safety
    /// `offset` must be the offset of an index inside the producer's layout.
    #[doc(hidden)]
    unsafe fn item(ptr: Self::Ptr, offset: isize) -> Self::Item;
}

impl<T> sealed::Sealed for NdView<'_, T> {}
impl<'a, T> NdProducer for NdView<'a, T> {
    type Item = &'a T;
    type Ptr = *const T;
    fn layout_parts(&self) -> (&[usize], [isize; MAX_DIMS]) {
        (self.layout.shape(), self.layout.strides)
    }
    fn base(&self) -> *const T {
        self.ptr
    }
    #[inline(always)]
    unsafe fn item(ptr: *const T, offset: isize) -> &'a T {
        // SAFETY: the caller passes offsets of indices inside the validated layout
        unsafe { &*ptr.wrapping_offset(offset) }
    }
}

impl<T> sealed::Sealed for NdViewMut<'_, T> {}
impl<'a, T> NdProducer for NdViewMut<'a, T> {
    type Item = &'a mut T;
    type Ptr = *mut T;
    fn layout_parts(&self) -> (&[usize], [isize; MAX_DIMS]) {
        (self.layout.shape(), self.layout.strides)
    }
    fn base(&self) -> *mut T {
        self.ptr
    }
    #[inline(always)]
    unsafe fn item(ptr: *mut T, offset: isize) -> &'a mut T {
        // SAFETY: inside the layout; mutable views are injective and a Zip visits each index once,
        // so no two live references alias
        unsafe { &mut *ptr.wrapping_offset(offset) }
    }
}

/// Walks several views of one shape together, calling a closure with one element of each:
/// element-wise computation over any layouts, fused into a single pass with no temporaries.
///
/// ```
/// use autodyne::signal::{NdArray, Zip};
///
/// let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let bias = NdArray::from_vec(vec![10.0, 20.0], &[2]).unwrap();
/// let mut out = NdArray::<f64>::zeros(&[2, 2]).unwrap();
/// // out = a * a + bias (the bias row repeats down the rows): one pass, no temporaries
/// Zip::from(out.view_mut()).and(a.view()).unwrap().and_broadcast(bias.view()).unwrap()
///     .for_each(|o, &x, &b| *o = x * x + b);
/// assert_eq!(out.as_slice(), [11.0, 24.0, 19.0, 36.0]);
/// ```
pub struct Zip<P> {
    producers: P,
    shape: [usize; MAX_DIMS],
    ndim: usize,
}

impl<P: NdProducer> Zip<(P,)> {
    /// Starts a zip over `producer`'s shape.
    pub fn from(producer: P) -> Self {
        let (shape, _) = producer.layout_parts();
        let mut s = [0; MAX_DIMS];
        s[..shape.len()].copy_from_slice(shape);
        let ndim = shape.len();
        Zip { producers: (producer,), shape: s, ndim }
    }
}

impl<P> Zip<P> {
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    fn check(&self, shape: &[usize]) -> Result<(), NdError> {
        if shape == self.shape() {
            return Ok(());
        }
        let axis = shape.iter().zip(self.shape()).position(|(a, b)| a != b).unwrap_or(0);
        Err(NdError::Broadcast { axis, from: shape.get(axis).copied().unwrap_or(0), to: self.shape().get(axis).copied().unwrap_or(0) })
    }
}

macro_rules! zip_arity {
    (@and $N:literal; $($P:ident $p:ident $i:tt),+ ; $Q:ident) => {
        /// Adds `producer`, which must have exactly the zip's shape.
        pub fn and<$Q: NdProducer>(self, producer: $Q) -> Result<Zip<($($P,)+ $Q,)>, NdError> {
            self.check(producer.layout_parts().0)?;
            Ok(Zip { producers: ($(self.producers.$i,)+ producer,), shape: self.shape, ndim: self.ndim })
        }
        /// Adds a read-only `view` repeated to the zip's shape (NumPy broadcasting).
        pub fn and_broadcast<'b, T>(self, view: NdView<'b, T>) -> Result<Zip<($($P,)+ NdView<'b, T>,)>, NdError> {
            let view = view.broadcast_to(&self.shape[..self.ndim])?;
            Ok(Zip { producers: ($(self.producers.$i,)+ view,), shape: self.shape, ndim: self.ndim })
        }
    };
    (@and $N:literal; $($P:ident $p:ident $i:tt),+ ; ) => {};
    ($N:literal; $($P:ident $p:ident $i:tt),+ ; $($Q:ident)?) => {

        impl<$($P: NdProducer),+> Zip<($($P,)+)> {
            /// Calls `f` with the elements at every index, in memory-friendly order.
            #[inline]
            pub fn for_each(self, mut f: impl FnMut($($P::Item),+)) {
                if self.shape().contains(&0) {
                    return;
                }
                let ($($p,)+) = self.producers;
                let strides: [[isize; MAX_DIMS]; $N] = [$($p.layout_parts().1),+];
                let bases = ($($p.base(),)+);
                let plan = plan(&self.shape[..self.ndim], &strides, 0);
                execute(&plan, |at, n, step| {
                    if step.iter().all(|&s| s == 1) {
                        for k in 0..n as isize {
                            // SAFETY: the plan only reaches indices inside every operand's layout
                            unsafe { f($($P::item(bases.$i, at[$i] + k)),+) }
                        }
                    } else {
                        for k in 0..n as isize {
                            // SAFETY: as above
                            unsafe { f($($P::item(bases.$i, at[$i] + k * step[$i])),+) }
                        }
                    }
                });
            }
            /// Combines every index's elements into one value, in the same order as `for_each`.
            pub fn fold<Acc>(self, init: Acc, mut f: impl FnMut(Acc, $($P::Item),+) -> Acc) -> Acc {
                let mut acc = Some(init);
                self.for_each(|$($p),+| acc = Some(f(acc.take().expect("set"), $($p),+)));
                acc.expect("set")
            }
            /// Collects `f` of every index into a new array of the zip's shape (allocates).
            pub fn map_collect<R: Copy + Default>(self, mut f: impl FnMut($($P::Item),+) -> R) -> NdArray<R> {
                let mut out = NdArray::<R>::zeros(self.shape()).expect("the zip's shape is valid");
                let zip = Zip { producers: (out.view_mut(), $(self.producers.$i,)+), shape: self.shape, ndim: self.ndim };
                zip.for_each_first(|o, $($p),+| *o = f($($p),+));
                out
            }
            zip_arity!(@and $N; $($P $p $i),+ ; $($Q)?);
        }
    };
}

zip_arity!(1; A a 0; B);
zip_arity!(2; A a 0, B b 1; C);
zip_arity!(3; A a 0, B b 1, C c 2; D);
zip_arity!(4; A a 0, B b 1, C c 2, D d 3; E);
zip_arity!(5; A a 0, B b 1, C c 2, D d 3, E e 4; );

macro_rules! for_each_first {
    ($N:literal; $($P:ident $p:ident $i:tt),+) => {
        impl<'o, O, $($P: NdProducer),+> Zip<(NdViewMut<'o, O>, $($P,)+)> {
            /// `for_each` for `map_collect`: the planning follows the inputs, not the new output.
            fn for_each_first(self, mut f: impl FnMut(&'o mut O, $($P::Item),+)) {
                if self.shape().contains(&0) {
                    return;
                }
                let (out, $($p,)+) = self.producers;
                let strides: [[isize; MAX_DIMS]; $N] = [out.layout_parts().1, $($p.layout_parts().1),+];
                let (ob, bases) = (out.base(), ($($p.base(),)+));
                let plan = plan(&self.shape[..self.ndim], &strides, 1);
                execute(&plan, |at, n, step| {
                    for k in 0..n as isize {
                        // SAFETY: inside every operand's layout; the output is injective
                        unsafe { f(<NdViewMut<'o, O> as NdProducer>::item(ob, at[0] + k * step[0]), $($P::item(bases.$i, at[$i + 1] + k * step[$i + 1])),+) }
                    }
                });
            }
        }
    };
}

for_each_first!(2; A a 0);
for_each_first!(3; A a 0, B b 1);
for_each_first!(4; A a 0, B b 1, C c 2);
for_each_first!(5; A a 0, B b 1, C c 2, D d 3);
for_each_first!(6; A a 0, B b 1, C c 2, D d 3, E e 4);

/// The shape two shapes broadcast to (NumPy rules), as a fixed-size array and its length.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<([usize; MAX_DIMS], usize), NdError> {
    let n = a.len().max(b.len());
    if n > MAX_DIMS {
        return Err(NdError::TooManyDims(n));
    }
    let mut out = [0; MAX_DIMS];
    for i in 0..n {
        let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (x, y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => return Err(NdError::Broadcast { axis: i, from: x.min(y), to: x.max(y) }),
        };
    }
    Ok((out, n))
}

// ELEMENT-WISE ====================================================================================

impl<'a, T> NdViewMut<'a, T> {
    /// Applies `f` to every element, in memory order.
    pub fn map_inplace(&mut self, f: impl FnMut(&mut T)) {
        Zip::from(self.reborrow()).for_each(f);
    }
    /// Calls `f(self_element, other_element)` at every index, with `other` broadcast to this
    /// view's shape.
    pub fn zip_mut_with<U>(&mut self, other: &NdView<'_, U>, f: impl FnMut(&mut T, &U)) -> Result<(), NdError> {
        Zip::from(self.reborrow()).and_broadcast(*other)?.for_each(f);
        Ok(())
    }
}

impl<'a, T: Copy> NdViewMut<'a, T> {
    /// Copies `src` (broadcast to this view's shape) into the view.
    pub fn assign(&mut self, src: &NdView<'_, T>) -> Result<(), NdError> {
        self.zip_mut_with(src, |a, &b| *a = b)
    }
}

impl<'a, T> NdView<'a, T> {
    /// `f` of every element, as a new array of the same shape (allocates).
    pub fn map<R: Copy + Default>(&self, f: impl FnMut(&T) -> R) -> NdArray<R> {
        Zip::from(*self).map_collect(f)
    }
}

impl<T> NdArray<T> {
    pub fn map_inplace(&mut self, f: impl FnMut(&mut T)) {
        self.view_mut().map_inplace(f);
    }
    pub fn map<R: Copy + Default>(&self, f: impl FnMut(&T) -> R) -> NdArray<R> {
        self.view().map(f)
    }
}

/// Element types usable as the scalar operand of array arithmetic (`array * 2.0`).
pub trait Scalar: Copy {}
macro_rules! scalars {
    ($($t:ty),+) => {$(impl Scalar for $t {})+};
}
scalars!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl<T: Float> Scalar for Complex<T> {}

macro_rules! arithmetic {
    ($Op:ident $op:ident $OpAssign:ident $op_assign:ident) => {
        /// In place with a scalar.
        impl<'a, T: Scalar + $Op<Output = T>> $OpAssign<T> for NdViewMut<'a, T> {
            fn $op_assign(&mut self, rhs: T) {
                self.map_inplace(|x| *x = $Op::$op(*x, rhs));
            }
        }
        /// In place with another view, broadcast to this one's shape. Panics if it can't be
        /// (use `zip_mut_with` to handle that as an error).
        impl<'a, 'b, T: Copy + $Op<Output = T>> $OpAssign<&NdView<'b, T>> for NdViewMut<'a, T> {
            fn $op_assign(&mut self, rhs: &NdView<'b, T>) {
                self.zip_mut_with(rhs, |x, &y| *x = $Op::$op(*x, y)).expect("operands broadcast");
            }
        }
        impl<T: Scalar + $Op<Output = T>> $OpAssign<T> for NdArray<T> {
            fn $op_assign(&mut self, rhs: T) {
                self.view_mut().$op_assign(rhs);
            }
        }
        impl<'b, T: Copy + $Op<Output = T>> $OpAssign<&NdView<'b, T>> for NdArray<T> {
            fn $op_assign(&mut self, rhs: &NdView<'b, T>) {
                self.view_mut().$op_assign(rhs);
            }
        }
        impl<T: Copy + $Op<Output = T>> $OpAssign<&NdArray<T>> for NdArray<T> {
            fn $op_assign(&mut self, rhs: &NdArray<T>) {
                self.view_mut().$op_assign(&rhs.view());
            }
        }
        /// A new array: both views broadcast to their common shape. Panics if they can't be.
        impl<'a, 'b, T: Copy + Default + $Op<Output = T>> $Op<NdView<'b, T>> for NdView<'a, T> {
            type Output = NdArray<T>;
            fn $op(self, rhs: NdView<'b, T>) -> NdArray<T> {
                let (shape, n) = broadcast_shapes(self.shape(), rhs.shape()).expect("operands broadcast");
                let (a, b) = (self.broadcast_to(&shape[..n]).expect("broadcasts"), rhs.broadcast_to(&shape[..n]).expect("broadcasts"));
                Zip::from(a).and(b).expect("same shape").map_collect(|&x, &y| $Op::$op(x, y))
            }
        }
        impl<'a, T: Scalar + Default + $Op<Output = T>> $Op<T> for NdView<'a, T> {
            type Output = NdArray<T>;
            fn $op(self, rhs: T) -> NdArray<T> {
                self.map(|&x| $Op::$op(x, rhs))
            }
        }
        impl<T: Copy + Default + $Op<Output = T>> $Op<&NdArray<T>> for &NdArray<T> {
            type Output = NdArray<T>;
            fn $op(self, rhs: &NdArray<T>) -> NdArray<T> {
                $Op::$op(self.view(), rhs.view())
            }
        }
        impl<T: Scalar + Default + $Op<Output = T>> $Op<T> for &NdArray<T> {
            type Output = NdArray<T>;
            fn $op(self, rhs: T) -> NdArray<T> {
                $Op::$op(self.view(), rhs)
            }
        }
    };
}

arithmetic!(Add add AddAssign add_assign);
arithmetic!(Sub sub SubAssign sub_assign);
arithmetic!(Mul mul MulAssign mul_assign);
arithmetic!(Div div DivAssign div_assign);

// REDUCTIONS ======================================================================================

/// Pairwise sum of `n` elements from `ptr` at `stride`: O(log n) rounding error growth instead of
/// O(n), in a fixed order (so results are reproducible). Blocks of contiguous elements are summed
/// with eight independent accumulators, which the compiler turns into vector adds.
fn pairwise_sum<T: Float>(ptr: *const T, n: usize, stride: isize) -> T {
    const BLOCK: usize = 256;
    if n <= BLOCK {
        if stride == 1 {
            // SAFETY: the caller passes a run of `n` consecutive elements inside a validated layout
            let run = unsafe { std::slice::from_raw_parts(ptr, n) };
            let mut acc = [T::_ZERO; 8];
            let (chunks, tail) = run.as_chunks::<8>();
            for chunk in chunks {
                for (a, &x) in acc.iter_mut().zip(chunk) {
                    *a = *a + x;
                }
            }
            let mut total = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for &x in tail {
                total = total + x;
            }
            return total;
        }
        let mut acc = T::_ZERO;
        for k in 0..n as isize {
            // SAFETY: as above, at `stride`
            acc = acc + unsafe { *ptr.wrapping_offset(k * stride) };
        }
        return acc;
    }
    let half = n / 2;
    pairwise_sum(ptr, half, stride) + pairwise_sum(ptr.wrapping_offset(half as isize * stride), n - half, stride)
}

/// How a reduction combines input elements into an output element: one at a time, or a whole run
/// that lands on the same output element (where a reducer can do better than element by element).
trait Reducer<T, U> {
    fn element(&mut self, acc: U, x: &T) -> U;
    /// `acc` combined with the `n` elements at `ptr`, `stride` apart.
    fn run(&mut self, mut acc: U, ptr: *const T, n: usize, stride: isize) -> U {
        for k in 0..n as isize {
            // SAFETY: the caller passes a run inside a validated layout
            acc = self.element(acc, unsafe { &*ptr.wrapping_offset(k * stride) });
        }
        acc
    }
}

struct FoldWith<F>(F);
impl<T, U, F: FnMut(U, &T) -> U> Reducer<T, U> for FoldWith<F> {
    #[inline(always)]
    fn element(&mut self, acc: U, x: &T) -> U {
        (self.0)(acc, x)
    }
}

struct Summing;
impl<T: Float> Reducer<T, T> for Summing {
    #[inline(always)]
    fn element(&mut self, acc: T, x: &T) -> T {
        acc + *x
    }
    fn run(&mut self, acc: T, ptr: *const T, n: usize, stride: isize) -> T {
        acc + pairwise_sum(ptr, n, stride)
    }
}

impl<'a, T> NdView<'a, T> {
    /// Combines every element into one value with `f`, in memory order.
    pub fn fold<A>(&self, init: A, f: impl FnMut(A, &T) -> A) -> A {
        Zip::from(*self).fold(init, f)
    }

    /// Reduces into `out`, whose shape broadcasts to this view's (e.g. `[rows, 1]` or `[rows]`...
    /// see below): every element of `out` is set to `init`, then combined with each input element
    /// that maps onto it, `*o = f(*o, x)`, in a fixed order. This is the write side of
    /// broadcasting: the axes where `out` has length 1 (or is missing, at the front) are reduced.
    pub fn fold_into<U: Copy>(&self, out: &mut NdViewMut<'_, U>, init: U, f: impl FnMut(U, &T) -> U) -> Result<(), NdError> {
        self.reduce_into(out, init, FoldWith(f))
    }

    fn reduce_into<U: Copy>(&self, out: &mut NdViewMut<'_, U>, init: U, mut reducer: impl Reducer<T, U>) -> Result<(), NdError> {
        // out's layout repeated over the reduced axes: only ever touched through raw pointers here,
        // one read-modify-write at a time, so the repetition is sound
        let target = out.layout.broadcast_to(self.shape())?;
        out.map_inplace(|o| *o = init);
        if self.is_empty() {
            return Ok(());
        }
        let strides = [target.strides, self.layout.strides];
        let (op, ip) = (out.ptr, self.ptr);
        let plan = plan(self.shape(), &strides, 1);
        execute(&plan, |at, n, step| {
            // SAFETY (both branches): every offset is inside its layout, and no reference to an
            // output element outlives its statement
            if step[0] == 0 {
                // the whole run lands on one output element: combine it in one go
                unsafe {
                    let o = op.wrapping_offset(at[0]);
                    *o = reducer.run(*o, ip.wrapping_offset(at[1]), n, step[1]);
                }
            } else {
                for k in 0..n as isize {
                    unsafe {
                        let o = op.wrapping_offset(at[0] + k * step[0]);
                        *o = reducer.element(*o, &*ip.wrapping_offset(at[1] + k * step[1]));
                    }
                }
            }
        });
        Ok(())
    }

    /// The smallest element by `PartialOrd` (NaNs are skipped); `None` when empty or all NaN.
    pub fn min(&self) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        self.fold(None, |m: Option<T>, &x| match m {
            Some(m) if x.partial_cmp(&m) != Some(std::cmp::Ordering::Less) => Some(m),
            _ if x.partial_cmp(&x).is_some() => Some(x),
            m => m,
        })
    }
    /// The largest element (NaNs are skipped).
    pub fn max(&self) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        self.fold(None, |m: Option<T>, &x| match m {
            Some(m) if x.partial_cmp(&m) != Some(std::cmp::Ordering::Greater) => Some(m),
            _ if x.partial_cmp(&x).is_some() => Some(x),
            m => m,
        })
    }
}

impl<'a, T: Float> NdView<'a, T> {
    /// Sum of every element (pairwise along each run, so accurate for long arrays).
    pub fn sum(&self) -> T {
        if self.is_empty() {
            return T::_ZERO;
        }
        let strides = [self.layout.strides];
        let ptr = self.ptr;
        let plan = plan(self.shape(), &strides, 0);
        let mut total = T::_ZERO;
        execute(&plan, |at, n, step| total = total + pairwise_sum(ptr.wrapping_offset(at[0]), n, step[0]));
        total
    }
    /// Mean of every element; `None` when empty.
    pub fn mean(&self) -> Option<T> {
        (!self.is_empty()).then(|| self.sum() / T::_lit(self.len() as f64))
    }
    /// Root mean square; `None` when empty.
    pub fn rms(&self) -> Option<T> {
        (!self.is_empty()).then(|| (self.fold(T::_ZERO, |acc, &x| acc + x * x) / T::_lit(self.len() as f64))._sqrt())
    }
    /// Largest absolute value (0 when empty).
    pub fn peak(&self) -> T {
        self.fold(T::_ZERO, |m, &x| m._max(x._abs()))
    }
    /// Sums into `out` over the axes where it broadcasts (see [`fold_into`](Self::fold_into)).
    pub fn sum_into(&self, out: &mut NdViewMut<'_, T>) -> Result<(), NdError> {
        self.reduce_into(out, T::_ZERO, Summing)
    }
    /// Sum along `axis`, which is removed (allocates the result).
    pub fn sum_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        let shape = self.layout.without_axis_checked(axis)?;
        let mut out = NdArray::full(shape.shape(), T::_ZERO)?;
        let mut target = out.view_mut().insert_axis(axis)?;
        self.sum_into(&mut target)?;
        Ok(out)
    }
    /// Mean along `axis`, which is removed (allocates the result).
    pub fn mean_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        let mut sums = self.sum_axis(axis)?;
        let n = T::_lit(self.shape()[axis] as f64);
        sums.map_inplace(|x| *x = *x / n);
        Ok(sums)
    }
}

impl<T: Float> NdArray<T> {
    /// Sum along `axis`, which is removed.
    pub fn sum_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        self.view().sum_axis(axis)
    }
    /// Mean along `axis`, which is removed.
    pub fn mean_axis(&self, axis: usize) -> Result<NdArray<T>, NdError> {
        self.view().mean_axis(axis)
    }
}

impl Layout {
    fn without_axis_checked(&self, axis: usize) -> Result<Layout, NdError> {
        if axis >= self.ndim {
            return Err(NdError::AxisOutOfRange { axis, ndim: self.ndim });
        }
        Ok(self.without_axis(axis))
    }
}

// PROCESSING ALONG LANES ==========================================================================

/// Elements staged per call when processing strided lanes.
pub const LANE_CHUNK: usize = 256;

impl<'a, T: Copy + Default> NdViewMut<'a, T> {
    /// Streams every lane along `axis` through `f(lane_index, piece)` in consecutive pieces, in
    /// order: a contiguous lane arrives whole, a strided one in pieces of up to [`LANE_CHUNK`]
    /// staged through a stack buffer and written back. Never allocates, so it suits the audio
    /// thread; any block processor (a filter, a compressor) works this way, since its state
    /// carries from one piece to the next. Use `for_each_lane` instead when `f` needs a whole lane
    /// at once (sorting, reversing).
    pub fn for_each_lane_chunked(&mut self, axis: usize, mut f: impl FnMut(usize, &mut [T])) -> Result<(), NdError> {
        let mut buffer = [T::default(); LANE_CHUNK];
        for (index, lane) in self.lanes_mut(axis)?.enumerate() {
            if lane.is_contiguous() {
                f(index, lane.into_slice().expect("contiguous"));
                continue;
            }
            let (n, stride, ptr) = (lane.layout.shape[0], lane.layout.strides[0], lane.ptr);
            let mut start = 0;
            while start < n {
                let len = (n - start).min(LANE_CHUNK);
                for (k, b) in buffer[..len].iter_mut().enumerate() {
                    // SAFETY: indices start..start + len of a validated 1-D lane
                    *b = unsafe { *ptr.wrapping_offset((start + k) as isize * stride) };
                }
                f(index, &mut buffer[..len]);
                for (k, &b) in buffer[..len].iter().enumerate() {
                    // SAFETY: as above, exclusively borrowed through `lane`
                    unsafe { *ptr.wrapping_offset((start + k) as isize * stride) = b };
                }
                start += len;
            }
        }
        Ok(())
    }

    /// Runs `processors[i]` over lane `i` along `axis` (e.g. one filter per channel of a
    /// `[batch, channel, time]` tensor along time). Allocation-free; errors unless there is one
    /// processor per lane.
    pub fn process_lanes<P: Processor<T>>(&mut self, axis: usize, processors: &mut [P]) -> Result<(), NdError>
    where
        T: Float,
    {
        let lanes = self.lanes_mut(axis)?.len();
        if lanes != processors.len() {
            return Err(NdError::ShapeMismatch { expected: lanes, got: processors.len() });
        }
        self.for_each_lane_chunked(axis, |i, piece| processors[i].process(piece))
    }
}

impl<T: Copy + Default> NdArray<T> {
    /// See [`NdViewMut::for_each_lane_chunked`].
    pub fn for_each_lane_chunked(&mut self, axis: usize, f: impl FnMut(usize, &mut [T])) -> Result<(), NdError> {
        self.view_mut().for_each_lane_chunked(axis, f)
    }
}

impl<T: Float + Default> NdArray<T> {
    /// See [`NdViewMut::process_lanes`].
    pub fn process_lanes<P: Processor<T>>(&mut self, axis: usize, processors: &mut [P]) -> Result<(), NdError> {
        self.view_mut().process_lanes(axis, processors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::osc::Noise;

    fn counting(shape: &[usize]) -> NdArray<f64> {
        let mut k = 0.0;
        NdArray::from_fn(shape, |_| {
            k += 1.0;
            k - 1.0
        })
        .unwrap()
    }

    #[test]
    fn plans_merge_contiguous_runs() {
        // a contiguous [4, 5, 6] array is one run of 120
        let p = plan(&[4, 5, 6], &[[30, 6, 1, 0, 0, 0, 0, 0]], 0);
        assert_eq!((p.ndim, p.dims[0], p.strides[0][0]), (1, 120, 1));
        // a column slice of a row-major matrix stays two loops
        let p = plan(&[4, 3], &[[10, 1, 0, 0, 0, 0, 0, 0]], 0);
        assert_eq!((p.ndim, &p.dims[..2]), (2, &[4, 3][..]));
        // a transposed array is walked in memory order
        let p = plan(&[6, 5], &[[1, 6, 0, 0, 0, 0, 0, 0]], 0);
        assert_eq!((p.ndim, p.dims[0], p.strides[0][0]), (1, 30, 1));
        // a broadcast operand blocks merging where its stride is 0
        let p = plan(&[4, 5], &[[5, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], 0);
        assert_eq!(p.ndim, 2);
    }

    #[test]
    fn zip_matches_index_by_index_reference_on_any_layout() {
        let a = counting(&[4, 6, 5]);
        let b = counting(&[5, 6, 4]);
        let views = [
            a.view(),
            a.view().flip(1).unwrap(),
            b.view().transpose(),
            a.view().step_axis(2, -2).unwrap().slice_axis(0, 1..4).unwrap(),
        ];
        for x in views {
            for y in views {
                if x.shape() != y.shape() {
                    continue;
                }
                let mut out = NdArray::<f64>::zeros(x.shape()).unwrap();
                let mut flipped = NdArray::<f64>::zeros(x.shape()).unwrap();
                Zip::from(out.view_mut()).and(x).unwrap().and(y).unwrap().for_each(|o, &p, &q| *o = 2.0 * p - q);
                // the output itself may be written through a reversed view
                Zip::from(flipped.view_mut().flip(0).unwrap()).and(x).unwrap().for_each(|o, &p| *o = p);
                let mut reference = Vec::new();
                for (p, q) in x.iter().zip(y.iter()) {
                    reference.push(2.0 * p - q);
                }
                assert_eq!(out.as_slice(), &reference[..]);
                assert_eq!(flipped.view().flip(0).unwrap().to_vec(), x.to_vec());
            }
        }
        assert!(Zip::from(a.view()).and(b.view()).is_err());
    }

    #[test]
    fn arithmetic_broadcasts_like_numpy() {
        let m = counting(&[2, 3]);
        let row = NdArray::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let col = NdArray::from_vec(vec![100.0, 200.0], &[2, 1]).unwrap();
        assert_eq!((&m + &row).as_slice(), [10.0, 21.0, 32.0, 13.0, 24.0, 35.0]);
        assert_eq!((col.view() + row.view()).shape(), [2, 3]);
        assert_eq!((col.view() + row.view()).as_slice(), [110.0, 120.0, 130.0, 210.0, 220.0, 230.0]);
        assert_eq!((&m * 2.0).as_slice(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let mut x = m.clone();
        x -= &row.view();
        x /= 2.0;
        assert_eq!(x.as_slice(), [-5.0, -9.5, -14.0, -3.5, -8.0, -12.5]);
        assert!(broadcast_shapes(&[2, 3], &[4]).is_err());
        assert_eq!(broadcast_shapes(&[5, 1, 3], &[4, 1]).map(|(s, n)| s[..n].to_vec()).unwrap(), [5, 4, 3]);
    }

    #[test]
    fn reductions_on_strided_views() {
        let a = counting(&[3, 4]);
        // a strided column, no copy
        let col = a.view().index_axis(1, 2).unwrap();
        assert_eq!((col.sum(), col.mean(), col.max(), col.min()), (2.0 + 6.0 + 10.0, Some(6.0), Some(10.0), Some(2.0)));
        assert_eq!(a.view().flip(0).unwrap().sum(), 66.0);
        assert_eq!(a.sum_axis(0).unwrap().as_slice(), [12.0, 15.0, 18.0, 21.0]);
        assert_eq!(a.sum_axis(1).unwrap().as_slice(), [6.0, 22.0, 38.0]);
        assert_eq!(a.mean_axis(1).unwrap().as_slice(), [1.5, 5.5, 9.5]);
        // fold_into: the write side of broadcasting (max over rows into a [1, 4] array)
        let mut out = NdArray::<f64>::zeros(&[1, 4]).unwrap();
        a.view().fold_into(&mut out.view_mut(), f64::MIN, |m, &x| m.max(x)).unwrap();
        assert_eq!(out.as_slice(), [8.0, 9.0, 10.0, 11.0]);
        assert!(a.view().sum_into(&mut NdArray::<f64>::zeros(&[2]).unwrap().view_mut()).is_err());
        let rms = NdArray::from_vec(vec![3.0, -4.0], &[2]).unwrap();
        assert_eq!((rms.view().rms(), rms.view().peak()), (Some(12.5f64.sqrt()), 4.0));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // ten million elements: too slow under Miri
    fn pairwise_summation_keeps_long_sums_accurate() {
        let tenths = NdArray::full(&[10_000_000], 0.1f32).unwrap();
        let sum = tenths.view().sum();
        assert!((sum - 1_000_000.0).abs() < 1.0, "{sum}");
        // a naive running sum is off by about 1%
        let naive = tenths.as_slice().iter().fold(0.0f32, |a, &x| a + x);
        assert!((naive - 1_000_000.0).abs() > 1_000.0);
    }

    #[test]
    fn lanes_process_in_chunks_without_allocating() {
        // [time, channel]: time is strided; filtering along it in pieces equals filtering whole
        let n = 2_000;
        let noise: Vec<f64> = Noise::new(3).take(n * 2).collect();
        let mut interleaved = NdArray::from_vec(noise.clone(), &[n, 2]).unwrap();
        let mut filters = [Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q), Biquad::highpass(500.0, 48_000.0, BUTTERWORTH_Q)];
        interleaved.process_lanes(0, &mut filters).unwrap();
        for ch in 0..2 {
            let mut reference: Vec<f64> = noise.iter().skip(ch).step_by(2).copied().collect();
            let mut f = if ch == 0 { Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q) } else { Biquad::highpass(500.0, 48_000.0, BUTTERWORTH_Q) };
            f.process(&mut reference);
            let got = interleaved.view().index_axis(1, ch).unwrap().to_vec();
            let err = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12, "channel {ch}: {err}");
        }
        assert!(interleaved.process_lanes(0, &mut filters[..1]).is_err());
    }
}
