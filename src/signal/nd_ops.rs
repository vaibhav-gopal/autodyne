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

use crate::alloc_prelude::*;
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ndarray::{Layout, NdArray, NdError, NdView, NdViewMut, MAX_DIMS};
use super::{Storage, StorageMut};
use crate::processor::Processor;
use crate::units::*;

// LOOP PLANNING ===================================================================================

/// A loop nest over operands that share a shape: the merged axes (outer first) and each operand's
/// stride along them.
struct Plan<const N: usize> {
    ndim: usize,
    dims: [usize; MAX_DIMS],
    strides: [[isize; MAX_DIMS]; N],
    /// walk the last two axes in TILE x TILE blocks (an operand runs fastest along the other one)
    tiled: bool,
}

/// Side of the blocks used when operands disagree on their fastest axis (a transpose): 64 x 64
/// elements keep every touched line in L1 cache.
const TILE: usize = 64;

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
    let mut p = Plan { ndim: 0, dims: [1; MAX_DIMS], strides: [[0; MAX_DIMS]; N], tiled: false };
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
    if p.ndim >= 2 {
        let (outer, inner) = (p.ndim - 2, p.ndim - 1);
        // an operand that steps faster along the outer axis than the inner one is read across
        // lines: block both axes so those lines are reused while they are still in cache
        let crosses = p.strides.iter().any(|s| s[outer] != 0 && s[outer].unsigned_abs() < s[inner].unsigned_abs());
        p.tiled = crosses && p.dims[outer] > TILE && p.dims[inner] > TILE;
    }
    p
}

/// Calls `f` with the operands' offsets at every index of the axes `0..axes` (row-major).
#[inline]
fn walk<const N: usize>(p: &Plan<N>, axes: usize, mut f: impl FnMut([isize; N])) {
    let mut index = [0usize; MAX_DIMS];
    let mut offsets = [0isize; N];
    loop {
        f(offsets);
        let mut axis = axes;
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

/// Runs the plan: `body(offsets, length, inner_strides)` for each inner run, where `offsets` are the
/// operands' element offsets at the run's start. Tiled plans give shorter runs, block by block.
#[inline]
fn execute<const N: usize>(p: &Plan<N>, mut body: impl FnMut([isize; N], usize, [isize; N])) {
    let inner = p.ndim - 1;
    let inner_strides: [isize; N] = core::array::from_fn(|k| p.strides[k][inner]);
    if !p.tiled {
        walk(p, inner, |at| body(at, p.dims[inner], inner_strides));
        return;
    }
    let outer = inner - 1;
    let (n_outer, n_inner) = (p.dims[outer], p.dims[inner]);
    walk(p, outer, |base| {
        let mut o0 = 0;
        while o0 < n_outer {
            let o1 = (o0 + TILE).min(n_outer);
            let mut i0 = 0;
            while i0 < n_inner {
                let len = TILE.min(n_inner - i0);
                for o in o0..o1 {
                    let at: [isize; N] = core::array::from_fn(|k| base[k] + o as isize * p.strides[k][outer] + i0 as isize * p.strides[k][inner]);
                    body(at, len, inner_strides);
                }
                i0 += len;
            }
            o0 = o1;
        }
    });
}
// ZIP =============================================================================================

mod sealed {
    pub trait Sealed {}
}

/// An operand of a [`Zip`]: a view (yielding `&T`) or a mutable view (yielding `&mut T`).
pub trait NdProducer: sealed::Sealed {
    /// What each element is seen as: `&T` or `&mut T`.
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
    /// The shape every operand is iterated over.
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
            /// Collects `f` of every index into a new array of the zip's shape (allocates; the new
            /// memory is written once, never zero-filled first).
            pub fn map_collect<R: Copy>(self, mut f: impl FnMut($($P::Item),+) -> R) -> NdArray<R> {
                let len: usize = self.shape().iter().product();
                let mut data: Vec<MaybeUninit<R>> = Vec::with_capacity(len);
                // SAFETY: MaybeUninit elements need no initialization
                unsafe { data.set_len(len) };
                let mut out = NdArray::from_vec(data, self.shape()).expect("the zip's shape is valid");
                let zip = Zip { producers: (out.view_mut(), $(self.producers.$i,)+), shape: self.shape, ndim: self.ndim };
                zip.for_each_first(|o, $($p),+| {
                    o.write(f($($p),+));
                });
                // every index was written exactly once (the zip visits each index once)
                let shape = out.layout.shape;
                let mut data = core::mem::ManuallyDrop::new(out.into_vec());
                // SAFETY: all `len` elements are initialized; MaybeUninit<R> has R's layout
                let data = unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<R>(), data.len(), data.capacity()) };
                NdArray::from_vec(data, &shape[..self.ndim]).expect("the zip's shape is valid")
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
                    // unit steps (contiguous runs) as their own loop, so it vectorizes
                    if step.iter().all(|&s| s == 1) {
                        for k in 0..n as isize {
                            // SAFETY: inside every operand's layout; the output is injective
                            unsafe { f(<NdViewMut<'o, O> as NdProducer>::item(ob, at[0] + k), $($P::item(bases.$i, at[$i + 1] + k)),+) }
                        }
                    } else {
                        for k in 0..n as isize {
                            // SAFETY: as above
                            unsafe { f(<NdViewMut<'o, O> as NdProducer>::item(ob, at[0] + k * step[0]), $($P::item(bases.$i, at[$i + 1] + k * step[$i + 1])),+) }
                        }
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

impl<T, S: StorageMut<Elem = T>> NdArray<T, S> {
    /// Applies `f` to every element in place.
    pub fn map_inplace(&mut self, f: impl FnMut(&mut T)) {
        self.view_mut().map_inplace(f);
    }
}

impl<T, S: Storage<Elem = T>> NdArray<T, S> {
    /// A new array of `f` applied to every element.
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
        impl<T: Scalar + $Op<Output = T>, S: StorageMut<Elem = T>> $OpAssign<T> for NdArray<T, S> {
            fn $op_assign(&mut self, rhs: T) {
                self.view_mut().$op_assign(rhs);
            }
        }
        impl<'b, T: Copy + $Op<Output = T>, S: StorageMut<Elem = T>> $OpAssign<&NdView<'b, T>> for NdArray<T, S> {
            fn $op_assign(&mut self, rhs: &NdView<'b, T>) {
                self.view_mut().$op_assign(rhs);
            }
        }
        impl<T: Copy + $Op<Output = T>, S: StorageMut<Elem = T>, S2: Storage<Elem = T>> $OpAssign<&NdArray<T, S2>> for NdArray<T, S> {
            fn $op_assign(&mut self, rhs: &NdArray<T, S2>) {
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
        impl<T: Copy + Default + $Op<Output = T>, S: Storage<Elem = T>, S2: Storage<Elem = T>> $Op<&NdArray<T, S2>> for &NdArray<T, S> {
            type Output = NdArray<T>;
            fn $op(self, rhs: &NdArray<T, S2>) -> NdArray<T> {
                $Op::$op(self.view(), rhs.view())
            }
        }
        impl<T: Scalar + Default + $Op<Output = T>, S: Storage<Elem = T>> $Op<T> for &NdArray<T, S> {
            type Output = NdArray<T>;
            fn $op(self, rhs: T) -> NdArray<T> {
                $Op::$op(self.view(), rhs)
            }
        }
        /// A new array from two owned ones, broadcast to their common shape. Panics if they can't be.
        impl<T: Copy + Default + $Op<Output = T>> $Op for NdArray<T> {
            type Output = NdArray<T>;
            fn $op(self, rhs: NdArray<T>) -> NdArray<T> {
                $Op::$op(self.view(), rhs.view())
            }
        }
    };
}

arithmetic!(Add add AddAssign add_assign);
arithmetic!(Sub sub SubAssign sub_assign);
arithmetic!(Mul mul MulAssign mul_assign);
arithmetic!(Div div DivAssign div_assign);

impl<T: Copy + Default + Neg<Output = T>> Neg for NdArray<T> {
    type Output = NdArray<T>;
    fn neg(mut self) -> NdArray<T> {
        self.map_inplace(|x| *x = -*x);
        self
    }
}

// REDUCTIONS ======================================================================================

/// Bytes of independent partial sums for contiguous runs: enough separate add chains to keep the
/// vector units busy (eight 128-bit vectors: 32 f32 or 16 f64), so summing in cache isn't
/// latency-bound, without spilling registers.
const LANE_BYTES: usize = 128;
/// Elements summed directly before pairwise combination takes over.
const BLOCK: usize = 1_024;

/// LANES partial sums of a contiguous run, pairwise: runs of up to BLOCK elements are summed lane by
/// lane, and longer runs split at a block boundary near the middle, their halves' partial sums added
/// lane-wise (recursive halving still reads memory front to back). Elements after the last whole
/// chunk of LANES are left out; only the final piece of a run can have them.
fn lanes_pairwise<T: Float, const LANES: usize>(run: &[T]) -> [T; LANES] {
    if run.len() <= BLOCK {
        let mut acc = [T::_ZERO; LANES];
        for chunk in run.as_chunks::<LANES>().0 {
            for (a, &x) in acc.iter_mut().zip(chunk) {
                *a = *a + x;
            }
        }
        return acc;
    }
    let mid = run.len().div_ceil(BLOCK) / 2 * BLOCK;
    let (a, b) = (lanes_pairwise::<T, LANES>(&run[..mid]), lanes_pairwise::<T, LANES>(&run[mid..]));
    core::array::from_fn(|i| a[i] + b[i])
}

fn contiguous_sum<T: Float, const LANES: usize>(run: &[T]) -> T {
    let mut acc = lanes_pairwise::<T, LANES>(run);
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] = acc[i] + acc[i + width];
        }
    }
    let n = run.len();
    run[n - n % LANES..].iter().fold(acc[0], |s, &x| s + x)
}

/// Pairwise sum of `n` elements from `ptr` at `stride`: O(log n) rounding error growth instead of
/// O(n), in a fixed order (so results are reproducible). Contiguous runs keep LANES independent
/// partial sums all the way up the tree and combine them once at the end.
fn pairwise_sum<T: Float>(ptr: *const T, n: usize, stride: isize) -> T {
    if stride == 1 {
        // SAFETY: the caller passes a run of `n` consecutive elements inside a validated layout
        let run = unsafe { core::slice::from_raw_parts(ptr, n) };
        return match LANE_BYTES / core::mem::size_of::<T>() {
            16 => contiguous_sum::<T, 16>(run),
            _ => contiguous_sum::<T, 32>(run),
        };
    }
    let strided = |start: usize, len: usize| {
        let mut acc = T::_ZERO;
        for k in start..start + len {
            // SAFETY: as above, at `stride`
            acc = acc + unsafe { *ptr.wrapping_offset(k as isize * stride) };
        }
        acc
    };
    fn halves<T: Float>(start: usize, len: usize, leaf: &impl Fn(usize, usize) -> T) -> T {
        if len <= BLOCK {
            return leaf(start, len);
        }
        let mid = len.div_ceil(BLOCK) / 2 * BLOCK;
        halves(start, mid, leaf) + halves(start + mid, len - mid, leaf)
    }
    halves(0, n, &strided)
}/// How a reduction combines input elements into an output element: one at a time, or a whole run
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
            } else if step[0] == 1 && step[1] == 1 {
                // both runs contiguous (e.g. column sums of a row-major matrix): plain slices, which
                // the compiler vectorizes
                unsafe {
                    let out = core::slice::from_raw_parts_mut(op.wrapping_offset(at[0]), n);
                    let input = core::slice::from_raw_parts(ip.wrapping_offset(at[1]), n);
                    for (o, x) in out.iter_mut().zip(input) {
                        *o = reducer.element(*o, x);
                    }
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
            Some(m) if x.partial_cmp(&m) != Some(core::cmp::Ordering::Less) => Some(m),
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
            Some(m) if x.partial_cmp(&m) != Some(core::cmp::Ordering::Greater) => Some(m),
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
        // in memory order, a summed axis that isn't the innermost sums whole rows at a time
        let n = self.ndim();
        let order = self.memory_order();
        let order = &order[..n];
        let at = order.iter().position(|&a| a == axis).expect("a permutation");
        if at + 1 < n {
            let p = self.permute(order)?;
            if let Some(data) = p.as_slice() {
                let ps = p.shape();
                let sums = sum_rows(data, &ps[..at], ps[at], ps[at + 1..].iter().product());
                // result axis i is the input's axis kept[i]; put them back in ascending order
                let kept: Vec<usize> = order.iter().copied().filter(|&a| a != axis).collect();
                let in_memory = NdArray::from_vec(sums, &kept.iter().map(|&a| self.shape()[a]).collect::<Vec<_>>())?;
                let mut sorted = kept.clone();
                sorted.sort_unstable();
                let perm: Vec<usize> = sorted.iter().map(|a| kept.iter().position(|b| b == a).expect("kept")).collect();
                if perm.iter().enumerate().all(|(i, &p)| i == p) {
                    return Ok(in_memory);
                }
                return Ok(in_memory.view().permute(&perm)?.to_owned());
            }
        }
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

/// Sums the `rows` rows (of `inner` elements) in each of the `outer` blocks of row-major `data`:
/// four rows per pass, so the output is read and written a quarter as often as row by row.
fn sum_rows<T: Float>(data: &[T], outer: &[usize], rows: usize, inner: usize) -> Vec<T> {
    let outer: usize = outer.iter().product();
    let mut out = vec![T::_ZERO; outer * inner];
    if inner == 0 {
        return out;
    }
    for (block, acc) in data.chunks_exact(rows * inner).zip(out.chunks_exact_mut(inner)) {
        let mut quads = block.chunks_exact(4 * inner);
        for quad in &mut quads {
            let (r0, rest) = quad.split_at(inner);
            let (r1, rest) = rest.split_at(inner);
            let (r2, r3) = rest.split_at(inner);
            for ((((a, &w), &x), &y), &z) in acc.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
                *a = *a + ((w + x) + (y + z));
            }
        }
        for row in quads.remainder().chunks_exact(inner) {
            for (a, &x) in acc.iter_mut().zip(row) {
                *a = *a + x;
            }
        }
    }
    out
}

impl<T: Float, S: Storage<Elem = T>> NdArray<T, S> {
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

impl<T: Copy + Default, S: StorageMut<Elem = T>> NdArray<T, S> {
    /// See [`NdViewMut::for_each_lane_chunked`].
    pub fn for_each_lane_chunked(&mut self, axis: usize, f: impl FnMut(usize, &mut [T])) -> Result<(), NdError> {
        self.view_mut().for_each_lane_chunked(axis, f)
    }
}

impl<T: Float + Default, S: StorageMut<Elem = T>> NdArray<T, S> {
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
        assert!(!p.tiled, "a broadcast row is not a transpose");
        // a large transpose against a contiguous output is walked in blocks
        let p = plan(&[500, 400], &[[400, 1, 0, 0, 0, 0, 0, 0], [1, 500, 0, 0, 0, 0, 0, 0]], 0);
        assert!(p.tiled);
    }

    #[test]
    fn tiled_walks_visit_every_index_once() {
        // odd sizes, not multiples of the tile, through a transposed operand
        let a = counting(&[3, 130, 70]);
        let t = a.view().permute(&[0, 2, 1]).unwrap();
        let mut out = NdArray::<f64>::zeros(t.shape()).unwrap();
        let mut visits = NdArray::<u32>::zeros(t.shape()).unwrap();
        Zip::from(out.view_mut()).and(t).unwrap().and(visits.view_mut()).unwrap().for_each(|o, &x, n| {
            *o = x;
            *n += 1;
        });
        assert!(visits.as_slice().iter().all(|&n| n == 1));
        assert_eq!(out.as_slice(), &t.to_vec()[..]);
        // and a reduction through the same blocks
        let col_sums = t.sum_axis(1).unwrap();
        let reference: Vec<f64> = (0..3).flat_map(|b| (0..130).map(move |j| (0..70).map(|i| (b * 9_100 + j * 70 + i) as f64).sum())).collect();
        assert_eq!(col_sums.as_slice(), &reference[..]);
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
    fn axis_sums_agree_in_every_memory_order() {
        // all six axis orders of a 3x4x5 array, summed along each axis (some through the row-block
        // path, some through the general one), against direct sums
        let a = counting(&[3, 4, 5]).map(|v| (v * 0.37).sin());
        for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let v = a.view().permute(&perm).unwrap();
            for axis in 0..3 {
                let got = v.sum_axis(axis).unwrap();
                let mut shape = v.shape().to_vec();
                shape.remove(axis);
                assert_eq!(got.shape(), shape.as_slice());
                for (k, g) in got.as_slice().iter().enumerate() {
                    // the output index, then the sum over the removed axis
                    let (mut rest, mut index) = (k, [0usize; 3]);
                    for d in (0..3).filter(|&d| d != axis).rev() {
                        let n = v.shape()[d];
                        index[d] = rest % n;
                        rest /= n;
                    }
                    let want: f64 = (0..v.shape()[axis]).map(|i| {
                        index[axis] = i;
                        *v.get(&index).unwrap()
                    }).sum();
                    assert!((g - want).abs() < 1e-12, "perm {perm:?} axis {axis} element {k}: {g} vs {want}");
                }
            }
        }
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
        let mut filters = [Biquad::lowpass(1_000.0, BUTTERWORTH_Q, 48_000.0), Biquad::highpass(500.0, BUTTERWORTH_Q, 48_000.0)];
        interleaved.process_lanes(0, &mut filters).unwrap();
        for ch in 0..2 {
            let mut reference: Vec<f64> = noise.iter().skip(ch).step_by(2).copied().collect();
            let mut f = if ch == 0 { Biquad::lowpass(1_000.0, BUTTERWORTH_Q, 48_000.0) } else { Biquad::highpass(500.0, BUTTERWORTH_Q, 48_000.0) };
            f.process(&mut reference);
            let got = interleaved.view().index_axis(1, ch).unwrap().to_vec();
            let err = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12, "channel {ch}: {err}");
        }
        assert!(interleaved.process_lanes(0, &mut filters[..1]).is_err());
    }
}
