//! N-dimensional arrays: owned [`NdArray`] and zero-copy views ([`NdView`], [`NdViewMut`]).
//!
//! The shape of tensors in ML frameworks and of multichannel / batched signals: e.g.
//! `[batch, channel, time]`. Owned arrays are contiguous and row-major (last axis fastest). Views
//! are a pointer plus a *layout* (shape and signed element strides) over memory someone else owns,
//! so every view transform is O(1) arithmetic on the layout and never touches the data:
//!
//! - slicing ([`slice_axis`](NdView::slice_axis)), steps and reversal ([`step_axis`](NdView::step_axis),
//!   [`flip`](NdView::flip)), fixing an index ([`index_axis`](NdView::index_axis)), splitting
//!   ([`split_at`](NdView::split_at));
//! - reordering axes ([`permute`](NdView::permute), [`transpose`](NdView::transpose),
//!   [`swap_axes`](NdView::swap_axes)), adding or removing length-1 axes;
//! - broadcasting ([`broadcast_to`](NdView::broadcast_to): repeated axes get stride 0) and reshaping
//!   ([`reshape`](NdView::reshape) when the layout allows it without a copy, or
//!   [`to_shape`](NdView::to_shape), which says when it had to copy).
//!
//! A layout is checked against its buffer once, when the view is made; after that, access needs no
//! bounds checks. Mutable views must be *injective* (no two indices reach the same element): every
//! transform keeps that property, broadcasting (which breaks it) only exists for read-only views, and
//! [`NdViewMut::from_parts`] checks it. That is what makes handing out disjoint mutable lanes
//! ([`lanes_mut`](NdViewMut::lanes_mut)) and `&mut` elements sound.
//!
//! Signals live inside arrays as *lanes*: the 1-D runs along one axis. [`lanes`](NdView::lanes)
//! iterates them as views without allocating (contiguous lanes expose `as_slice`, so every `Signal`
//! method applies), and [`for_each_lane`](NdViewMut::for_each_lane) applies a `SignalMut` operation
//! or a `Processor` along any axis. Axes can be labelled ([`Axis`]) so code can find "the time axis"
//! instead of hard-coding positions. Shapes hold at most [`MAX_DIMS`] axes in fixed-size arrays, so
//! making and transforming views never allocates.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;

use thiserror::Error;

use super::{Signal, SignalError, SignalMut, SignalOwned, Storage, StorageMut};
use crate::units::*;

/// Maximum number of axes.
pub const MAX_DIMS: usize = 8;

/// What an axis means. `Unlabeled` by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Axis {
    #[default]
    Unlabeled,
    /// independent examples / instances
    Batch,
    /// samples over time
    Time,
    /// audio channels or feature maps
    Channel,
    /// spatial position (image rows/columns, sensor position)
    Spatial,
    /// feature / embedding dimension
    Feature,
    Named(&'static str),
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NdError {
    #[error("{0} dimensions exceeds the maximum of {MAX_DIMS}")]
    TooManyDims(usize),
    #[error("shape holds {expected} elements but {got} were given")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("axis {axis} is out of range for {ndim} dimensions")]
    AxisOutOfRange { axis: usize, ndim: usize },
    #[error("index or range is out of bounds")]
    OutOfBounds,
    #[error("not a permutation of the axes")]
    InvalidPermutation,
    #[error("{0} labels given for {1} dimensions")]
    LabelCount(usize, usize),
    #[error("a step of 0 is not allowed")]
    ZeroStep,
    #[error("axis {0} has length {1}, not 1")]
    NotUnitAxis(usize, usize),
    #[error("axis {axis} of length {from} can't be broadcast to {to}")]
    Broadcast { axis: usize, from: usize, to: usize },
    #[error("this layout can't be reshaped without copying (to_shape copies when needed)")]
    ReshapeNeedsCopy,
    #[error("several indices reach the same element, so the layout can't be written through")]
    Overlapping,
}

fn check_axis(axis: usize, ndim: usize) -> Result<(), NdError> {
    if axis < ndim { Ok(()) } else { Err(NdError::AxisOutOfRange { axis, ndim }) }
}

/// Calls `f` with every index of `shape` in row-major order.
fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut idx = [0usize; MAX_DIMS];
    let n = shape.len();
    loop {
        f(&idx[..n]);
        // odometer increment, last axis fastest
        let mut i = n;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            idx[i] += 1;
            if idx[i] < shape[i] {
                break;
            }
            idx[i] = 0;
        }
    }
}

// LAYOUT ==========================================================================================

/// Shape, signed element strides and labels: where each index of a view lives relative to its
/// origin (the element at index `[0, 0, ...]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Layout {
    pub(super) ndim: usize,
    pub(super) shape: [usize; MAX_DIMS],
    pub(super) strides: [isize; MAX_DIMS],
    pub(super) labels: [Axis; MAX_DIMS],
}

impl Layout {
    pub(super) fn new(shape: &[usize], strides: &[isize]) -> Result<Self, NdError> {
        if shape.len() > MAX_DIMS {
            return Err(NdError::TooManyDims(shape.len()));
        }
        if strides.len() != shape.len() {
            return Err(NdError::ShapeMismatch { expected: shape.len(), got: strides.len() });
        }
        let mut l = Self { ndim: shape.len(), shape: [0; MAX_DIMS], strides: [0; MAX_DIMS], labels: [Axis::Unlabeled; MAX_DIMS] };
        l.shape[..shape.len()].copy_from_slice(shape);
        l.strides[..strides.len()].copy_from_slice(strides);
        Ok(l)
    }
    pub(super) fn row_major(shape: &[usize]) -> Result<Self, NdError> {
        let mut strides = [0isize; MAX_DIMS];
        let n = shape.len().min(MAX_DIMS);
        let mut acc = 1isize;
        for i in (0..n).rev() {
            strides[i] = acc;
            acc = acc.saturating_mul(shape[i] as isize);
        }
        Self::new(shape, &strides[..n])
    }
    pub(super) fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    pub(super) fn strides(&self) -> &[isize] {
        &self.strides[..self.ndim]
    }
    fn labels(&self) -> &[Axis] {
        &self.labels[..self.ndim]
    }
    pub(super) fn len(&self) -> usize {
        self.shape().iter().product()
    }
    fn offset_of(&self, index: &[usize]) -> Option<isize> {
        if index.len() != self.ndim || index.iter().zip(self.shape()).any(|(&i, &d)| i >= d) {
            return None;
        }
        Some(index.iter().zip(self.strides()).map(|(&i, &s)| i as isize * s).sum())
    }
    /// Lowest and highest element offset from the origin, or `None` when empty (or the arithmetic
    /// overflows, which no real buffer allows).
    fn span(&self) -> Option<(isize, isize)> {
        if self.len() == 0 {
            return None;
        }
        let (mut lo, mut hi) = (0isize, 0isize);
        for (&d, &s) in self.shape().iter().zip(self.strides()) {
            let reach = (d as isize - 1).checked_mul(s)?;
            if reach < 0 {
                lo = lo.checked_add(reach)?;
            } else {
                hi = hi.checked_add(reach)?;
            }
        }
        Some((lo, hi))
    }
    /// Row-major contiguous (length-1 axes may have any stride).
    fn is_standard(&self) -> bool {
        if self.len() == 0 {
            return true;
        }
        let mut expected = 1isize;
        for i in (0..self.ndim).rev() {
            if self.shape[i] != 1 {
                if self.strides[i] != expected {
                    return false;
                }
                expected *= self.shape[i] as isize;
            }
        }
        true
    }
    /// A sufficient test that no two indices reach the same element: sorted by stride size, every
    /// axis must step past everything the faster axes can reach. Exact for all layouts made by
    /// slicing, stepping, permuting and reshaping; may reject exotic hand-made interleavings.
    fn is_injective(&self) -> bool {
        if self.len() == 0 {
            return true;
        }
        let mut axes = [(0usize, 0usize); MAX_DIMS]; // (|stride|, length) of axes longer than 1
        let mut n = 0;
        for (&d, &s) in self.shape().iter().zip(self.strides()) {
            if d > 1 {
                axes[n] = (s.unsigned_abs(), d);
                n += 1;
            }
        }
        axes[..n].sort_unstable();
        let mut reach = 0usize;
        for &(stride, d) in &axes[..n] {
            if stride <= reach {
                return false;
            }
            reach = match stride.checked_mul(d - 1).and_then(|r| reach.checked_add(r)) {
                Some(r) => r,
                None => return false,
            };
        }
        true
    }
    fn permute(&self, axes: &[usize]) -> Result<Layout, NdError> {
        let n = self.ndim;
        let mut seen = [false; MAX_DIMS];
        if axes.len() != n || axes.iter().any(|&a| a >= n || std::mem::replace(&mut seen[a], true)) {
            return Err(NdError::InvalidPermutation);
        }
        let mut out = *self;
        for (i, &a) in axes.iter().enumerate() {
            out.shape[i] = self.shape[a];
            out.strides[i] = self.strides[a];
            out.labels[i] = self.labels[a];
        }
        Ok(out)
    }
    /// (layout, origin shift)
    fn slice_axis(&self, axis: usize, range: Range<usize>) -> Result<(Layout, isize), NdError> {
        check_axis(axis, self.ndim)?;
        if range.start > range.end || range.end > self.shape[axis] {
            return Err(NdError::OutOfBounds);
        }
        let mut out = *self;
        out.shape[axis] = range.end - range.start;
        Ok((out, range.start as isize * self.strides[axis]))
    }
    /// Every `step`-th index along `axis`; a negative step walks backwards from the last index.
    fn step_axis(&self, axis: usize, step: isize) -> Result<(Layout, isize), NdError> {
        check_axis(axis, self.ndim)?;
        if step == 0 {
            return Err(NdError::ZeroStep);
        }
        let (d, s) = (self.shape[axis], self.strides[axis]);
        let mut out = *self;
        out.shape[axis] = d.div_ceil(step.unsigned_abs());
        out.strides[axis] = s * step;
        let shift = if step < 0 && d > 0 { (d as isize - 1) * s } else { 0 };
        Ok((out, shift))
    }
    fn index_axis(&self, axis: usize, index: usize) -> Result<(Layout, isize), NdError> {
        check_axis(axis, self.ndim)?;
        if index >= self.shape[axis] {
            return Err(NdError::OutOfBounds);
        }
        Ok((self.without_axis(axis), index as isize * self.strides[axis]))
    }
    pub(super) fn without_axis(&self, axis: usize) -> Layout {
        let mut out = *self;
        for i in axis..self.ndim - 1 {
            out.shape[i] = self.shape[i + 1];
            out.strides[i] = self.strides[i + 1];
            out.labels[i] = self.labels[i + 1];
        }
        out.ndim -= 1;
        out.shape[out.ndim] = 0;
        out.strides[out.ndim] = 0;
        out.labels[out.ndim] = Axis::Unlabeled;
        out
    }
    pub(super) fn insert_axis(&self, at: usize) -> Result<Layout, NdError> {
        if self.ndim == MAX_DIMS {
            return Err(NdError::TooManyDims(MAX_DIMS + 1));
        }
        check_axis(at, self.ndim + 1)?;
        let mut out = *self;
        for i in (at..self.ndim).rev() {
            out.shape[i + 1] = self.shape[i];
            out.strides[i + 1] = self.strides[i];
            out.labels[i + 1] = self.labels[i];
        }
        out.shape[at] = 1;
        out.strides[at] = 0;
        out.labels[at] = Axis::Unlabeled;
        out.ndim += 1;
        Ok(out)
    }
    fn remove_axis(&self, axis: usize) -> Result<Layout, NdError> {
        check_axis(axis, self.ndim)?;
        if self.shape[axis] != 1 {
            return Err(NdError::NotUnitAxis(axis, self.shape[axis]));
        }
        Ok(self.without_axis(axis))
    }
    /// NumPy rules: shapes align at the last axis; a length-1 axis repeats (stride 0) and missing
    /// leading axes are added (stride 0).
    pub(super) fn broadcast_to(&self, shape: &[usize]) -> Result<Layout, NdError> {
        if shape.len() < self.ndim {
            return Err(NdError::Broadcast { axis: 0, from: self.ndim, to: shape.len() });
        }
        let mut out = Layout::new(shape, &[0; MAX_DIMS][..shape.len().min(MAX_DIMS)])?;
        let lead = shape.len() - self.ndim;
        for i in 0..self.ndim {
            let (from, to) = (self.shape[i], shape[lead + i]);
            out.labels[lead + i] = self.labels[i];
            if from == to {
                out.strides[lead + i] = self.strides[i];
            } else if from != 1 {
                return Err(NdError::Broadcast { axis: i, from, to });
            }
        }
        Ok(out)
    }
    /// The same elements in row-major order under a new shape, without copying, when the layout
    /// allows it (NumPy's no-copy reshape). Labels are cleared.
    fn reshape(&self, shape: &[usize]) -> Result<Layout, NdError> {
        let target: usize = shape.iter().product();
        if target != self.len() {
            return Err(NdError::ShapeMismatch { expected: target, got: self.len() });
        }
        if self.len() <= 1 {
            return Layout::row_major(shape);
        }
        // the old axes without length-1 ones (their strides don't matter)
        let (mut od, mut os, mut no) = ([0usize; MAX_DIMS], [0isize; MAX_DIMS], 0);
        for (&d, &s) in self.shape().iter().zip(self.strides()) {
            if d != 1 {
                od[no] = d;
                os[no] = s;
                no += 1;
            }
        }
        let mut out = Layout::row_major(shape)?;
        let nn = shape.len();
        let (mut oi, mut oj, mut ni, mut nj) = (0, 1, 0, 1);
        while ni < nn && oi < no {
            // grow the smaller side until both groups hold the same number of elements
            let (mut np, mut op) = (shape[ni], od[oi]);
            while np != op {
                if np < op {
                    np *= shape[nj];
                    nj += 1;
                } else {
                    op *= od[oj];
                    oj += 1;
                }
            }
            // the old axes of the group must be one contiguous run...
            for k in oi..oj - 1 {
                if os[k] != os[k + 1] * od[k + 1] as isize {
                    return Err(NdError::ReshapeNeedsCopy);
                }
            }
            // ...which the new axes then split from the fastest stride up
            out.strides[nj - 1] = os[oj - 1];
            for k in (ni + 1..nj).rev() {
                out.strides[k - 1] = out.strides[k] * shape[k] as isize;
            }
            (ni, nj, oi, oj) = (nj, nj + 1, oj, oj + 1);
        }
        Ok(out)
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layout").field("shape", &self.shape()).field("strides", &self.strides()).finish()
    }
}

/// Walks the offsets of a layout in row-major index order (last axis fastest).
#[derive(Clone, Copy)]
struct Odometer {
    index: [usize; MAX_DIMS],
    offset: isize,
    remaining: usize,
}

impl Odometer {
    fn new(layout: &Layout) -> Self {
        Self { index: [0; MAX_DIMS], offset: 0, remaining: layout.len() }
    }
    #[inline]
    fn next(&mut self, layout: &Layout) -> Option<isize> {
        if self.remaining == 0 {
            return None;
        }
        let current = self.offset;
        self.remaining -= 1;
        if self.remaining > 0 {
            let mut i = layout.ndim;
            loop {
                i -= 1;
                self.index[i] += 1;
                self.offset += layout.strides[i];
                if self.index[i] < layout.shape[i] {
                    break;
                }
                self.offset -= layout.strides[i] * layout.shape[i] as isize;
                self.index[i] = 0;
            }
        }
        Some(current)
    }
}

// OWNED ARRAY =====================================================================================

/// A contiguous, row-major n-dimensional array over some [`Storage`]: `Vec<T>` by default (owned),
/// `Box<[T]>`, `Arc<[T]>` (shared, copy-on-write) or foreign memory (DLPack). All of them give the
/// same views, so every operation works on every kind of array.
pub struct NdArray<T, S = Vec<T>> {
    pub(super) data: S,
    pub(super) layout: Layout,
    pub(super) _elem: PhantomData<T>,
}

/// An array whose elements are shared between owners: clones are cheap, and writing copies the
/// elements first if another owner still holds them.
pub type SharedArray<T> = NdArray<T, Arc<[T]>>;

impl<T, S: Clone> Clone for NdArray<T, S> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), layout: self.layout, _elem: PhantomData }
    }
}

/// Equal shapes, labels and elements, whatever the storage.
impl<T: PartialEq, S: Storage<Elem = T>, S2: Storage<Elem = T>> PartialEq<NdArray<T, S2>> for NdArray<T, S> {
    fn eq(&self, other: &NdArray<T, S2>) -> bool {
        self.layout.shape() == other.layout.shape() && self.layout.labels() == other.layout.labels() && self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug, S: Storage<Elem = T>> fmt::Debug for NdArray<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdArray").field("shape", &self.layout.shape()).field("labels", &self.layout.labels()).field("data", &self.as_slice()).finish()
    }
}

impl<T> NdArray<T> {
    /// Wraps `data` (row-major) as an array of `shape`.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, NdError> {
        Self::from_storage(data, shape)
    }
    /// Builds each element from its index.
    pub fn from_fn(shape: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Result<Self, NdError> {
        let layout = Layout::row_major(shape)?;
        let mut data = Vec::with_capacity(layout.len());
        for_each_index(layout.shape(), |idx| data.push(f(idx)));
        Ok(Self { data, layout, _elem: PhantomData })
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    /// Moves the elements into shared storage (one copy into the shared allocation), after which
    /// clones are cheap.
    pub fn into_shared(self) -> SharedArray<T> {
        NdArray { data: Arc::from(self.data), layout: self.layout, _elem: PhantomData }
    }
}

impl<T: Copy> NdArray<T> {
    /// An array of `shape` filled with `value`.
    pub fn full(shape: &[usize], value: T) -> Result<Self, NdError> {
        let layout = Layout::row_major(shape)?;
        Ok(Self { data: vec![value; layout.len()], layout, _elem: PhantomData })
    }
    pub fn zeros(shape: &[usize]) -> Result<Self, NdError>
    where
        T: Default,
    {
        Self::full(shape, T::default())
    }
}

impl<T, S: Storage<Elem = T>> NdArray<T, S> {
    /// Wraps `storage` (row-major elements) as an array of `shape`.
    pub fn from_storage(storage: S, shape: &[usize]) -> Result<Self, NdError> {
        let layout = Layout::row_major(shape)?;
        if layout.len() != storage.as_slice().len() {
            return Err(NdError::ShapeMismatch { expected: layout.len(), got: storage.as_slice().len() });
        }
        Ok(Self { data: storage, layout, _elem: PhantomData })
    }
    /// Labels every axis; errors unless exactly one label per axis is given.
    pub fn with_labels(mut self, labels: &[Axis]) -> Result<Self, NdError> {
        if labels.len() != self.layout.ndim {
            return Err(NdError::LabelCount(labels.len(), self.layout.ndim));
        }
        self.layout.labels[..labels.len()].copy_from_slice(labels);
        Ok(self)
    }
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    pub fn ndim(&self) -> usize {
        self.layout.ndim
    }
    pub fn len(&self) -> usize {
        self.layout.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn labels(&self) -> &[Axis] {
        self.layout.labels()
    }
    /// Position of the first axis with this label.
    pub fn axis_of(&self, label: Axis) -> Option<usize> {
        self.labels().iter().position(|&l| l == label)
    }
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
    pub fn storage(&self) -> &S {
        &self.data
    }
    pub fn into_storage(self) -> S {
        self.data
    }
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        self.layout.offset_of(index).map(|o| &self.as_slice()[o as usize])
    }
    /// Same elements, new shape (the element count must match). Labels are cleared. No copy.
    pub fn reshape(self, shape: &[usize]) -> Result<Self, NdError> {
        Self::from_storage(self.data, shape)
    }
    pub fn view(&self) -> NdView<'_, T> {
        NdView { ptr: self.as_slice().as_ptr(), layout: self.layout, _borrow: PhantomData }
    }
    /// The 1-D lanes along `axis` (see [`NdView::lanes`]).
    pub fn lanes(&self, axis: usize) -> Result<Lanes<'_, T>, NdError> {
        self.view().lanes(axis)
    }
}

impl<T, S: StorageMut<Elem = T>> NdArray<T, S> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        self.layout.offset_of(index).map(move |o| &mut self.data.as_mut_slice()[o as usize])
    }
    pub fn view_mut(&mut self) -> NdViewMut<'_, T> {
        NdViewMut { ptr: self.data.as_mut_slice().as_mut_ptr(), layout: self.layout, _borrow: PhantomData }
    }
    /// The 1-D lanes along `axis`, mutable (see [`NdViewMut::lanes_mut`]).
    pub fn lanes_mut(&mut self, axis: usize) -> Result<LanesMut<'_, T>, NdError> {
        self.view_mut().into_lanes_mut(axis)
    }
    /// Applies `f` to every 1-D lane along `axis` (e.g. filter every channel along time).
    pub fn for_each_lane(&mut self, axis: usize, f: impl FnMut(&mut [T])) -> Result<(), NdError>
    where
        T: Copy,
    {
        self.view_mut().for_each_lane(axis, f)
    }
}

impl<T: Clone> SharedArray<T> {
    /// Whether no other array shares these elements (writing then needs no copy).
    pub fn is_unique(&mut self) -> bool {
        Arc::get_mut(&mut self.data).is_some()
    }
    /// The elements for writing, copied first if another array still shares them.
    pub fn make_mut(&mut self) -> &mut [T] {
        if Arc::get_mut(&mut self.data).is_none() {
            self.data = Arc::from(self.data.to_vec());
        }
        Arc::get_mut(&mut self.data).expect("unique after copying")
    }
    /// A mutable view, copying the elements first if they are shared.
    pub fn make_view_mut(&mut self) -> NdViewMut<'_, T> {
        let layout = self.layout;
        NdViewMut { ptr: self.make_mut().as_mut_ptr(), layout, _borrow: PhantomData }
    }
    /// A copy in a plain `Vec`-backed array.
    pub fn to_unshared(&self) -> NdArray<T> {
        NdArray { data: self.data.to_vec(), layout: self.layout, _elem: PhantomData }
    }
}

impl<T, S: Storage<Elem = T>> Index<&[usize]> for NdArray<T, S> {
    type Output = T;
    /// Panics if the index is out of bounds or has the wrong number of axes.
    fn index(&self, index: &[usize]) -> &T {
        self.get(index).expect("NdArray index out of bounds")
    }
}

impl<T, S: StorageMut<Elem = T>> IndexMut<&[usize]> for NdArray<T, S> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        self.get_mut(index).expect("NdArray index out of bounds")
    }
}

/// An array is a signal over all its elements (row-major).
impl<T: Float, S: Storage<Elem = T>> Signal for NdArray<T, S> {
    type Sample = T;
    fn samples(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Float, S: StorageMut<Elem = T>> SignalMut for NdArray<T, S> {
    fn samples_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Float> SignalOwned for NdArray<T> {
    type Container = Vec<T>;
    /// A 1-D array.
    fn from_container(container: Vec<T>) -> Self {
        let n = container.len();
        Self::from_vec(container, &[n]).expect("1-D shape always matches")
    }
    fn into_container(self) -> Vec<T> {
        self.data
    }
    fn as_container(&self) -> &Vec<T> {
        &self.data
    }
    fn as_container_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
    fn from_samples(samples: &[T]) -> Result<Self, SignalError> {
        Ok(Self::from_container(samples.to_vec()))
    }
}
// VIEWS ===========================================================================================

/// Borrowed, possibly strided view of n-dimensional data. `Copy`: views are cheap descriptions.
pub struct NdView<'a, T> {
    /// the element at index [0, 0, ...] (any address for an empty view; never read then)
    pub(super) ptr: *const T,
    pub(super) layout: Layout,
    pub(super) _borrow: PhantomData<&'a [T]>,
}

/// Mutable, possibly strided view of n-dimensional data. Its layout is always injective.
pub struct NdViewMut<'a, T> {
    pub(super) ptr: *mut T,
    pub(super) layout: Layout,
    pub(super) _borrow: PhantomData<&'a mut [T]>,
}

impl<T> Clone for NdView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for NdView<'_, T> {}

// SAFETY: a view is a shared borrow of `T`s (like `&[T]`); a mutable view an exclusive one (like
// `&mut [T]`), with disjoint elements per index.
unsafe impl<T: Sync> Send for NdView<'_, T> {}
unsafe impl<T: Sync> Sync for NdView<'_, T> {}
unsafe impl<T: Send> Send for NdViewMut<'_, T> {}
unsafe impl<T: Sync> Sync for NdViewMut<'_, T> {}

impl<T> fmt::Debug for NdView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdView").field("shape", &self.layout.shape()).field("strides", &self.layout.strides()).finish()
    }
}

impl<T> fmt::Debug for NdViewMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdViewMut").field("shape", &self.layout.shape()).field("strides", &self.layout.strides()).finish()
    }
}

/// Validates a layout over `len` elements with its origin at `offset`; returns the layout.
fn checked_layout(len: usize, shape: &[usize], strides: &[isize], offset: usize) -> Result<Layout, NdError> {
    let layout = Layout::new(shape, strides)?;
    match layout.span() {
        None if layout.len() == 0 => Ok(layout),
        None => Err(NdError::OutOfBounds),
        Some((lo, hi)) => {
            let origin = isize::try_from(offset).map_err(|_| NdError::OutOfBounds)?;
            match (origin.checked_add(lo), origin.checked_add(hi)) {
                (Some(start), Some(end)) if start >= 0 && end < len as isize => Ok(layout),
                _ => Err(NdError::OutOfBounds),
            }
        }
    }
}

macro_rules! view_common {
    ($View:ident) => {
        impl<'a, T> $View<'a, T> {
            pub fn shape(&self) -> &[usize] {
                self.layout.shape()
            }
            /// Element strides per axis (negative for reversed axes, 0 for broadcast ones).
            pub fn strides(&self) -> &[isize] {
                self.layout.strides()
            }
            pub fn ndim(&self) -> usize {
                self.layout.ndim
            }
            pub fn len(&self) -> usize {
                self.layout.len()
            }
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            pub fn labels(&self) -> &[Axis] {
                self.layout.labels()
            }
            pub fn axis_of(&self, label: Axis) -> Option<usize> {
                self.labels().iter().position(|&l| l == label)
            }
            /// Whether the elements are one contiguous row-major block.
            pub fn is_contiguous(&self) -> bool {
                self.layout.is_standard()
            }
            /// Address of the element at index `[0, 0, ...]` (for FFI; other elements are at
            /// `strides`-weighted element offsets from it). Dangling for empty views.
            pub fn as_ptr(&self) -> *const T {
                self.ptr as *const T
            }
            pub fn get(&self, index: &[usize]) -> Option<&T> {
                // SAFETY: the offset of a valid index is inside the validated layout
                self.layout.offset_of(index).map(|o| unsafe { &*self.ptr.wrapping_offset(o) })
            }
            /// Labels every axis of the view.
            pub fn with_labels(mut self, labels: &[Axis]) -> Result<Self, NdError> {
                if labels.len() != self.ndim() {
                    return Err(NdError::LabelCount(labels.len(), self.ndim()));
                }
                self.layout.labels[..labels.len()].copy_from_slice(labels);
                Ok(self)
            }
            fn relayout(self, layout: Layout, shift: isize) -> Self {
                Self { ptr: self.ptr.wrapping_offset(shift), layout, _borrow: PhantomData }
            }
            /// Reorders axes: new axis i is old axis `axes[i]`. No copy.
            pub fn permute(self, axes: &[usize]) -> Result<Self, NdError> {
                let layout = self.layout.permute(axes)?;
                Ok(self.relayout(layout, 0))
            }
            /// Reverses the axis order (a matrix transpose for 2-D). No copy.
            pub fn transpose(self) -> Self {
                let n = self.ndim();
                let mut axes = [0; MAX_DIMS];
                for (i, a) in axes[..n].iter_mut().enumerate() {
                    *a = n - 1 - i;
                }
                self.permute(&axes[..n]).expect("reversed axes are a permutation")
            }
            /// Exchanges two axes. No copy.
            pub fn swap_axes(self, a: usize, b: usize) -> Result<Self, NdError> {
                check_axis(a, self.ndim())?;
                check_axis(b, self.ndim())?;
                let mut axes = [0; MAX_DIMS];
                for (i, x) in axes[..self.ndim()].iter_mut().enumerate() {
                    *x = if i == a { b } else if i == b { a } else { i };
                }
                let n = self.ndim();
                self.permute(&axes[..n])
            }
            /// Restricts `axis` to `range`. No copy.
            pub fn slice_axis(self, axis: usize, range: Range<usize>) -> Result<Self, NdError> {
                let (layout, shift) = self.layout.slice_axis(axis, range)?;
                Ok(self.relayout(layout, shift))
            }
            /// Keeps every `step`-th index along `axis`; a negative step runs backwards from the
            /// last index (-1 reverses). No copy.
            pub fn step_axis(self, axis: usize, step: isize) -> Result<Self, NdError> {
                let (layout, shift) = self.layout.step_axis(axis, step)?;
                Ok(self.relayout(layout, shift))
            }
            /// Reverses `axis`. No copy.
            pub fn flip(self, axis: usize) -> Result<Self, NdError> {
                self.step_axis(axis, -1)
            }
            /// Fixes `axis` at `index`, dropping that axis. No copy.
            pub fn index_axis(self, axis: usize, index: usize) -> Result<Self, NdError> {
                let (layout, shift) = self.layout.index_axis(axis, index)?;
                Ok(self.relayout(layout, shift))
            }
            /// Adds a length-1 axis at position `at`. No copy.
            pub fn insert_axis(self, at: usize) -> Result<Self, NdError> {
                let layout = self.layout.insert_axis(at)?;
                Ok(self.relayout(layout, 0))
            }
            /// Removes a length-1 axis. No copy.
            pub fn remove_axis(self, axis: usize) -> Result<Self, NdError> {
                let layout = self.layout.remove_axis(axis)?;
                Ok(self.relayout(layout, 0))
            }
            /// The same elements (in row-major order) under a new shape, without copying; errors
            /// with [`NdError::ReshapeNeedsCopy`] when the layout can't express it (e.g. flattening
            /// a transposed matrix). Labels are cleared.
            pub fn reshape(self, shape: &[usize]) -> Result<Self, NdError> {
                let layout = self.layout.reshape(shape)?;
                Ok(self.relayout(layout, 0))
            }
            /// Splits `axis` at `index` into two views (`..index` and `index..`). No copy; for
            /// mutable views the halves are disjoint.
            pub fn split_at(self, axis: usize, index: usize) -> Result<(Self, Self), NdError> {
                check_axis(axis, self.ndim())?;
                if index > self.layout.shape[axis] {
                    return Err(NdError::OutOfBounds);
                }
                let (first, _) = self.layout.slice_axis(axis, 0..index)?;
                let (second, shift) = self.layout.slice_axis(axis, index..self.layout.shape[axis])?;
                let ptr = self.ptr;
                Ok((Self { ptr, layout: first, _borrow: PhantomData }, Self { ptr: ptr.wrapping_offset(shift), layout: second, _borrow: PhantomData }))
            }
            /// Elements in logical (row-major) order.
            pub fn iter(&self) -> Iter<'_, T> {
                Iter { ptr: self.ptr, layout: self.layout, odometer: Odometer::new(&self.layout), _borrow: PhantomData }
            }
        }

        impl<'a, T: Copy> $View<'a, T> {
            /// Copies the elements (in logical order) into a new contiguous array, keeping labels.
            pub fn to_owned(&self) -> NdArray<T> {
                let data: Vec<T> = self.iter().copied().collect();
                let mut layout = Layout::row_major(self.shape()).expect("valid shape");
                layout.labels = self.layout.labels;
                NdArray { data, layout, _elem: PhantomData }
            }
            /// The elements in logical order, copied into a `Vec`.
            pub fn to_vec(&self) -> Vec<T> {
                self.iter().copied().collect()
            }
        }
    };
}

view_common!(NdView);
view_common!(NdViewMut);

/// A view that may own its data: what [`NdView::to_shape`] returns, so callers can see whether a
/// copy was needed.
#[derive(Debug, Clone)]
pub enum CowArray<'a, T> {
    View(NdView<'a, T>),
    Owned(NdArray<T>),
}

impl<'a, T: Copy> CowArray<'a, T> {
    pub fn view(&self) -> NdView<'_, T> {
        match self {
            CowArray::View(v) => *v,
            CowArray::Owned(a) => a.view(),
        }
    }
    pub fn is_view(&self) -> bool {
        matches!(self, CowArray::View(_))
    }
    pub fn into_owned(self) -> NdArray<T> {
        match self {
            CowArray::View(v) => v.to_owned(),
            CowArray::Owned(a) => a,
        }
    }
}

impl<'a, T> NdView<'a, T> {
    /// A view over external memory (e.g. a buffer from another runtime): the element at index
    /// `[0, 0, ...]` is `data[offset]`, and moving one step along axis `i` moves `strides[i]`
    /// elements (negative strides walk backwards). Errors if any element of the layout would fall
    /// outside `data`.
    pub fn from_parts(data: &'a [T], shape: &[usize], strides: &[isize], offset: usize) -> Result<Self, NdError> {
        let layout = checked_layout(data.len(), shape, strides, offset)?;
        let ptr = if layout.len() == 0 { data.as_ptr() } else { data.as_ptr().wrapping_add(offset) };
        Ok(Self { ptr, layout, _borrow: PhantomData })
    }
    /// A view of memory described only by a pointer (FFI, other array libraries).
    ///
    /// # Safety
    /// Every element the layout reaches from `ptr` (the element at index `[0, 0, ...]`) must be
    /// valid for reads, and not written through anything else, for the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *const T, shape: &[usize], strides: &[isize]) -> Result<Self, NdError> {
        let layout = Layout::new(shape, strides)?;
        if layout.len() > 0 && layout.span().is_none() {
            return Err(NdError::OutOfBounds);
        }
        Ok(Self { ptr, layout, _borrow: PhantomData })
    }
    /// A 1-D view of a slice.
    pub fn from_slice(data: &'a [T]) -> Self {
        Self::from_parts(data, &[data.len()], &[1], 0).expect("a slice is a valid 1-D layout")
    }
    /// The elements as one slice, if the view is contiguous (row-major).
    pub fn as_slice(&self) -> Option<&'a [T]> {
        // SAFETY: a standard layout covers exactly `len` consecutive elements from the origin
        self.is_contiguous().then(|| unsafe { std::slice::from_raw_parts(self.ptr, self.len()) })
    }
    /// Elements in logical (row-major) order, borrowed for the view's whole lifetime.
    pub fn iter_all(&self) -> Iter<'a, T> {
        Iter { ptr: self.ptr, layout: self.layout, odometer: Odometer::new(&self.layout), _borrow: PhantomData }
    }
    /// Repeats the view to `shape` (NumPy broadcasting: axes line up at the end; a length-1 axis
    /// repeats and missing leading axes are added, both with stride 0). No copy; read-only, since
    /// the repeated indices share elements.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<NdView<'a, T>, NdError> {
        let layout = self.layout.broadcast_to(shape)?;
        Ok(NdView { ptr: self.ptr, layout, _borrow: PhantomData })
    }
    /// The 1-D lanes along `axis` as views, in row-major order of the other axes; no allocation.
    /// A lane along the last axis of a contiguous view is contiguous, so `lane.as_slice()` gives a
    /// `Signal`.
    pub fn lanes(&self, axis: usize) -> Result<Lanes<'a, T>, NdError> {
        check_axis(axis, self.ndim())?;
        Ok(Lanes { inner: LaneWalker::new(&self.layout, axis), ptr: self.ptr, _borrow: PhantomData })
    }
    /// The sub-views at each index along `axis` (that axis dropped), e.g. the items of a batch.
    pub fn axis_iter(&self, axis: usize) -> Result<AxisIter<'a, T>, NdError> {
        check_axis(axis, self.ndim())?;
        Ok(AxisIter { ptr: self.ptr, sub: self.layout.without_axis(axis), stride: self.layout.strides[axis], index: 0, count: self.layout.shape[axis], _borrow: PhantomData })
    }
}

impl<'a, T: Copy> NdView<'a, T> {
    /// The view under a new shape: borrowed when the layout allows it, otherwise copied (in
    /// row-major order) into an owned array. [`CowArray::is_view`] tells which happened.
    pub fn to_shape(&self, shape: &[usize]) -> Result<CowArray<'a, T>, NdError> {
        match self.reshape(shape) {
            Ok(view) => Ok(CowArray::View(view)),
            Err(NdError::ReshapeNeedsCopy) => Ok(CowArray::Owned(NdArray::from_vec(self.to_vec(), shape)?)),
            Err(e) => Err(e),
        }
    }
}

impl<'a, T> NdViewMut<'a, T> {
    /// A mutable view over external memory (see [`NdView::from_parts`]). Errors if the layout falls
    /// outside `data`, or if it may reach one element through several indices
    /// ([`NdError::Overlapping`]: zero strides, or axes whose ranges interleave).
    pub fn from_parts(data: &'a mut [T], shape: &[usize], strides: &[isize], offset: usize) -> Result<Self, NdError> {
        let layout = checked_layout(data.len(), shape, strides, offset)?;
        if !layout.is_injective() {
            return Err(NdError::Overlapping);
        }
        let ptr = if layout.len() == 0 { data.as_mut_ptr() } else { data.as_mut_ptr().wrapping_add(offset) };
        Ok(Self { ptr, layout, _borrow: PhantomData })
    }
    /// A mutable view of memory described only by a pointer; errors if the layout may reach one
    /// element through several indices ([`NdError::Overlapping`]).
    ///
    /// # Safety
    /// Every element the layout reaches from `ptr` must be valid for reads and writes, and not
    /// accessed through anything else, for the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: &[usize], strides: &[isize]) -> Result<Self, NdError> {
        let layout = Layout::new(shape, strides)?;
        if layout.len() > 0 && layout.span().is_none() {
            return Err(NdError::OutOfBounds);
        }
        if !layout.is_injective() {
            return Err(NdError::Overlapping);
        }
        Ok(Self { ptr, layout, _borrow: PhantomData })
    }
    /// A 1-D view of a slice.
    pub fn from_slice(data: &'a mut [T]) -> Self {
        let n = data.len();
        Self::from_parts(data, &[n], &[1], 0).expect("a slice is a valid 1-D layout")
    }
    /// A read-only view of the same elements.
    pub fn view(&self) -> NdView<'_, T> {
        NdView { ptr: self.ptr, layout: self.layout, _borrow: PhantomData }
    }
    /// Writable address of the element at index `[0, 0, ...]` (see `as_ptr`).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    /// A shorter-lived mutable view of the same elements, leaving this one usable afterwards.
    pub fn reborrow(&mut self) -> NdViewMut<'_, T> {
        NdViewMut { ptr: self.ptr, layout: self.layout, _borrow: PhantomData }
    }
    /// Gives up mutability, keeping the full lifetime.
    pub fn into_view(self) -> NdView<'a, T> {
        NdView { ptr: self.ptr, layout: self.layout, _borrow: PhantomData }
    }
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        // SAFETY: valid index -> inside the layout; `&mut self` makes the reference exclusive
        self.layout.offset_of(index).map(|o| unsafe { &mut *self.ptr.wrapping_offset(o) })
    }
    /// The elements as one mutable slice, if the view is contiguous (row-major).
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        let len = self.len();
        // SAFETY: as `NdView::as_slice`, exclusively borrowed
        self.is_contiguous().then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr, len) })
    }
    /// Converts into the contiguous slice for the view's whole lifetime (`None` if the view isn't
    /// contiguous: check [`is_contiguous`](Self::is_contiguous) first to keep the view otherwise).
    pub fn into_slice(self) -> Option<&'a mut [T]> {
        // SAFETY: as above; `self` is consumed, so the borrow moves to the slice
        self.is_contiguous().then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len()) })
    }
    /// Every element, mutably, in logical order.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut { ptr: self.ptr, layout: self.layout, odometer: Odometer::new(&self.layout), _borrow: PhantomData }
    }
    /// Calls `f` on every element.
    pub fn for_each_mut(&mut self, f: impl FnMut(&mut T)) {
        self.iter_mut().for_each(f);
    }
    /// The 1-D lanes along `axis` as mutable views (disjoint, since the layout is injective).
    pub fn lanes_mut(&mut self, axis: usize) -> Result<LanesMut<'_, T>, NdError> {
        self.reborrow().into_lanes_mut(axis)
    }
    /// [`lanes_mut`](Self::lanes_mut) for the view's whole lifetime.
    pub fn into_lanes_mut(self, axis: usize) -> Result<LanesMut<'a, T>, NdError> {
        check_axis(axis, self.ndim())?;
        Ok(LanesMut { inner: LaneWalker::new(&self.layout, axis), ptr: self.ptr, _borrow: PhantomData })
    }
    /// The mutable sub-views at each index along `axis` (that axis dropped).
    pub fn axis_iter_mut(&mut self, axis: usize) -> Result<AxisIterMut<'_, T>, NdError> {
        check_axis(axis, self.ndim())?;
        Ok(AxisIterMut { ptr: self.ptr, sub: self.layout.without_axis(axis), stride: self.layout.strides[axis], index: 0, count: self.layout.shape[axis], _borrow: PhantomData })
    }
}

impl<'a, T: Copy> NdViewMut<'a, T> {
    pub fn fill(&mut self, value: T) {
        self.for_each_mut(|x| *x = value);
    }
    /// Applies `f` to every 1-D lane along `axis`, in row-major order of the other axes. Contiguous
    /// lanes are passed in place without allocating; strided ones are copied to a scratch buffer (one
    /// allocation per call) and written back, so prefer a layout where `axis` is last in real-time code.
    pub fn for_each_lane(&mut self, axis: usize, mut f: impl FnMut(&mut [T])) -> Result<(), NdError> {
        let mut scratch = Vec::new();
        for mut lane in self.lanes_mut(axis)? {
            if lane.is_contiguous() {
                f(lane.into_slice().expect("contiguous"));
            } else {
                scratch.clear();
                scratch.extend(lane.iter().copied());
                f(&mut scratch);
                for (x, &v) in lane.iter_mut().zip(&scratch) {
                    *x = v;
                }
            }
        }
        Ok(())
    }
}

// ITERATORS =======================================================================================

/// Elements of a view in logical order.
pub struct Iter<'a, T> {
    ptr: *const T,
    layout: Layout,
    odometer: Odometer,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        // SAFETY: the odometer only yields offsets of indices inside the validated layout
        self.odometer.next(&self.layout).map(|o| unsafe { &*self.ptr.wrapping_offset(o) })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.odometer.remaining, Some(self.odometer.remaining))
    }
}
impl<T> ExactSizeIterator for Iter<'_, T> {}

/// Mutable elements of a view in logical order.
pub struct IterMut<'a, T> {
    ptr: *mut T,
    layout: Layout,
    odometer: Odometer,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        // SAFETY: inside the layout; injective layouts yield each element once, so the references
        // never alias
        self.odometer.next(&self.layout).map(|o| unsafe { &mut *self.ptr.wrapping_offset(o) })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.odometer.remaining, Some(self.odometer.remaining))
    }
}
impl<T> ExactSizeIterator for IterMut<'_, T> {}

/// Walks the starting offsets of the lanes along one axis.
#[derive(Clone, Copy)]
struct LaneWalker {
    /// the other axes
    outer: Layout,
    odometer: Odometer,
    /// the 1-D layout of every lane
    lane: Layout,
}

impl LaneWalker {
    fn new(layout: &Layout, axis: usize) -> Self {
        let outer = layout.without_axis(axis);
        let mut lane = Layout::new(&[layout.shape[axis]], &[layout.strides[axis]]).expect("1-D layout");
        lane.labels[0] = layout.labels[axis];
        let mut odometer = Odometer::new(&outer);
        if layout.shape[axis] == 0 {
            odometer.remaining = 0;
        }
        Self { outer, odometer, lane }
    }
    #[inline]
    fn next(&mut self) -> Option<isize> {
        self.odometer.next(&self.outer)
    }
}

/// The 1-D lanes of a view along one axis.
pub struct Lanes<'a, T> {
    inner: LaneWalker,
    ptr: *const T,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T> Iterator for Lanes<'a, T> {
    type Item = NdView<'a, T>;
    fn next(&mut self) -> Option<NdView<'a, T>> {
        let start = self.inner.next()?;
        Some(NdView { ptr: self.ptr.wrapping_offset(start), layout: self.inner.lane, _borrow: PhantomData })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.odometer.remaining, Some(self.inner.odometer.remaining))
    }
}
impl<T> ExactSizeIterator for Lanes<'_, T> {}

/// The 1-D lanes of a mutable view along one axis (pairwise disjoint).
pub struct LanesMut<'a, T> {
    inner: LaneWalker,
    ptr: *mut T,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for LanesMut<'a, T> {
    type Item = NdViewMut<'a, T>;
    fn next(&mut self) -> Option<NdViewMut<'a, T>> {
        let start = self.inner.next()?;
        Some(NdViewMut { ptr: self.ptr.wrapping_offset(start), layout: self.inner.lane, _borrow: PhantomData })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.odometer.remaining, Some(self.inner.odometer.remaining))
    }
}
impl<T> ExactSizeIterator for LanesMut<'_, T> {}

/// The sub-views at each index along one axis.
pub struct AxisIter<'a, T> {
    ptr: *const T,
    sub: Layout,
    stride: isize,
    index: usize,
    count: usize,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T> Iterator for AxisIter<'a, T> {
    type Item = NdView<'a, T>;
    fn next(&mut self) -> Option<NdView<'a, T>> {
        (self.index < self.count).then(|| {
            let ptr = self.ptr.wrapping_offset(self.index as isize * self.stride);
            self.index += 1;
            NdView { ptr, layout: self.sub, _borrow: PhantomData }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count - self.index, Some(self.count - self.index))
    }
}
impl<T> ExactSizeIterator for AxisIter<'_, T> {}

/// The mutable sub-views at each index along one axis (pairwise disjoint).
pub struct AxisIterMut<'a, T> {
    ptr: *mut T,
    sub: Layout,
    stride: isize,
    index: usize,
    count: usize,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for AxisIterMut<'a, T> {
    type Item = NdViewMut<'a, T>;
    fn next(&mut self) -> Option<NdViewMut<'a, T>> {
        (self.index < self.count).then(|| {
            let ptr = self.ptr.wrapping_offset(self.index as isize * self.stride);
            self.index += 1;
            NdViewMut { ptr, layout: self.sub, _borrow: PhantomData }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count - self.index, Some(self.count - self.index))
    }
}
impl<T> ExactSizeIterator for AxisIterMut<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::osc::Sine;

    fn counting(shape: &[usize]) -> NdArray<f64> {
        let mut k = 0.0;
        NdArray::from_fn(shape, |_| {
            k += 1.0;
            k - 1.0
        })
        .unwrap()
    }

    #[test]
    fn construction_and_indexing() {
        let a = counting(&[2, 3]);
        assert_eq!(a.shape(), [2, 3]);
        assert_eq!(a[&[1, 2][..]], 5.0);
        assert_eq!(a.get(&[2, 0]), None);
        assert_eq!(a.get(&[0]), None);
        assert_eq!(NdArray::from_vec(vec![1.0f32; 5], &[2, 3]), Err(NdError::ShapeMismatch { expected: 6, got: 5 }));
        assert_eq!(NdArray::<f32>::zeros(&[1; 9]), Err(NdError::TooManyDims(9)));
        let r = a.reshape(&[3, 2]).unwrap();
        assert_eq!(r[&[2, 1][..]], 5.0);
    }

    #[test]
    fn transpose_and_permute_without_copying() {
        let a = counting(&[2, 3, 4]);
        let t = a.view().transpose();
        assert_eq!(t.shape(), [4, 3, 2]);
        assert!(!t.is_contiguous());
        for_each_index(a.shape(), |i| assert_eq!(a.get(i), t.get(&[i[2], i[1], i[0]])));
        let p = a.view().permute(&[1, 0, 2]).unwrap();
        assert_eq!(p.get(&[2, 1, 3]), a.get(&[1, 2, 3]));
        assert!(a.view().permute(&[0, 0, 1]).is_err());
        let owned = t.to_owned();
        assert_eq!(owned.shape(), [4, 3, 2]);
        assert!(owned.view().is_contiguous());
        assert_eq!(owned.get(&[3, 2, 1]), a.get(&[1, 2, 3]));
        assert_eq!(a.view().swap_axes(0, 2).unwrap().shape(), [4, 3, 2]);
    }

    #[test]
    fn slicing_stepping_and_indexing_axes() {
        let a = counting(&[3, 4]);
        let s = a.view().slice_axis(1, 1..3).unwrap();
        assert_eq!(s.shape(), [3, 2]);
        assert_eq!(s.to_vec(), [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        let row = a.view().index_axis(0, 2).unwrap();
        assert_eq!(row.as_slice(), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        let col = a.view().index_axis(1, 1).unwrap();
        assert_eq!(col.as_slice(), None);
        assert_eq!(col.to_vec(), [1.0, 5.0, 9.0]);
        assert!(a.view().slice_axis(0, 2..4).is_err());
        assert!(a.view().index_axis(2, 0).is_err());
        // steps and reversal
        assert_eq!(a.view().step_axis(1, 2).unwrap().to_vec(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(a.view().step_axis(1, -3).unwrap().to_vec(), [3.0, 0.0, 7.0, 4.0, 11.0, 8.0]);
        assert_eq!(a.view().flip(0).unwrap().index_axis(1, 0).unwrap().to_vec(), [8.0, 4.0, 0.0]);
        assert_eq!(a.view().step_axis(0, 0).err(), Some(NdError::ZeroStep));
        // unit axes
        let e = a.view().insert_axis(1).unwrap();
        assert_eq!(e.shape(), [3, 1, 4]);
        assert_eq!(e.remove_axis(1).unwrap().to_vec(), a.as_slice());
        assert_eq!(a.view().remove_axis(0).err(), Some(NdError::NotUnitAxis(0, 3)));
        // split
        let (top, bottom) = a.view().split_at(0, 1).unwrap();
        assert_eq!((top.shape(), bottom.shape()), (&[1, 4][..], &[2, 4][..]));
        assert_eq!(bottom.get(&[0, 0]), Some(&4.0));
    }

    /// A slow reference: the same transforms done by copying into dense row-major data.
    #[derive(Clone, Debug, PartialEq)]
    struct Dense {
        shape: Vec<usize>,
        data: Vec<f64>,
    }

    impl Dense {
        fn at(&self, idx: &[usize]) -> f64 {
            let mut o = 0;
            for (i, &x) in idx.iter().enumerate() {
                o = o * self.shape[i] + x;
            }
            self.data[o]
        }
        fn build(shape: Vec<usize>, f: impl Fn(&[usize]) -> f64) -> Self {
            let mut data = Vec::new();
            for_each_index(&shape, |i| data.push(f(i)));
            Self { shape, data }
        }
    }

    #[test]
    fn random_transform_chains_match_a_copying_reference() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for _ in 0..400 {
            let a = counting(&[4, 5, 3]);
            let mut view = a.view();
            let mut reference = Dense { shape: vec![4, 5, 3], data: a.as_slice().to_vec() };
            for _ in 0..6 {
                let n = view.ndim();
                if n == 0 {
                    break;
                }
                let axis = rand(n);
                let d = reference.shape[axis];
                match rand(6) {
                    0 if d > 0 => {
                        let (s, e) = (rand(d), rand(d + 1));
                        let (s, e) = (s.min(e), s.max(e));
                        view = view.slice_axis(axis, s..e).unwrap();
                        let r = reference.clone();
                        let mut shape = r.shape.clone();
                        shape[axis] = e - s;
                        reference = Dense::build(shape, |i| {
                            let mut j = i.to_vec();
                            j[axis] += s;
                            r.at(&j)
                        });
                    }
                    1 => {
                        let step = [1isize, 2, 3, -1, -2][rand(5)];
                        view = view.step_axis(axis, step).unwrap();
                        let r = reference.clone();
                        let mut shape = r.shape.clone();
                        shape[axis] = d.div_ceil(step.unsigned_abs());
                        reference = Dense::build(shape, |i| {
                            let mut j = i.to_vec();
                            j[axis] = if step > 0 { i[axis] * step as usize } else { d - 1 - i[axis] * step.unsigned_abs() };
                            r.at(&j)
                        });
                    }
                    2 if n > 1 => {
                        let b = rand(n);
                        view = view.swap_axes(axis, b).unwrap();
                        let r = reference.clone();
                        let mut shape = r.shape.clone();
                        shape.swap(axis, b);
                        reference = Dense::build(shape, |i| {
                            let mut j = i.to_vec();
                            j.swap(axis, b);
                            r.at(&j)
                        });
                    }
                    3 if d > 0 && n > 1 => {
                        let k = rand(d);
                        view = view.index_axis(axis, k).unwrap();
                        let r = reference.clone();
                        let mut shape = r.shape.clone();
                        shape.remove(axis);
                        reference = Dense::build(shape, |i| {
                            let mut j = i.to_vec();
                            j.insert(axis, k);
                            r.at(&j)
                        });
                    }
                    4 if n < MAX_DIMS => {
                        view = view.insert_axis(axis).unwrap();
                        reference.shape.insert(axis, 1);
                    }
                    _ => {
                        // reshape: flatten when possible, else the copying route
                        let len = reference.data.len();
                        match view.to_shape(&[len]).unwrap() {
                            CowArray::View(v) => {
                                assert_eq!(v.to_vec(), reference.data);
                                view = v;
                                reference.shape = vec![len];
                            }
                            CowArray::Owned(o) => assert_eq!(o.into_vec(), reference.data),
                        }
                    }
                }
                assert_eq!(view.shape(), &reference.shape[..]);
                assert_eq!(view.to_vec(), reference.data, "after a transform");
            }
        }
    }

    #[test]
    fn reshape_without_copying_when_the_layout_allows() {
        let a = counting(&[4, 6]);
        // contiguous: any shape
        let r = a.view().reshape(&[2, 2, 3, 2]).unwrap();
        assert_eq!(r.to_vec(), a.as_slice());
        // rows sliced: the columns stay one run, so splitting them works...
        let s = a.view().slice_axis(1, 0..4).unwrap();
        let split = s.reshape(&[4, 2, 2]).unwrap();
        assert_eq!(split.to_vec(), s.to_vec());
        // ...but merging rows across the gap doesn't
        assert_eq!(s.reshape(&[16]).err(), Some(NdError::ReshapeNeedsCopy));
        let cow = s.to_shape(&[16]).unwrap();
        assert!(!cow.is_view());
        assert_eq!(cow.view().to_vec(), s.to_vec());
        // a transposed matrix flattens only by copying; splitting an axis is fine
        let t = a.view().transpose();
        assert!(t.to_shape(&[24]).map(|c| !c.is_view()).unwrap());
        assert_eq!(t.reshape(&[3, 2, 4]).unwrap().to_vec(), t.to_vec());
        // reversed axes stay viewable
        let f = a.view().flip(1).unwrap();
        assert_eq!(f.reshape(&[4, 3, 2]).unwrap().to_vec(), f.to_vec());
        assert_eq!(a.view().reshape(&[5, 5]).err(), Some(NdError::ShapeMismatch { expected: 25, got: 24 }));
    }

    #[test]
    fn broadcasting_repeats_without_copying() {
        let row = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let b = row.view().broadcast_to(&[2, 4, 3]).unwrap();
        assert_eq!(b.shape(), [2, 4, 3]);
        assert_eq!(b.strides(), [0, 0, 1]);
        assert_eq!(b.to_vec(), [1.0, 2.0, 3.0].repeat(8));
        assert_eq!(row.view().broadcast_to(&[2, 2]).err(), Some(NdError::Broadcast { axis: 1, from: 3, to: 2 }));
        // a column against rows
        let col = NdArray::from_vec(vec![10.0, 20.0], &[2, 1]).unwrap();
        assert_eq!(col.view().broadcast_to(&[2, 3]).unwrap().to_vec(), [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn mutable_views_must_be_injective() {
        let mut data = [0.0f32; 12];
        // padded rows and column-major layouts are fine
        assert!(NdViewMut::from_parts(&mut data, &[2, 3], &[6, 1], 0).is_ok());
        assert!(NdViewMut::from_parts(&mut data, &[3, 4], &[1, 3], 0).is_ok());
        // reversed: the origin is the last row
        let mut v = NdViewMut::from_parts(&mut data, &[3, 4], &[-4, 1], 8).unwrap();
        *v.get_mut(&[0, 0]).unwrap() = 1.0;
        assert_eq!(data[8], 1.0);
        // overlapping: a zero stride, or rows that run into each other
        assert_eq!(NdViewMut::from_parts(&mut data, &[2, 3], &[0, 1], 0).err(), Some(NdError::Overlapping));
        assert_eq!(NdViewMut::from_parts(&mut data, &[3, 4], &[2, 1], 0).err(), Some(NdError::Overlapping));
        // read-only views may overlap (that is what broadcasting is)
        assert!(NdView::from_parts(&data, &[3, 4], &[2, 1], 0).is_ok());
        // bounds include negative reach
        assert_eq!(NdView::from_parts(&data, &[3], &[-1], 1).err(), Some(NdError::OutOfBounds));
    }

    #[test]
    fn lanes_are_signals() {
        // [channel, time]: 2 channels of a sine at different levels
        let tone: Vec<f64> = Sine::new(1_000.0, 48_000.0).take(4_800).collect();
        let a = NdArray::from_fn(&[2, 4_800], |i| tone[i[1]] * (i[0] + 1) as f64).unwrap().with_labels(&[Axis::Channel, Axis::Time]).unwrap();
        let time = a.axis_of(Axis::Time).unwrap();
        let rms: Vec<f64> = a.lanes(time).unwrap().map(|l| l.as_slice().unwrap().rms().unwrap()).collect();
        assert_eq!(rms.len(), 2);
        assert!((rms[1] / rms[0] - 2.0).abs() < 1e-12);
        // lanes along the other axis are strided
        let across = a.lanes(0).unwrap();
        assert_eq!(across.len(), 4_800);
        assert_eq!(a.lanes(0).unwrap().nth(10).unwrap().to_vec(), [tone[10], 2.0 * tone[10]]);
        assert_eq!(a.lanes(0).unwrap().next().unwrap().labels(), [Axis::Channel]);
        // the items of a batch
        let items: Vec<Vec<f64>> = counting(&[3, 2]).view().axis_iter(0).unwrap().map(|v| v.to_vec()).collect();
        assert_eq!(items, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
    }

    #[test]
    fn mutable_lanes_write_disjointly() {
        let mut a = counting(&[3, 4]);
        // scale each column by its index, through strided mutable lanes
        for (c, mut lane) in a.lanes_mut(0).unwrap().enumerate() {
            lane.for_each_mut(|x| *x *= c as f64);
        }
        assert_eq!(a.as_slice(), [0.0, 1.0, 4.0, 9.0, 0.0, 5.0, 12.0, 21.0, 0.0, 9.0, 20.0, 33.0]);
        // two disjoint halves at once
        let (mut left, mut right) = a.view_mut().split_at(1, 2).unwrap();
        left.fill(-1.0);
        right.iter_mut().for_each(|x| *x += 0.5);
        assert_eq!(a.as_slice()[..4], [-1.0, -1.0, 4.5, 9.5]);
        for mut item in a.view_mut().axis_iter_mut(0).unwrap() {
            item.fill(7.0);
        }
        assert!(a.as_slice().iter().all(|&x| x == 7.0));
    }

    #[test]
    fn processing_along_a_strided_axis_matches_contiguous() {
        // [time, channel] layout (interleaved): time is NOT the last axis
        let input: Vec<f64> = Sine::new(9_000.0, 48_000.0).take(2_000).collect();
        let mut interleaved = NdArray::from_fn(&[2_000, 2], |i| input[i[0]]).unwrap();
        interleaved.for_each_lane(0, |lane| Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q).process(lane)).unwrap();

        let mut reference = input.clone();
        Biquad::lowpass(1_000.0, 48_000.0, BUTTERWORTH_Q).process(&mut reference);
        for (t, &r) in reference.iter().enumerate() {
            assert_eq!(interleaved[&[t, 0][..]], r);
            assert_eq!(interleaved[&[t, 1][..]], r);
        }
    }

    #[test]
    fn external_memory_views_are_bounds_checked() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = NdView::from_parts(&data, &[2, 2], &[3, 1], 1).unwrap(); // rows [2,3], [5,6]
        assert_eq!(v.to_vec(), [2.0, 3.0, 5.0, 6.0]);
        assert_eq!(NdView::from_parts(&data, &[2, 3], &[3, 1], 1).err(), Some(NdError::OutOfBounds));
        let mut buf = [0.0f32; 4];
        let mut m = NdViewMut::from_parts(&mut buf, &[2, 2], &[1, 2], 0).unwrap(); // column-major
        *m.get_mut(&[1, 0]).unwrap() = 7.0;
        assert_eq!(buf, [0.0, 7.0, 0.0, 0.0]);
        // empty views are fine anywhere
        assert!(NdView::from_parts(&data, &[0, 5], &[100, 1], 99).is_ok());
    }

    #[test]
    fn storage_kinds_share_one_api() {
        // boxed storage: owned and writable
        let mut boxed = NdArray::from_storage(vec![1.0f64, 2.0, 3.0, 4.0].into_boxed_slice(), &[2, 2]).unwrap();
        boxed[&[1, 1][..]] = 40.0;
        assert_eq!(boxed.view().transpose().to_vec(), [1.0, 3.0, 2.0, 40.0]);
        // shared storage: cheap clones, copy on write
        let shared = counting(&[2, 3]).into_shared();
        let mut writer = shared.clone();
        assert!(std::ptr::eq(shared.as_slice(), writer.as_slice()), "a clone shares the elements");
        writer.make_view_mut().fill(9.0);
        assert_eq!(shared.as_slice(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "the original is untouched");
        assert_eq!(writer.as_slice(), [9.0; 6]);
        assert!(writer.is_unique());
        writer.make_mut()[0] = 1.0; // unique now: no copy
        assert_eq!(writer.to_unshared().as_slice()[..2], [1.0, 9.0]);
        // equality and signals ignore the storage kind
        assert_eq!(shared, counting(&[2, 3]));
        assert_eq!(shared.sum(), 15.0);
        assert_eq!(shared.reshape(&[3, 2]).unwrap().view().index_axis(0, 2).unwrap().to_vec(), [4.0, 5.0]);
        assert!(NdArray::from_storage(vec![1.0f32; 5], &[2, 3]).is_err());
    }
    #[test]
    fn arrays_are_owned_signals() {
        let mut a = counting(&[2, 2]);
        assert_eq!(a.sum(), 6.0);
        a.scale(2.0);
        assert_eq!(a.max(), Some(6.0));
        assert_eq!(NdArray::from_samples(&[1.0f64, 2.0]).unwrap().shape(), [2]);
    }
}
