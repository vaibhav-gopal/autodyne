//! N-dimensional arrays: owned [`NdArray`] and zero-copy views ([`NdView`], [`NdViewMut`]).
//!
//! The shape of tensors in ML frameworks and of multichannel / batched signals: e.g.
//! `[batch, channel, time]`. Owned arrays are contiguous and row-major (last axis fastest); views carry
//! strides, so `permute`, `transpose`, `slice_axis` and `index_axis` never copy. Shapes hold at most
//! [`MAX_DIMS`] axes in fixed-size arrays, so creating and slicing views never allocates.
//!
//! Signals live inside arrays as *lanes*: the 1-D runs along one axis. `lanes(axis)` yields them as
//! views (contiguous lanes expose `as_slice`, so every `Signal` method applies), and
//! `for_each_lane(axis, f)` applies a `SignalMut` operation or a `Processor` along any axis.
//! Axes can be labelled ([`Axis`]) so code can find "the time axis" instead of hard-coding positions.

use std::fmt;
use std::ops::{Index, IndexMut, Range};

use thiserror::Error;

use super::{Signal, SignalError, SignalMut, SignalOwned};
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
}

// SHAPE ===========================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Dims {
    d: [usize; MAX_DIMS],
    n: usize,
}

impl Dims {
    fn new(dims: &[usize]) -> Result<Self, NdError> {
        if dims.len() > MAX_DIMS {
            return Err(NdError::TooManyDims(dims.len()));
        }
        let mut d = [0; MAX_DIMS];
        d[..dims.len()].copy_from_slice(dims);
        Ok(Self { d, n: dims.len() })
    }
    fn as_slice(&self) -> &[usize] {
        &self.d[..self.n]
    }
    fn product(&self) -> usize {
        self.as_slice().iter().product()
    }
    fn row_major_strides(&self) -> Dims {
        let mut s = Dims { d: [0; MAX_DIMS], n: self.n };
        let mut acc = 1;
        for i in (0..self.n).rev() {
            s.d[i] = acc;
            acc *= self.d[i];
        }
        s
    }
    fn remove(&self, axis: usize) -> Dims {
        let mut out = Dims { d: [0; MAX_DIMS], n: self.n - 1 };
        for (j, i) in (0..self.n).filter(|&i| i != axis).enumerate() {
            out.d[j] = self.d[i];
        }
        out
    }
}

impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
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

// LAYOUT (shared by views) ========================================================================

/// Where a view's elements are: shape, strides, offset and labels.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Layout {
    shape: Dims,
    strides: Dims,
    offset: usize,
    labels: [Axis; MAX_DIMS],
}

impl Layout {
    fn contiguous(shape: Dims, labels: [Axis; MAX_DIMS]) -> Self {
        Self { shape, strides: shape.row_major_strides(), offset: 0, labels }
    }
    fn offset_of(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.n || index.iter().zip(self.shape.as_slice()).any(|(&i, &d)| i >= d) {
            return None;
        }
        Some(self.offset + index.iter().zip(self.strides.as_slice()).map(|(i, s)| i * s).sum::<usize>())
    }
    fn is_contiguous(&self) -> bool {
        self.shape.product() <= 1 || self.strides == self.shape.row_major_strides()
            || self.shape.as_slice().iter().zip(self.strides.as_slice()).zip(self.shape.row_major_strides().as_slice())
                .all(|((&d, &s), &rm)| d == 1 || s == rm)
    }
    /// Largest element offset + 1, or 0 for an empty view.
    fn extent(&self) -> usize {
        if self.shape.product() == 0 {
            return 0;
        }
        self.offset + self.shape.as_slice().iter().zip(self.strides.as_slice()).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1
    }
    fn permute(&self, axes: &[usize]) -> Result<Layout, NdError> {
        let n = self.shape.n;
        let mut seen = [false; MAX_DIMS];
        if axes.len() != n || axes.iter().any(|&a| a >= n || std::mem::replace(&mut seen[a], true)) {
            return Err(NdError::InvalidPermutation);
        }
        let mut out = *self;
        for (i, &a) in axes.iter().enumerate() {
            out.shape.d[i] = self.shape.d[a];
            out.strides.d[i] = self.strides.d[a];
            out.labels[i] = self.labels[a];
        }
        Ok(out)
    }
    fn slice_axis(&self, axis: usize, range: Range<usize>) -> Result<Layout, NdError> {
        check_axis(axis, self.shape.n)?;
        if range.start > range.end || range.end > self.shape.d[axis] {
            return Err(NdError::OutOfBounds);
        }
        let mut out = *self;
        out.shape.d[axis] = range.end - range.start;
        if out.shape.product() > 0 {
            out.offset += range.start * self.strides.d[axis];
        }
        Ok(out)
    }
    fn index_axis(&self, axis: usize, index: usize) -> Result<Layout, NdError> {
        check_axis(axis, self.shape.n)?;
        if index >= self.shape.d[axis] {
            return Err(NdError::OutOfBounds);
        }
        let mut labels = [Axis::Unlabeled; MAX_DIMS];
        for (j, i) in (0..self.shape.n).filter(|&i| i != axis).enumerate() {
            labels[j] = self.labels[i];
        }
        Ok(Layout {
            shape: self.shape.remove(axis),
            strides: self.strides.remove(axis),
            offset: self.offset + index * self.strides.d[axis],
            labels,
        })
    }
    /// Offsets of the start of every lane along `axis`, in row-major order of the other axes.
    fn lane_starts(&self, axis: usize, mut f: impl FnMut(usize)) {
        let others = self.shape.remove(axis);
        let other_strides = self.strides.remove(axis);
        if self.shape.d[axis] == 0 {
            return;
        }
        for_each_index(others.as_slice(), |idx| {
            f(self.offset + idx.iter().zip(other_strides.as_slice()).map(|(i, s)| i * s).sum::<usize>());
        });
    }
    fn lane(&self, axis: usize, start: usize) -> Layout {
        let mut labels = [Axis::Unlabeled; MAX_DIMS];
        labels[0] = self.labels[axis];
        Layout {
            shape: Dims { d: [self.shape.d[axis], 0, 0, 0, 0, 0, 0, 0], n: 1 },
            strides: Dims { d: [self.strides.d[axis], 0, 0, 0, 0, 0, 0, 0], n: 1 },
            offset: start,
            labels,
        }
    }
}

// OWNED ARRAY =====================================================================================

/// Owned, contiguous, row-major n-dimensional array.
#[derive(Clone, PartialEq)]
pub struct NdArray<T> {
    data: Vec<T>,
    shape: Dims,
    labels: [Axis; MAX_DIMS],
}

impl<T: fmt::Debug> fmt::Debug for NdArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdArray").field("shape", &self.shape).field("labels", &&self.labels[..self.shape.n]).field("data", &self.data).finish()
    }
}

impl<T: Copy> NdArray<T> {
    /// An array of `shape` filled with `value`.
    pub fn full(shape: &[usize], value: T) -> Result<Self, NdError> {
        let dims = Dims::new(shape)?;
        Ok(Self { data: vec![value; dims.product()], shape: dims, labels: [Axis::Unlabeled; MAX_DIMS] })
    }
    pub fn zeros(shape: &[usize]) -> Result<Self, NdError>
    where
        T: Default,
    {
        Self::full(shape, T::default())
    }
    /// Wraps `data` (row-major) as an array of `shape`.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, NdError> {
        let dims = Dims::new(shape)?;
        if dims.product() != data.len() {
            return Err(NdError::ShapeMismatch { expected: dims.product(), got: data.len() });
        }
        Ok(Self { data, shape: dims, labels: [Axis::Unlabeled; MAX_DIMS] })
    }
    /// Builds each element from its index.
    pub fn from_fn(shape: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Result<Self, NdError> {
        let dims = Dims::new(shape)?;
        let mut data = Vec::with_capacity(dims.product());
        for_each_index(dims.as_slice(), |idx| data.push(f(idx)));
        Ok(Self { data, shape: dims, labels: [Axis::Unlabeled; MAX_DIMS] })
    }
    /// Labels every axis; errors unless exactly one label per axis is given.
    pub fn with_labels(mut self, labels: &[Axis]) -> Result<Self, NdError> {
        if labels.len() != self.shape.n {
            return Err(NdError::LabelCount(labels.len(), self.shape.n));
        }
        self.labels[..labels.len()].copy_from_slice(labels);
        Ok(self)
    }
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }
    pub fn ndim(&self) -> usize {
        self.shape.n
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn labels(&self) -> &[Axis] {
        &self.labels[..self.shape.n]
    }
    /// Position of the first axis with this label.
    pub fn axis_of(&self, label: Axis) -> Option<usize> {
        self.labels().iter().position(|&l| l == label)
    }
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        self.layout().offset_of(index).map(|o| &self.data[o])
    }
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        self.layout().offset_of(index).map(move |o| &mut self.data[o])
    }
    /// Same elements, new shape (the element count must match). Labels are cleared.
    pub fn reshape(self, shape: &[usize]) -> Result<Self, NdError> {
        Self::from_vec(self.data, shape)
    }
    pub fn view(&self) -> NdView<'_, T> {
        NdView { data: &self.data, layout: self.layout() }
    }
    pub fn view_mut(&mut self) -> NdViewMut<'_, T> {
        let layout = self.layout();
        NdViewMut { data: &mut self.data, layout }
    }
    /// Applies `f` to every 1-D lane along `axis` (e.g. filter every channel along time).
    pub fn for_each_lane(&mut self, axis: usize, f: impl FnMut(&mut [T])) -> Result<(), NdError> {
        self.view_mut().for_each_lane(axis, f)
    }
    fn layout(&self) -> Layout {
        Layout::contiguous(self.shape, self.labels)
    }
}

impl<T: Copy> Index<&[usize]> for NdArray<T> {
    type Output = T;
    /// Panics if the index is out of bounds or has the wrong number of axes.
    fn index(&self, index: &[usize]) -> &T {
        self.get(index).expect("NdArray index out of bounds")
    }
}

impl<T: Copy> IndexMut<&[usize]> for NdArray<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        self.get_mut(index).expect("NdArray index out of bounds")
    }
}

/// An array is a signal over all its elements (row-major).
impl<T: Float> Signal for NdArray<T> {
    type Sample = T;
    fn samples(&self) -> &[T] {
        &self.data
    }
}

impl<T: Float> SignalMut for NdArray<T> {
    fn samples_mut(&mut self) -> &mut [T] {
        &mut self.data
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

/// Borrowed, possibly strided view of n-dimensional data.
#[derive(Clone, Copy)]
pub struct NdView<'a, T> {
    data: &'a [T],
    layout: Layout,
}

/// Mutable, possibly strided view of n-dimensional data.
pub struct NdViewMut<'a, T> {
    data: &'a mut [T],
    layout: Layout,
}

impl<T> fmt::Debug for NdView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = &self.layout;
        f.debug_struct("NdView").field("shape", &l.shape).field("strides", &l.strides).field("offset", &l.offset).finish()
    }
}

impl<T> fmt::Debug for NdViewMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = &self.layout;
        f.debug_struct("NdViewMut").field("shape", &l.shape).field("strides", &l.strides).field("offset", &l.offset).finish()
    }
}

/// Checks that a layout stays inside `len` elements.
fn checked_layout(len: usize, shape: &[usize], strides: &[usize], offset: usize) -> Result<Layout, NdError> {
    let shape = Dims::new(shape)?;
    let strides = Dims::new(strides)?;
    if shape.n != strides.n {
        return Err(NdError::ShapeMismatch { expected: shape.n, got: strides.n });
    }
    let layout = Layout { shape, strides, offset, labels: [Axis::Unlabeled; MAX_DIMS] };
    if layout.extent() > len {
        return Err(NdError::OutOfBounds);
    }
    Ok(layout)
}

macro_rules! view_common {
    ($View:ident) => {
        impl<'a, T: Copy> $View<'a, T> {
            pub fn shape(&self) -> &[usize] {
                self.layout.shape.as_slice()
            }
            /// Element strides per axis.
            pub fn strides(&self) -> &[usize] {
                self.layout.strides.as_slice()
            }
            pub fn ndim(&self) -> usize {
                self.layout.shape.n
            }
            pub fn len(&self) -> usize {
                self.layout.shape.product()
            }
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            pub fn labels(&self) -> &[Axis] {
                &self.layout.labels[..self.layout.shape.n]
            }
            pub fn axis_of(&self, label: Axis) -> Option<usize> {
                self.labels().iter().position(|&l| l == label)
            }
            /// Whether the elements are one contiguous row-major block.
            pub fn is_contiguous(&self) -> bool {
                self.layout.is_contiguous()
            }
            pub fn get(&self, index: &[usize]) -> Option<&T> {
                self.layout.offset_of(index).map(|o| &self.data[o])
            }
            /// Copies the elements (in logical order) into a new contiguous array, keeping labels.
            pub fn to_owned(&self) -> NdArray<T> {
                let mut data = Vec::with_capacity(self.len());
                for_each_index(self.shape(), |idx| data.push(self.data[self.layout.offset_of(idx).unwrap()]));
                NdArray { data, shape: self.layout.shape, labels: self.layout.labels }
            }
            /// Labels every axis of the view.
            pub fn with_labels(mut self, labels: &[Axis]) -> Result<Self, NdError> {
                if labels.len() != self.ndim() {
                    return Err(NdError::LabelCount(labels.len(), self.ndim()));
                }
                self.layout.labels[..labels.len()].copy_from_slice(labels);
                Ok(self)
            }
            /// Reorders axes: new axis i is old axis `axes[i]`. No copy.
            pub fn permute(self, axes: &[usize]) -> Result<Self, NdError> {
                let layout = self.layout.permute(axes)?;
                Ok(Self { layout, ..self })
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
            /// Restricts `axis` to `range`. No copy.
            pub fn slice_axis(self, axis: usize, range: Range<usize>) -> Result<Self, NdError> {
                let layout = self.layout.slice_axis(axis, range)?;
                Ok(Self { layout, ..self })
            }
            /// Fixes `axis` at `index`, dropping that axis. No copy.
            pub fn index_axis(self, axis: usize, index: usize) -> Result<Self, NdError> {
                let layout = self.layout.index_axis(axis, index)?;
                Ok(Self { layout, ..self })
            }
        }
    };
}

view_common!(NdView);
view_common!(NdViewMut);

impl<'a, T: Copy> NdView<'a, T> {
    /// A view over external memory (e.g. a buffer from another runtime). Errors if any element of the
    /// described layout would fall outside `data`.
    pub fn from_parts(data: &'a [T], shape: &[usize], strides: &[usize], offset: usize) -> Result<Self, NdError> {
        let layout = checked_layout(data.len(), shape, strides, offset)?;
        Ok(Self { data, layout })
    }
    /// The elements as one slice, if the view is contiguous.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        self.is_contiguous().then(|| &self.data[self.layout.offset..self.layout.offset + self.len()])
    }
    /// Elements in logical (row-major) order.
    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        let mut offsets = Vec::with_capacity(self.len());
        for_each_index(self.shape(), |idx| offsets.push(self.layout.offset_of(idx).unwrap()));
        let data = self.data;
        offsets.into_iter().map(move |o| &data[o])
    }
    /// The 1-D lanes along `axis`, as views. A lane along the last axis of a contiguous view is
    /// contiguous, so `lane.as_slice()` gives a `Signal`.
    pub fn lanes(&self, axis: usize) -> Result<Vec<NdView<'a, T>>, NdError> {
        check_axis(axis, self.ndim())?;
        let mut out = Vec::new();
        self.layout.lane_starts(axis, |start| out.push(NdView { data: self.data, layout: self.layout.lane(axis, start) }));
        Ok(out)
    }
    /// Copies a 1-D view's elements into a Vec (for lanes that aren't contiguous).
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().copied().collect()
    }
}

impl<'a, T: Copy> NdViewMut<'a, T> {
    /// A mutable view over external memory; errors if the layout would fall outside `data`.
    pub fn from_parts(data: &'a mut [T], shape: &[usize], strides: &[usize], offset: usize) -> Result<Self, NdError> {
        let layout = checked_layout(data.len(), shape, strides, offset)?;
        Ok(Self { data, layout })
    }
    /// A read-only view of the same elements.
    pub fn view(&self) -> NdView<'_, T> {
        NdView { data: self.data, layout: self.layout }
    }
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        self.layout.offset_of(index).map(move |o| &mut self.data[o])
    }
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        let (start, len) = (self.layout.offset, self.len());
        self.is_contiguous().then(move || &mut self.data[start..start + len])
    }
    pub fn fill(&mut self, value: T) {
        self.for_each_mut(|x| *x = value);
    }
    /// Calls `f` on every element.
    pub fn for_each_mut(&mut self, mut f: impl FnMut(&mut T)) {
        let layout = self.layout;
        for_each_index(layout.shape.as_slice(), |idx| f(&mut self.data[layout.offset_of(idx).unwrap()]));
    }
    /// Applies `f` to every 1-D lane along `axis`, in row-major order of the other axes. Contiguous
    /// lanes are passed in place; strided ones are copied to a scratch buffer and written back.
    pub fn for_each_lane(&mut self, axis: usize, mut f: impl FnMut(&mut [T])) -> Result<(), NdError> {
        check_axis(axis, self.ndim())?;
        let layout = self.layout;
        let (n, stride) = (layout.shape.d[axis], layout.strides.d[axis]);
        let mut scratch = Vec::new();
        let data = &mut *self.data;
        layout.lane_starts(axis, |start| {
            if stride == 1 {
                f(&mut data[start..start + n]);
            } else {
                scratch.clear();
                scratch.extend((0..n).map(|i| data[start + i * stride]));
                f(&mut scratch);
                for (i, &v) in scratch.iter().enumerate() {
                    data[start + i * stride] = v;
                }
            }
        });
        Ok(())
    }
}

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
        // back to contiguous by copying
        let owned = t.to_owned();
        assert_eq!(owned.shape(), [4, 3, 2]);
        assert!(owned.view().is_contiguous());
        assert_eq!(owned.get(&[3, 2, 1]), a.get(&[1, 2, 3]));
    }

    #[test]
    fn slicing_and_indexing_axes() {
        let a = counting(&[3, 4]);
        let s = a.view().slice_axis(1, 1..3).unwrap();
        assert_eq!(s.shape(), [3, 2]);
        assert_eq!(s.to_owned().into_vec(), [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        let row = a.view().index_axis(0, 2).unwrap();
        assert_eq!(row.as_slice(), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        let col = a.view().index_axis(1, 1).unwrap();
        assert_eq!(col.as_slice(), None);
        assert_eq!(col.to_vec(), [1.0, 5.0, 9.0]);
        assert!(a.view().slice_axis(0, 2..4).is_err());
        assert!(a.view().index_axis(2, 0).is_err());
    }

    #[test]
    fn lanes_are_signals() {
        // [channel, time]: 2 channels of a sine at different levels
        let tone: Vec<f64> = Sine::new(1_000.0, 48_000.0).take(4_800).collect();
        let a = NdArray::from_fn(&[2, 4_800], |i| tone[i[1]] * (i[0] + 1) as f64).unwrap().with_labels(&[Axis::Channel, Axis::Time]).unwrap();
        let time = a.axis_of(Axis::Time).unwrap();
        let lanes = a.view().lanes(time).unwrap();
        assert_eq!(lanes.len(), 2);
        let rms: Vec<f64> = lanes.iter().map(|l| l.as_slice().unwrap().rms().unwrap()).collect();
        assert!((rms[1] / rms[0] - 2.0).abs() < 1e-12);
        // lanes along the other axis are strided
        let across = a.view().lanes(0).unwrap();
        assert_eq!(across.len(), 4_800);
        assert_eq!(across[10].to_vec(), [tone[10], 2.0 * tone[10]]);
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
