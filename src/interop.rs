//! Zero-copy conversions to and from other array libraries.
//!
//! - The [`ndarray`](https://docs.rs/ndarray) crate (feature `ndarray`): views convert both ways
//!   without copying (any strides, including reversed axes); owned arrays move their `Vec` across
//!   when the layout is standard and copy otherwise.
//! - Python, PyTorch, JAX, CuPy...: through [`dlpack`](crate::dlpack).
//! - Apache Arrow: its primitive arrays are contiguous buffers, so `NdView::from_slice(array.values())`
//!   is already a zero-copy view.

#[cfg(feature = "ndarray")]
mod ndarray_crate {
    use ndarray::{ArrayD, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMutD, Axis as NdAxis, Dimension, IxDyn, ShapeBuilder};

    use crate::signal::{NdArray, NdError, NdView, NdViewMut, MAX_DIMS};

    /// (pointer to the lowest-addressed element, absolute strides, axes to flip back)
    fn lowest<T>(ptr: *const T, shape: &[usize], strides: &[isize]) -> (*const T, [usize; MAX_DIMS], [bool; MAX_DIMS]) {
        let mut abs = [0usize; MAX_DIMS];
        let mut flip = [false; MAX_DIMS];
        let mut low = ptr;
        let empty = shape.contains(&0);
        for (i, (&d, &s)) in shape.iter().zip(strides).enumerate() {
            abs[i] = s.unsigned_abs();
            if s < 0 && d > 1 {
                flip[i] = true;
                if !empty {
                    low = low.wrapping_offset((d as isize - 1) * s);
                }
            }
        }
        (low, abs, flip)
    }

    impl<'a, T, D: Dimension> TryFrom<ArrayView<'a, T, D>> for NdView<'a, T> {
        type Error = NdError;
        /// The same elements, no copy. Errors beyond `MAX_DIMS` axes.
        fn try_from(view: ArrayView<'a, T, D>) -> Result<Self, NdError> {
            let strides: Vec<isize> = view.strides().to_vec();
            // SAFETY: an ndarray view guarantees its elements are valid for reads for 'a
            unsafe { NdView::from_raw_parts(view.as_ptr(), view.shape(), &strides) }
        }
    }

    impl<'a, T, D: Dimension> TryFrom<ArrayViewMut<'a, T, D>> for NdViewMut<'a, T> {
        type Error = NdError;
        fn try_from(mut view: ArrayViewMut<'a, T, D>) -> Result<Self, NdError> {
            let strides: Vec<isize> = view.strides().to_vec();
            let shape: Vec<usize> = view.shape().to_vec();
            // SAFETY: an ndarray mutable view grants exclusive access to its elements for 'a
            unsafe { NdViewMut::from_raw_parts(view.as_mut_ptr(), &shape, &strides) }
        }
    }

    impl<'a, T> From<NdView<'a, T>> for ArrayViewD<'a, T> {
        /// The same elements, no copy.
        fn from(view: NdView<'a, T>) -> Self {
            let (low, abs, flip) = lowest(view.as_ptr(), view.shape(), view.strides());
            let n = view.ndim();
            // SAFETY: the layout was validated against live memory for 'a; from the lowest element
            // with absolute strides every reachable element is inside it, and flipping the reversed
            // axes restores the original order
            let mut out = unsafe { ArrayView::from_shape_ptr(IxDyn(view.shape()).strides(IxDyn(&abs[..n])), low) };
            for (i, &f) in flip[..n].iter().enumerate() {
                if f {
                    out.invert_axis(NdAxis(i));
                }
            }
            out
        }
    }

    impl<'a, T> From<NdViewMut<'a, T>> for ArrayViewMutD<'a, T> {
        fn from(mut view: NdViewMut<'a, T>) -> Self {
            let (low, abs, flip) = lowest(view.as_mut_ptr().cast_const(), view.shape(), view.strides());
            let n = view.ndim();
            let shape: Vec<usize> = view.shape().to_vec();
            // SAFETY: as above; the view was exclusive (and injective) for 'a
            let mut out = unsafe { ArrayViewMut::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&abs[..n])), low.cast_mut()) };
            for (i, &f) in flip[..n].iter().enumerate() {
                if f {
                    out.invert_axis(NdAxis(i));
                }
            }
            out
        }
    }

    impl<T> From<NdArray<T>> for ArrayD<T> {
        /// Moves the `Vec` across, no copy. Axis labels are dropped.
        fn from(array: NdArray<T>) -> Self {
            let shape = array.shape().to_vec();
            ArrayD::from_shape_vec(IxDyn(&shape), array.into_vec()).expect("an NdArray's shape matches its elements")
        }
    }

    impl<T: Clone, D: Dimension> TryFrom<ndarray::Array<T, D>> for NdArray<T> {
        type Error = NdError;
        /// Moves the `Vec` across when the array is standard row-major from the start of its buffer;
        /// copies otherwise.
        fn try_from(array: ndarray::Array<T, D>) -> Result<Self, NdError> {
            let shape = array.shape().to_vec();
            if array.is_standard_layout() {
                let (data, offset) = array.into_raw_vec_and_offset();
                if offset.unwrap_or(0) == 0 && data.len() == shape.iter().product::<usize>() {
                    return NdArray::from_vec(data, &shape);
                }
                return NdArray::from_vec(data[offset.unwrap_or(0)..][..shape.iter().product::<usize>()].to_vec(), &shape);
            }
            NdArray::from_vec(array.iter().cloned().collect(), &shape)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::{s, Array2, Array3};

        #[test]
        fn views_cross_both_ways_without_copying() {
            let a = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f64);
            // ndarray -> autodyne: a reversed, stepped, transposed view
            let theirs = a.slice(s![.., ..;-1, 1..;2]).reversed_axes();
            let ours = NdView::try_from(theirs.view()).unwrap();
            assert_eq!(ours.shape(), theirs.shape());
            assert_eq!(ours.to_vec(), theirs.iter().copied().collect::<Vec<_>>());
            assert_eq!(ours.sum(), theirs.sum());
            // autodyne -> ndarray: reversed and transposed back
            let mine = NdArray::from_fn(&[3, 5], |i| (i[0] * 5 + i[1]) as f32).unwrap();
            let flipped = mine.view().flip(1).unwrap().transpose();
            let back = ArrayViewD::from(flipped);
            assert_eq!(back.shape(), [5, 3]);
            assert_eq!(back.iter().copied().collect::<Vec<_>>(), flipped.to_vec());
            assert_eq!(back.as_ptr(), flipped.get(&[0, 0]).unwrap() as *const f32, "the same memory");
        }

        #[test]
        fn mutable_views_and_owned_arrays() {
            let mut a = Array2::<f32>::zeros((3, 4));
            let mut ours = NdViewMut::try_from(a.slice_mut(s![.., ..;-1])).unwrap();
            *ours.get_mut(&[0, 0]).unwrap() = 7.0;
            assert_eq!(a[[0, 3]], 7.0, "written through the reversed view");
            let mut mine = NdArray::<f32>::zeros(&[2, 3]).unwrap();
            ArrayViewMutD::from(mine.view_mut().flip(0).unwrap()).fill(1.5);
            assert_eq!(mine.view().sum(), 9.0);
            // owned: the Vec moves across
            let owned = NdArray::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
            let address = owned.as_slice().as_ptr();
            let theirs = ArrayD::from(owned);
            assert_eq!(theirs.as_ptr(), address);
            let again = NdArray::try_from(theirs).unwrap();
            assert_eq!(again.as_slice().as_ptr(), address);
            // a transposed array copies into row-major order
            let t = NdArray::try_from(Array2::from_shape_fn((2, 3), |(i, j)| i * 3 + j).reversed_axes()).unwrap();
            assert_eq!(t.as_slice(), [0, 3, 1, 4, 2, 5]);
        }
    }
}