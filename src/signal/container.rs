//! Capability tiers for signal *containers*, and the operations each tier unlocks.
//!
//! | tier | capability | implemented for |
//! |---|---|---|
//! | `Signal` | read the samples | `[T]`, `Vec<T>`, `[T; N]`, `NdArray<T>` |
//! | `SignalMut` | change samples in place (same length) | `[T]`, `Vec<T>`, `[T; N]`, `NdArray<T>` |
//! | [`SignalOwned`] | owns its storage: build from / turn back into its container | `Vec<T>`, `[T; N]`, `NdArray<T>` |
//! | [`SignalResizable`] | change its length | `Vec<T>` |
//!
//! Operation traits are classified by the tier they need, and implemented automatically for every
//! type with that capability:
//! - [`SigOwnedOps`] (needs `SignalOwned + Clone`): return a transformed copy of the same type
//!   (`normalized_peak`, `differenced`, `reversed`, ...).
//! - [`SigResizeOps`] (needs `SignalResizable`): length-changing operations in place (`convolve`,
//!   `pad`, `trim_silence`, ...). Sample-rate conversion is `resample::Resample` (`resampled`,
//!   `resample`), in the prelude.
//!
//! Naming: past tense returns new data (`convolved`), imperative mutates (`convolve`).

use super::{Broadcast, SignalError, SignalMut};
use crate::units::*;

// TIERS ===========================================================================================

/// A signal that owns its storage.
pub trait SignalOwned: SignalMut + Sized {
    /// The storage type (`Vec<T>` for vectors and n-d arrays, `[T; N]` for arrays).
    type Container;

    /// Wraps a container without copying.
    fn from_container(container: Self::Container) -> Self;
    /// Gives back the container.
    fn into_container(self) -> Self::Container;
    /// The container.
    fn as_container(&self) -> &Self::Container;
    /// The container, mutably.
    fn as_container_mut(&mut self) -> &mut Self::Container;
    /// An owned signal holding a copy of `samples`. Fixed-size types error if the length doesn't fit.
    fn from_samples(samples: &[Self::Sample]) -> Result<Self, SignalError>;
}

/// An owned signal whose length can change.
pub trait SignalResizable: SignalOwned {
    /// Changes the length, filling new samples with `value`.
    fn resize(&mut self, len: usize, value: Self::Sample);
    /// Removes every sample.
    fn clear(&mut self);
    /// Appends one sample.
    fn push(&mut self, sample: Self::Sample);
    /// Appends a copy of `samples` at the end.
    fn append_samples(&mut self, samples: &[Self::Sample]);
    /// Shortens to `len` samples (no effect if already shorter).
    fn truncate(&mut self, len: usize);
    /// Reserves room for at least `additional` more samples, so later growth doesn't allocate.
    fn reserve(&mut self, additional: usize);
}

impl<T: Float> SignalOwned for Vec<T> {
    type Container = Vec<T>;
    fn from_container(container: Vec<T>) -> Self {
        container
    }
    fn into_container(self) -> Vec<T> {
        self
    }
    fn as_container(&self) -> &Vec<T> {
        self
    }
    fn as_container_mut(&mut self) -> &mut Vec<T> {
        self
    }
    fn from_samples(samples: &[T]) -> Result<Self, SignalError> {
        Ok(samples.to_vec())
    }
}

impl<T: Float> SignalResizable for Vec<T> {
    fn resize(&mut self, len: usize, value: T) {
        Vec::resize(self, len, value)
    }
    fn clear(&mut self) {
        Vec::clear(self)
    }
    fn push(&mut self, sample: T) {
        Vec::push(self, sample)
    }
    fn append_samples(&mut self, samples: &[T]) {
        self.extend_from_slice(samples)
    }
    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len)
    }
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional)
    }
}

impl<T: Float, const N: usize> SignalOwned for [T; N] {
    type Container = [T; N];
    fn from_container(container: [T; N]) -> Self {
        container
    }
    fn into_container(self) -> [T; N] {
        self
    }
    fn as_container(&self) -> &[T; N] {
        self
    }
    fn as_container_mut(&mut self) -> &mut [T; N] {
        self
    }
    fn from_samples(samples: &[T]) -> Result<Self, SignalError> {
        samples.try_into().map_err(|_| SignalError::LengthMismatch(N, samples.len()))
    }
}

// OPERATIONS BY TIER ==============================================================================

/// Operations that return a transformed copy of the same signal type. Available on every
/// `SignalOwned + Clone` type; each mirrors a `SignalMut` method.
pub trait SigOwnedOps: SignalOwned + Clone {
    /// A copy with `f` applied to every sample.
    fn mapped(&self, f: impl FnMut(Self::Sample) -> Self::Sample) -> Self {
        let mut out = self.clone();
        out.apply(f);
        out
    }
    /// A copy multiplied by `gain`.
    fn scaled_by(&self, gain: Self::Sample) -> Self {
        let mut out = self.clone();
        out.scale(gain);
        out
    }
    /// A copy with `value` added to every sample.
    fn offset_by(&self, value: Self::Sample) -> Self {
        let mut out = self.clone();
        out.offset(value);
        out
    }
    /// A copy hard-clipped into `[lo, hi]`.
    fn clipped(&self, lo: Self::Sample, hi: Self::Sample) -> Self {
        let mut out = self.clone();
        out.clip(lo, hi);
        out
    }
    /// A copy scaled so its peak is `target`.
    fn normalized_peak(&self, target: Self::Sample) -> Self {
        let mut out = self.clone();
        out.normalize_peak(target);
        out
    }
    /// A copy scaled so its rms level is `target`.
    fn normalized_rms(&self, target: Self::Sample) -> Self {
        let mut out = self.clone();
        out.normalize_rms(target);
        out
    }
    /// A copy with the mean subtracted.
    fn dc_removed(&self) -> Self {
        let mut out = self.clone();
        out.remove_dc();
        out
    }
    /// A copy faded in linearly over the first `len` samples.
    fn faded_in(&self, len: usize) -> Self {
        let mut out = self.clone();
        out.fade_in(len);
        out
    }
    /// A copy faded out linearly over the last `len` samples.
    fn faded_out(&self, len: usize) -> Self {
        let mut out = self.clone();
        out.fade_out(len);
        out
    }
    /// Running sum (see `SignalMut::cumsum`).
    fn integrated(&self) -> Self {
        let mut out = self.clone();
        out.cumsum();
        out
    }
    /// First difference (see `SignalMut::diff`).
    fn differenced(&self) -> Self {
        let mut out = self.clone();
        out.diff();
        out
    }
    /// A copy in reverse order.
    fn reversed(&self) -> Self {
        let mut out = self.clone();
        out.samples_mut().reverse();
        out
    }
    /// A copy projected onto `basis` (see `SignalMut::project_onto`).
    fn projected_onto(&self, basis: &[Self::Sample]) -> Result<Self, SignalError> {
        let mut out = self.clone();
        out.project_onto(basis)?;
        Ok(out)
    }
    /// A copy combined with `other` sample by sample (see `SignalMut::zip_apply`).
    fn zipped_with(
        &self,
        other: &[Self::Sample],
        mode: Broadcast<Self::Sample>,
        f: impl FnMut(Self::Sample, Self::Sample) -> Self::Sample,
    ) -> Result<Self, SignalError> {
        let mut out = self.clone();
        out.zip_apply(other, mode, f)?;
        Ok(out)
    }
}

impl<S: SignalOwned + Clone> SigOwnedOps for S {}

/// Length-changing operations applied in place. Available on every `SignalResizable` type.
pub trait SigResizeOps: SignalResizable {
    /// Replaces the signal with its full convolution with `kernel` (grows by `kernel.len() - 1`).
    fn convolve(&mut self, kernel: &[Self::Sample]) {
        let out = self.convolved(kernel);
        self.replace_with(&out);
    }
    /// Replaces the signal with its full cross-correlation with `other` (see `Signal::correlated`).
    fn correlate(&mut self, other: &[Self::Sample]) {
        let out = self.correlated(other);
        self.replace_with(&out);
    }
    /// Adds `before` samples at the start and `after` at the end, all equal to `value`.
    fn pad(&mut self, before: usize, after: usize, value: Self::Sample) {
        let len = self.samples().len();
        self.resize(len + before + after, value);
        let x = self.samples_mut();
        x.copy_within(0..len, before);
        x[..before].iter_mut().for_each(|s| *s = value);
    }
    /// Removes leading and trailing samples whose magnitude is at most `threshold`; returns how many
    /// were removed from (start, end). An all-quiet signal becomes empty.
    fn trim_silence(&mut self, threshold: Self::Sample) -> (usize, usize) {
        let x = self.samples();
        let loud = |s: &Self::Sample| s._abs() > threshold;
        let Some(first) = x.iter().position(loud) else {
            let removed = x.len();
            self.clear();
            return (removed, 0);
        };
        let last = x.iter().rposition(loud).unwrap_or(first);
        let (len, kept_end) = (x.len(), last + 1);
        self.samples_mut().copy_within(first..kept_end, 0);
        self.truncate(kept_end - first);
        (first, len - kept_end)
    }
    /// Replaces the contents with a copy of `samples`, reusing the allocation when it is big enough.
    fn replace_with(&mut self, samples: &[Self::Sample]) {
        self.clear();
        self.append_samples(samples);
    }
}

impl<S: SignalResizable> SigResizeOps for S {}

// EDGE EXTENSION ==================================================================================

/// How [`extended`] continues a signal past its ends (SciPy's `odd_ext`, `even_ext`, `const_ext`,
/// zeros).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edge {
    /// Point reflection about the end sample.
    Odd,
    /// Mirror image, without repeating the end sample.
    Even,
    /// The end sample repeated.
    Constant,
    /// Zeros.
    Zeros,
}

/// `x` extended by `n` samples at both ends (reflections stop at the far end when `n >= len`).
pub(crate) fn extended(x: &[f64], n: usize, edge: Edge) -> Vec<f64> {
    if n == 0 || x.is_empty() {
        return x.to_vec();
    }
    let len = x.len();
    let (first, last) = (x[0], x[len - 1]);
    let at = |i: usize| x[i.min(len - 1)];
    let left = (1..=n).rev().map(|i| match edge {
        Edge::Odd => 2.0 * first - at(i),
        Edge::Even => at(i),
        Edge::Constant => first,
        Edge::Zeros => 0.0,
    });
    let right = (1..=n).map(|i| match edge {
        Edge::Odd => 2.0 * last - x[len - 1 - i.min(len - 1)],
        Edge::Even => x[len - 1 - i.min(len - 1)],
        Edge::Constant => last,
        Edge::Zeros => 0.0,
    });
    left.chain(x.iter().copied()).chain(right).collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_containers_roundtrip() {
        let v = Vec::from_samples(&[1.0f64, 2.0]).unwrap();
        assert_eq!(v.as_container(), &vec![1.0, 2.0]);
        let a: [f64; 2] = SignalOwned::from_samples(&[3.0, 4.0]).unwrap();
        assert_eq!(a.into_container(), [3.0, 4.0]);
        assert_eq!(<[f64; 3]>::from_samples(&[1.0]), Err(SignalError::LengthMismatch(3, 1)));
    }

    #[test]
    fn owned_ops_return_copies_and_leave_the_original() {
        let x = vec![1.0f64, 2.0, 3.0];
        assert_eq!(x.reversed(), [3.0, 2.0, 1.0]);
        assert_eq!(x.differenced(), [1.0, 1.0, 1.0]);
        assert_eq!(x.differenced().integrated(), x);
        assert_eq!(x.normalized_peak(1.5), [0.5, 1.0, 1.5]);
        assert_eq!(x.zipped_with(&[1.0], Broadcast::Latch, |a, b| a - b).unwrap(), [0.0, 1.0, 2.0]);
        assert_eq!(x, [1.0, 2.0, 3.0]);
        // fixed-size arrays get them too
        let arr = [2.0f32, -4.0];
        assert_eq!(arr.scaled_by(0.5), [1.0, -2.0]);
        assert_eq!(arr.mapped(f32::abs), [2.0, 4.0]);
    }

    #[test]
    fn resizable_ops() {
        let mut x = vec![1.0f64, 2.0];
        x.convolve(&[1.0, 1.0]);
        assert_eq!(x, [1.0, 3.0, 2.0]);
        x.pad(2, 1, 0.0);
        assert_eq!(x, [0.0, 0.0, 1.0, 3.0, 2.0, 0.0]);
        assert_eq!(x.trim_silence(0.5), (2, 1));
        assert_eq!(x, [1.0, 3.0, 2.0]);
        let mut quiet = vec![0.01f64, -0.02];
        assert_eq!(quiet.trim_silence(0.1), (2, 0));
        assert!(quiet.is_empty());

        SignalResizable::push(&mut x, 1.0);
        assert_eq!(x.len(), 4);
    }
}
