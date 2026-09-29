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
//!   `resample`, `pad`, `trim_silence`, ...).
//!
//! Naming: past tense returns new data (`convolved`), imperative mutates (`convolve`).

use super::{Broadcast, SignalError, SignalMut};
use crate::units::*;

// TIERS ===========================================================================================

/// A signal that owns its storage.
pub trait SignalOwned: SignalMut + Sized {
    /// The storage type (`Vec<T>` for vectors and n-d arrays, `[T; N]` for arrays).
    type Container;

    fn from_container(container: Self::Container) -> Self;
    fn into_container(self) -> Self::Container;
    fn as_container(&self) -> &Self::Container;
    fn as_container_mut(&mut self) -> &mut Self::Container;
    /// An owned signal holding a copy of `samples`. Fixed-size types error if the length doesn't fit.
    fn from_samples(samples: &[Self::Sample]) -> Result<Self, SignalError>;
}

/// An owned signal whose length can change.
pub trait SignalResizable: SignalOwned {
    /// Changes the length, filling new samples with `value`.
    fn resize(&mut self, len: usize, value: Self::Sample);
    fn clear(&mut self);
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
    fn mapped(&self, f: impl FnMut(Self::Sample) -> Self::Sample) -> Self {
        let mut out = self.clone();
        out.apply(f);
        out
    }
    fn scaled_by(&self, gain: Self::Sample) -> Self {
        let mut out = self.clone();
        out.scale(gain);
        out
    }
    fn offset_by(&self, value: Self::Sample) -> Self {
        let mut out = self.clone();
        out.offset(value);
        out
    }
    fn clipped(&self, lo: Self::Sample, hi: Self::Sample) -> Self {
        let mut out = self.clone();
        out.clip(lo, hi);
        out
    }
    fn normalized_peak(&self, target: Self::Sample) -> Self {
        let mut out = self.clone();
        out.normalize_peak(target);
        out
    }
    fn normalized_rms(&self, target: Self::Sample) -> Self {
        let mut out = self.clone();
        out.normalize_rms(target);
        out
    }
    fn dc_removed(&self) -> Self {
        let mut out = self.clone();
        out.remove_dc();
        out
    }
    fn faded_in(&self, len: usize) -> Self {
        let mut out = self.clone();
        out.fade_in(len);
        out
    }
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
    fn reversed(&self) -> Self {
        let mut out = self.clone();
        out.samples_mut().reverse();
        out
    }
    fn projected_onto(&self, basis: &[Self::Sample]) -> Result<Self, SignalError> {
        let mut out = self.clone();
        out.project_onto(basis)?;
        Ok(out)
    }
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
    /// Converts the sample rate in place (see `Signal::resampled`).
    fn resample(&mut self, from_rate: u32, to_rate: u32) {
        let out = self.resampled(from_rate, to_rate);
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

        let mut tone: Vec<f64> = (0..4_800).map(|n| (n as f64 * 0.1).sin()).collect();
        tone.resample(48_000, 16_000);
        assert_eq!(tone.len(), 1_600);
        SignalResizable::push(&mut tone, 1.0);
        assert_eq!(tone.len(), 1_601);
    }
}
