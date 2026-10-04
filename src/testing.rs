//! Helpers shared by the unit tests: deterministic data and tolerance checks. Compiled for tests
//! only.

// the data generators serve feature-gated modules' tests (linalg, flux, gpu)
#![allow(dead_code)]

use crate::signal::NdArray;

/// Asserts `|got - want| <= tol`, naming the check in the failure message.
pub(crate) fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: {got} vs {want} (tol {tol})");
}

/// `n` deterministic `f32` values in [-1, 1) from a linear congruential generator: the same seed
/// gives the same values on every platform.
pub(crate) fn noise_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// [`noise_f32`] values in an array of `shape`.
pub(crate) fn random_f32(shape: &[usize], seed: u32) -> NdArray<f32> {
    NdArray::from_vec(noise_f32(shape.iter().product(), seed), shape).expect("the shape holds the values")
}

/// `n` deterministic `f64` values in [-1, 1) from a 64-bit linear congruential generator (53-bit
/// mantissas).
pub(crate) fn noise_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// [`noise_f64`] values in an array of `shape`.
pub(crate) fn random_f64(shape: &[usize], seed: u64) -> NdArray<f64> {
    NdArray::from_vec(noise_f64(shape.iter().product(), seed), shape).expect("the shape holds the values")
}
