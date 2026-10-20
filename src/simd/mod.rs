//! Vectorized inner-loop kernels on stable Rust.
//!
//! There is no portable SIMD type on stable yet, so the kernels are written in a shape LLVM reliably
//! vectorizes: fixed-size chunks and several independent accumulators. A single accumulator would
//! chain every add onto the previous one; since float addition isn't associative, the compiler must
//! keep that order and can't use SIMD. Independent lanes remove the dependency.
//!
//! On x86-64 the same code is also compiled with AVX2 and chosen at runtime when the CPU supports it
//! (twice the width of the SSE2 baseline every x86-64 CPU has).
//!
//! `dot` checks the CPU on every call, which is fine for one-off use. Hot loops should instead pick
//! the instruction set once per block: write the loop as an `#[inline(always)]` body using
//! `dot_kernel`, add an `#[target_feature(enable = "avx2")]` wrapper around it, and branch on
//! `avx2_available()` (see `Fir::process`). The kernel then inlines into each version of the loop,
//! with no per-sample call or check.
//!
//! [`lanes`] has the lane types (`F32x4`, `F32x8`, `F32x16`) kernels are written over, and
//! [`Level`] the instruction sets they're compiled for.
//!
//! tend: Core / simd

pub mod lanes;

use crate::units::*;

/// Independent accumulators per kernel: two AVX registers of f32 (four of f64), enough to hide
/// floating-point add latency.
const LANES: usize = 16;

/// Dot product: `sum of a[i] * b[i]`. Panics if the lengths differ.
///
/// Summation order differs from a plain left-to-right loop, so results can differ in the last few
/// bits (usually more accurate, since the partial sums are smaller).
#[inline]
pub fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "dot product needs equal lengths");
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: the CPU was just checked for AVX2, the only feature `dot_avx2` is compiled with.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_kernel(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2<T: Float>(a: &[T], b: &[T]) -> T {
    dot_kernel(a, b)
}

/// Whether the AVX2 versions of the hot loops can run on this CPU. With `std` it's detected at run
/// time (cached by std, so calling it once per block is cheap); without, it's whether the build
/// targets AVX2 (`-C target-feature=+avx2`). Always false off x86-64.
#[inline]
pub fn avx2_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        cfg!(target_feature = "avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the CPU also has fused multiply-add (with AVX2), for loops written with `mul_add`.
/// Decided like [`avx2_available`]. Always false off x86-64.
#[inline]
pub fn avx2_fma_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        cfg!(all(target_feature = "avx2", target_feature = "fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the CPU has AVX-512 (the foundation set: sixteen `f32` lanes), for loops compiled with
/// `#[target_feature(enable = "avx512f")]`. Decided like [`avx2_available`]. Always false off
/// x86-64.
#[inline]
pub fn avx512_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        cfg!(target_feature = "avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The widest instruction set [`lanes`] kernels are compiled for that this CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// The target's baseline: SSE2 on x86-64, NEON on aarch64, simd128 on wasm32 built with it, or
    /// scalar code.
    Baseline,
    /// AVX2 and FMA (eight `f32` lanes).
    Avx2,
    /// AVX-512F (sixteen `f32` lanes).
    Avx512,
}

impl Level {
    /// The best level here, detected once per call (std caches the detection).
    #[inline]
    pub fn detect() -> Self {
        if avx512_available() {
            Level::Avx512
        } else if avx2_fma_available() {
            Level::Avx2
        } else {
            Level::Baseline
        }
    }
}

/// The dot-product body, for use inside loops that do their own per-block dispatch (see the module
/// docs). `inline(always)` so each caller compiles its own copy with its own instruction set.
/// Unlike `dot`, lengths are not checked: the shorter slice wins.
#[inline(always)]
pub fn dot_kernel<T: Float>(a: &[T], b: &[T]) -> T {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]); // equal lengths, so the remainders line up
    let mut acc = [T::_ZERO; LANES];
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks) {
        // fixed-size arrays: no bounds checks, so the loop body is straight-line SIMD
        for i in 0..LANES {
            acc[i] = acc[i] + ca[i] * cb[i];
        }
    }
    let mut tail = T::_ZERO;
    for (&x, &y) in a_rest.iter().zip(b_rest) {
        tail = tail + x * y;
    }
    // pairwise reduction keeps rounding error low
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] = acc[i] + acc[i + width];
        }
    }
    acc[0] + tail
}

/// Sum of all elements, with the same independent-accumulator shape as `dot` (so it vectorizes).
#[inline]
pub fn sum<T: Float>(a: &[T]) -> T {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: the CPU was just checked for AVX2, the only feature `sum_avx2` is compiled with.
            return unsafe { sum_avx2(a) };
        }
    }
    sum_kernel(a)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2<T: Float>(a: &[T]) -> T {
    sum_kernel(a)
}

/// The sum body, for loops with their own per-block dispatch (see the module docs).
#[inline(always)]
pub fn sum_kernel<T: Float>(a: &[T]) -> T {
    let mut acc = [T::_ZERO; LANES];
    let (chunks, rest) = a.as_chunks::<LANES>();
    for c in chunks {
        for i in 0..LANES {
            acc[i] = acc[i] + c[i];
        }
    }
    let tail = rest.iter().fold(T::_ZERO, |s, &x| s + x);
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] = acc[i] + acc[i + width];
        }
    }
    acc[0] + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` values in [-1, 1) from `seed` (xorshift; simd sits below osc, so no Noise here).
    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn naive(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn matches_naive_for_every_length_and_remainder() {
        let a: Vec<f64> = noise(1, 300);
        let b: Vec<f64> = noise(2, 300);
        for len in 0..300 {
            let (x, y) = (&a[..len], &b[..len]);
            assert!((dot(x, y) - naive(x, y)).abs() < 1e-12, "len {len}");
            assert!((dot_kernel(x, y) - naive(x, y)).abs() < 1e-12, "baseline kernel, len {len}");
        }
    }

    #[test]
    fn sum_matches_naive_for_every_length() {
        let a: Vec<f64> = noise(7, 300);
        for len in 0..300 {
            let expected: f64 = a[..len].iter().sum();
            assert!((sum(&a[..len]) - expected).abs() < 1e-12, "len {len}");
        }
    }

    #[test]
    fn f32_agrees_with_f64() {
        let a: Vec<f64> = noise(3, 1000);
        let b: Vec<f64> = noise(4, 1000);
        let (a32, b32): (Vec<f32>, Vec<f32>) = (a.iter().map(|&x| x as f32).collect(), b.iter().map(|&x| x as f32).collect());
        assert!((dot(&a32, &b32) as f64 - naive(&a, &b)).abs() < 1e-4);
    }

    #[test]
    fn exact_when_only_one_product_is_nonzero() {
        // e.g. filtering an impulse must reproduce the taps bit-for-bit
        let mut a = vec![0.0f32; 37];
        a[20] = 1.0;
        let b: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        assert_eq!(dot(&a, &b), b[20]);
    }

    #[test]
    fn kernel_uses_the_shorter_length() {
        let a: Vec<f64> = noise(5, 50);
        let b: Vec<f64> = noise(6, 20);
        assert!((dot_kernel(&a, &b) - naive(&a[..20], &b)).abs() < 1e-12);
        assert!((dot_kernel(&b, &a) - naive(&b, &a[..20])).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn rejects_mismatched_lengths() {
        dot(&[1.0f64, 2.0], &[1.0]);
    }
}
