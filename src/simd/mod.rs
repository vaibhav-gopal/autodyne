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

/// Timing candidates for the AVX-512 copy (see the ignored `timing` test).
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512_lanes<T: Float, const L: usize>(a: &[T], b: &[T]) -> T {
    dot_lanes::<T, L>(a, b)
}
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2_lanes<T: Float, const L: usize>(a: &[T], b: &[T]) -> T {
    dot_lanes::<T, L>(a, b)
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
    dot_lanes::<T, LANES>(a, b)
}

/// [`dot_kernel`] with `L` independent accumulators (a power of two): more for wider registers,
/// so each register still has several add chains in flight.
#[inline(always)]
fn dot_lanes<T: Float, const L: usize>(a: &[T], b: &[T]) -> T {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]); // equal lengths, so the remainders line up
    let mut acc = [T::_ZERO; L];
    let (a_chunks, a_rest) = a.as_chunks::<L>();
    let (b_chunks, b_rest) = b.as_chunks::<L>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks) {
        // fixed-size arrays: no bounds checks, so the loop body is straight-line SIMD
        for i in 0..L {
            acc[i] = acc[i] + ca[i] * cb[i];
        }
    }
    let mut tail = T::_ZERO;
    for (&x, &y) in a_rest.iter().zip(b_rest) {
        tail = tail + x * y;
    }
    // pairwise reduction keeps rounding error low
    let mut width = L;
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

    /// Dot-product throughput by copy and accumulator count, in cache (4 Ki) and out (1 Mi):
    /// `cargo test --release -p autodyne --lib simd::tests::timing -- --ignored --nocapture`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn timing() {
        fn best(f: &mut dyn FnMut() -> f64, reps: usize) -> f64 {
            let mut best = f64::MAX;
            for _ in 0..21 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(f());
                }
                best = best.min(t.elapsed().as_secs_f64() / reps as f64);
            }
            best
        }
        for n in [4_096usize, 1 << 20] {
            let reps = (1 << 24) / n;
            let a64 = noise(1, n);
            let b64 = noise(2, n);
            let a32: Vec<f32> = a64.iter().map(|&x| x as f32).collect();
            let b32: Vec<f32> = b64.iter().map(|&x| x as f32).collect();
            let gflops = |s: f64| 2.0 * n as f64 / s / 1e9;
            let mut row = format!("TIMING n {n}:");
            let mut add = |name: &str, s: f64| row.push_str(&format!(" {name} {:.1}", gflops(s)));
            add("f32 base16", best(&mut || dot_kernel(&a32, &b32) as f64, reps));
            if avx2_available() {
                add("avx2-16", best(&mut || unsafe { dot_avx2_lanes::<f32, 16>(&a32, &b32) } as f64, reps));
                add("avx2-32", best(&mut || unsafe { dot_avx2_lanes::<f32, 32>(&a32, &b32) } as f64, reps));
            }
            if avx512_available() {
                add("512-16", best(&mut || unsafe { dot_avx512_lanes::<f32, 16>(&a32, &b32) } as f64, reps));
                add("512-64", best(&mut || unsafe { dot_avx512_lanes::<f32, 64>(&a32, &b32) } as f64, reps));
            }
            add("| f64 base16", best(&mut || dot_kernel(&a64, &b64), reps));
            if avx2_available() {
                add("avx2-16", best(&mut || unsafe { dot_avx2_lanes::<f64, 16>(&a64, &b64) }, reps));
            }
            if avx512_available() {
                add("512-16", best(&mut || unsafe { dot_avx512_lanes::<f64, 16>(&a64, &b64) }, reps));
                add("512-32", best(&mut || unsafe { dot_avx512_lanes::<f64, 32>(&a64, &b64) }, reps));
            }
            eprintln!("{row} (GFLOP/s)");
        }
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
