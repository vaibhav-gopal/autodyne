//! Lanes of `f32` on stable Rust: [`F32s<N>`](F32s) (with [`F32x4`], [`F32x8`] and [`F32x16`]),
//! arrays whose arithmetic is written lane by lane in the shape LLVM vectorizes.
//!
//! Code using them is compiled into one copy per instruction set: as the target builds it (SSE2
//! on x86-64, NEON on aarch64, simd128 on wasm32 built with it) and, on x86-64, inside
//! `#[target_feature(enable = "avx2,fma")]` and `"avx512f"` wrappers, chosen once per call by
//! [`Level::detect`](super::Level::detect). No operation here contracts a multiply and an add into
//! a fused one, so every copy computes the same bits.
//!
//! They suit a group of values computed together (sixteen pixels' distances to a shape). For a
//! kernel over a whole row, a plain per-element loop in the loop vectorizer's shape, compiled the
//! same way, vectorizes more reliably than a loop over lane groups, whose straight-line body LLVM
//! may leave scalar: `geometry::raster` measured 4.3x / 8.8x / 10.8x a plain loop (SSE2 / AVX2 /
//! AVX-512) the first way, and under 3x the second.
//!
//! tend: Geometry and lanes / simd lanes

use core::ops::{Add, Div, Mul, Neg, Sub};

/// `N` lanes of `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct F32s<const N: usize>(pub [f32; N]);

/// Four lanes (one SSE, NEON or simd128 register).
pub type F32x4 = F32s<4>;
/// Eight lanes (one AVX register).
pub type F32x8 = F32s<8>;
/// Sixteen lanes (one AVX-512 register).
pub type F32x16 = F32s<16>;

/// `[f(0), f(1), ..., f(N - 1)]` as an index loop: `core::array::from_fn` and `map` don't always
/// inline (their drop guards make them too big), and a lane operation behind a call can't vectorize.
#[inline(always)]
fn lanes<const N: usize>(f: impl Fn(usize) -> f32) -> [f32; N] {
    let mut out = [0.0; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = f(i);
    }
    out
}

impl<const N: usize> F32s<N> {
    /// Every lane `x`.
    #[inline(always)]
    pub fn splat(x: f32) -> Self {
        F32s([x; N])
    }
    /// `start`, `start + step`, `start + 2 step`, ...
    #[inline(always)]
    pub fn ramp(start: f32, step: f32) -> Self {
        F32s(lanes(|i| start + i as f32 * step))
    }
    /// The first `N` values of `s`; lanes past its end are `fill`.
    #[inline(always)]
    pub fn load_or(s: &[f32], fill: f32) -> Self {
        F32s(lanes(|i| if i < s.len() { s[i] } else { fill }))
    }
    /// Writes the first `min(N, s.len())` lanes into `s`.
    #[inline(always)]
    pub fn store(self, s: &mut [f32]) {
        for (d, v) in s.iter_mut().zip(self.0) {
            *d = v;
        }
    }
    /// Each lane through `f`.
    #[inline(always)]
    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        F32s(lanes(|i| f(self.0[i])))
    }
    /// Lane-wise `f(self, other)`.
    #[inline(always)]
    pub fn zip(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        F32s(lanes(|i| f(self.0[i], other.0[i])))
    }
    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        self.zip(o, |a, b| if a < b { a } else { b })
    }
    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        self.zip(o, |a, b| if a > b { a } else { b })
    }
    /// Lane-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        self.map(f32::abs)
    }
    /// Each lane clamped to `lo..=hi`.
    #[inline(always)]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        self.max(Self::splat(lo)).min(Self::splat(hi))
    }
    /// Each lane clamped to 0..=1.
    #[inline(always)]
    pub fn saturate(self) -> Self {
        self.clamp(0.0, 1.0)
    }
    /// Lane-wise square root (the instruction, not a library call, with and without `std`).
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        F32s(sqrt_lanes(self.0))
    }
    /// The length of the 2-vectors `(x, y)`, lane by lane.
    #[inline(always)]
    pub fn length2(x: Self, y: Self) -> Self {
        (x * x + y * y).sqrt()
    }
    /// Each lane rounded half up and saturated into a byte (for values meant to be within 0..=255).
    #[inline(always)]
    pub fn to_u8(self) -> [u8; N] {
        // `as` truncates toward zero and saturates (NaN to 0)
        let mut out = [0u8; N];
        for (o, v) in out.iter_mut().zip(self.0) {
            *o = (v + 0.5) as u8;
        }
        out
    }
}

/// Lane-wise square roots: std's `sqrt` (the instruction) with `std`; without it, the target's own
/// square-root instruction four lanes at a time, or `libm` where there is none.
#[inline(always)]
fn sqrt_lanes<const N: usize>(v: [f32; N]) -> [f32; N] {
    #[cfg(feature = "std")]
    {
        lanes(|i| v[i].sqrt())
    }
    #[cfg(not(feature = "std"))]
    {
        let mut out = v;
        let (quads, rest) = out.as_chunks_mut::<4>();
        for quad in quads {
            *quad = sqrt4(*quad);
        }
        for x in rest {
            *x = libm::sqrtf(*x);
        }
        out
    }
}

#[cfg(all(not(feature = "std"), target_arch = "x86_64"))]
#[inline(always)]
fn sqrt4(v: [f32; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;
    let mut out = [0.0; 4];
    // SAFETY: SSE is part of every x86-64 CPU; four readable and writable f32s, unaligned allowed.
    #[allow(unused_unsafe)]
    unsafe {
        _mm_storeu_ps(out.as_mut_ptr(), _mm_sqrt_ps(_mm_loadu_ps(v.as_ptr())))
    };
    out
}

#[cfg(all(not(feature = "std"), target_arch = "aarch64"))]
#[inline(always)]
fn sqrt4(v: [f32; 4]) -> [f32; 4] {
    use core::arch::aarch64::*;
    let mut out = [0.0; 4];
    // SAFETY: NEON is part of every aarch64 CPU; four readable and writable f32s.
    #[allow(unused_unsafe)]
    unsafe {
        vst1q_f32(out.as_mut_ptr(), vsqrtq_f32(vld1q_f32(v.as_ptr())))
    };
    out
}

#[cfg(all(not(feature = "std"), target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn sqrt4(v: [f32; 4]) -> [f32; 4] {
    use core::arch::wasm32::*;
    let r = f32x4_sqrt(f32x4(v[0], v[1], v[2], v[3]));
    [f32x4_extract_lane::<0>(r), f32x4_extract_lane::<1>(r), f32x4_extract_lane::<2>(r), f32x4_extract_lane::<3>(r)]
}

#[cfg(all(not(feature = "std"), not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128")))))]
#[inline(always)]
fn sqrt4(v: [f32; 4]) -> [f32; 4] {
    v.map(libm::sqrtf)
}

macro_rules! lane_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const N: usize> $trait for F32s<N> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, o: Self) -> Self {
                F32s(lanes(|i| self.0[i] $op o.0[i]))
            }
        }
        impl<const N: usize> $trait<f32> for F32s<N> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, o: f32) -> Self {
                F32s(lanes(|i| self.0[i] $op o))
            }
        }
    };
}
lane_op!(Add, add, +);
lane_op!(Sub, sub, -);
lane_op!(Mul, mul, *);
lane_op!(Div, div, /);

impl<const N: usize> Neg for F32s<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_arithmetic_is_element_wise() {
        let a = F32x8::ramp(1.0, 1.0);
        let b = F32x8::splat(2.0);
        assert_eq!((a + b).0, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!((a * b - a / b).0[3], 4.0 * 2.0 - 4.0 / 2.0);
        assert_eq!((-a).abs(), a);
        assert_eq!(F32x4::ramp(-1.0, 1.0).clamp(0.0, 1.5).0, [0.0, 0.0, 1.0, 1.5]);
        assert_eq!(F32x4::ramp(-0.5, 0.5).saturate().0, [0.0, 0.0, 0.5, 1.0]);
        assert_eq!(a.min(b).0[..3], [1.0, 2.0, 2.0]);
        assert_eq!(a.max(b).0[..3], [2.0, 2.0, 3.0]);
    }

    #[test]
    fn square_roots_and_lengths_are_exact_where_they_can_be() {
        // sqrt is correctly rounded on every path (the instruction, or libm's)
        let x = F32x16::ramp(0.0, 1.0);
        let r = x.sqrt();
        for (i, v) in r.0.iter().enumerate() {
            assert_eq!(*v, libm::sqrtf(i as f32));
        }
        let l = F32x4::length2(F32x4::splat(3.0), F32x4::splat(4.0));
        assert_eq!(l.0, [5.0; 4]);
    }

    #[test]
    fn loads_stores_and_bytes() {
        let s = [1.0, 2.0, 3.0];
        let v = F32x4::load_or(&s, -1.0);
        assert_eq!(v.0, [1.0, 2.0, 3.0, -1.0]);
        let mut out = [0.0; 2];
        v.store(&mut out);
        assert_eq!(out, [1.0, 2.0]);
        // half up, saturating, NaN to 0
        assert_eq!(F32s([0.4, 0.5, 1.5, 254.5, 255.4, 300.0, -3.0, f32::NAN]).to_u8(), [0, 1, 2, 255, 255, 255, 0, 0]);
    }
}
