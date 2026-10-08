//! Four `f32` lanes: the single-value matrix kernels (4x4 products, matrix-vector products and the
//! 4x4 inverse), written once over [`Quad`], a handful of lane operations each instruction set
//! provides: SSE2 on x86-64 (part of the baseline), NEON on aarch64, simd128 on wasm32 built with
//! `+simd128`. [`Native`] is the target's, when it has one; [`Portable`] is plain arrays, so the
//! kernels' arithmetic is tested on every host.
//!
//! The kernels take and return columns, matching the column-major matrices: each column is one
//! register.

/// A shuffle pattern in SSE's encoding: two bits per output lane, lane 0 lowest. `imm(a, b, c, d)`
/// takes lanes `a, b` from the first operand and `c, d` from the second (from the only operand for
/// [`Quad::shuf`]).
pub(crate) const fn imm(a: i32, b: i32, c: i32, d: i32) -> i32 {
    a | (b << 2) | (c << 4) | (d << 6)
}

/// Output lane `k`'s source lane in a pattern made by [`imm`].
#[allow(dead_code)] // the non-x86 backends and Portable decode patterns; SSE takes them as they are
const fn lane(pattern: i32, k: u32) -> usize {
    ((pattern >> (2 * k)) & 3) as usize
}

/// Four `f32` lanes and the operations the kernels need.
pub(crate) trait Quad: Copy {
    /// Lanes from an array.
    fn load(a: &[f32; 4]) -> Self;
    /// The lanes as an array.
    fn store(self) -> [f32; 4];
    /// The lanes written into `out`.
    #[inline(always)]
    fn store_to(self, out: &mut [f32; 4]) {
        *out = self.store();
    }
    /// Every lane `x`.
    fn splat(x: f32) -> Self;
    /// Lane-wise sum.
    fn add(self, o: Self) -> Self;
    /// Lane-wise difference.
    fn sub(self, o: Self) -> Self;
    /// Lane-wise product.
    fn mul(self, o: Self) -> Self;
    /// Lane-wise quotient.
    fn div(self, o: Self) -> Self;
    /// Lanes of `self` rearranged by `P` (see [`imm`]).
    fn shuf<const P: i32>(self) -> Self;
    /// Lanes 0 and 1 from `self`, 2 and 3 from `o`, chosen by `P` (SSE's `shufps`).
    fn shuf2<const P: i32>(self, o: Self) -> Self;
}

/// `m * v`, `m` given by its columns, summed column after column (as glam does).
#[inline(always)]
pub(crate) fn mul_vec4<Q: Quad>(m: &[Q; 4], v: Q) -> Q {
    let [x, y, z, w] = broadcast(v);
    m[0].mul(x).add(m[1].mul(y)).add(m[2].mul(z)).add(m[3].mul(w))
}

/// `a * b`, both given by their columns. Each column's sum goes in pairs: four independent columns
/// keep the pipeline busy either way, and pairs shorten each chain (measured: 0.92x glam summed in
/// sequence, 1.01x in pairs; for a single `m * v` the sequence measured faster).
#[inline(always)]
pub(crate) fn mul_mat4<Q: Quad>(a: &[Q; 4], b: &[Q; 4]) -> [Q; 4] {
    [mul_col(a, b[0]), mul_col(a, b[1]), mul_col(a, b[2]), mul_col(a, b[3])]
}

/// One column of [`mul_mat4`].
#[inline(always)]
fn mul_col<Q: Quad>(a: &[Q; 4], c: Q) -> Q {
    let [x, y, z, w] = broadcast(c);
    a[0].mul(x).add(a[1].mul(y)).add(a[2].mul(z).add(a[3].mul(w)))
}

/// Each lane of `v` in every lane.
#[inline(always)]
fn broadcast<Q: Quad>(v: Q) -> [Q; 4] {
    [v.shuf::<{ imm(0, 0, 0, 0) }>(), v.shuf::<{ imm(1, 1, 1, 1) }>(), v.shuf::<{ imm(2, 2, 2, 2) }>(), v.shuf::<{ imm(3, 3, 3, 3) }>()]
}

// 2x2 blocks below are packed row-major in one register: lanes (x00, x01, x10, x11).

/// `x * y` for 2x2 blocks.
#[inline(always)]
fn mul2<Q: Quad>(x: Q, y: Q) -> Q {
    let l = x.shuf::<{ imm(0, 0, 2, 2) }>().mul(y.shuf::<{ imm(0, 1, 0, 1) }>());
    l.add(x.shuf::<{ imm(1, 1, 3, 3) }>().mul(y.shuf::<{ imm(2, 3, 2, 3) }>()))
}

/// `adj(x) * y` for 2x2 blocks (`adj` the adjugate: `x * adj(x) = det(x) I`).
#[inline(always)]
fn adj_mul<Q: Quad>(x: Q, y: Q) -> Q {
    let l = x.shuf::<{ imm(3, 3, 0, 0) }>().mul(y);
    l.sub(x.shuf::<{ imm(1, 1, 2, 2) }>().mul(y.shuf::<{ imm(2, 3, 0, 1) }>()))
}

/// `x * adj(y)` for 2x2 blocks.
#[inline(always)]
fn mul_adj<Q: Quad>(x: Q, y: Q) -> Q {
    let l = x.mul(y.shuf::<{ imm(3, 0, 3, 0) }>());
    l.sub(x.shuf::<{ imm(1, 0, 3, 2) }>().mul(y.shuf::<{ imm(2, 1, 2, 1) }>()))
}

/// The sum of the lanes, in every lane.
#[inline(always)]
fn sum<Q: Quad>(x: Q) -> Q {
    let x = x.add(x.shuf::<{ imm(2, 3, 0, 1) }>());
    x.add(x.shuf::<{ imm(1, 0, 3, 2) }>())
}

/// The inverse of a 4x4 matrix given by its columns (non-finite lanes when it's singular).
///
/// Block-wise: with `M = [A B; C D]` in 2x2 blocks and `#` the adjugate,
/// `det M = |A||D| + |B||C| - tr(A#B D#C)`, and the blocks of `M⁻¹` are
/// `adj(|D|A - B D#C)`, `adj(|B|C - D adj(A#B))`, `adj(|C|B - A adj(D#C))` and
/// `adj(|A|D - C A#B)` (top-left, top-right, bottom-left, bottom-right), each over `det M`.
/// These are polynomial identities (they need no block to be invertible), and the products
/// `A#B` and `D#C` are shared, so the whole inverse is about 60 lane operations against about
/// 150 scalar ones by cofactors. Columns go in where the derivation has rows: the inverse of the
/// transpose is the transpose of the inverse, so columns come out.
#[inline(always)]
pub(crate) fn inverse<Q: Quad>(r: &[Q; 4]) -> [Q; 4] {
    let a = r[0].shuf2::<{ imm(0, 1, 0, 1) }>(r[1]);
    let b = r[0].shuf2::<{ imm(2, 3, 2, 3) }>(r[1]);
    let c = r[2].shuf2::<{ imm(0, 1, 0, 1) }>(r[3]);
    let d = r[2].shuf2::<{ imm(2, 3, 2, 3) }>(r[3]);
    // |A|, |B|, |C|, |D| in lanes 0..3
    let det = r[0].shuf2::<{ imm(0, 2, 0, 2) }>(r[2]).mul(r[1].shuf2::<{ imm(1, 3, 1, 3) }>(r[3]));
    let det = det.sub(r[0].shuf2::<{ imm(1, 3, 1, 3) }>(r[2]).mul(r[1].shuf2::<{ imm(0, 2, 0, 2) }>(r[3])));
    let det_a = det.shuf::<{ imm(0, 0, 0, 0) }>();
    let det_b = det.shuf::<{ imm(1, 1, 1, 1) }>();
    let det_c = det.shuf::<{ imm(2, 2, 2, 2) }>();
    let det_d = det.shuf::<{ imm(3, 3, 3, 3) }>();

    let a_b = adj_mul(a, b);
    let d_c = adj_mul(d, c);
    let x = det_d.mul(a).sub(mul2(b, d_c));
    let w = det_a.mul(d).sub(mul2(c, a_b));
    let y = det_b.mul(c).sub(mul_adj(d, a_b));
    let z = det_c.mul(b).sub(mul_adj(a, d_c));

    // |A||D| + |B||C| in every lane: (AD, BC, CB, DA) plus itself swapped in pairs
    let ad_bc = det.mul(det.shuf::<{ imm(3, 2, 1, 0) }>());
    let ad_bc = ad_bc.add(ad_bc.shuf::<{ imm(1, 0, 3, 2) }>());
    // tr(P Q) for 2x2 blocks is the lane sum of P times Q's transpose
    let det_m = ad_bc.sub(sum(a_b.mul(d_c.shuf::<{ imm(0, 2, 1, 3) }>())));
    let inv = Q::splat(1.0).div(det_m);
    let pn = Q::load(&[1.0, -1.0, 1.0, -1.0]).mul(inv);
    let np = Q::load(&[-1.0, 1.0, -1.0, 1.0]).mul(inv);
    // adj(x) is (x11, -x01, -x10, x00): one shuffle per output row takes two blocks' adjugates
    [
        x.shuf2::<{ imm(3, 1, 3, 1) }>(y).mul(pn),
        x.shuf2::<{ imm(2, 0, 2, 0) }>(y).mul(np),
        z.shuf2::<{ imm(3, 1, 3, 1) }>(w).mul(pn),
        z.shuf2::<{ imm(2, 0, 2, 0) }>(w).mul(np),
    ]
}

/// Plain arrays: the kernels' arithmetic on any host (the tests check the native backend against
/// it and both against the scalar cofactors).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)] // used by the tests, and as the reference on targets without a backend
pub(crate) struct Portable(pub(crate) [f32; 4]);

impl Quad for Portable {
    #[inline(always)]
    fn load(a: &[f32; 4]) -> Self {
        Portable(*a)
    }
    #[inline(always)]
    fn store(self) -> [f32; 4] {
        self.0
    }
    #[inline(always)]
    fn splat(x: f32) -> Self {
        Portable([x; 4])
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Portable(core::array::from_fn(|i| self.0[i] + o.0[i]))
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Portable(core::array::from_fn(|i| self.0[i] - o.0[i]))
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Portable(core::array::from_fn(|i| self.0[i] * o.0[i]))
    }
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        Portable(core::array::from_fn(|i| self.0[i] / o.0[i]))
    }
    #[inline(always)]
    fn shuf<const P: i32>(self) -> Self {
        Portable(core::array::from_fn(|k| self.0[lane(P, k as u32)]))
    }
    #[inline(always)]
    fn shuf2<const P: i32>(self, o: Self) -> Self {
        Portable(core::array::from_fn(|k| if k < 2 { self.0[lane(P, k as u32)] } else { o.0[lane(P, k as u32)] }))
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use sse::F32x4 as Native;
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::F32x4 as Native;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::F32x4 as Native;

/// Whether this target has a [`Native`] backend.
pub(crate) const HAS_NATIVE: bool =
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128")));

#[cfg(target_arch = "x86_64")]
mod sse {
    use core::arch::x86_64::*;

    use super::Quad;

    /// An SSE register (SSE2 is part of every x86-64 CPU, so no detection).
    #[derive(Clone, Copy)]
    pub(crate) struct F32x4(__m128);

    // SAFETY (every block below): SSE2 is enabled on every x86-64 target; newer compilers call these
    // safely, hence the allow.
    #[allow(unused_unsafe)]
    impl Quad for F32x4 {
        #[inline(always)]
        fn load(a: &[f32; 4]) -> Self {
            // SAFETY: four readable f32s; unaligned loads are allowed.
            F32x4(unsafe { _mm_loadu_ps(a.as_ptr()) })
        }
        #[inline(always)]
        fn store(self) -> [f32; 4] {
            let mut out = [0.0; 4];
            // SAFETY: four writable f32s; unaligned stores are allowed.
            unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) };
            out
        }
        #[inline(always)]
        fn store_to(self, out: &mut [f32; 4]) {
            // SAFETY: four writable f32s; unaligned stores are allowed.
            unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) };
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            F32x4(unsafe { _mm_set1_ps(x) })
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F32x4(unsafe { _mm_add_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F32x4(unsafe { _mm_sub_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F32x4(unsafe { _mm_mul_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F32x4(unsafe { _mm_div_ps(self.0, o.0) })
        }
        #[inline(always)]
        fn shuf<const P: i32>(self) -> Self {
            F32x4(unsafe { _mm_shuffle_ps::<P>(self.0, self.0) })
        }
        #[inline(always)]
        fn shuf2<const P: i32>(self, o: Self) -> Self {
            F32x4(unsafe { _mm_shuffle_ps::<P>(self.0, o.0) })
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{lane, Quad};

    /// A NEON register (NEON is part of every aarch64 CPU).
    #[derive(Clone, Copy)]
    pub(crate) struct F32x4(float32x4_t);

    /// The byte indices `tbl` takes for pattern `p`; lanes 2 and 3 index the second register
    /// (`second` 16) in a two-register table.
    const fn bytes(p: i32, second: u8) -> [u8; 16] {
        let mut out = [0u8; 16];
        let mut k = 0;
        while k < 4 {
            let base = (lane(p, k as u32) as u8) * 4 + if k >= 2 { second } else { 0 };
            let mut j = 0;
            while j < 4 {
                out[k * 4 + j] = base + j as u8;
                j += 1;
            }
            k += 1;
        }
        out
    }

    #[allow(unused_unsafe)] // NEON intrinsics are safe to call where NEON is enabled, as on every aarch64 target
    impl Quad for F32x4 {
        #[inline(always)]
        fn load(a: &[f32; 4]) -> Self {
            // SAFETY: four readable f32s.
            F32x4(unsafe { vld1q_f32(a.as_ptr()) })
        }
        #[inline(always)]
        fn store(self) -> [f32; 4] {
            let mut out = [0.0; 4];
            // SAFETY: four writable f32s.
            unsafe { vst1q_f32(out.as_mut_ptr(), self.0) };
            out
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            F32x4(unsafe { vdupq_n_f32(x) })
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F32x4(unsafe { vaddq_f32(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F32x4(unsafe { vsubq_f32(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F32x4(unsafe { vmulq_f32(self.0, o.0) })
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F32x4(unsafe { vdivq_f32(self.0, o.0) })
        }
        #[inline(always)]
        fn shuf<const P: i32>(self) -> Self {
            let idx = const { bytes(P, 0) };
            // SAFETY: sixteen readable bytes.
            unsafe { F32x4(vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(self.0), vld1q_u8(idx.as_ptr())))) }
        }
        #[inline(always)]
        fn shuf2<const P: i32>(self, o: Self) -> Self {
            let idx = const { bytes(P, 16) };
            // SAFETY: sixteen readable bytes; indices 16..32 select from the second register.
            unsafe {
                let t = uint8x16x2_t(vreinterpretq_u8_f32(self.0), vreinterpretq_u8_f32(o.0));
                F32x4(vreinterpretq_f32_u8(vqtbl2q_u8(t, vld1q_u8(idx.as_ptr()))))
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::*;

    use super::{lane, Quad};

    /// A simd128 register (on wasm32 built with `-C target-feature=+simd128`).
    #[derive(Clone, Copy)]
    pub(crate) struct F32x4(v128);

    /// The byte indices `i8x16_swizzle` takes for pattern `p`, lanes `from..to` only (indices of
    /// 16 and above give zero bytes, which leaves the other lanes for a second swizzle).
    const fn bytes(p: i32, from: usize, to: usize) -> v128 {
        let mut out = [0x80u8; 16];
        let mut k = from;
        while k < to {
            let base = (lane(p, k as u32) as u8) * 4;
            let mut j = 0;
            while j < 4 {
                out[k * 4 + j] = base + j as u8;
                j += 1;
            }
            k += 1;
        }
        u8x16(
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12],
            out[13], out[14], out[15],
        )
    }

    impl Quad for F32x4 {
        #[inline(always)]
        fn load(a: &[f32; 4]) -> Self {
            F32x4(f32x4(a[0], a[1], a[2], a[3]))
        }
        #[inline(always)]
        fn store(self) -> [f32; 4] {
            [
                f32x4_extract_lane::<0>(self.0),
                f32x4_extract_lane::<1>(self.0),
                f32x4_extract_lane::<2>(self.0),
                f32x4_extract_lane::<3>(self.0),
            ]
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            F32x4(f32x4_splat(x))
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            F32x4(f32x4_add(self.0, o.0))
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            F32x4(f32x4_sub(self.0, o.0))
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            F32x4(f32x4_mul(self.0, o.0))
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            F32x4(f32x4_div(self.0, o.0))
        }
        #[inline(always)]
        fn shuf<const P: i32>(self) -> Self {
            F32x4(i8x16_swizzle(self.0, const { bytes(P, 0, 4) }))
        }
        #[inline(always)]
        fn shuf2<const P: i32>(self, o: Self) -> Self {
            let lo = i8x16_swizzle(self.0, const { bytes(P, 0, 2) });
            let hi = i8x16_swizzle(o.0, const { bytes(P, 2, 4) });
            F32x4(v128_or(lo, hi))
        }
    }
}
