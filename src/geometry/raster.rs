//! A software rasterizer's per-pixel work, sixteen pixels at a time: the signed distance to a
//! rounded rectangle, antialiased coverage from it, and premultiplied blending into RGBA8 pixels,
//! over the lanes of [`crate::simd::lanes`] with the instruction set chosen at run time.
//!
//! tend: Geometry and lanes / simd lanes

use super::vector::Vec2;
use crate::simd::lanes::F32s;
use crate::simd::Level;

/// Pixels per lane group.
const W: usize = 16;

/// A rectangle with rounded corners, in pixel coordinates (`y` down or up, as long as it's the same
/// as the pixels').
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundedRect {
    /// One corner.
    pub min: Vec2<f32>,
    /// The opposite corner.
    pub max: Vec2<f32>,
    /// The corners' radius (at most half the shorter side; larger is clamped).
    pub radius: f32,
}

impl RoundedRect {
    /// The centre, the half extents and the clamped radius.
    #[inline(always)]
    fn shape(&self) -> (Vec2<f32>, Vec2<f32>, f32) {
        let centre = (self.min + self.max) * 0.5;
        let half = ((self.max - self.min) * 0.5).abs();
        let r = self.radius.max(0.0).min(half.x.min(half.y));
        (centre, half, r)
    }

    /// The signed distance from `(x, y)` to the edge: negative inside (Quílez's rounded box).
    pub fn distance(&self, x: f32, y: f32) -> f32 {
        let (c, h, r) = self.shape();
        sdf((x - c.x).abs(), (y - c.y).abs(), h, r)
    }

    /// [`distance`](Self::distance) for sixteen pixels in a row at once (the same arithmetic in
    /// the same order, so the same bits).
    #[inline(always)]
    pub fn distance_lanes(&self, x: F32s<W>, y: f32) -> F32s<W> {
        let (c, h, r) = self.shape();
        let qx = (x - c.x).abs() - h.x + r;
        let qy = F32s::splat((y - c.y).abs() - h.y + r);
        let zero = F32s::splat(0.0);
        F32s::length2(qx.max(zero), qy.max(zero)) + qx.max(qy).min(zero) - r
    }
}

/// The rounded box's signed distance from the point's offsets to the centre (`ax`, `ay`, both
/// non-negative), the half extents and the radius.
#[inline(always)]
fn sdf(ax: f32, ay: f32, h: Vec2<f32>, r: f32) -> f32 {
    let (qx, qy) = (ax - h.x + r, ay - h.y + r);
    len2(qx.max(0.0), qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

/// `sqrt(x² + y²)` as the lanes compute it (no fused multiply-add, so the same bits).
#[inline(always)]
fn len2(x: f32, y: f32) -> f32 {
    let s = x * x + y * y;
    #[cfg(feature = "std")]
    {
        s.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(s)
    }
}

/// Antialiased coverage from a signed distance: `saturate(0.5 - d / width)`, a ramp `width` pixels
/// wide centred on the edge (1 for a one-pixel ramp).
#[inline(always)]
pub fn coverage(distance: f32, width: f32) -> f32 {
    (0.5 - distance / width).clamp(0.0, 1.0)
}

/// Fills a row of RGBA8 pixels with a rounded rectangle in `color` (premultiplied, each channel
/// 0..=1), antialiased over `aa_width` pixels, blended over what's there (premultiplied source
/// over). Pixel `i`'s centre is at `(x + i, y)`. Picks the widest instruction set the CPU runs.
pub fn fill_rounded_rect(dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    fill_rounded_rect_at(Level::detect(), dst, x, y, rect, color, aa_width);
}

/// [`fill_rounded_rect`] compiled for `level` (for comparing them). Every level computes the same
/// bytes. Panics if this CPU can't run `level`.
pub fn fill_rounded_rect_at(level: Level, dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    match level {
        Level::Baseline => fill_row(dst, x, y, rect, color, aa_width),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => {
            assert!(crate::simd::avx2_fma_available(), "this CPU has no AVX2 and FMA");
            // SAFETY: the CPU was just checked for both features `fill_avx2` is compiled with.
            unsafe { fill_avx2(dst, x, y, rect, color, aa_width) }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => {
            assert!(crate::simd::avx512_available(), "this CPU has no AVX-512");
            // SAFETY: the CPU was just checked for AVX-512F, the feature `fill_avx512` is compiled with.
            unsafe { fill_avx512(dst, x, y, rect, color, aa_width) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        other => panic!("{other:?} is x86-64 only"),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fill_avx2(dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    fill_row(dst, x, y, rect, color, aa_width)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fill_avx512(dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    fill_row(dst, x, y, rect, color, aa_width)
}

/// The kernel, inlined into each instruction set's copy: one pixel per iteration in the shape
/// LLVM's loop vectorizer takes, which then runs as many pixels at once as the copy's registers
/// hold. Written for it: the pixel index converted from 32 bits (64-bit integers have no vector
/// conversion to float before AVX-512DQ), each pixel read and written as one word with channels
/// taken apart by shifts, and bytes rounded by clamping first and truncating after (a saturating
/// float-to-byte cast doesn't vectorize). The same arithmetic as the plain loop, so the same bytes.
#[inline(always)]
fn fill_row(dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    let (c, h, r) = rect.shape();
    let ay = (y - c.y).abs();
    let inv_width = 1.0 / aa_width;
    let scaled = [color[0] * 255.0, color[1] * 255.0, color[2] * 255.0, color[3] * 255.0];
    for (i, p) in dst.iter_mut().enumerate() {
        let px = x + (i as u32 as i32) as f32;
        let d = sdf((px - c.x).abs(), ay, h, r);
        let cov = (0.5 - d * inv_width).clamp(0.0, 1.0);
        let keep = 1.0 - cov * color[3];
        let word = u32::from_le_bytes(*p);
        let mut out = 0u32;
        for (ch, &s) in scaled.iter().enumerate() {
            let shift = 8 * ch as u32;
            let under = ((word >> shift) & 0xff) as i32 as f32;
            // NaN to 0 (max returns the number), then within a byte: truncating needs no checks.
            // Not clamp: clamp passes NaN through, which to_int_unchecked must never see.
            #[allow(clippy::manual_clamp)]
            let v = (cov * s + under * keep + 0.5).max(0.0).min(255.0);
            // SAFETY: v is finite and within 0..=255
            out |= (unsafe { v.to_int_unchecked::<i32>() } as u32) << shift;
        }
        *p = out.to_le_bytes();
    }
}

/// [`fill_rounded_rect`] one pixel at a time, as a plain loop writes it: the reference the lanes
/// are tested against (byte for byte) and measured against.
pub fn fill_rounded_rect_scalar(dst: &mut [[u8; 4]], x: f32, y: f32, rect: &RoundedRect, color: [f32; 4], aa_width: f32) {
    let inv_width = 1.0 / aa_width;
    for (i, p) in dst.iter_mut().enumerate() {
        let d = rect.distance(x + i as f32, y);
        let cov = (0.5 - d * inv_width).clamp(0.0, 1.0);
        let keep = 1.0 - cov * color[3];
        for c in 0..4 {
            p[c] = (cov * (color[c] * 255.0) + p[c] as f32 * keep + 0.5) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> RoundedRect {
        RoundedRect { min: Vec2::new(10.3, 4.0), max: Vec2::new(70.6, 40.2), radius: 9.5 }
    }

    #[test]
    fn distances_are_the_rounded_box() {
        let r = rect();
        // the centre is the half height inside; a point straight off a side is its gap
        assert!((r.distance(40.45, 22.1) + 18.1).abs() < 1e-4);
        assert!((r.distance(80.6, 22.1) - 10.0).abs() < 1e-4);
        // off a corner: the distance to the corner circle
        let (cx, cy) = (70.6 - 9.5, 40.2 - 9.5);
        assert!((r.distance(cx + 6.0, cy + 8.0) - (10.0 - 9.5)).abs() < 1e-4);
        // the lanes give the same bits as one at a time
        let xs = F32s::ramp(0.5, 5.25);
        let d = r.distance_lanes(xs, 39.5);
        for i in 0..W {
            assert_eq!(d.0[i], r.distance(xs.0[i], 39.5), "lane {i}");
        }
        assert_eq!(coverage(-3.0, 1.0), 1.0);
        assert_eq!(coverage(0.0, 1.0), 0.5);
        assert_eq!(coverage(0.25, 2.0), 0.375);
    }

    /// A background with every byte different, so blending errors show.
    fn background(n: usize) -> Vec<[u8; 4]> {
        (0..n).map(|i| [(i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8, 255 - (i % 200) as u8]).collect()
    }

    #[test]
    fn every_level_matches_the_plain_loop_byte_for_byte() {
        let r = rect();
        let color = [0.2, 0.5, 0.1, 0.6]; // premultiplied
        let mut levels = vec![Level::Baseline];
        if crate::simd::avx2_fma_available() {
            levels.push(Level::Avx2);
        }
        if crate::simd::avx512_available() {
            levels.push(Level::Avx512);
        }
        // spans of every remainder, across edges and corners, rows inside and outside
        for (len, x, y) in [(83, 0.5, 20.5), (83, 0.5, 4.5), (17, 60.5, 38.7), (5, 68.5, 40.5), (100, -3.5, 50.5), (1, 40.5, 22.5)] {
            let mut want = background(len);
            fill_rounded_rect_scalar(&mut want, x, y, &r, color, 1.0);
            for &level in &levels {
                let mut got = background(len);
                fill_rounded_rect_at(level, &mut got, x, y, &r, color, 1.0);
                assert_eq!(got, want, "{level:?}, span of {len} at ({x}, {y})");
            }
        }
    }

    #[test]
    fn coverage_shapes_the_fill() {
        // opaque white over black: inside 255, outside untouched, the edge pixel half way
        let r = RoundedRect { min: Vec2::new(2.0, 0.0), max: Vec2::new(6.0, 10.0), radius: 0.0 };
        let mut row = vec![[0, 0, 0, 255]; 8];
        fill_rounded_rect(&mut row, 0.5, 5.0, &r, [1.0; 4], 1.0);
        let red: Vec<u8> = row.iter().map(|p| p[0]).collect();
        assert_eq!(red, [0, 0, 255, 255, 255, 255, 0, 0]);
        // the edge at x = 2 through a pixel centred on it: half covered
        let mut row = vec![[0, 0, 0, 255]; 1];
        fill_rounded_rect(&mut row, 2.0, 5.0, &r, [1.0; 4], 1.0);
        assert_eq!(row[0][0], 128);
    }
}
