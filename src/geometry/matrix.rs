//! [`Mat3`] and [`Mat4`], column-major: construction, products, transposes, determinants and
//! inverses. Single `f32` values take the four-lane kernels in [`quad`](super::quad) where the
//! target has them; everything else is plain arithmetic.

use core::any::TypeId;
use core::ops::Mul;

use super::quad::{self, Quad};
use super::vector::{Vec3, Vec4};
use crate::units::*;

/// `x` as an `&B` when `A` and `B` are the same type (decided at compile time, so each generic
/// instance keeps one branch).
#[inline(always)]
fn same<A: 'static, B: 'static>(x: &A) -> Option<&B> {
    if TypeId::of::<A>() == TypeId::of::<B>() {
        // SAFETY: A and B are the same type.
        Some(unsafe { &*(x as *const A as *const B) })
    } else {
        None
    }
}

/// A 3x3 matrix, column-major.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3<T> {
    /// The columns.
    pub cols: [Vec3<T>; 3],
}

impl<T: Float> Mat3<T> {
    /// The identity.
    pub fn identity() -> Self {
        let (o, z) = (T::_ONE, T::_ZERO);
        Mat3::from_cols(Vec3::new(o, z, z), Vec3::new(z, o, z), Vec3::new(z, z, o))
    }

    /// A matrix from its columns.
    #[inline(always)]
    pub const fn from_cols(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Mat3 { cols: [c0, c1, c2] }
    }

    /// Column `i`. Panics if `i > 2`.
    #[inline(always)]
    pub fn col(&self, i: usize) -> Vec3<T> {
        self.cols[i]
    }

    /// The components, column by column.
    #[inline(always)]
    pub fn to_cols_array(self) -> [T; 9] {
        let [a, b, c] = self.cols;
        [a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z]
    }

    /// The transpose.
    #[inline(always)]
    pub fn transpose(self) -> Self {
        let [a, b, c] = self.cols;
        Mat3::from_cols(Vec3::new(a.x, b.x, c.x), Vec3::new(a.y, b.y, c.y), Vec3::new(a.z, b.z, c.z))
    }

    /// The determinant.
    #[inline(always)]
    pub fn determinant(self) -> T {
        let [a, b, c] = self.cols;
        a.dot(b.cross(c))
    }

    /// The inverse, by cross products (non-finite components when the matrix is singular, as
    /// glam's).
    #[inline]
    pub fn inverse(self) -> Self {
        let [a, b, c] = self.cols;
        let (r0, r1, r2) = (b.cross(c), c.cross(a), a.cross(b));
        let inv_det = T::_ONE / a.dot(r0);
        Mat3::from_cols(r0, r1, r2).transpose() * inv_det
    }

    /// Whether every component is finite.
    pub fn is_finite(self) -> bool {
        self.cols.iter().all(|c| c.is_finite())
    }
}

impl<T: Float> Default for Mat3<T> {
    /// The identity.
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Float> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;
    #[inline(always)]
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
}

impl<T: Float> Mul for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Mat3::from_cols(self * o.cols[0], self * o.cols[1], self * o.cols[2])
    }
}

impl<T: Float> Mul<T> for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: T) -> Self {
        Mat3::from_cols(self.cols[0] * k, self.cols[1] * k, self.cols[2] * k)
    }
}

/// A 4x4 matrix, column-major.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4<T> {
    /// The columns.
    pub cols: [Vec4<T>; 4],
}

impl<T: Float> Mat4<T> {
    /// The identity.
    pub fn identity() -> Self {
        let (o, z) = (T::_ONE, T::_ZERO);
        Mat4::from_cols(Vec4::new(o, z, z, z), Vec4::new(z, o, z, z), Vec4::new(z, z, o, z), Vec4::new(z, z, z, o))
    }

    /// A matrix from its columns.
    #[inline(always)]
    pub const fn from_cols(c0: Vec4<T>, c1: Vec4<T>, c2: Vec4<T>, c3: Vec4<T>) -> Self {
        Mat4 { cols: [c0, c1, c2, c3] }
    }

    /// Column `i`. Panics if `i > 3`.
    #[inline(always)]
    pub fn col(&self, i: usize) -> Vec4<T> {
        self.cols[i]
    }

    /// A translation.
    pub fn from_translation(t: Vec3<T>) -> Self {
        let mut m = Self::identity();
        m.cols[3] = t.extend(T::_ONE);
        m
    }

    /// A scale along each axis.
    pub fn from_scale(s: Vec3<T>) -> Self {
        let z = T::_ZERO;
        Mat4::from_cols(Vec4::new(s.x, z, z, z), Vec4::new(z, s.y, z, z), Vec4::new(z, z, s.z, z), Vec4::new(z, z, z, T::_ONE))
    }

    /// A rotation about x by `angle` radians (counter-clockwise looking down the axis).
    pub fn from_rotation_x(angle: T) -> Self {
        let (s, co) = angle._sin_cos();
        let (o, z) = (T::_ONE, T::_ZERO);
        Mat4::from_cols(Vec4::new(o, z, z, z), Vec4::new(z, co, s, z), Vec4::new(z, -s, co, z), Vec4::new(z, z, z, o))
    }

    /// A rotation about y by `angle` radians.
    pub fn from_rotation_y(angle: T) -> Self {
        let (s, co) = angle._sin_cos();
        let (o, z) = (T::_ONE, T::_ZERO);
        Mat4::from_cols(Vec4::new(co, z, -s, z), Vec4::new(z, o, z, z), Vec4::new(s, z, co, z), Vec4::new(z, z, z, o))
    }

    /// A rotation about z by `angle` radians.
    pub fn from_rotation_z(angle: T) -> Self {
        let (s, co) = angle._sin_cos();
        let (o, z) = (T::_ONE, T::_ZERO);
        Mat4::from_cols(Vec4::new(co, s, z, z), Vec4::new(-s, co, z, z), Vec4::new(z, z, o, z), Vec4::new(z, z, z, o))
    }

    /// The components, column by column.
    #[inline(always)]
    pub fn to_cols_array(self) -> [T; 16] {
        let [a, b, c, d] = self.cols;
        [a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w]
    }

    /// The transpose.
    #[inline(always)]
    pub fn transpose(self) -> Self {
        let [a, b, c, d] = self.cols;
        Mat4::from_cols(
            Vec4::new(a.x, b.x, c.x, d.x),
            Vec4::new(a.y, b.y, c.y, d.y),
            Vec4::new(a.z, b.z, c.z, d.z),
            Vec4::new(a.w, b.w, c.w, d.w),
        )
    }

    /// The determinant.
    pub fn determinant(self) -> T {
        let (s, c) = self.minors();
        s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]
    }

    /// The 2x2 minors of the top two and bottom two rows, over each pair of columns
    /// (01, 02, 03, 12, 13, 23).
    #[inline(always)]
    fn minors(self) -> ([T; 6], [T; 6]) {
        let [a, b, c, d] = self.cols;
        let top = [
            a.x * b.y - b.x * a.y,
            a.x * c.y - c.x * a.y,
            a.x * d.y - d.x * a.y,
            b.x * c.y - c.x * b.y,
            b.x * d.y - d.x * b.y,
            c.x * d.y - d.x * c.y,
        ];
        let bottom = [
            a.z * b.w - b.z * a.w,
            a.z * c.w - c.z * a.w,
            a.z * d.w - d.z * a.w,
            b.z * c.w - c.z * b.w,
            b.z * d.w - d.z * b.w,
            c.z * d.w - d.z * c.w,
        ];
        (top, bottom)
    }

    /// The inverse (non-finite components when the matrix is singular, as glam's). `f32` takes the
    /// block-wise SIMD kernel where the target has one; otherwise cofactors.
    #[inline]
    pub fn inverse(self) -> Self {
        if quad::HAS_NATIVE {
            if let Some(m) = same::<Self, Mat4<f32>>(&self) {
                let r = m.via_quad(quad::inverse::<NativeQuad>);
                return *same::<Mat4<f32>, Self>(&r).expect("T is f32");
            }
        }
        self.inverse_cofactors()
    }

    /// The inverse by cofactors, in plain arithmetic.
    fn inverse_cofactors(self) -> Self {
        let [a, b, c, d] = self.cols;
        let (s, k) = self.minors();
        let det = s[0] * k[5] - s[1] * k[4] + s[2] * k[3] + s[3] * k[2] - s[4] * k[1] + s[5] * k[0];
        let inv = T::_ONE / det;
        Mat4::from_cols(
            Vec4::new(
                (b.y * k[5] - c.y * k[4] + d.y * k[3]) * inv,
                (-a.y * k[5] + c.y * k[2] - d.y * k[1]) * inv,
                (a.y * k[4] - b.y * k[2] + d.y * k[0]) * inv,
                (-a.y * k[3] + b.y * k[1] - c.y * k[0]) * inv,
            ),
            Vec4::new(
                (-b.x * k[5] + c.x * k[4] - d.x * k[3]) * inv,
                (a.x * k[5] - c.x * k[2] + d.x * k[1]) * inv,
                (-a.x * k[4] + b.x * k[2] - d.x * k[0]) * inv,
                (a.x * k[3] - b.x * k[1] + c.x * k[0]) * inv,
            ),
            Vec4::new(
                (b.w * s[5] - c.w * s[4] + d.w * s[3]) * inv,
                (-a.w * s[5] + c.w * s[2] - d.w * s[1]) * inv,
                (a.w * s[4] - b.w * s[2] + d.w * s[0]) * inv,
                (-a.w * s[3] + b.w * s[1] - c.w * s[0]) * inv,
            ),
            Vec4::new(
                (-b.z * s[5] + c.z * s[4] - d.z * s[3]) * inv,
                (a.z * s[5] - c.z * s[2] + d.z * s[1]) * inv,
                (-a.z * s[4] + b.z * s[2] - d.z * s[0]) * inv,
                (a.z * s[3] - b.z * s[1] + c.z * s[0]) * inv,
            ),
        )
    }

    /// Transforms a point (w = 1), dividing by the resulting w.
    #[inline(always)]
    pub fn project_point3(self, p: Vec3<T>) -> Vec3<T> {
        let r = self * p.extend(T::_ONE);
        r.truncate() / r.w
    }

    /// Whether every component is finite.
    pub fn is_finite(self) -> bool {
        self.cols.iter().all(|c| c.is_finite())
    }

    /// `self * o` in plain arithmetic: `f64`, and `f32` on targets without a four-lane backend.
    #[inline(always)]
    pub(crate) fn mul_plain(self, o: Self) -> Self {
        // summed in pairs, as the SIMD kernel sums (so f32 gets the same bits on every path)
        let col = |v: Vec4<T>| (self.cols[0] * v.x + self.cols[1] * v.y) + (self.cols[2] * v.z + self.cols[3] * v.w);
        Mat4::from_cols(col(o.cols[0]), col(o.cols[1]), col(o.cols[2]), col(o.cols[3]))
    }

    /// `self * v` in plain arithmetic, summed in sequence as the SIMD kernel sums, so `f32` gets the
    /// same bits on every path.
    #[inline(always)]
    fn mul_vec_plain(self, v: Vec4<T>) -> Vec4<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z + self.cols[3] * v.w
    }
}

/// The target's four-lane backend (or the portable one, never called, where it has none).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128")))]
type NativeQuad = quad::Native;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
type NativeQuad = quad::Portable;

/// A `Vec4<f32>` as the array it is.
#[inline(always)]
fn lanes(v: &Vec4<f32>) -> &[f32; 4] {
    // SAFETY: Vec4 is #[repr(C)] with four f32 fields: the layout of [f32; 4].
    unsafe { &*(v as *const Vec4<f32> as *const [f32; 4]) }
}

/// A `Vec4<f32>` as the array it is, to write.
#[inline(always)]
fn lanes_mut(v: &mut Vec4<f32>) -> &mut [f32; 4] {
    // SAFETY: as in `lanes`.
    unsafe { &mut *(v as *mut Vec4<f32> as *mut [f32; 4]) }
}

impl Mat4<f32> {
    /// The columns as lanes.
    #[inline(always)]
    fn quads<Q: Quad>(&self) -> [Q; 4] {
        [Q::load(lanes(&self.cols[0])), Q::load(lanes(&self.cols[1])), Q::load(lanes(&self.cols[2])), Q::load(lanes(&self.cols[3]))]
    }

    /// The columns as lanes, through `f`, and back (each stored straight into the result).
    #[inline(always)]
    fn via_quad<Q: Quad>(&self, f: impl FnOnce(&[Q; 4]) -> [Q; 4]) -> Self {
        let r = f(&self.quads());
        let mut out = Mat4 { cols: [Vec4::zero(); 4] };
        for (q, c) in r.into_iter().zip(&mut out.cols) {
            q.store_to(lanes_mut(c));
        }
        out
    }
}

impl<T: Float> Default for Mat4<T> {
    /// The identity.
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Float> Mul<Vec4<T>> for Mat4<T> {
    type Output = Vec4<T>;
    #[inline(always)]
    fn mul(self, v: Vec4<T>) -> Vec4<T> {
        if quad::HAS_NATIVE {
            if let (Some(m), Some(v)) = (same::<Self, Mat4<f32>>(&self), same::<Vec4<T>, Vec4<f32>>(&v)) {
                let mut r = Vec4::zero();
                quad::mul_vec4(&m.quads::<NativeQuad>(), NativeQuad::load(lanes(v))).store_to(lanes_mut(&mut r));
                return *same::<Vec4<f32>, Vec4<T>>(&r).expect("T is f32");
            }
        }
        self.mul_vec_plain(v)
    }
}

impl<T: Float> Mul for Mat4<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        if quad::HAS_NATIVE {
            if let (Some(a), Some(b)) = (same::<Self, Mat4<f32>>(&self), same::<Self, Mat4<f32>>(&o)) {
                let bc = b.quads::<NativeQuad>();
                let r = a.via_quad(|ac: &[NativeQuad; 4]| quad::mul_mat4(ac, &bc));
                return *same::<Mat4<f32>, Self>(&r).expect("T is f32");
            }
        }
        self.mul_plain(o)
    }
}

// SAFETY: `#[repr(C)]` arrays of Pod vectors: no padding, every bit pattern valid.
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for Mat3<f32> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for Mat3<f32> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for Mat4<f32> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for Mat4<f32> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for Mat3<f64> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for Mat3<f64> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for Mat4<f64> {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for Mat4<f64> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::quad::Portable;

    /// Matrices with known exact inverses: integer, determinant ±1 (unimodular), built from
    /// elementary row operations, plus scaled rotations and translations.
    fn unimodular() -> Vec<([f64; 16], [f64; 16])> {
        // column-major M and M⁻¹: M is unit upper triangular, rows (1 2 3 5), (0 1 4 6), (0 0 1 7),
        // (0 0 0 1); back substitution gives rows (1 -2 5 -28), (0 1 -4 22), (0 0 1 -7), (0 0 0 1)
        vec![
            (
                [1., 0., 0., 0., 2., 1., 0., 0., 3., 4., 1., 0., 5., 6., 7., 1.],
                [1., 0., 0., 0., -2., 1., 0., 0., 5., -4., 1., 0., -28., 22., -7., 1.],
            ),
            // a permutation swapping 0 with 1 and 2 with 3: its own inverse
            (
                [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
                [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
            ),
        ]
    }

    fn from_cols_array<T: Float>(a: &[f64; 16]) -> Mat4<T> {
        let v = |i: usize| Vec4::new(T::lit(a[i]), T::lit(a[i + 1]), T::lit(a[i + 2]), T::lit(a[i + 3]));
        Mat4::from_cols(v(0), v(4), v(8), v(12))
    }

    /// Well-conditioned dense matrices: rotations, scales and shears about random axes, from a fixed
    /// seed (no dependence on another library).
    fn dense(n: usize) -> Vec<Mat4<f64>> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        (0..n)
            .map(|_| {
                let m = Mat4::from_translation(Vec3::new(r() * 50.0, r() * 50.0, r() * 50.0))
                    * Mat4::from_rotation_x(r() * 3.0)
                    * Mat4::from_rotation_y(r() * 3.0)
                    * Mat4::from_rotation_z(r() * 3.0)
                    * Mat4::from_scale(Vec3::new(0.5 + r().abs() * 2.0, 0.5 + r().abs() * 2.0, 0.5 + r().abs() * 2.0));
                // a full bottom row too (projective), so every cofactor is exercised
                let mut m = m;
                m.cols[0].w = r() * 0.1;
                m.cols[1].w = r() * 0.1;
                m.cols[2].w = r() * 0.1;
                m
            })
            .collect()
    }

    fn widen(m: Mat4<f32>) -> [f64; 16] {
        m.to_cols_array().map(f64::from)
    }

    #[test]
    fn inverses_of_unimodular_matrices_are_exact() {
        for (m, inv) in unimodular() {
            assert_eq!(from_cols_array::<f64>(&m).inverse().to_cols_array(), inv);
            assert_eq!(from_cols_array::<f64>(&m).determinant().abs(), 1.0);
            // f32: the native kernel (and cofactors) on small integers are exact too
            assert_eq!(widen(from_cols_array::<f32>(&m).inverse()), inv);
            assert_eq!(widen(from_cols_array::<f32>(&m).inverse_cofactors()), inv);
            let p = from_cols_array::<f32>(&m).via_quad(quad::inverse::<Portable>);
            assert_eq!(widen(p), inv);
        }
    }

    #[test]
    fn block_inverse_is_as_accurate_as_cofactors() {
        // errors as multiples of cond(M) * f32's epsilon, the scale inversion's rounding error has
        // (normwise, infinity norm)
        let norm = |a: [f64; 16]| (0..4).map(|row| (0..4).map(|col| a[col * 4 + row].abs()).sum::<f64>()).fold(0.0, f64::max);
        let (mut block, mut cofactors) = (0.0f64, 0.0f64);
        for m64 in dense(2000) {
            let m: Mat4<f32> = Mat4 { cols: m64.cols.map(|c| Vec4::from_array(c.to_array().map(|x| x as f32))) };
            // the reference: the f64 inverse of the same (rounded) matrix
            let exact = Mat4::<f64> { cols: m.cols.map(|c| Vec4::from_array(c.to_array().map(f64::from))) }.inverse().to_cols_array();
            let unit = norm(widen(m)) * norm(exact) * f64::from(f32::EPSILON) * norm(exact);
            let portable = m.via_quad(quad::inverse::<Portable>);
            // the native backend does the portable arithmetic in the same order: bit for bit
            assert_eq!(m.inverse().to_cols_array().map(f32::to_bits), portable.to_cols_array().map(f32::to_bits));
            let err = |got: Mat4<f32>| norm(core::array::from_fn(|i| widen(got)[i] - exact[i])) / unit;
            block = block.max(err(portable));
            cofactors = cofactors.max(err(m.inverse_cofactors()));
        }
        // a few units each (the bound's constant is small for a 4x4), and no worse than cofactors
        assert!(block < 16.0 && cofactors < 16.0, "worst errors in units of cond * eps: block {block}, cofactors {cofactors}");
        assert!(block < 2.0 * cofactors, "block {block} against cofactors {cofactors}");
    }

    #[test]
    fn products_agree_with_plain_arithmetic() {
        let ms: Vec<Mat4<f32>> =
            dense(64).into_iter().map(|m| Mat4 { cols: m.cols.map(|c| Vec4::from_array(c.to_array().map(|x| x as f32))) }).collect();
        for w in ms.windows(2) {
            let (a, b) = (w[0], w[1]);
            // the same products and sums in the same order: bit-identical
            assert_eq!((a * b).to_cols_array().map(f32::to_bits), a.mul_plain(b).to_cols_array().map(f32::to_bits));
            let v = b.cols[1];
            assert_eq!((a * v).to_array().map(f32::to_bits), a.mul_vec_plain(v).to_array().map(f32::to_bits));
            let c = a.cols.map(|c| Portable::load(&c.to_array()));
            assert_eq!(quad::mul_vec4(&c, Portable::load(&v.to_array())).store().map(f32::to_bits), a.mul_vec_plain(v).to_array().map(f32::to_bits));
        }
    }

    #[test]
    fn singular_matrices_give_non_finite_inverses() {
        let mut m = Mat4::<f32>::identity();
        m.cols[2] = m.cols[1];
        assert!(!m.inverse().is_finite());
        assert!(!m.inverse_cofactors().is_finite());
        assert!(!Mat3::<f64>::from_cols(Vec3::splat(1.0), Vec3::splat(2.0), Vec3::new(0.0, 1.0, 0.0)).inverse().is_finite());
    }

    #[test]
    fn transforms_follow_glams_conventions() {
        use core::f64::consts::FRAC_PI_2;
        // rotations are counter-clockwise for positive angles: x to y about z, y to z about x, z to x about y
        let near = |a: Vec4<f64>, b: Vec4<f64>| (a - b).abs().to_array().iter().all(|d| *d < 1e-12);
        let (x, y, z) = (Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(0.0, 1.0, 0.0, 0.0), Vec4::new(0.0, 0.0, 1.0, 0.0));
        assert!(near(Mat4::from_rotation_z(FRAC_PI_2) * x, y));
        assert!(near(Mat4::from_rotation_x(FRAC_PI_2) * y, z));
        assert!(near(Mat4::from_rotation_y(FRAC_PI_2) * z, x));
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)) * Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(m.project_point3(Vec3::new(1.0, 1.0, 1.0)), Vec3::new(3.0, 5.0, 7.0));
        assert_eq!(m.transpose().transpose(), m);
        assert_eq!(m.determinant(), 24.0);
        assert_eq!(Mat3::from_cols(Vec3::new(2.0, 0.0, 1.0), Vec3::new(1.0, 3.0, 0.0), Vec3::new(0.0, 1.0, 4.0)).determinant(), 25.0);
    }

    #[test]
    fn mat3_inverse_is_exact_on_a_unimodular_matrix() {
        // det 1: columns (1,0,0), (2,1,0), (3,4,1); inverse columns (1,0,0), (-2,1,0), (5,-4,1)
        let m = Mat3::from_cols(Vec3::new(1.0f32, 0.0, 0.0), Vec3::new(2.0, 1.0, 0.0), Vec3::new(3.0, 4.0, 1.0));
        assert_eq!(m.inverse().to_cols_array(), [1.0, 0.0, 0.0, -2.0, 1.0, 0.0, 5.0, -4.0, 1.0]);
        assert_eq!(m * m.inverse(), Mat3::identity());
        assert_eq!(m.col(2), Vec3::new(3.0, 4.0, 1.0));
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn pod_layouts() {
        assert_eq!(size_of::<Mat4<f32>>(), 64);
        assert_eq!(size_of::<Vec4<f64>>(), 32);
        assert_eq!(bytemuck::bytes_of(&Mat4::<f32>::identity()).len(), 64);
        let v = Vec3::new(1.0f32, 2.0, 3.0);
        assert_eq!(bytemuck::pod_read_unaligned::<[f32; 3]>(bytemuck::bytes_of(&v)), [1.0, 2.0, 3.0]);
    }
}
