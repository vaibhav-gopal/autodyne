//! Batched kernels: one tight loop over contiguous arrays, for work done over thousands of elements
//! at a time (a UI's transform pass every frame).
//!
//! tend: Geometry and lanes / batched kernels

use super::matrix::{Mat3, Mat4};
use crate::units::*;

/// Points through a homography (a projective 3x3 matrix of 2-D points), in place: each `(x, y)`
/// becomes `(m * (x, y, 1)).xy / w`. The coordinates are separate columns, so the loop vectorizes
/// (projective division included); points on the horizon (`w = 0`) come out infinite.
///
/// # Panics
///
/// If `xs` and `ys` differ in length.
pub fn transform_points<T: Float>(m: &Mat3<T>, xs: &mut [T], ys: &mut [T]) {
    assert_eq!(xs.len(), ys.len(), "transform_points: xs and ys differ in length");
    let [c0, c1, c2] = m.cols;
    for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
        let (px, py) = (*x, *y);
        let w = c0.z * px + c1.z * py + c2.z;
        let inv = T::_ONE / w;
        *x = (c0.x * px + c1.x * py + c2.x) * inv;
        *y = (c0.y * px + c1.y * py + c2.y) * inv;
    }
}

/// No parent: a root, in [`compose_world`]'s `parents`.
pub const NO_PARENT: u32 = u32::MAX;

/// World matrices for a forest in one pass: `world[i] = world[parents[i]] * locals[i]`, or
/// `locals[i]` for roots ([`NO_PARENT`]). Every parent must come before its children (any preorder
/// or breadth-first order does).
///
/// Each product is `Mat4`'s own (four SIMD lanes for `f32` where the target has them), which keeps
/// the parent in registers and measures level with glam's loop.
///
/// # Panics
///
/// If the slices differ in length, or a parent doesn't come before its child.
pub fn compose_world<T: Float>(parents: &[u32], locals: &[Mat4<T>], world: &mut [Mat4<T>]) {
    assert!(parents.len() == locals.len() && locals.len() == world.len(), "compose_world: slices differ in length");
    for i in 0..world.len() {
        let p = parents[i];
        world[i] = if p == NO_PARENT {
            locals[i]
        } else {
            assert!((p as usize) < i, "compose_world: parent {p} doesn't come before {i}");
            world[p as usize] * locals[i]
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    #[test]
    fn world_matrices_follow_parents() {
        let t = |x: f32| Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
        let locals = [t(1.0), t(10.0), t(100.0), t(1000.0), t(5.0)];
        // two trees: 0 <- 1 <- 2, 0 <- 3; and 4 alone
        let parents = [NO_PARENT, 0, 1, 0, NO_PARENT];
        let mut world = [Mat4::identity(); 5];
        compose_world(&parents, &locals, &mut world);
        assert_eq!(world.map(|m| m.cols[3].x), [1.0, 11.0, 111.0, 1001.0, 5.0]);
        // the same as composing by hand, product by product
        assert_eq!(world[2], locals[0] * locals[1] * locals[2]);
    }

    #[test]
    fn homographies_move_points_as_single_products_do() {
        use crate::geometry::Vec3;
        // a perspective map: the unit square to a trapezoid
        let m = Mat3::from_cols(Vec3::new(2.0, 0.1, 0.05), Vec3::new(0.3, 1.5, 0.02), Vec3::new(4.0, -1.0, 1.0));
        let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = (0..37).map(|i| (i as f64 * 0.37 - 5.0, (i * i % 11) as f64 - 3.0)).unzip();
        let (x0, y0) = (xs.clone(), ys.clone());
        transform_points(&m, &mut xs, &mut ys);
        for i in 0..xs.len() {
            let p = m * Vec3::new(x0[i], y0[i], 1.0);
            assert!((xs[i] - p.x / p.z).abs() < 1e-12 && (ys[i] - p.y / p.z).abs() < 1e-12, "point {i}");
        }
        // the inverse maps them back
        transform_points(&m.inverse(), &mut xs, &mut ys);
        assert!(xs.iter().zip(&x0).chain(ys.iter().zip(&y0)).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    #[should_panic(expected = "parent 2 doesn't come before 1")]
    fn parents_after_children_are_refused() {
        let m = Mat4::<f64>::identity();
        compose_world(&[NO_PARENT, 2, 0], &[m; 3], &mut [m; 3]);
    }
}
