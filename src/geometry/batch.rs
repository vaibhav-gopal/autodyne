//! Batched kernels: one tight loop over contiguous arrays, for work done over thousands of elements
//! at a time (a UI's transform pass every frame).

use super::matrix::Mat4;
use crate::units::*;

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
    #[should_panic(expected = "parent 2 doesn't come before 1")]
    fn parents_after_children_are_refused() {
        let m = Mat4::<f64>::identity();
        compose_world(&[NO_PARENT, 2, 0], &[m; 3], &mut [m; 3]);
    }
}
