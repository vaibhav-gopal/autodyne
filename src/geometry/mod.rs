//! Small fixed-size geometry: [`Vec2`], [`Vec3`], [`Vec4`], [`Mat3`] and [`Mat4`], generic over
//! [`Float`](crate::units::Float) like the rest of autodyne, and batched kernels over arrays of
//! them ([`compose_world`]).
//!
//! - Column-major and `#[repr(C)]`, so arrays of them go into GPU buffers as they are; with the
//!   `bytemuck` feature they're `Pod` for `f32` and `f64`.
//! - Single `f32` values take four-lane SIMD paths where the target has them (SSE2 on x86-64, NEON
//!   on aarch64, simd128 on wasm32 built with `+simd128`): matrix products, matrix-vector products
//!   and the 4x4 inverse. The choice is made per instance at compile time, so `f64` and other
//!   targets carry no branch.
//! - Batched kernels run the same products over contiguous arrays in one loop.
//!
//! Conventions follow glam's (column vectors, `m * v`, rotations counter-clockwise for positive
//! angles), so code moving from glam keeps its meaning. Each type is benchmarked against glam in
//! `bench/geometry` before it's offered as a replacement.
//!
//! ```
//! use autodyne::geometry::{Mat4, Vec3, Vec4};
//!
//! let m = Mat4::from_translation(Vec3::new(10.0f32, 20.0, 0.0)) * Mat4::from_rotation_z(core::f32::consts::FRAC_PI_2);
//! let p = m * Vec4::new(1.0, 0.0, 0.0, 1.0); // rotated to (0, 1), then moved
//! assert!((p.x - 10.0).abs() < 1e-5 && (p.y - 21.0).abs() < 1e-5);
//! let i = (m * m.inverse()).to_cols_array();
//! assert!(i.iter().zip(Mat4::<f32>::identity().to_cols_array()).all(|(a, b)| (a - b).abs() < 1e-5));
//! ```
//!
//! tend: Geometry and lanes / small types

mod batch;
mod matrix;
mod quad;
mod vector;

pub use batch::{compose_world, NO_PARENT};
pub use matrix::{Mat3, Mat4};
pub use vector::{Vec2, Vec3, Vec4};
