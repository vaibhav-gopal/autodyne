//! [`Vec2`], [`Vec3`] and [`Vec4`]: component-wise arithmetic, dot products, lengths, and the
//! conversions between sizes.

use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::units::*;

macro_rules! vector {
    ($name:ident { $($f:ident = $i:literal),+ }, $n:literal) => {
        /// A vector of
        #[doc = stringify!($n)]
        /// components.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name<T> {
            $(
                #[doc = concat!("component ", stringify!($i))]
                pub $f: T,
            )+
        }

        impl<T: Float> $name<T> {
            /// A vector from its components.
            #[inline(always)]
            pub const fn new($($f: T),+) -> Self {
                $name { $($f),+ }
            }

            /// Every component `v`.
            #[inline(always)]
            pub fn splat(v: T) -> Self {
                $name { $($f: v),+ }
            }

            /// All zeros.
            #[inline(always)]
            pub fn zero() -> Self {
                Self::splat(T::_ZERO)
            }

            /// The dot product.
            #[inline(always)]
            pub fn dot(self, o: Self) -> T {
                let mut s = T::_ZERO;
                $(s = s + self.$f * o.$f;)+
                s
            }

            /// The length.
            #[inline(always)]
            pub fn length(self) -> T {
                self.dot(self)._sqrt()
            }

            /// Component-wise minimum.
            #[inline(always)]
            pub fn min(self, o: Self) -> Self {
                $name { $($f: self.$f._min(o.$f)),+ }
            }

            /// Component-wise maximum.
            #[inline(always)]
            pub fn max(self, o: Self) -> Self {
                $name { $($f: self.$f._max(o.$f)),+ }
            }

            /// Component-wise absolute value.
            #[inline(always)]
            pub fn abs(self) -> Self {
                $name { $($f: self.$f._abs()),+ }
            }

            /// Whether every component is finite.
            #[inline(always)]
            pub fn is_finite(self) -> bool {
                true $(&& self.$f._is_finite())+
            }

            /// The components as an array.
            #[inline(always)]
            pub fn to_array(self) -> [T; $n] {
                [$(self.$f),+]
            }

            /// A vector from an array of its components.
            #[inline(always)]
            pub fn from_array(a: [T; $n]) -> Self {
                $name { $($f: a[$i]),+ }
            }
        }

        impl<T: Float> Default for $name<T> {
            /// All zeros.
            #[inline(always)]
            fn default() -> Self {
                Self::zero()
            }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: usize) -> &T {
                match i {
                    $($i => &self.$f,)+
                    _ => panic!("{} index {i} out of range", stringify!($name)),
                }
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut T {
                match i {
                    $($i => &mut self.$f,)+
                    _ => panic!("{} index {i} out of range", stringify!($name)),
                }
            }
        }

        impl<T: Float> Add for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                $name { $($f: self.$f + o.$f),+ }
            }
        }

        impl<T: Float> AddAssign for $name<T> {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }

        impl<T: Float> Sub for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                $name { $($f: self.$f - o.$f),+ }
            }
        }

        impl<T: Float> SubAssign for $name<T> {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }

        impl<T: Float> Mul for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                $name { $($f: self.$f * o.$f),+ }
            }
        }

        impl<T: Float> Div for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn div(self, o: Self) -> Self {
                $name { $($f: self.$f / o.$f),+ }
            }
        }

        impl<T: Float> Mul<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, k: T) -> Self {
                $name { $($f: self.$f * k),+ }
            }
        }

        impl<T: Float> Div<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn div(self, k: T) -> Self {
                $name { $($f: self.$f / k),+ }
            }
        }

        impl<T: Float> Add<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn add(self, k: T) -> Self {
                $name { $($f: self.$f + k),+ }
            }
        }

        impl<T: Float> Sub<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, k: T) -> Self {
                $name { $($f: self.$f - k),+ }
            }
        }

        impl<T: Float> Neg for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                $name { $($f: -self.$f),+ }
            }
        }

        // SAFETY: `#[repr(C)]` of identical `f32` / `f64` fields: no padding, every bit pattern valid.
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name<f32> {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name<f32> {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name<f64> {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name<f64> {}
    };
}

vector!(Vec2 { x = 0, y = 1 }, 2);
vector!(Vec3 { x = 0, y = 1, z = 2 }, 3);
// Not 16-byte aligned (glam's Vec4 is): measured in parda's prototype, alignment made matrix-vector
// products slower (737 vs 600 ns per 1,024) and SIMD loads don't need it.
vector!(Vec4 { x = 0, y = 1, z = 2, w = 3 }, 4);

impl<T: Float> Vec2<T> {
    /// `(x, y, z)`.
    #[inline(always)]
    pub fn extend(self, z: T) -> Vec3<T> {
        Vec3::new(self.x, self.y, z)
    }
}

impl<T: Float> Vec3<T> {
    /// `(x, y, z, w)`.
    #[inline(always)]
    pub fn extend(self, w: T) -> Vec4<T> {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// `(x, y)`.
    #[inline(always)]
    pub fn truncate(self) -> Vec2<T> {
        Vec2::new(self.x, self.y)
    }

    /// The cross product.
    #[inline(always)]
    pub fn cross(self, o: Self) -> Self {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
}

impl<T: Float> Vec4<T> {
    /// `(x, y, z)`.
    #[inline(always)]
    pub fn truncate(self) -> Vec3<T> {
        Vec3::new(self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_lengths_and_conversions() {
        let a = Vec3::new(1.0f64, 2.0, 2.0);
        assert_eq!(a.length(), 3.0);
        assert_eq!(a.dot(Vec3::new(4.0, -1.0, 0.5)), 3.0);
        // the right-handed basis: x cross y = z
        assert_eq!(Vec3::new(1.0f32, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(Vec2::new(1.0f32, 2.0).extend(3.0).extend(4.0).truncate().truncate(), Vec2::new(1.0, 2.0));
        assert_eq!(Vec4::from_array([1.0f64, 2.0, 3.0, 4.0]).to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Vec3::new(-1.0f32, 5.0, 2.0).min(Vec3::splat(1.0)).max(Vec3::splat(0.0)).abs(), Vec3::new(0.0, 1.0, 1.0));
        assert!(!Vec2::new(1.0f32, f32::NAN).is_finite() && Vec2::<f64>::default().is_finite());
    }

    #[test]
    fn arithmetic_is_component_wise() {
        let mut v = Vec4::new(1.0f32, 2.0, 3.0, 4.0);
        v += Vec4::splat(1.0);
        v -= Vec4::new(0.0, 0.0, 0.0, 5.0);
        assert_eq!(v, Vec4::new(2.0, 3.0, 4.0, 0.0));
        assert_eq!(-(v * 2.0 + 1.0) / 2.0 - 0.5, Vec4::new(-3.0, -4.0, -5.0, -1.0));
        assert_eq!(v * v / Vec4::splat(2.0), Vec4::new(2.0, 4.5, 8.0, 0.0));
        v[3] = 9.0;
        assert_eq!((v[0], v[3]), (2.0, 9.0));
    }

    #[test]
    #[should_panic(expected = "Vec3 index 3 out of range")]
    fn indexing_past_the_end_panics() {
        let _ = Vec3::new(1.0f32, 2.0, 3.0)[3];
    }
}
