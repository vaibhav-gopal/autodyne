//! Runtime type information for sample types, for crossing API boundaries (game engines, DAWs,
//! ML frameworks) where the element type of a buffer is only known at runtime.
//!
//! [`DType`] names an element type as a value, the same idea as a NumPy / PyTorch / ONNX dtype, plus
//! 24-bit integers for audio PCM. [`Reflection`] links a Rust type to its `DType`.

use std::any::{Any, TypeId};
use std::fmt::{self, Debug, Display};

use super::Complex;

/// Element type of a buffer, as a runtime value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 24-bit signed integer, packed in 3 bytes (common in audio files and interfaces)
    I24,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// Complex number of two 32-bit floats (real, imaginary).
    ComplexF32,
    /// Complex number of two 64-bit floats (real, imaginary).
    ComplexF64,
}

impl DType {
    /// Every element type, in declaration order.
    pub const ALL: [DType; 13] = [
        DType::F32, DType::F64, DType::I8, DType::I16, DType::I24, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64, DType::ComplexF32, DType::ComplexF64,
    ];

    /// Size of one element in bytes (3 for `I24`).
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 => 2,
            DType::I24 => 3,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 | DType::ComplexF32 => 8,
            DType::ComplexF64 => 16,
        }
    }
    /// Short lowercase name, as used by NumPy-style APIs ("f32", "i16", "c64" style names are not used).
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I24 => "i24",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::ComplexF32 => "complex_f32",
            DType::ComplexF64 => "complex_f64",
        }
    }
    /// Parses a name produced by `name` (case-insensitive).
    pub fn from_name(name: &str) -> Option<DType> {
        DType::ALL.into_iter().find(|d| d.name().eq_ignore_ascii_case(name))
    }
    /// `F32` or `F64`.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
    /// `ComplexF32` or `ComplexF64`.
    pub const fn is_complex(self) -> bool {
        matches!(self, DType::ComplexF32 | DType::ComplexF64)
    }
    /// Any of the signed or unsigned integers (`I24` included).
    pub const fn is_integer(self) -> bool {
        !self.is_float() && !self.is_complex()
    }
    /// Whether negative values are representable (floats and complex numbers included).
    pub const fn is_signed(self) -> bool {
        !matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Runtime reflection for element types: which `DType` a Rust type is, its name and identity.
///
/// Implemented for the primitive numeric types and `Complex<f32>` / `Complex<f64>`: every type that
/// has a `DType`. (There is no Rust type for `DType::I24`; 24-bit data is converted on read/write.)
pub trait Reflection: Any + Send + Sync + Copy + Debug + PartialEq + 'static {
    /// The runtime tag for this type.
    const DTYPE: DType;
    /// The runtime tag of a value's type (same as `Self::DTYPE`).
    fn dtype(&self) -> DType {
        Self::DTYPE
    }
    /// Unique identifier of the type (named to avoid clashing with `Any::type_id`).
    fn reflect_type_id() -> TypeId {
        TypeId::of::<Self>()
    }
    /// Human-readable type name.
    fn type_name() -> &'static str {
        std::any::type_name::<Self>()
    }
    /// Human-readable type name as an owned String.
    fn type_name_string() -> String {
        Self::type_name().to_string()
    }
    /// Equality comparison (IEEE semantics for floats: NaN is not equal to itself).
    fn reflect_eq(&self, other: Self) -> bool {
        *self == other
    }
}

macro_rules! impl_reflection {
    ($($T:ty => $D:ident),+ $(,)?) => {$(
        impl Reflection for $T {
            const DTYPE: DType = DType::$D;
        }
    )+};
}

impl_reflection!(
    f32 => F32, f64 => F64,
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    Complex<f32> => ComplexF32, Complex<f64> => ComplexF64,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtypes_match_rust_types() {
        fn check<T: Reflection>() {
            if T::DTYPE != DType::I24 {
                assert_eq!(T::DTYPE.size_bytes(), std::mem::size_of::<T>(), "{}", T::type_name());
            }
        }
        check::<f32>();
        check::<f64>();
        check::<i8>();
        check::<i16>();
        check::<i64>();
        check::<u32>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
        assert_eq!(1.0f32.dtype(), DType::F32);
        assert_eq!(<f64 as Reflection>::type_name(), "f64");
        assert_eq!(<i16 as Reflection>::reflect_type_id(), TypeId::of::<i16>());
    }

    #[test]
    fn names_roundtrip_and_classify() {
        for d in DType::ALL {
            assert_eq!(DType::from_name(d.name()), Some(d));
        }
        assert_eq!(DType::from_name("F32"), Some(DType::F32));
        assert_eq!(DType::from_name("f16"), None);
        assert!(DType::F64.is_float() && !DType::F64.is_integer());
        assert!(DType::I24.is_integer() && DType::I24.is_signed() && DType::I24.size_bytes() == 3);
        assert!(DType::ComplexF32.is_complex() && !DType::U8.is_signed());
    }

    #[test]
    fn reflect_eq_follows_ieee() {
        assert!(1.5f64.reflect_eq(1.5));
        assert!(!f32::NAN.reflect_eq(f32::NAN));
    }
}
