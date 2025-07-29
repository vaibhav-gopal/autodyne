pub trait FromPrimitive: Copy {
    fn from_i64(n: i64) -> Option<Self>;
    fn from_u64(n: u64) -> Option<Self>;
    fn from_isize(n: isize) -> Option<Self>;
    fn from_i8(n: i8) -> Option<Self>;
    fn from_i16(n: i16) -> Option<Self>;
    fn from_i32(n: i32) -> Option<Self>;
    fn from_i128(n: i128) -> Option<Self>;
    fn from_usize(n: usize) -> Option<Self>;
    fn from_u8(n: u8) -> Option<Self>;
    fn from_u16(n: u16) -> Option<Self>;
    fn from_u32(n: u32) -> Option<Self>;
    fn from_u128(n: u128) -> Option<Self>;
    fn from_f32(n: f32) -> Option<Self>;
    fn from_f64(n: f64) -> Option<Self>;
    fn from_str(n: &str) -> Option<Self>;
    fn from(n: impl ToPrimitive) -> Option<Self>;
}

pub trait ToPrimitive: Copy {
    fn to_i64(&self) -> Option<i64>;
    fn to_u64(&self) -> Option<u64>;
    fn to_isize(&self) -> Option<isize>;
    fn to_i8(&self) -> Option<i8>;
    fn to_i16(&self) -> Option<i16>;
    fn to_i32(&self) -> Option<i32>;
    fn to_i128(&self) -> Option<i128>;
    fn to_usize(&self) -> Option<usize>;
    fn to_u8(&self) -> Option<u8>;
    fn to_u16(&self) -> Option<u16>;
    fn to_u32(&self) -> Option<u32>;
    fn to_u128(&self) -> Option<u128>;
    fn to_f32(&self) -> Option<f32>;
    fn to_f64(&self) -> Option<f64>;
    fn to_str(&self) -> Option<String>;
    fn to(&self) -> Option<impl FromPrimitive>;
}

pub trait CastPrimitive: FromPrimitive + ToPrimitive {}
impl<T> CastPrimitive for T
where T:
    FromPrimitive + ToPrimitive {}

macro_rules! impl_to_primitive_int_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let min = $DstT::MIN as $SrcT;
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || (min <= *self && *self <= max) {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_int_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if 0 <= *self && (size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max) {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_int {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_int_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_int_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> Option<f32> {
                Some(*self as f32)
            }
            #[inline]
            fn to_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
        }
    };
}

macro_rules! impl_to_primitive_uint_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() < size_of::<$DstT>() || *self <= max {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_uint_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_uint {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_uint_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_uint_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> Option<f32> {
                Some(*self as f32)
            }
            #[inline]
            fn to_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
        }
    };
}

macro_rules! impl_to_primitive_float_to_float {
    ($SrcT:ident : $( fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        fn $method(&self) -> Option<$DstT> {
            // We can safely cast all values, whether NaN, +-inf, or finite.
            // Finite values that are reducing size may saturate to +-inf.
            Some(*self as $DstT)
        }
    )*}
}

macro_rules! float_to_int_unchecked {
    // SAFETY: Must not be NaN or infinite; must be representable as the integer after truncating.
    // We already checked that the float is in the exclusive range `(MIN-1, MAX+1)`.
    ($float:expr => $int:ty) => {
        unsafe { $float.to_int_unchecked::<$int>() }
    };
}

macro_rules! impl_to_primitive_float_to_signed_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $i:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$i> {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(MIN-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$i>() {
                // With a larger size, we can represent the range exactly.
                const MIN_M1: $f = $i::MIN as $f - 1.0;
                const MAX_P1: $f = $i::MAX as $f + 1.0;
                if *self > MIN_M1 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $i));
                }
            } else {
                // We can't represent `MIN-1` exactly, but there's no fractional part
                // at this magnitude, so we can just use a `MIN` inclusive boundary.
                const MIN: $f = $i::MIN as $f;
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                const MAX_P1: $f = $i::MAX as $f;
                if *self >= MIN && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $i));
                }
            }
            None
        }
    )*}
}

macro_rules! impl_to_primitive_float_to_unsigned_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $u:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$u> {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$u>() {
                // With a larger size, we can represent the range exactly.
                const MAX_P1: $f = $u::MAX as $f + 1.0;
                if *self > -1.0 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $u));
                }
            } else {
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                // (`u128::MAX as f32` is infinity, but this is still ok.)
                const MAX_P1: $f = $u::MAX as $f;
                if *self > -1.0 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $u));
                }
            }
            None
        }
    )*}
}

macro_rules! impl_to_primitive_float {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_float_to_signed_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_float_to_unsigned_int! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            impl_to_primitive_float_to_float! { $T:
                fn to_f32 -> f32;
                fn to_f64 -> f64;
            }
        }
    };
}

impl_to_primitive_int!(i8);
impl_to_primitive_int!(i16);
impl_to_primitive_int!(i32);
impl_to_primitive_int!(i64);
impl_to_primitive_int!(i128);
impl_to_primitive_int!(isize);

impl_to_primitive_int!(u8);
impl_to_primitive_int!(u16);
impl_to_primitive_int!(u32);
impl_to_primitive_int!(u64);
impl_to_primitive_int!(u128);
impl_to_primitive_int!(usize);

impl_to_primitive_float!(f32);
impl_to_primitive_float!(f64);

macro_rules! impl_from_primitive {
    ($T:ty, $to_ty:ident) => {
        #[allow(deprecated)]
        impl FromPrimitive for $T {
            #[inline]
            fn from_isize(n: isize) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_i8(n: i8) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_i16(n: i16) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_i32(n: i32) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_i64(n: i64) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_i128(n: i128) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_usize(n: usize) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_u8(n: u8) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_u16(n: u16) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_u32(n: u32) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_u64(n: u64) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_u128(n: u128) -> Option<$T> {
                n.$to_ty()
            }

            #[inline]
            fn from_f32(n: f32) -> Option<$T> {
                n.$to_ty()
            }
            #[inline]
            fn from_f64(n: f64) -> Option<$T> {
                n.$to_ty()
            }
        }
    };
}

impl_from_primitive!(isize, to_isize);
impl_from_primitive!(i8, to_i8);
impl_from_primitive!(i16, to_i16);
impl_from_primitive!(i32, to_i32);
impl_from_primitive!(i64, to_i64);
impl_from_primitive!(i128, to_i128);
impl_from_primitive!(usize, to_usize);
impl_from_primitive!(u8, to_u8);
impl_from_primitive!(u16, to_u16);
impl_from_primitive!(u32, to_u32);
impl_from_primitive!(u64, to_u64);
impl_from_primitive!(u128, to_u128);
impl_from_primitive!(f32, to_f32);
impl_from_primitive!(f64, to_f64);