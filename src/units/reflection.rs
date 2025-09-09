use std::any::{Any, TypeId};
use std::fmt::Debug;

/// Type reflection for a unit
pub trait Reflection: Any + Send + Sync + Copy + Debug + 'static {
    /// Unique type identifier
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
    /// Human-readable type name
    fn type_name() -> &'static str;
    /// Human-readable type name as a owned String
    fn type_name_string() -> String {
        Self::type_name().to_string()
    }
    /// Equality comparison
    fn reflect_eq(&self, other: Self) -> bool;
}
