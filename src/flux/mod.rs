/// Going to follow a similar flow to JAX
/// First JIT compile / trace via operator overloads or compile time macros to some primitive operator representation
/// Afterwards can do the following:
/// - Automatic differentiation via primitive operations
/// - Automatic SIMD and vectorization and kernel calls (BIG MAYBE, THIS IS AN EXTREMELY LONG PROCESS)

/// MLIR is important here, its a graph IR framework for representing many computation graphs in different 'dialects' of computation (hardware-levels: linalg -> gpu -> llvm -> nvgpu -> amdgpu etc)
/// - use melior for MLIR rust bindings if you want
/// - create your own IR framework for graphs, maybe as two parts the IR part and dialect part
/// - will need to be able to transpile to a different IR (LLVM or NVVM)

/// Essentially end-goal would be to replace JAX and OpenXLA for Rust instead
/// 1. Parse High-Level Graph (using some IR framework)
///          ↓
/// 2. Graph-Level Optimizations:
///    - Fusion (combine ops)
///    - Layout optimization
///    - Constant folding
///          ↓
/// 3. Device Placement:
///    - Analyze computation costs
///    - Consider memory constraints
///    - Assign ops to CPU/GPU/TPU
///          ↓
/// 4. Memory Planning:
///    - Buffer lifetime analysis
///    - Memory reuse opportunities
///    - Cross-device transfer insertion
///          ↓
/// 5. Kernel Selection:
///    - For each op, choose best implementation
///    - BLAS library vs custom kernel
///    - Consider input shapes/types
///          ↓
/// 6. Code Generation (choose a backend which has its own IR and simply transpile between them):
///    - LLVM for CPU (can automatically use SIMD instructions)
///    - Cranelift for CPU (rust JIT compiler, faster than LLVM and of course native rust)
///    - NVVM (PTX) for Nvidia GPU (lower level than LLVM, need to use BLAS in step before rather than here)

/// Since the code generation part (Backend IR) handles some optimizations on its own, kernel selection should only handle the optimizations that LLVM or NVVM can't handle

/// Phase 1: Core Value System ← WE ARE HERE
/// ├── Value trait definition
/// ├── IRType enum with extensibility  
/// ├── Basic type system operations
/// └── Value creation/management
/// 
/// Phase 2: Operation System
/// ├── Operation struct design
/// ├── Opcode system
/// ├── Input/output relationships
/// └── Attributes system
/// 
/// Phase 3: IR Module Structure
/// ├── IRModule container
/// ├── Value and operation storage
/// ├── Function definitions
/// └── Module validation
/// 
/// Phase 4: Dialect System
/// ├── Dialect trait
/// ├── Built-in dialects (Core)
/// ├── Registration system
/// └── Type inference
/// 
/// Phase 5: Pass System  
/// ├── Pass trait
/// ├── PassManager
/// ├── Basic optimization passes
/// └── Pass ordering
/// 
/// Phase 6: Backend System
/// ├── Backend trait
/// ├── Cranelift implementation
/// ├── Code generation
/// └── JIT execution
/// 
/// Phase 7: User API & Ergonomics
/// ├── Builder patterns
/// ├── Proc macros
/// ├── Error handling
/// └── Documentation

/// Notes:
/// - Zero external deps; everything is `std`.
/// - Opaque types support pluggable payloads with stable hash/eq using `ErasedHashEq`.
/// - No interning yet; that can be added later in Phase 4 (Dialects) or Phase 3 (Modules).
/// - Thread-safety: `IRContext` uses `AtomicU32` for IDs; otherwise cloning is cheap-ish via `Arc`.

use std::any::Any;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/* ===================== Symbols ===================== */

/// Interned-ish symbol: for Phase 1 we just wrap an `Arc<str>`.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Symbol(Arc<str>);

impl Symbol {
    pub fn new<S: AsRef<str>>(s: S) -> Self { Self(Arc::<str>::from(s.as_ref())) }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl From<&str> for Symbol { fn from(s: &str) -> Self { Symbol::new(s) } }
impl From<String> for Symbol { fn from(s: String) -> Self { Symbol::new(s) } }
impl Debug for Symbol { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_tuple("Symbol").field(&self.as_str()).finish() } }
impl Display for Symbol { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) } }

/* ===================== Opaque payload erasure ===================== */

/// Trait for erased, hashable, equatable, downcastable payloads behind dialect-defined types.
/// Blanket-implemented for any `'static + Send + Sync + Debug + Hash + Eq`.
pub trait ErasedHashEq: Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;
    fn type_id(&self) -> std::any::TypeId;
    fn erased_hash(&self, state: &mut dyn Hasher);
    fn erased_eq(&self, other: &dyn ErasedHashEq) -> bool;
}
impl<T> ErasedHashEq for T
where
    T: 'static + Send + Sync + Debug + Hash + Eq,
{
    fn as_any(&self) -> &dyn Any { self }
    fn type_id(&self) -> std::any::TypeId { std::any::TypeId::of::<T>() }
    fn erased_hash(&self, mut state: &mut dyn Hasher) { self.hash(&mut state) }
    fn erased_eq(&self, other: &dyn ErasedHashEq) -> bool {
        if self.type_id() == other.type_id() {
            // Safe: same concrete type
            if let Some(rhs) = other.as_any().downcast_ref::<T>() { return self == rhs; }
        }
        false
    }
}

/* ===================== Type System ===================== */

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntWidth { W1, W8, W16, W32, W64 }
impl IntWidth { pub fn bits(self) -> u32 { match self { Self::W1 => 1, Self::W8 => 8, Self::W16 => 16, Self::W32 => 32, Self::W64 => 64 } } }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FloatWidth { F16, BF16, F32, F64 }
impl FloatWidth { pub fn bits(self) -> u32 { match self { Self::F16 => 16, Self::BF16 => 16, Self::F32 => 32, Self::F64 => 64 } } }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dim { Dynamic, Known(usize) }

#[derive(Clone, Debug)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub elem: Type,
}

/// Extensible IR type tree.
#[derive(Clone, Debug)]
pub enum Type {
    // Core scalars
    Int(IntWidth, /* signed? */ bool),
    Float(FloatWidth),
    Index, // target-sized index type

    // Aggregates
    Tensor(TensorType),
    Tuple(Vec<Type>),

    // Dialect-defined structural leaf with optional payload used for eq/hash
    Opaque { dialect: Symbol, name: Symbol, payload: Option<Arc<dyn ErasedHashEq>> },
}

impl Type {
    /* --- Constructors --- */
    pub fn i(bits: u32, signed: bool) -> Self {
        let w = match bits { 1 => IntWidth::W1, 8 => IntWidth::W8, 16 => IntWidth::W16, 32 => IntWidth::W32, 64 => IntWidth::W64, _ => panic!("unsupported int width: {}", bits) };
        Self::Int(w, signed)
    }
    pub fn f(bits: u32) -> Self {
        let w = match bits { 16 => FloatWidth::F16, 32 => FloatWidth::F32, 64 => FloatWidth::F64, _ => panic!("unsupported float width: {}", bits) };
        Self::Float(w)
    }
    pub fn bf16() -> Self { Self::Float(FloatWidth::BF16) }
    pub fn index() -> Self { Self::Index }
    pub fn tensor(shape: impl Into<Vec<Dim>>, elem: Type) -> Self { Self::Tensor(TensorType { shape: shape.into(), elem }) }
    pub fn tuple(elems: impl Into<Vec<Type>>) -> Self { Self::Tuple(elems.into()) }
    pub fn opaque<D: Into<Symbol>, N: Into<Symbol>>(dialect: D, name: N) -> Self {
        Self::Opaque { dialect: dialect.into(), name: name.into(), payload: None }
    }
    pub fn opaque_with<D, N, P>(dialect: D, name: N, payload: P) -> Self
    where
        D: Into<Symbol>, N: Into<Symbol>, P: 'static + Send + Sync + Debug + Hash + Eq,
    {
        Self::Opaque { dialect: dialect.into(), name: name.into(), payload: Some(Arc::new(payload)) }
    }

    /* --- Queries --- */
    pub fn is_int(&self) -> bool { matches!(self, Type::Int(_, _)) }
    pub fn is_float(&self) -> bool { matches!(self, Type::Float(_)) }
    pub fn is_tensor(&self) -> bool { matches!(self, Type::Tensor(_)) }

    /// Return width in bits for fixed-size scalar types; `None` for Index or aggregates.
    pub fn scalar_bit_width(&self) -> Option<u32> {
        match self {
            Type::Int(w, _) => Some(w.bits()),
            Type::Float(w) => Some(w.bits()),
            Type::Index | Type::Tensor(_) | Type::Tuple(_) | Type::Opaque { .. } => None,
        }
    }

    /// Returns byte size if statically known (no dynamic dims, all element sizes known, non-Index).
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            Type::Int(w, _) => Some((w.bits() as usize + 7) / 8),
            Type::Float(w) => Some((w.bits() as usize + 7) / 8),
            Type::Index => None, // target-dependent
            Type::Tuple(elems) => {
                let mut sum = 0usize;
                for t in elems { sum += t.byte_size()?; } // fail fast if any unknown
                Some(sum)
            }
            Type::Tensor(TensorType { shape, elem }) => {
                let elem_bytes = elem.byte_size()?;
                let mut elements = 1usize;
                for d in shape {
                    match d { Dim::Known(k) => { elements = elements.checked_mul(*k)?; }, Dim::Dynamic => return None }
                }
                Some(elements.checked_mul(elem_bytes)?)
            }
            Type::Opaque { .. } => None,
        }
    }

    /// Deep equality including opaque payloads.
    fn eq_impl(&self, other: &Self) -> bool {
        use Type::*;
        match (self, other) {
            (Int(a, sa), Int(b, sb)) => a == b && sa == sb,
            (Float(a), Float(b)) => a == b,
            (Index, Index) => true,
            (Tuple(a), Tuple(b)) => a == b,
            (Tensor(a), Tensor(b)) => a.shape == b.shape && a.elem == b.elem,
            (Opaque { dialect: da, name: na, payload: pa }, Opaque { dialect: db, name: nb, payload: pb }) => {
                if da != db || na != nb { return false; }
                match (pa, pb) {
                    (None, None) => true,
                    (Some(x), Some(y)) => x.erased_eq(&**y),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Hash including opaque payloads.
    fn hash_impl<H: Hasher>(&self, state: &mut H) {
        use Type::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Int(w, s) => { w.hash(state); s.hash(state); }
            Float(w) => { w.hash(state); }
            Index => {}
            Tuple(elems) => { for t in elems { t.hash_impl(state); } }
            Tensor(TensorType { shape, elem }) => {
                for d in shape { d.hash(state); }
                elem.hash_impl(state);
            }
            Opaque { dialect, name, payload } => {
                dialect.hash(state); name.hash(state);
                if let Some(p) = payload { p.erased_hash(state); }
            }
        }
    }
}
impl PartialEq for Type { fn eq(&self, other: &Self) -> bool { self.eq_impl(other) } }
impl Eq for Type {}
impl Hash for Type { fn hash<H: Hasher>(&self, state: &mut H) { self.hash_impl(state) } }

/* ===================== Values ===================== */

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ValueId(NonZeroU32);
impl ValueId {
    fn new(n: u32) -> Self { Self(NonZeroU32::new(n).expect("ValueId must be nonzero")) }
    pub fn get(self) -> u32 { self.0.get() }
}

/// Minimal value handle for Phase 1.
#[derive(Clone, Debug)]
pub struct Value {
    id: ValueId,
    ty: Type,
    // TODO(phase2): back-pointer to defining op or block-arg descriptor
    // TODO(phase2): uses list (def-use chain)
    // TODO(phase7): metadata/attrs map
}

pub trait ValueLike {
    fn id(&self) -> ValueId;
    fn ty(&self) -> &Type;
}
impl ValueLike for Value {
    fn id(&self) -> ValueId { self.id }
    fn ty(&self) -> &Type { &self.ty }
}
impl Value {
    pub fn ty(&self) -> &Type { &self.ty }
}

/* ===================== IR Context ===================== */

/// `IRContext` is a factory for IDs and (later) the home for interning registries.
#[derive(Debug)]
pub struct IRContext { next_id: AtomicU32 }
impl IRContext {
    pub fn new() -> Self { Self { next_id: AtomicU32::new(1) } }

    fn alloc_id(&self) -> ValueId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        ValueId::new(id)
    }

    /// Create a new `Value` with the given type. In Phase 1 this is a free-floating SSA value
    /// with no defining operation; Phase 2 will attach values to `Operation` results.
    pub fn make_value(&self, ty: Type) -> Value { Value { id: self.alloc_id(), ty } }
}

/* ===================== Example Dialect Payload ===================== */

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct LayoutPayload { pub tag: Symbol }

/* ===================== Tests ===================== */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_roundtrip() {
        let s1 = Symbol::new("linalg");
        let s2 = Symbol::from("linalg");
        assert_eq!(s1, s2);
        assert_eq!(s1.as_str(), "linalg");
    }

    #[test]
    fn type_helpers_and_sizes() {
        let i32t = Type::i(32, true);
        let f32t = Type::f(32);
        assert!(i32t.is_int());
        assert!(f32t.is_float());
        assert_eq!(i32t.scalar_bit_width(), Some(32));
        assert_eq!(f32t.byte_size(), Some(4));

        let t = Type::tensor([Dim::Known(2), Dim::Known(3)], Type::f(64));
        assert_eq!(t.byte_size(), Some(2 * 3 * 8));

        let dyn_t = Type::tensor([Dim::Known(2), Dim::Dynamic], Type::i(8, false));
        assert_eq!(dyn_t.byte_size(), None);

        let tup = Type::tuple(vec![Type::i(8, true), Type::f(32)]);
        assert_eq!(tup.byte_size(), Some(1 + 4));
    }

    #[test]
    fn opaque_with_payload_eq_hash() {
        let layout_a = Type::opaque_with("core", "layout", LayoutPayload { tag: Symbol::new("NHWC") });
        let layout_b = Type::opaque_with("core", "layout", LayoutPayload { tag: Symbol::new("NHWC") });
        let layout_c = Type::opaque_with("core", "layout", LayoutPayload { tag: Symbol::new("NCHW") });

        assert_eq!(layout_a, layout_b);
        assert_ne!(layout_a, layout_c);

        use std::collections::hash_map::DefaultHasher;
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        layout_a.hash(&mut ha);
        layout_b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn make_values() {
        let cx = IRContext::new();
        let v0 = cx.make_value(Type::i(32, true));
        let v1 = cx.make_value(Type::i(32, true));
        assert_ne!(v0.id(), v1.id());
        assert_eq!(v0.ty(), v1.ty());
    }
}