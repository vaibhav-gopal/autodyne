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

use std::any::Any;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;

//! Rust-first MLIR-ish IR — Phase 1 (updated for B) + Phase 2 (Operations)
//! ----------------------------------------------------------------------------
//! This single-file crate carries:
//!   Phase 1 (updated): Symbols, Types, ValueId/Value/ValueKind (design B), IRContext
//!   Phase 2: Operation, Opcode as Symbol, Attributes, Def–Use chains, basic builder
//!
//! Design notes:
//! - `Value` is the edge handle. Identity is `ValueId`; origin is in `ValueKind` side-table.
//! - `Operation` stores `ValueId`s for inputs/results. IRContext owns arenas and tables.
//! - Def–use is tracked incrementally on op creation and on RAUW.
//! - Attributes: minimal `Attr` enum plus `AttrMap`.
//! - No control-flow regions/blocks yet; add in Phase 3.
//! - No interning; easy to add inside IRContext later.

/* ===================== Symbols ===================== */

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Symbol(Arc<str>);
impl Symbol { pub fn new<S: AsRef<str>>(s: S) -> Self { Self(Arc::<str>::from(s.as_ref())) } pub fn as_str(&self) -> &str { &self.0 } }
impl From<&str> for Symbol { fn from(s: &str) -> Self { Symbol::new(s) } }
impl From<String> for Symbol { fn from(s: String) -> Self { Symbol::new(s) } }
impl Debug for Symbol { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_tuple("Symbol").field(&self.as_str()).finish() } }
impl Display for Symbol { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) } }

/* ===================== Opaque payload erasure ===================== */

pub trait ErasedHashEq: Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any; fn type_id(&self) -> std::any::TypeId; fn erased_hash(&self, state: &mut dyn Hasher); fn erased_eq(&self, other: &dyn ErasedHashEq) -> bool;
}
impl<T> ErasedHashEq for T where T: 'static + Send + Sync + Debug + Hash + Eq {
    fn as_any(&self) -> &dyn Any { self }
    fn type_id(&self) -> std::any::TypeId { std::any::TypeId::of::<T>() }
    fn erased_hash(&self, mut state: &mut dyn Hasher) { self.hash(&mut state) }
    fn erased_eq(&self, other: &dyn ErasedHashEq) -> bool { if self.type_id() == other.type_id() { if let Some(rhs) = other.as_any().downcast_ref::<T>() { return self == rhs; } } false }
}

/* ===================== Type System ===================== */

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntWidth { W1, W8, W16, W32, W64 }
impl IntWidth { pub fn bits(self) -> u32 { match self { Self::W1=>1, Self::W8=>8, Self::W16=>16, Self::W32=>32, Self::W64=>64 } } }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FloatWidth { F16, BF16, F32, F64 }
impl FloatWidth { pub fn bits(self) -> u32 { match self { Self::F16|Self::BF16=>16, Self::F32=>32, Self::F64=>64 } } }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dim { Dynamic, Known(usize) }

#[derive(Clone, Debug)]
pub struct TensorType { pub shape: Vec<Dim>, pub elem: Type }

#[derive(Clone, Debug)]
pub enum Type {
    Int(IntWidth, bool),
    Float(FloatWidth),
    Index,
    Tensor(TensorType),
    Tuple(Vec<Type>),
    Opaque { dialect: Symbol, name: Symbol, payload: Option<Arc<dyn ErasedHashEq>> },
}

impl Type {
    pub fn i(bits: u32, signed: bool) -> Self { let w = match bits {1=>IntWidth::W1,8=>IntWidth::W8,16=>IntWidth::W16,32=>IntWidth::W32,64=>IntWidth::W64,_=>panic!("unsupported int width: {bits}")}; Self::Int(w, signed) }
    pub fn f(bits: u32) -> Self { let w = match bits {16=>FloatWidth::F16,32=>FloatWidth::F32,64=>FloatWidth::F64,_=>panic!("unsupported float width: {bits}")}; Self::Float(w) }
    pub fn bf16() -> Self { Self::Float(FloatWidth::BF16) }
    pub fn index() -> Self { Self::Index }
    pub fn tensor(shape: impl Into<Vec<Dim>>, elem: Type) -> Self { Self::Tensor(TensorType{shape:shape.into(), elem}) }
    pub fn tuple(elems: impl Into<Vec<Type>>) -> Self { Self::Tuple(elems.into()) }
    pub fn opaque<D:Into<Symbol>,N:Into<Symbol>>(dialect:D,name:N)->Self{ Self::Opaque{dialect:dialect.into(),name:name.into(),payload:None} }
    pub fn opaque_with<D:Into<Symbol>,N:Into<Symbol>,P:'static+Send+Sync+Debug+Hash+Eq>(dialect:D,name:N,payload:P)->Self{ Self::Opaque{dialect:dialect.into(),name:name.into(),payload:Some(Arc::new(payload))} }

    pub fn is_int(&self)->bool{matches!(self,Type::Int(_, _))}
    pub fn is_float(&self)->bool{matches!(self,Type::Float(_))}
    pub fn is_tensor(&self)->bool{matches!(self,Type::Tensor(_))}

    pub fn scalar_bit_width(&self)->Option<u32>{match self{Type::Int(w,_ )=>Some(w.bits()),Type::Float(w)=>Some(w.bits()),Type::Index|Type::Tensor(_)|Type::Tuple(_)|Type::Opaque{..}=>None}}

    pub fn byte_size(&self)->Option<usize>{
        match self{
            Type::Int(w,_)|Type::Float(w)=>Some((w.bits() as usize +7)/8),
            Type::Index=>None,
            Type::Tuple(elems)=>{let mut sum=0usize;for t in elems{sum+=t.byte_size()?;}Some(sum)}
            Type::Tensor(TensorType{shape,elem})=>{let eb=elem.byte_size()?;let mut n=1usize;for d in shape{match d{Dim::Known(k)=>{n=n.checked_mul(*k)?},Dim::Dynamic=>return None}}Some(n.checked_mul(eb)?)},
            Type::Opaque{..}=>None,
        }
    }

    fn eq_impl(&self, other:&Self)->bool{use Type::*;match(self,other){(Int(a,sa),Int(b,sb))=>a==b&&sa==sb,(Float(a),Float(b))=>a==b,(Index,Index)=>true,(Tuple(a),Tuple(b))=>a==b,(Tensor(a),Tensor(b))=>a.shape==b.shape&&a.elem==b.elem,(Opaque{dialect:da,name:na,payload:pa},Opaque{dialect:db,name:nb,payload:pb})=>{if da!=db||na!=nb{return false;}match(pa,pb){(None,None)=>true,(Some(x),Some(y))=>x.erased_eq(&**y),_=>false}},_=>false}}
    fn hash_impl<H:Hasher>(&self,state:&mut H){use Type::*;std::mem::discriminant(self).hash(state);match self{Int(w,s)=>{w.hash(state);s.hash(state);}Float(w)=>{w.hash(state);}Index=>{}Tuple(elems)=>{for t in elems{t.hash_impl(state);}}Tensor(TensorType{shape,elem})=>{for d in shape{d.hash(state);}elem.hash_impl(state);}Opaque{dialect,name,payload}=>{dialect.hash(state);name.hash(state);if let Some(p)=payload{p.erased_hash(state);}}}}
}
impl PartialEq for Type{fn eq(&self,other:&Self)->bool{self.eq_impl(other)}}
impl Eq for Type{}
impl Hash for Type{fn hash<H:Hasher>(&self,state:&mut H){self.hash_impl(state)}}

/* ===================== Values ===================== */

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ValueId(NonZeroU32);
impl ValueId{fn new(n:u32)->Self{Self(NonZeroU32::new(n).expect("ValueId must be nonzero"))} pub fn get(self)->u32{self.0.get()}}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct OpId(NonZeroU32);
impl OpId{fn new(n:u32)->Self{Self(NonZeroU32::new(n).expect("OpId must be nonzero"))} pub fn get(self)->u32{self.0.get()}}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ValueKind { OpResult { op: OpId, idx: u32 } /* Phase3: BlockArg { block: BlockId, idx: u32 } */ }

#[derive(Clone, Debug)]
pub struct Value { id: ValueId, ty: Type }
impl Value { 
    pub fn id(&self)->ValueId{self.id} pub fn ty(&self)->&Type{&self.ty}
    pub fn uses<'a>(&'a self, cx: &'a IRContext) -> impl Iterator<Item = &'a Use> {
        cx.users(self.id)
    }
}

pub trait ValueLike { fn id(&self)->ValueId; fn ty(&self)->&Type; }
impl ValueLike for Value { fn id(&self)->ValueId{self.id} fn ty(&self)->&Type{&self.ty} }

/* ===================== Attributes (minimal) ===================== */

#[derive(Clone, Debug, PartialEq)]
pub enum Attr { Bool(bool), I64(i64), F64(f64), Str(Symbol), Type(Type), Opaque { dialect: Symbol, name: Symbol, payload: Option<Arc<dyn ErasedHashEq>> } }

type AttrMap = HashMap<Symbol, Attr>;

/* ===================== Operation & IR ===================== */

#[derive(Debug)]
pub struct Operation { pub id: OpId, pub opcode: Symbol, pub attrs: AttrMap, pub inputs: Vec<ValueId>, pub results: Vec<ValueId> }

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Use { pub user: OpId, pub input_index: u32 }

#[derive(Debug)]
pub struct IRContext {
    next_value: AtomicU32,
    next_op: AtomicU32,
    kinds: HashMap<ValueId, ValueKind>,
    types: HashMap<ValueId, Type>,
    uses: HashMap<ValueId, Vec<Use>>,
    ops: HashMap<OpId, Operation>,
}

impl IRContext {
    pub fn new()->Self{Self{ next_value:AtomicU32::new(1), next_op:AtomicU32::new(1), kinds:HashMap::new(), types:HashMap::new(), uses:HashMap::new(), ops:HashMap::new() }}
    fn alloc_value(&self)->ValueId{ let id=self.next_value.fetch_add(1,Ordering::Relaxed); ValueId::new(id) }
    fn alloc_op(&self)->OpId{ let id=self.next_op.fetch_add(1,Ordering::Relaxed); OpId::new(id) }

    /* ---- Phase 1 API (updated) ---- */
    pub fn make_op_result(&mut self, op: OpId, idx: u32, ty: Type)->Value{ let vid=self.alloc_value(); self.kinds.insert(vid, ValueKind::OpResult{op,idx}); self.types.insert(vid, ty.clone()); Value{ id:vid, ty } }
    pub fn value_kind(&self, v: ValueId)->Option<&ValueKind>{ self.kinds.get(&v) }
    pub fn value_type(&self, v: ValueId)->Option<&Type>{ self.types.get(&v) }
    pub fn users(&self, v: ValueId)->impl Iterator<Item=&Use>{ self.uses.get(&v).into_iter().flatten() }

    /* ---- Phase 2: Operation construction ---- */
    pub fn build_op(&mut self, opcode: impl Into<Symbol>, inputs: &[ValueId], result_types: &[Type], attrs: AttrMap) -> OpId {
        let id=self.alloc_op();
        for (i,&vin) in inputs.iter().enumerate(){ self.uses.entry(vin).or_default().push(Use{user:id, input_index:i as u32}); }
        let mut results=Vec::with_capacity(result_types.len());
        for (i,ty) in result_types.iter().cloned().enumerate(){ let v=self.make_op_result(id, i as u32, ty); results.push(v.id()); }
        let op=Operation{ id, opcode: opcode.into(), attrs, inputs: inputs.to_vec(), results };
        self.ops.insert(id, op);
        id
    }

    pub fn op(&self, id: OpId)->Option<&Operation>{ self.ops.get(&id) }
    pub fn op_mut(&mut self, id: OpId)->Option<&mut Operation>{ self.ops.get_mut(&id) }

    /// Replace-all-uses-of (RAUW): replace every use of `from` with `to`.
    pub fn rauw(&mut self, from: ValueId, to: ValueId){ if from==to { return; } let uses = match self.uses.remove(&from) { Some(u)=>u, None=>return }; for u in &uses { if let Some(op) = self.ops.get_mut(&u.user) { let idx = u.input_index as usize; op.inputs[idx]=to; } self.uses.entry(to).or_default().push(Use{ user: u.user, input_index: u.input_index }); } }
}

/* ===================== Example Dialect Payload ===================== */

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct LayoutPayload { pub tag: Symbol }

/* ===================== Tests ===================== */

#[cfg(test)]
mod tests { use super::*; use std::collections::hash_map::DefaultHasher;
    #[test] fn symbol_roundtrip(){ let s1=Symbol::new("linalg"); let s2=Symbol::from("linalg"); assert_eq!(s1,s2); assert_eq!(s1.as_str(),"linalg"); }
    #[test] fn type_helpers_and_sizes(){ let i32t=Type::i(32,true); let f32t=Type::f(32); assert!(i32t.is_int()); assert!(f32t.is_float()); assert_eq!(i32t.scalar_bit_width(),Some(32)); assert_eq!(f32t.byte_size(),Some(4)); let t=Type::tensor([Dim::Known(2),Dim::Known(3)], Type::f(64)); assert_eq!(t.byte_size(),Some(2*3*8)); let dyn_t=Type::tensor([Dim::Known(2),Dim::Dynamic], Type::i(8,false)); assert_eq!(dyn_t.byte_size(),None); let tup=Type::tuple(vec![Type::i(8,true),Type::f(32)]); assert_eq!(tup.byte_size(),Some(1+4)); }
    #[test] fn opaque_with_payload_eq_hash(){ let a=Type::opaque_with("core","layout", LayoutPayload{tag:Symbol::new("NHWC")}); let b=Type::opaque_with("core","layout", LayoutPayload{tag:Symbol::new("NHWC")}); let c=Type::opaque_with("core","layout", LayoutPayload{tag:Symbol::new("NCHW")}); assert_eq!(a,b); assert_ne!(a,c); let mut ha=DefaultHasher::new(); let mut hb=DefaultHasher::new(); a.hash(&mut ha); b.hash(&mut hb); assert_eq!(ha.finish(),hb.finish()); }
    #[test] fn build_simple_op_and_rauw(){ let mut cx=IRContext::new(); let op_const_a=cx.build_op(Symbol::new("core.const"), &[], &[Type::f(32)], AttrMap::new()); let va = cx.op(op_const_a).unwrap().results[0]; let op_const_b=cx.build_op(Symbol::new("core.const"), &[], &[Type::f(32)], AttrMap::new()); let vb = cx.op(op_const_b).unwrap().results[0]; let add_id=cx.build_op(Symbol::new("arith.addf"), &[va, vb], &[Type::f(32)], AttrMap::new()); let add = cx.op(add_id).unwrap(); assert_eq!(add.inputs.len(),2); assert_eq!(add.results.len(),1); cx.rauw(vb, va); let add = cx.op(add_id).unwrap(); assert_eq!(add.inputs[0], va); assert_eq!(add.inputs[1], va); let res = add.results[0]; let ty = cx.value_type(res).unwrap(); assert_eq!(ty, &Type::f(32)); }
}
