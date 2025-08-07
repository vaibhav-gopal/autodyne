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

pub trait IRValue: std::fmt::Debug + Send + Sync + 'static {
    fn type_name(&self) -> &'static str;
    fn clone_boxed(&self) -> Box<dyn IRValue>;
    fn eq_boxed(&self, other: &dyn IRValue) -> bool;
    fn as_bytes(&self) -> &[u8];
    fn size_bytes(&self) -> usize;
}

pub struct Node {
    pub id: OperationId,
    pub dialect: DialectId,
    pub op_type: OperationType,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub attributes: HashMap<String, Attribute>,
}
