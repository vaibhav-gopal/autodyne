﻿/// Module for extending the primitive types and introducing a system for working with primitive data (numbers)
/// todo:
/// - abstract SIMD operations (way later ; or just use nightly)
/// - implement the fixed-point type (way later)
/// - implement the bigint type (later)
/// - add tests (next)
/// - implement the complex type (next)
/// - add benchmarks (next)

mod cast;
pub use cast::*;

mod properties;
pub use properties::*;

mod ops;
pub use ops::*;

mod units;
pub use units::*;

mod integer;
pub use integer::*;

mod float;
pub use float::*;
