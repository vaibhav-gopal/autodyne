/// Module for extending the primitive types and introducing a system for working with primitive data (numbers)
/// todo:
/// - abstract SIMD operations (way later ; or just use nightly)
/// - implement the bigint, fixed-point and bigfloat types (way later)
/// - add tests and benchmarks (next)
/// - implement the complex type (later)

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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add() {
        let a = 2f32;
        let b = 3f32;
        assert_eq!(a + b, 5f32)
    }
    
    #[test]
    fn test_constants() {
        let pi = f32::_PI;
        let tau = f32::_TAU;
        assert_eq!(pi * 2.0, tau);
        
        let pi6 = f64::_PI;
        let tau6 = f64::_TAU;
        assert_eq!(pi6 * 2.0, tau6);
        
        assert_eq!(pi, pi6 as f32);
    }
}