//! Decibels: the logarithmic level scale (`20 log10` of an amplitude ratio).

use super::Real;

/// Decibels to linear amplitude: 0 dB -> 1.0, +6.02 dB -> 2.0, -20 dB -> 0.1.
pub fn db_to_gain<T: Real>(db: T) -> T {
    T::lit(10.0).powf(db / T::lit(20.0))
}

/// Linear amplitude to decibels; 0.0 gives negative infinity.
pub fn gain_to_db<T: Real>(gain: T) -> T {
    T::lit(20.0) * gain.log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decibel_conversions() {
        assert_eq!(db_to_gain(0.0), 1.0);
        assert!((db_to_gain(-20.0f64) - 0.1).abs() < 1e-12);
        assert!((db_to_gain(20.0 * 2f64.log10()) - 2.0).abs() < 1e-12);
        for g in [0.001f64, 0.5, 1.0, 3.0] {
            assert!((db_to_gain(gain_to_db(g)) - g).abs() < 1e-12);
        }
        assert_eq!(gain_to_db(0.0f64), f64::NEG_INFINITY);
    }
}
