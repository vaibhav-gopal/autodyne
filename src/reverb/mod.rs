//! Reverb: an algorithmic feedback delay network and FFT convolution.
//!
//! - [`Reverb`]: 8-line feedback delay network with diffusion, damping, pre-delay and decorrelated
//!   stereo. Cheap, fully adjustable, and its decay time is exact by construction.
//! - [`Convolver`]: partitioned FFT convolution with any impulse response (a recorded room, a
//!   cabinet, [`synthetic_ir`]), with a fixed latency of one block.
//!
//! tend: Audio / reverb

mod params;
mod convolver;
mod fdn;

pub use convolver::*;
pub use fdn::*;

use crate::osc::Noise;
use crate::units::*;

/// A simple room-like impulse response: noise under an exponential envelope that falls 60 dB in
/// `rt60` seconds, normalized to unit energy. Useful with a `Convolver` when there is no recorded
/// response at hand; different `seed`s give decorrelated responses (e.g. for left and right).
pub fn synthetic_ir<T: Float>(rt60: T, sample_rate: T, seed: u64) -> Vec<T> {
    let rt = rt60.to_f64().unwrap_or(1.0).max(0.01);
    let fs = sample_rate.to_f64().unwrap_or(48_000.0);
    let len = (rt * fs).ceil() as usize;
    let decay_per_sample = 10f64.powf(-3.0 / (rt * fs)); // amplitude: -60 dB after rt * fs samples
    let mut noise = Noise::<f64>::new(seed);
    let mut envelope = 1.0;
    let mut ir: Vec<f64> = (0..len)
        .map(|_| {
            let s = noise.next_sample() * envelope;
            envelope *= decay_per_sample;
            s
        })
        .collect();
    let norm = ir.iter().map(|x| x * x).sum::<f64>().sqrt();
    ir.iter_mut().for_each(|x| *x /= norm);
    ir.into_iter().map(T::_lit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;

    #[test]
    fn synthetic_ir_decays_at_the_requested_rate() {
        let ir: Vec<f64> = synthetic_ir(0.5, 48_000.0, 1);
        assert_eq!(ir.len(), 24_000);
        assert!((ir.energy() - 1.0).abs() < 1e-9);
        let (early, late) = (ir[..2_400].rms().unwrap(), ir[21_600..].rms().unwrap());
        // the last 10% sits ~54 dB below the first 10% (60 dB over the length, 10% windows)
        let drop_db = 20.0 * (early / late).log10();
        assert!((drop_db - 54.0).abs() < 4.0, "{drop_db:.1} dB");
        assert_ne!(synthetic_ir::<f32>(0.1, 48_000.0, 1), synthetic_ir::<f32>(0.1, 48_000.0, 2));
    }
}
