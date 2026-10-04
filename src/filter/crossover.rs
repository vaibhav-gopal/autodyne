//! Linkwitz-Riley crossovers: splitting a signal into bands that add back up flat.

use super::{Biquad, BiquadCoeffs, BUTTERWORTH_Q};
use crate::units::*;

/// Most bands a [`Crossover`] makes.
pub const MAX_BANDS: usize = 4;

/// A 4th-order (24 dB/octave) Linkwitz-Riley two-way split: each side is two cascaded Butterworth
/// sections, so both are -6 dB at the crossover and their sum has a perfectly flat magnitude (it is
/// a 2nd-order all-pass, see [`allpass`](Self::allpass)).
#[derive(Debug, Clone, Copy)]
pub struct LinkwitzRiley<T: Float> {
    frequency: T,
    sample_rate: T,
    low: [Biquad<T>; 2],
    high: [Biquad<T>; 2],
}

impl<T: Float> LinkwitzRiley<T> {
    pub fn new(frequency: T, sample_rate: T) -> Self {
        let f = clamp_frequency(frequency, sample_rate);
        let q = T::_lit(BUTTERWORTH_Q);
        let (lp, hp) = (Biquad::lowpass(f, q, sample_rate), Biquad::highpass(f, q, sample_rate));
        Self { frequency: f, sample_rate, low: [lp; 2], high: [hp; 2] }
    }
    /// Moves the crossover, keeping the filters' state (no click).
    pub fn set_frequency(&mut self, hz: T) {
        let f = clamp_frequency(hz, self.sample_rate);
        self.frequency = f;
        let q = T::_lit(BUTTERWORTH_Q);
        let (lp, hp) = (BiquadCoeffs::lowpass(f, q, self.sample_rate), BiquadCoeffs::highpass(f, q, self.sample_rate));
        self.low.iter_mut().for_each(|b| b.set_coeffs(lp));
        self.high.iter_mut().for_each(|b| b.set_coeffs(hp));
    }
    pub fn frequency(&self) -> T {
        self.frequency
    }
    /// The all-pass that the two bands sum to: run it on signals that bypass this split, so they
    /// stay in phase with the ones that went through it.
    pub fn allpass(&self) -> Biquad<T> {
        Biquad::new(BiquadCoeffs::allpass(self.frequency, T::_lit(BUTTERWORTH_Q), self.sample_rate))
    }
    pub fn reset(&mut self) {
        self.low.iter_mut().chain(self.high.iter_mut()).for_each(Biquad::reset);
    }
    pub fn flush_denormals(&mut self) {
        self.low.iter_mut().chain(self.high.iter_mut()).for_each(Biquad::flush_denormals);
    }
    /// (low band, high band) of one sample.
    #[inline]
    pub fn split(&mut self, x: T) -> (T, T) {
        let low = self.low[0].process_sample(x);
        let low = self.low[1].process_sample(low);
        let high = self.high[0].process_sample(x);
        let high = self.high[1].process_sample(high);
        (low, high)
    }
    /// |H| of the (low, high) bands at `frequency`.
    pub fn magnitudes_at(&self, frequency: T) -> (T, T) {
        let (l, h) = (self.low[0].magnitude_at(frequency, self.sample_rate), self.high[0].magnitude_at(frequency, self.sample_rate));
        (l * l, h * h)
    }
}

fn clamp_frequency<T: Float>(hz: T, sample_rate: T) -> T {
    hz._max(T::_lit(10.0))._min(T::_lit(0.45) * sample_rate)
}

/// Splits a signal into 2 to [`MAX_BANDS`] bands with Linkwitz-Riley crossovers whose sum is flat.
///
/// The splits cascade (low band off the first, the rest split again, ...); each lower band then
/// passes through the all-passes of the splits above it, so every band carries the same phase
/// shift and adding them back gives the input through one all-pass chain, unchanged in level at
/// every frequency. Changing a crossover keeps it between its neighbours.
#[derive(Debug, Clone, Copy)]
pub struct Crossover<T: Float> {
    splits: usize,
    lr: [LinkwitzRiley<T>; MAX_BANDS - 1],
    /// compensation[k][j]: band k's all-pass for split j (j > k)
    compensation: [[Biquad<T>; MAX_BANDS - 1]; MAX_BANDS - 1],
}

impl<T: Float> Crossover<T> {
    /// One band more than `frequencies` (1 to MAX_BANDS - 1 of them, ascending). Panics otherwise.
    pub fn new(frequencies: &[T], sample_rate: T) -> Self {
        assert!((1..MAX_BANDS).contains(&frequencies.len()), "a crossover takes 1 to {} frequencies", MAX_BANDS - 1);
        assert!(frequencies.windows(2).all(|w| w[0] < w[1]), "crossover frequencies must ascend");
        let lr = std::array::from_fn(|i| LinkwitzRiley::new(frequencies.get(i).copied().unwrap_or(T::_lit(1_000.0)), sample_rate));
        let mut c = Self { splits: frequencies.len(), lr, compensation: [[lr[0].allpass(); MAX_BANDS - 1]; MAX_BANDS - 1] };
        c.redesign_compensation();
        c
    }
    pub fn bands(&self) -> usize {
        self.splits + 1
    }
    pub fn frequency(&self, split: usize) -> T {
        self.lr[split].frequency()
    }
    /// Moves crossover `split`, kept at least a third of an octave from its neighbours.
    pub fn set_frequency(&mut self, split: usize, hz: T) {
        let gap = T::_lit(2f64.powf(1.0 / 3.0));
        let mut f = hz;
        if split > 0 {
            f = f._max(self.lr[split - 1].frequency() * gap);
        }
        if split + 1 < self.splits {
            f = f._min(self.lr[split + 1].frequency() / gap);
        }
        self.lr[split].set_frequency(f);
        self.redesign_compensation();
    }
    fn redesign_compensation(&mut self) {
        for k in 0..self.splits {
            for j in k + 1..self.splits {
                let coeffs = *self.lr[j].allpass().coeffs();
                self.compensation[k][j].set_coeffs(coeffs);
            }
        }
    }
    pub fn reset(&mut self) {
        self.lr.iter_mut().for_each(LinkwitzRiley::reset);
        self.compensation.iter_mut().flatten().for_each(Biquad::reset);
    }
    /// Call about once per block when splitting sample by sample: decayed states become exact zeros.
    pub fn flush_denormals(&mut self) {
        self.lr.iter_mut().for_each(LinkwitzRiley::flush_denormals);
        self.compensation.iter_mut().flatten().for_each(Biquad::flush_denormals);
    }
    /// Splits one sample into `bands()` values (the rest of `out` is zeroed).
    #[inline]
    pub fn split(&mut self, x: T, out: &mut [T; MAX_BANDS]) {
        *out = [T::_ZERO; MAX_BANDS];
        let mut rest = x;
        for (i, lr) in self.lr[..self.splits].iter_mut().enumerate() {
            let (low, high) = lr.split(rest);
            out[i] = low;
            rest = high;
        }
        out[self.splits] = rest;
        for (k, band) in out[..self.splits].iter_mut().enumerate() {
            for ap in &mut self.compensation[k][k + 1..self.splits] {
                *band = ap.process_sample(*band);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn sine(freq: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (std::f64::consts::TAU * freq * i as f64 / FS).sin()).collect()
    }
    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn linkwitz_riley_bands_are_minus_6_db_at_the_crossover_and_sum_flat() {
        let lr = LinkwitzRiley::new(1_000.0, FS);
        let (l, h) = lr.magnitudes_at(1_000.0);
        assert!((20.0 * l.log10() + 6.02).abs() < 0.01 && (20.0 * h.log10() + 6.02).abs() < 0.01);
        for freq in [50.0, 700.0, 1_000.0, 1_400.0, 9_000.0] {
            let mut lr = LinkwitzRiley::new(1_000.0, FS);
            let x = sine(freq, 48_000);
            let y: Vec<f64> = x.iter().map(|&s| {
                let (l, h) = lr.split(s);
                l + h
            }).collect();
            let gain_db = 20.0 * (rms(&y[24_000..]) / rms(&x[24_000..])).log10();
            assert!(gain_db.abs() < 1e-3, "{freq} Hz: {gain_db} dB");
        }
    }

    #[test]
    fn multiband_splits_recombine_flat_and_separate() {
        for frequencies in [&[1_000.0][..], &[200.0, 2_000.0], &[120.0, 1_000.0, 6_000.0]] {
            for freq in [40.0, 120.0, 400.0, 1_000.0, 3_000.0, 6_000.0, 15_000.0] {
                let mut c = Crossover::new(frequencies, FS);
                let x = sine(freq, 48_000);
                let mut bands = vec![vec![0.0; x.len()]; c.bands()];
                let mut sum = vec![0.0; x.len()];
                let mut out = [0.0; MAX_BANDS];
                for (i, &s) in x.iter().enumerate() {
                    c.split(s, &mut out);
                    for (b, band) in bands.iter_mut().enumerate() {
                        band[i] = out[b];
                    }
                    sum[i] = out.iter().sum();
                }
                let gain_db = 20.0 * (rms(&sum[24_000..]) / rms(&x[24_000..])).log10();
                assert!(gain_db.abs() < 1e-3, "{frequencies:?} at {freq} Hz: sum {gain_db} dB");
                // the tone lands mostly in its own band
                let band_of = frequencies.iter().filter(|&&f| f < freq).count();
                let loudest = (0..c.bands()).max_by(|&a, &b| rms(&bands[a][24_000..]).total_cmp(&rms(&bands[b][24_000..]))).unwrap();
                if frequencies.iter().all(|&f| (freq / f).log2().abs() > 0.5) {
                    assert_eq!(loudest, band_of, "{frequencies:?} at {freq} Hz");
                }
            }
        }
    }

    #[test]
    fn crossover_frequencies_stay_ordered() {
        let mut c = Crossover::new(&[200.0, 2_000.0, 8_000.0], FS);
        c.set_frequency(1, 50.0);
        assert!(c.frequency(1) > c.frequency(0) * 1.25);
        c.set_frequency(1, 20_000.0);
        assert!(c.frequency(1) < c.frequency(2) / 1.25);
    }
}
