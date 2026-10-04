use crate::gain::SmoothedValue;
use crate::fft::RealFft;
use crate::units::*;

/// Convolution with a long impulse response (convolution reverb, cabinet or room simulation, long
/// FIR filters), done in the frequency domain.
///
/// Uniformly partitioned overlap-save: the impulse response is cut into `block`-sample partitions, each
/// transformed once in `new`. Each input block is transformed once, multiplied with every partition
/// against the matching past input block, and transformed back. Cost per block: two real FFTs of
/// `2 * block` plus one spectrum multiply-add per partition, instead of `block * ir_len` multiplies.
///
/// Output is delayed by `latency()` = `block` samples; the dry path is delayed to match, so `mix`
/// blends aligned signals. Any host block size works. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct Convolver<T: Float> {
    block: usize,
    fft: RealFft<T>,
    bins: usize,
    /// partition spectra, `partitions` x `bins`
    ir_spectra: Vec<Complex<T>>,
    /// spectra of the last `partitions` input frames (a ring)
    history: Vec<Complex<T>>,
    head: usize,
    partitions: usize,
    frame: Vec<T>,
    acc: Vec<Complex<T>>,
    result: Vec<T>,
    input: Vec<T>,
    wet: Vec<T>,
    dry: Vec<T>,
    pos: usize,
    mix: SmoothedValue<T>,
}

impl<T: Float> Convolver<T> {
    /// `block` (a power of two) trades latency for efficiency: 64-256 for live use, larger offline.
    /// Starts fully wet. Panics unless `block` is a power of two.
    pub fn new(ir: &[T], block: usize, sample_rate: T) -> Self {
        assert!(block.is_power_of_two(), "block must be a power of two, got {block}");
        let n = 2 * block;
        let mut fft = RealFft::new(n);
        let bins = fft.spectrum_len();
        let partitions = ir.len().div_ceil(block).max(1);
        let mut ir_spectra = vec![Complex::zero(); partitions * bins];
        let mut padded = vec![T::_ZERO; n];
        for (k, spectrum) in ir_spectra.chunks_exact_mut(bins).enumerate() {
            let part = ir.get(k * block..).map_or(&[][..], |rest| &rest[..rest.len().min(block)]);
            padded.iter_mut().for_each(|s| *s = T::_ZERO);
            padded[..part.len()].copy_from_slice(part);
            fft.forward(&padded, spectrum);
        }
        Self {
            block,
            fft,
            bins,
            ir_spectra,
            history: vec![Complex::zero(); partitions * bins],
            head: 0,
            partitions,
            frame: vec![T::_ZERO; n],
            acc: vec![Complex::zero(); bins],
            result: vec![T::_ZERO; n],
            input: vec![T::_ZERO; block],
            wet: vec![T::_ZERO; block],
            dry: vec![T::_ZERO; block],
            pos: 0,
            mix: SmoothedValue::new(T::_ONE).with_ramp_seconds(T::_lit(0.02), sample_rate),
        }
    }
    /// Delay of the output (wet and dry alike), in samples.
    pub fn latency(&self) -> usize {
        self.block
    }
    /// 0 = dry only (delayed by the latency), 1 = convolved only.
    pub fn set_mix(&mut self, mix: T) {
        self.mix.set_target(mix._clamp(T::_ZERO, T::_ONE));
    }
    /// Target dry / wet mix.
    pub fn mix(&self) -> T {
        self.mix.target()
    }
    /// Clears the convolution and the dry delay.
    pub fn reset(&mut self) {
        for buf in [&mut self.frame, &mut self.input, &mut self.wet, &mut self.dry] {
            buf.iter_mut().for_each(|s| *s = T::_ZERO);
        }
        self.history.iter_mut().for_each(|z| *z = Complex::zero());
        self.pos = 0;
        self.head = 0;
    }

    /// Consumes the collected input block and produces the next block of output.
    fn run_block(&mut self) {
        let (b, bins) = (self.block, self.bins);
        // overlap-save frame: the previous block followed by the new one
        self.frame.copy_within(b.., 0);
        self.frame[b..].copy_from_slice(&self.input);
        self.fft.forward(&self.frame, &mut self.history[self.head * bins..][..bins]);

        self.acc.iter_mut().for_each(|z| *z = Complex::zero());
        for k in 0..self.partitions {
            // partition k multiplies the input frame from k blocks ago
            let frame = (self.head + self.partitions - k) % self.partitions;
            let x = &self.history[frame * bins..][..bins];
            let h = &self.ir_spectra[k * bins..][..bins];
            for ((acc, &x), &h) in self.acc.iter_mut().zip(x).zip(h) {
                *acc += x * h;
            }
        }
        self.fft.inverse(&self.acc, &mut self.result);
        // the second half of the circular result is the valid linear convolution
        self.wet.copy_from_slice(&self.result[b..]);
        self.dry.copy_from_slice(&self.input);
        self.head = (self.head + 1) % self.partitions;
    }

    /// Processes `block` in place (any length).
    pub fn process(&mut self, block: &mut [T]) {
        for s in block {
            let (wet, dry) = (self.wet[self.pos], self.dry[self.pos]);
            self.input[self.pos] = *s;
            self.pos += 1;
            if self.pos == self.block {
                self.run_block();
                self.pos = 0;
            }
            *s = dry + self.mix.next_value() * (wet - dry);
        }
    }
}

crate::processor::forward_processor!(Convolver);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;
    use crate::osc::Noise;

    #[test]
    fn equals_direct_convolution_delayed_by_one_block() {
        let input: Vec<f64> = Noise::new(1).take(3_000).collect();
        for (ir_len, block) in [(1, 64), (5, 64), (64, 64), (1_000, 64), (300, 256), (37, 8)] {
            let ir: Vec<f64> = Noise::new(ir_len as u64).take(ir_len).collect();
            let reference = input.convolved(&ir);
            let mut conv = Convolver::new(&ir, block, 48_000.0);
            let mut out = input.clone();
            // uneven host block sizes, crossing the internal block boundaries
            let mut start = 0;
            for size in [1, 100, 7, 513, 64].iter().cycle() {
                if start == out.len() {
                    break;
                }
                let end = (start + size).min(out.len());
                conv.process(&mut out[start..end]);
                start = end;
            }
            for (n, &y) in out.iter().enumerate() {
                let expected = if n < block { 0.0 } else { reference[n - block] };
                assert!((y - expected).abs() < 1e-9, "ir {ir_len}, block {block}, sample {n}: {y} vs {expected}");
            }
        }
    }

    #[test]
    fn dry_mix_is_the_input_delayed() {
        let input: Vec<f64> = Noise::new(3).take(500).collect();
        let mut conv = Convolver::new(&[0.5, 0.25], 32, 48_000.0);
        conv.set_mix(0.0);
        conv.mix.set_immediate(0.0);
        let mut out = input.clone();
        conv.process(&mut out);
        assert!(out[..32].iter().all(|&s| s == 0.0));
        assert_eq!(&out[32..], &input[..468]);
    }
}
