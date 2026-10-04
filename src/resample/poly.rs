//! Whole-array rational resampling, as `scipy.signal` has it: [`upfirdn`] (upsample, FIR filter,
//! downsample, in one polyphase pass) and [`resample_poly`] (`upfirdn` with a Kaiser-windowed
//! low-pass, aligned so the output starts at the input's first sample).

use thiserror::Error;

use crate::signal::{lanes_f64, NdArray, NdView};
use crate::units::*;

/// Errors from whole-array resampling.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResampleError {
    /// An argument is out of range (a zero rate, an empty filter).
    #[error("invalid resampling: {0}")]
    Invalid(String),
    /// An n-d layout error (an axis out of range).
    #[error(transparent)]
    Nd(#[from] crate::signal::NdError),
    /// Designing the anti-aliasing filter failed.
    #[cfg(feature = "faer")]
    #[error(transparent)]
    Filter(#[from] crate::filter::FilterError),
}

impl ResampleError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        ResampleError::Invalid(message.into())
    }
}

/// The number of outputs of [`upfirdn`] for a filter of `taps` taps and `len >= 1` inputs:
/// `ceil(((len - 1) up + taps) / down)`, the full convolution of the upsampled signal (without the
/// zeros after its last sample) decimated.
pub fn upfirdn_len(taps: usize, len: usize, up: usize, down: usize) -> usize {
    ((len - 1) * up + taps).div_ceil(down)
}

/// Upsamples each lane of `x` along `axis` by `up` (inserting `up - 1` zeros after each sample),
/// filters it with the FIR `h`, and keeps every `down`-th sample (`scipy.signal.upfirdn`). Computed
/// in one polyphase pass, so the inserted zeros and the dropped samples cost nothing, and
/// vectorized across outputs. The output has [`upfirdn_len`] samples.
pub fn upfirdn<T: Float + Default>(h: &[f64], x: NdView<'_, T>, up: usize, down: usize, axis: usize) -> Result<NdArray<T>, ResampleError> {
    if h.is_empty() {
        return Err(ResampleError::invalid("the filter is empty"));
    }
    if up == 0 || down == 0 {
        return Err(ResampleError::invalid(format!("up ({up}) and down ({down}) must be positive")));
    }
    let lanes = lanes_f64(&x, axis)?;
    let n = x.shape()[axis];
    let out_len = if n == 0 { 0 } else { upfirdn_len(h.len(), n, up, down) };
    let mut shape = x.shape().to_vec();
    shape[axis] = out_len;
    let mut out = NdArray::<T>::zeros(&shape)?;
    if out_len == 0 {
        return Ok(out);
    }
    let mut plan = Polyphase::new(h, up, down, n, out_len);
    let mut y = vec![0.0; out_len];
    for (lane, mut dst) in lanes.iter().zip(out.lanes_mut(axis)?) {
        plan.run(lane, &mut y);
        for (d, &v) in dst.iter_mut().zip(&y) {
            *d = T::_lit(v);
        }
    }
    Ok(out)
}

/// Outputs computed together by [`Polyphase`]: four AVX registers of f64.
const OUT_BLOCK: usize = 16;

/// `upfirdn` for lanes of one length, vectorized across outputs.
///
/// Output `m` sits at upsampled position `m down = base up + r` and is `sum_q h[r + up q]
/// x[base - q]`. With `g = gcd(up, down)`, the outputs `m0, m0 + up/g, m0 + 2 up/g, ...` (a
/// *phase*) share the branch `r` while `base` steps by `step = down/g`. So the input is split into
/// `step` interleaved streams (`stream_s[k] = x[k step + s]`, zero-padded in front), after which every
/// tap of a phase reads a contiguous run of one stream, and blocks of [`OUT_BLOCK`] outputs
/// accumulate tap by tap in registers, as in direct convolution.
struct Polyphase {
    /// outputs per lane
    out_len: usize,
    /// outputs between members of a phase (`up / g`)
    period: usize,
    /// stream length
    stream_len: usize,
    /// input samples between members of a phase, and the streams' count (`down / g`)
    step: usize,
    /// zeros before the signal (a multiple of `step`, at least the longest branch minus one)
    front: usize,
    /// for each phase: its first output and its taps as (coefficient, offset of its run in `streams`)
    phases: Vec<(usize, Vec<(f64, usize)>)>,
    /// the interleaved streams, `stream_len` each
    streams: Vec<f64>,
}

impl Polyphase {
    fn new(h: &[f64], up: usize, down: usize, n: usize, out_len: usize) -> Self {
        let g = gcd(up, down);
        let (period, step) = (up / g, down / g);
        let longest = h.len().div_ceil(up);
        let front = (longest - 1).div_ceil(step) * step;
        let per_phase = out_len.div_ceil(period).div_ceil(OUT_BLOCK) * OUT_BLOCK;
        // the furthest run starts at (front + base0) / step <= front / step + 1
        let stream_len = (front + n).div_ceil(step).max(front / step + 2 + per_phase) + 1;
        let phases = (0..period.min(out_len))
            .map(|m0| {
                let (base0, r) = (m0 * down / up, m0 * down % up);
                let taps = h.iter().skip(r).step_by(up).enumerate().map(|(q, &c)| {
                    let p = front + base0 - q;
                    (c, (p % step) * stream_len + p / step)
                });
                (m0, taps.collect())
            })
            .collect();
        Self { out_len, period, stream_len, step, front, phases, streams: vec![0.0; step * stream_len] }
    }

    /// Every output of one lane into `out` (AVX2 chosen once when the CPU has it).
    fn run(&mut self, lane: &[f64], out: &mut [f64]) {
        let (step, len, front) = (self.step, self.stream_len, self.front);
        self.streams.iter_mut().for_each(|v| *v = 0.0);
        for (i, &v) in lane.iter().enumerate() {
            let p = front + i;
            self.streams[(p % step) * len + p / step] = v;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2_available() {
            // SAFETY: the CPU was just checked for AVX2, the only feature `phases_avx2` is compiled with.
            unsafe { phases_avx2(self, out) };
            return;
        }
        phases_kernel(self, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn phases_avx2(plan: &Polyphase, out: &mut [f64]) {
    phases_kernel(plan, out)
}

#[inline(always)]
fn phases_kernel(plan: &Polyphase, out: &mut [f64]) {
    for (m0, taps) in &plan.phases {
        let count = (plan.out_len - m0).div_ceil(plan.period);
        for block in (0..count).step_by(OUT_BLOCK) {
            let mut acc = [0.0; OUT_BLOCK];
            for &(c, offset) in taps {
                let run = plan.streams[offset + block..].first_chunk::<OUT_BLOCK>().expect("streams padded past the last block");
                for t in 0..OUT_BLOCK {
                    acc[t] += c * run[t];
                }
            }
            for (t, &v) in acc.iter().enumerate().take(count - block) {
                out[m0 + (block + t) * plan.period] = v;
            }
        }
    }
}

/// Resamples each lane of `x` along `axis` by `up / down` (`scipy.signal.resample_poly` with
/// `padtype="constant"`): [`upfirdn`] with a low-pass of `20 max(up, down) + 1` taps cut off at
/// the lower Nyquist frequency, windowed by `window` (SciPy's default is
/// `WindowSpec::Kaiser { beta: 5.0 }`), and the filter's delay removed so output `k` lies at input
/// time `k down / up`. The output has `ceil(len up / down)` samples; the signal is taken as zero
/// outside.
#[cfg(feature = "faer")]
pub fn resample_poly<T: Float + Default>(x: NdView<'_, T>, up: usize, down: usize, axis: usize, window: crate::spectral::WindowSpec) -> Result<NdArray<T>, ResampleError> {
    if up == 0 || down == 0 {
        return Err(ResampleError::invalid(format!("up ({up}) and down ({down}) must be positive")));
    }
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return Ok(x.map(|&v| v));
    }
    let n = x.shape().get(axis).copied().ok_or(crate::signal::NdError::OutOfBounds)?;
    let wanted = (n * up).div_ceil(down);
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let mut h = crate::filter::design::firwin(2 * half_len + 1, &[1.0 / max_rate as f64], window, true, true, 2.0)?;
    h.iter_mut().for_each(|v| *v *= up as f64);
    // zero-pad the filter so its centre lands on an output sample
    let pre_pad = down - half_len % down;
    let mut post_pad = 0;
    let pre_remove = (half_len + pre_pad) / down;
    while upfirdn_len(h.len() + pre_pad + post_pad, n, up, down) < wanted + pre_remove {
        post_pad += 1;
    }
    let padded: Vec<f64> = std::iter::repeat_n(0.0, pre_pad).chain(h).chain(std::iter::repeat_n(0.0, post_pad)).collect();
    let y = upfirdn(&padded, x, up, down, axis)?;
    Ok(y.view().slice_axis(axis, pre_remove..pre_remove + wanted)?.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upfirdn_matches_upsample_filter_downsample() {
        let x: Vec<f64> = (0..23).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let h = [0.5, -1.0, 2.0, 0.25, 1.5];
        for (up, down) in [(1, 1), (3, 2), (2, 3), (1, 4), (5, 1), (4, 4)] {
            // the definition: zero-stuff, convolve, decimate
            let mut stuffed = vec![0.0; x.len() * up];
            for (i, &v) in x.iter().enumerate() {
                stuffed[i * up] = v;
            }
            let full = crate::signal::Signal::convolved(&stuffed[..], &h);
            let len = upfirdn_len(h.len(), x.len(), up, down);
            let want: Vec<f64> = (0..len).map(|m| full.get(m * down).copied().unwrap_or(0.0)).collect();
            let got = upfirdn(&h, NdArray::from_vec(x.clone(), &[x.len()]).unwrap().view(), up, down, 0).unwrap();
            assert_eq!(got.as_slice().len(), want.len(), "up {up} down {down}");
            assert!(got.as_slice().iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-12), "up {up} down {down}");
        }
        assert!(upfirdn::<f64>(&[], NdArray::zeros(&[3]).unwrap().view(), 1, 1, 0).is_err());
    }
}
