//! Spectral estimation along an axis of n-d data, as in `scipy.signal`: short-time Fourier transforms
//! ([`stft`], [`istft`]), power spectral densities ([`welch`], [`periodogram`]), cross spectra and
//! coherence ([`csd`], [`coherence`]), and [`spectrogram`]s.
//!
//! Inputs are views with any strides; the frequency axis replaces the input's `axis`, and segment
//! times (where there are several) form a new last axis. Computation is in f64.

use std::f64::consts::PI;

use thiserror::Error;

use super::{get_window, WindowSpec};
use crate::fft::{Fft, RealFft};
use crate::signal::{extended, lanes_f64, Edge, NdArray, NdView};
use crate::units::*;

/// Errors from spectral estimation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SpectralError {
    /// An argument is out of range or inconsistent with the others.
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// An n-d layout error (an axis out of range, a slice outside the input).
    #[error(transparent)]
    Nd(#[from] crate::signal::NdError),
}

impl SpectralError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        SpectralError::Invalid(message.into())
    }
}

/// Frequencies, segment times, and values (frequency on the input's axis, time last).
pub type TimeFrequency<E> = (Vec<f64>, Vec<f64>, NdArray<E>);

/// What is removed from each segment before transforming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detrend {
    /// Nothing.
    None,
    /// The mean.
    Constant,
    /// The least-squares line.
    Linear,
}

/// The units of power estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// Power spectral density (V²/Hz).
    Density,
    /// Power spectrum (V²).
    Spectrum,
}

/// How segment estimates are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Average {
    /// The mean (SciPy's default).
    Mean,
    /// Robust to outliers (bias-corrected, as in SciPy).
    Median,
}

/// How the signal is extended at both ends by half a segment before an STFT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Zeros.
    Zeros,
    /// A mirror image without repeating the end sample.
    Even,
    /// A point reflection about the end sample.
    Odd,
    /// The end sample repeated.
    Constant,
    /// No extension.
    None,
}

/// Segmenting and windowing parameters shared by the estimators; `None` fields take each
/// function's SciPy default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segments {
    /// The window applied to each segment.
    pub window: WindowSpec,
    /// Samples per segment (default 256, at most the signal length).
    pub nperseg: Option<usize>,
    /// Samples shared by consecutive segments.
    pub noverlap: Option<usize>,
    /// FFT length (zero-padded segments), at least `nperseg`.
    pub nfft: Option<usize>,
    /// What is removed from each segment.
    pub detrend: Detrend,
    /// Only the non-negative frequencies (real input).
    pub onesided: bool,
    /// The units of the result.
    pub scaling: Scaling,
}

impl Segments {
    /// `welch`'s defaults: Hann, 256 samples, half overlap, mean removed, density.
    pub fn welch() -> Self {
        Segments { window: WindowSpec::Hann, nperseg: None, noverlap: None, nfft: None, detrend: Detrend::Constant, onesided: true, scaling: Scaling::Density }
    }
    /// `spectrogram`'s defaults: Tukey 0.25, 256 samples, 1/8 overlap, mean removed, density.
    pub fn spectrogram() -> Self {
        Segments { window: WindowSpec::Tukey { alpha: 0.25 }, ..Self::welch() }
    }
    /// `stft`'s defaults: Hann, 256 samples, half overlap, no detrending, spectrum scaling.
    pub fn stft() -> Self {
        Segments { detrend: Detrend::None, scaling: Scaling::Spectrum, ..Self::welch() }
    }
    /// Samples per segment.
    pub fn nperseg(mut self, n: usize) -> Self {
        self.nperseg = Some(n);
        self
    }
    /// Samples shared by consecutive segments.
    pub fn noverlap(mut self, n: usize) -> Self {
        self.noverlap = Some(n);
        self
    }
    /// FFT length (zero-padded segments).
    pub fn nfft(mut self, n: usize) -> Self {
        self.nfft = Some(n);
        self
    }
    /// The window.
    pub fn window(mut self, w: WindowSpec) -> Self {
        self.window = w;
        self
    }
    /// What is removed from each segment.
    pub fn detrend(mut self, d: Detrend) -> Self {
        self.detrend = d;
        self
    }
    /// The units of the result.
    pub fn scaling(mut self, s: Scaling) -> Self {
        self.scaling = s;
        self
    }
    /// Only the non-negative frequencies.
    pub fn onesided(mut self, onesided: bool) -> Self {
        self.onesided = onesided;
        self
    }
}

/// Resolved segmenting.
struct Plan {
    win: Vec<f64>,
    nperseg: usize,
    noverlap: usize,
    nfft: usize,
    detrend: Detrend,
    onesided: bool,
    scale: f64,
}

impl Plan {
    fn new(seg: &Segments, len: usize, default_overlap: impl Fn(usize) -> usize, fs: f64, stft: bool) -> Result<Plan, SpectralError> {
        let nperseg = seg.nperseg.unwrap_or(256).min(len);
        if nperseg < 1 {
            return Err(SpectralError::invalid("nperseg must be at least 1"));
        }
        let nfft = seg.nfft.unwrap_or(nperseg);
        if nfft < nperseg {
            return Err(SpectralError::invalid("nfft must be at least nperseg"));
        }
        let noverlap = seg.noverlap.unwrap_or(default_overlap(nperseg));
        if noverlap >= nperseg {
            return Err(SpectralError::invalid("noverlap must be less than nperseg"));
        }
        let win = get_window(seg.window, nperseg, true);
        let mut scale = match seg.scaling {
            Scaling::Density => 1.0 / (fs * win.iter().map(|w| w * w).sum::<f64>()),
            Scaling::Spectrum => 1.0 / win.iter().sum::<f64>().powi(2),
        };
        if stft {
            scale = scale.sqrt();
        }
        Ok(Plan { win, nperseg, noverlap, nfft, detrend: seg.detrend, onesided: seg.onesided, scale })
    }

    fn nfreq(&self) -> usize {
        if self.onesided { self.nfft / 2 + 1 } else { self.nfft }
    }

    fn freqs(&self, fs: f64) -> Vec<f64> {
        let n = self.nfft as f64;
        (0..self.nfreq())
            .map(|k| {
                // two-sided: the second half are negative frequencies, as numpy.fft.fftfreq
                let k = if !self.onesided && k >= self.nfft.div_ceil(2) { k as f64 - n } else { k as f64 };
                k * fs / n
            })
            .collect()
    }

    /// The windowed transforms of every segment of `x`: `[segment][frequency]`.
    fn transforms(&self, x: &[f64], fwd: &mut Transform) -> Vec<Vec<Complex<f64>>> {
        let step = self.nperseg - self.noverlap;
        let count = if x.len() < self.noverlap { 0 } else { (x.len() - self.noverlap) / step };
        (0..count)
            .map(|s| {
                let mut seg = x[s * step..s * step + self.nperseg].to_vec();
                detrend(&mut seg, self.detrend);
                for (v, w) in seg.iter_mut().zip(&self.win) {
                    *v *= w;
                }
                seg.resize(self.nfft, 0.0);
                fwd.run(&seg)
            })
            .collect()
    }
}

/// A real-input FFT, one-sided or two-sided.
enum Transform {
    Real(RealFft<f64>),
    Full(Fft<f64>),
}

impl Transform {
    fn new(nfft: usize, onesided: bool) -> Self {
        if onesided { Transform::Real(RealFft::new(nfft)) } else { Transform::Full(Fft::new(nfft)) }
    }
    fn run(&mut self, x: &[f64]) -> Vec<Complex<f64>> {
        match self {
            Transform::Real(f) => {
                let mut out = vec![Complex::zero(); f.spectrum_len()];
                f.forward(x, &mut out);
                out
            }
            Transform::Full(f) => {
                let mut out = vec![Complex::zero(); x.len()];
                f.forward_real(x, &mut out);
                out
            }
        }
    }
}

fn detrend(x: &mut [f64], kind: Detrend) {
    let n = x.len() as f64;
    match kind {
        Detrend::None => {}
        Detrend::Constant => {
            let mean = x.iter().sum::<f64>() / n;
            x.iter_mut().for_each(|v| *v -= mean);
        }
        Detrend::Linear => {
            // least squares a + b t over t = 0..n
            let tm = (n - 1.0) / 2.0;
            let ym = x.iter().sum::<f64>() / n;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (i, &v) in x.iter().enumerate() {
                let dt = i as f64 - tm;
                sxy += dt * (v - ym);
                sxx += dt * dt;
            }
            let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            for (i, v) in x.iter_mut().enumerate() {
                *v -= ym + b * (i as f64 - tm);
            }
        }
    }
}

/// The lanes of `x` along `axis` as f64 vectors.
fn lanes<T: Float>(x: NdView<'_, T>, axis: usize) -> Result<Vec<Vec<f64>>, SpectralError> {
    Ok(lanes_f64(&x, axis)?)
}

/// Assembles per-lane results (`[nfreq * extra]`, frequency-major) into the shape of `x` with
/// `axis` replaced by `nfreq`, plus a trailing axis of `extra` when `extra` is `Some`.
fn assemble<E: Copy + Default>(shape: &[usize], axis: usize, nfreq: usize, extra: Option<usize>, per_lane: Vec<Vec<E>>) -> NdArray<E> {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = nfreq;
    let m = extra.unwrap_or(1);
    if let Some(e) = extra {
        out_shape.push(e);
    }
    let other: Vec<usize> = shape.iter().enumerate().filter(|&(i, _)| i != axis).map(|(_, &n)| n).collect();
    let strides = {
        let mut s = vec![1usize; out_shape.len()];
        for i in (0..out_shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };
    let mut data = vec![E::default(); out_shape.iter().product()];
    for (li, values) in per_lane.into_iter().enumerate() {
        // the lane's index over the other axes, row-major
        let mut rem = li;
        let mut idx = vec![0usize; other.len()];
        for k in (0..other.len()).rev() {
            idx[k] = rem % other[k];
            rem /= other[k];
        }
        let mut base = 0;
        let mut k = 0;
        for (d, &stride) in strides.iter().enumerate().take(shape.len()) {
            if d != axis {
                base += idx[k] * stride;
                k += 1;
            }
        }
        for f in 0..nfreq {
            for s in 0..m {
                data[base + f * strides[axis] + s] = values[f * m + s];
            }
        }
    }
    NdArray::from_vec(data, &out_shape).expect("valid shape")
}

/// One-sided densities double every bin but DC (and Nyquist for even lengths).
fn fold_onesided(p: &mut [f64], plan: &Plan) {
    if !plan.onesided {
        return;
    }
    let end = if plan.nfft.is_multiple_of(2) { p.len() - 1 } else { p.len() };
    for v in &mut p[1..end] {
        *v *= 2.0;
    }
}

fn median_bias(n: usize) -> f64 {
    1.0 + (1..=(n - 1) / 2).map(|i| 1.0 / (2 * i + 1) as f64 - 1.0 / (2 * i) as f64).sum::<f64>()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// The cross spectral density of `x` and `y` along `axis` (`scipy.signal.csd`): returns the
/// frequencies and `conj(X) Y` averaged over segments.
pub fn csd<T: Float + Default>(x: NdView<'_, T>, y: NdView<'_, T>, fs: f64, axis: usize, seg: &Segments, average: Average) -> Result<(Vec<f64>, NdArray<Complex<T>>), SpectralError> {
    if x.shape() != y.shape() {
        return Err(SpectralError::invalid("x and y must have the same shape"));
    }
    let (xl, yl) = (lanes(x, axis)?, lanes(y, axis)?);
    let len = x.shape()[axis];
    let plan = Plan::new(seg, len, |n| n / 2, fs, false)?;
    let mut fwd = Transform::new(plan.nfft, plan.onesided);
    let nfreq = plan.nfreq();
    let per_lane: Vec<Vec<Complex<T>>> = xl
        .iter()
        .zip(&yl)
        .map(|(a, b)| {
            let (xa, xb) = (plan.transforms(a, &mut fwd), plan.transforms(b, &mut fwd));
            let segs: Vec<Vec<Complex<f64>>> = xa.iter().zip(&xb).map(|(p, q)| p.iter().zip(q).map(|(u, v)| u.conj() * *v * plan.scale).collect()).collect();
            let mut re: Vec<f64> = vec![0.0; nfreq];
            let mut im: Vec<f64> = vec![0.0; nfreq];
            for f in 0..nfreq {
                let mut rs: Vec<f64> = segs.iter().map(|s| s[f].re).collect();
                let mut is: Vec<f64> = segs.iter().map(|s| s[f].im).collect();
                (re[f], im[f]) = match average {
                    Average::Median if segs.len() > 1 => {
                        let bias = median_bias(segs.len());
                        (median(&mut rs) / bias, median(&mut is) / bias)
                    }
                    _ => (rs.iter().sum::<f64>() / segs.len().max(1) as f64, is.iter().sum::<f64>() / segs.len().max(1) as f64),
                };
            }
            fold_onesided(&mut re, &plan);
            fold_onesided(&mut im, &plan);
            re.iter().zip(&im).map(|(&r, &i)| Complex::new(T::_lit(r), T::_lit(i))).collect()
        })
        .collect();
    Ok((plan.freqs(fs), assemble(x.shape(), axis, nfreq, None, per_lane)))
}

/// The power spectral density by Welch's method (`scipy.signal.welch`): the average of modified
/// periodograms of overlapping windowed segments.
pub fn welch<T: Float + Default>(x: NdView<'_, T>, fs: f64, axis: usize, seg: &Segments, average: Average) -> Result<(Vec<f64>, NdArray<T>), SpectralError> {
    // one transform per segment: |X|², not conj(X) X through csd
    let xl = lanes(x, axis)?;
    let len = x.shape()[axis];
    let plan = Plan::new(seg, len, |n| n / 2, fs, false)?;
    let mut fwd = Transform::new(plan.nfft, plan.onesided);
    let nfreq = plan.nfreq();
    let per_lane: Vec<Vec<T>> = xl
        .iter()
        .map(|lane| {
            let segs = plan.transforms(lane, &mut fwd);
            let mut p: Vec<f64> = match average {
                Average::Median if segs.len() > 1 => {
                    let bias = median_bias(segs.len());
                    (0..nfreq).map(|f| median(&mut segs.iter().map(|s| s[f].norm_sqr() * plan.scale).collect::<Vec<_>>()) / bias).collect()
                }
                _ => {
                    let mut acc = vec![0.0; nfreq];
                    for s in &segs {
                        for (a, z) in acc.iter_mut().zip(s) {
                            *a += z.norm_sqr();
                        }
                    }
                    acc.iter().map(|a| a * plan.scale / segs.len().max(1) as f64).collect()
                }
            };
            fold_onesided(&mut p, &plan);
            p.into_iter().map(T::_lit).collect()
        })
        .collect();
    Ok((plan.freqs(fs), assemble(x.shape(), axis, nfreq, None, per_lane)))
}

/// The periodogram: one segment covering the signal (`scipy.signal.periodogram`; window default
/// boxcar). `nfft` shorter than the signal truncates it, longer zero-pads.
pub fn periodogram<T: Float + Default>(x: NdView<'_, T>, fs: f64, axis: usize, window: WindowSpec, nfft: Option<usize>, detrend: Detrend, scaling: Scaling) -> Result<(Vec<f64>, NdArray<T>), SpectralError> {
    if axis >= x.ndim() {
        return Err(SpectralError::invalid(format!("axis {axis} is out of range for shape {:?}", x.shape())));
    }
    let len = x.shape()[axis];
    let (view, nperseg, nfft) = match nfft {
        Some(n) if n < len => (x.slice_axis(axis, 0..n)?, n, n),
        Some(n) => (x, len, n),
        None => (x, len, len),
    };
    let seg = Segments { window, nperseg: Some(nperseg), noverlap: Some(0), nfft: Some(nfft), detrend, onesided: true, scaling };
    welch(view, fs, axis, &seg, Average::Mean)
}

/// The magnitude-squared coherence `|Pxy|² / (Pxx Pyy)` (`scipy.signal.coherence`).
pub fn coherence<T: Float + Default>(x: NdView<'_, T>, y: NdView<'_, T>, fs: f64, axis: usize, seg: &Segments) -> Result<(Vec<f64>, NdArray<T>), SpectralError> {
    let (f, pxx) = welch(x, fs, axis, seg, Average::Mean)?;
    let (_, pyy) = welch(y, fs, axis, seg, Average::Mean)?;
    let (_, pxy) = csd(x, y, fs, axis, seg, Average::Mean)?;
    let data: Vec<T> = pxy.as_slice().iter().zip(pxx.as_slice().iter().zip(pyy.as_slice())).map(|(z, (&a, &b))| z.norm_sqr() / a / b).collect();
    Ok((f, NdArray::from_vec(data, pxx.shape()).expect("same shape")))
}

/// What a [`spectrogram`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrogramMode {
    /// Power (density or spectrum).
    Psd,
    /// Magnitude.
    Magnitude,
    /// Phase angle in radians.
    Angle,
    /// Phase, unwrapped along frequency.
    Phase,
}

/// A spectrogram (`scipy.signal.spectrogram`): frequencies, segment center times, and values with
/// frequency on `axis` and time last. Overlap defaults to 1/8 of a segment.
pub fn spectrogram<T: Float + Default>(x: NdView<'_, T>, fs: f64, axis: usize, seg: &Segments, mode: SpectrogramMode) -> Result<TimeFrequency<T>, SpectralError> {
    let xl = lanes(x, axis)?;
    let len = x.shape()[axis];
    let plan = Plan::new(seg, len, |n| n / 8, fs, mode != SpectrogramMode::Psd)?;
    let mut fwd = Transform::new(plan.nfft, plan.onesided);
    let nfreq = plan.nfreq();
    let mut nseg = 0;
    let per_lane: Vec<Vec<T>> = xl
        .iter()
        .map(|lane| {
            let segs = plan.transforms(lane, &mut fwd);
            nseg = segs.len();
            let mut out = vec![T::_ZERO; nfreq * nseg];
            let mut columns: Vec<Vec<f64>> = segs
                .iter()
                .map(|s| match mode {
                    SpectrogramMode::Psd => {
                        let mut p: Vec<f64> = s.iter().map(|z| z.norm_sqr() * plan.scale).collect();
                        fold_onesided(&mut p, &plan);
                        p
                    }
                    SpectrogramMode::Magnitude => s.iter().map(|z| z.norm() * plan.scale).collect(),
                    SpectrogramMode::Angle | SpectrogramMode::Phase => s.iter().map(|z| (*z * plan.scale).arg()).collect(),
                })
                .collect();
            if mode == SpectrogramMode::Phase {
                columns.iter_mut().for_each(|c| unwrap(c));
            }
            for (si, c) in columns.iter().enumerate() {
                for (f, &v) in c.iter().enumerate() {
                    out[f * nseg + si] = T::_lit(v);
                }
            }
            out
        })
        .collect();
    let times = segment_times(len, &plan, fs, false);
    Ok((plan.freqs(fs), times, assemble(x.shape(), axis, nfreq, Some(nseg), per_lane)))
}

/// Unwraps phase jumps larger than π (`numpy.unwrap`).
fn unwrap(p: &mut [f64]) {
    let mut offset = 0.0;
    for i in 1..p.len() {
        let d = p[i] + offset - p[i - 1];
        // numpy: ddmod = mod(d + π, 2π) - π, with ddmod = π where it is -π and d > 0
        let mut dd = (d + PI).rem_euclid(2.0 * PI) - PI;
        if dd == -PI && d > 0.0 {
            dd = PI;
        }
        let correction = dd - d;
        if d.abs() >= PI {
            offset += correction;
        }
        p[i] += offset;
    }
}

/// Segment center times for a signal of `len` samples (after any boundary extension when
/// `extended`).
fn segment_times(len: usize, plan: &Plan, fs: f64, extended: bool) -> Vec<f64> {
    let step = (plan.nperseg - plan.noverlap) as f64;
    let half = plan.nperseg as f64 / 2.0;
    let mut t = half;
    let mut out = Vec::new();
    while t <= len as f64 - half + 1e-9 {
        out.push((if extended { t - half } else { t }) / fs);
        t += step;
    }
    out
}

/// Options of [`stft`] beyond segmenting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StftOptions {
    /// Extension by half a segment at both ends (default zeros), so the first and last samples
    /// sit at a segment center.
    pub boundary: Boundary,
    /// Zero-pad the end to a whole number of segments (default true).
    pub padded: bool,
}

impl Default for StftOptions {
    fn default() -> Self {
        StftOptions { boundary: Boundary::Zeros, padded: true }
    }
}

/// The short-time Fourier transform (`scipy.signal.stft`): frequencies, segment times, and the
/// complex spectra with frequency on `axis` and time last.
pub fn stft<T: Float + Default>(x: NdView<'_, T>, fs: f64, axis: usize, seg: &Segments, options: StftOptions) -> Result<TimeFrequency<Complex<T>>, SpectralError> {
    let xl = lanes(x, axis)?;
    let len = x.shape()[axis];
    let plan = Plan::new(seg, len, |n| n / 2, fs, true)?;
    let mut fwd = Transform::new(plan.nfft, plan.onesided);
    let nfreq = plan.nfreq();
    let step = plan.nperseg - plan.noverlap;
    let prepare = |lane: &[f64]| -> Vec<f64> {
        let edge = match options.boundary {
                Boundary::Odd => Some(Edge::Odd),
                Boundary::Even => Some(Edge::Even),
                Boundary::Constant => Some(Edge::Constant),
                Boundary::Zeros => Some(Edge::Zeros),
                Boundary::None => None,
            };
            let mut v = edge.map_or_else(|| lane.to_vec(), |e| extended(lane, plan.nperseg / 2, e));
        if options.padded {
            let rem = (v.len() as isize - plan.nperseg as isize).rem_euclid(step as isize) as usize;
            let nadd = ((step - rem) % step) % plan.nperseg;
            v.resize(v.len() + nadd, 0.0);
        }
        v
    };
    let ext_len = prepare(&vec![0.0; len]).len();
    let mut nseg = 0;
    let per_lane: Vec<Vec<Complex<T>>> = xl
        .iter()
        .map(|lane| {
            let segs = plan.transforms(&prepare(lane), &mut fwd);
            nseg = segs.len();
            let mut out = vec![Complex::zero(); nfreq * nseg];
            for (si, s) in segs.iter().enumerate() {
                for (f, z) in s.iter().enumerate() {
                    out[f * nseg + si] = Complex::new(T::_lit(z.re * plan.scale), T::_lit(z.im * plan.scale));
                }
            }
            out
        })
        .collect();
    let times = segment_times(ext_len, &plan, fs, options.boundary != Boundary::None);
    Ok((plan.freqs(fs), times, assemble(x.shape(), axis, nfreq, Some(nseg), per_lane)))
}

/// Options of [`istft`] (SciPy's defaults when `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IstftOptions {
    /// The window the STFT used.
    pub window: WindowSpec,
    /// Samples per segment.
    pub nperseg: Option<usize>,
    /// Samples shared by consecutive segments.
    pub noverlap: Option<usize>,
    /// FFT length.
    pub nfft: Option<usize>,
    /// The spectra hold only non-negative frequencies (default true).
    pub onesided: bool,
    /// The STFT extended the signal at both ends (default true): remove it.
    pub boundary: bool,
    /// The scaling the STFT used.
    pub scaling: Scaling,
}

impl Default for IstftOptions {
    fn default() -> Self {
        IstftOptions { window: WindowSpec::Hann, nperseg: None, noverlap: None, nfft: None, onesided: true, boundary: true, scaling: Scaling::Spectrum }
    }
}

/// The inverse STFT by weighted overlap-add (`scipy.signal.istft`): `z` holds spectra with
/// frequency on axis `ndim - 2` and time last; returns sample times and the signal (its last axis
/// time). Errors if the window does not satisfy the nonzero overlap-add condition.
pub fn istft<T: Float + Default>(z: NdView<'_, Complex<T>>, fs: f64, options: IstftOptions) -> Result<(Vec<f64>, NdArray<T>), SpectralError> {
    let nd = z.ndim();
    if nd < 2 {
        return Err(SpectralError::invalid("istft needs at least two axes (frequency, time)"));
    }
    let (nfreq, nseg) = (z.shape()[nd - 2], z.shape()[nd - 1]);
    let n_default = if options.onesided { 2 * (nfreq - 1) } else { nfreq };
    let nperseg = options.nperseg.unwrap_or(n_default);
    let nfft = match options.nfft {
        Some(n) if n < nperseg => return Err(SpectralError::invalid("nfft must be at least nperseg")),
        Some(n) => n,
        None if options.onesided && nperseg == n_default + 1 => nperseg,
        None => n_default,
    };
    let noverlap = options.noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SpectralError::invalid("noverlap must be less than nperseg"));
    }
    let step = nperseg - noverlap;
    let win = get_window(options.window, nperseg, true);
    let gain = match options.scaling {
        Scaling::Spectrum => win.iter().sum::<f64>(),
        Scaling::Density => (fs * win.iter().map(|w| w * w).sum::<f64>()).sqrt(),
    };
    let outlen = nperseg + (nseg - 1) * step;
    let mut norm = vec![0.0; outlen];
    for s in 0..nseg {
        for (k, w) in win.iter().enumerate() {
            norm[s * step + k] += w * w;
        }
    }
    let (lo, hi) = if options.boundary { (nperseg / 2, outlen - nperseg / 2) } else { (0, outlen) };
    if norm[lo..hi].iter().any(|&v| v <= 1e-10) {
        return Err(SpectralError::invalid("the window fails the nonzero overlap-add condition: the STFT is not invertible"));
    }
    let batches: usize = z.shape()[..nd - 2].iter().product();
    let contiguous = z.to_owned();
    let data = contiguous.as_slice();
    let mut inv_real = if options.onesided { Some(RealFft::<f64>::new(nfft)) } else { None };
    let mut inv_full = if options.onesided { None } else { Some(Fft::<f64>::new(nfft)) };
    let mut out = Vec::with_capacity(batches * (hi - lo));
    for b in 0..batches {
        let mut x = vec![0.0; outlen];
        for s in 0..nseg {
            let column: Vec<Complex<f64>> = (0..nfreq)
                .map(|f| {
                    let v = data[b * nfreq * nseg + f * nseg + s];
                    Complex::new(v.re.to_f64().unwrap_or(f64::NAN), v.im.to_f64().unwrap_or(f64::NAN))
                })
                .collect();
            let segment: Vec<f64> = if let Some(fft) = &mut inv_real {
                // numpy's irfft takes the first n/2 + 1 bins, zero-filling missing ones
                let mut spec = vec![Complex::zero(); nfft / 2 + 1];
                for (d, s) in spec.iter_mut().zip(&column) {
                    *d = *s;
                }
                let mut seg = vec![0.0; nfft];
                fft.inverse(&spec, &mut seg);
                seg
            } else {
                let mut buf = vec![Complex::zero(); nfft];
                for (d, s) in buf.iter_mut().zip(&column) {
                    *d = *s;
                }
                inv_full.as_mut().expect("two-sided").inverse(&mut buf);
                buf.iter().map(|c| c.re).collect()
            };
            for k in 0..nperseg {
                x[s * step + k] += segment[k] * gain * win[k];
            }
        }
        out.extend(x[lo..hi].iter().zip(&norm[lo..hi]).map(|(v, n)| T::_lit(v / if *n > 1e-10 { *n } else { 1.0 })));
    }
    let mut shape = z.shape()[..nd - 2].to_vec();
    shape.push(hi - lo);
    let times = (0..hi - lo).map(|i| i as f64 / fs).collect();
    Ok((times, NdArray::from_vec(out, &shape).expect("valid shape")))
}
