//! FIR design: windowed sinc (`firwin`), frequency sampling (`firwin2`), least squares (`firls`),
//! equiripple Parks-McClellan (`remez`), and Kaiser window estimates.

use std::f64::consts::PI;

use super::DesignError;
use crate::spectral::{get_window, RealFft, WindowSpec};
use crate::units::Complex;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The Kaiser `beta` for `attenuation` dB of stopband attenuation (`scipy.signal.kaiser_beta`).
pub fn kaiser_beta(attenuation: f64) -> f64 {
    if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation > 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    }
}

/// The attenuation in dB of a Kaiser FIR with `numtaps` taps and transition `width` (a fraction of
/// the Nyquist frequency) (`scipy.signal.kaiser_atten`).
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    2.285 * (numtaps as f64 - 1.0) * PI * width + 7.95
}

/// The taps and Kaiser `beta` meeting `ripple` dB (positive) with transition `width` (a fraction of
/// the Nyquist frequency) (`scipy.signal.kaiserord`).
pub fn kaiserord(ripple: f64, width: f64) -> Result<(usize, f64), DesignError> {
    let a = ripple.abs();
    if a < 8.0 {
        return Err(DesignError::Invalid(format!("requested maximum ripple attenuation {a} is too small for the Kaiser formula")));
    }
    let numtaps = (a - 7.95) / 2.285 / (PI * width) + 1.0;
    Ok((numtaps.ceil() as usize, kaiser_beta(a)))
}

/// A linear-phase FIR by the window method (`scipy.signal.firwin`): `cutoff` edges in Hz at sample
/// rate `fs` (increasing, between 0 and Nyquist), `pass_zero` true if the band containing 0 Hz
/// passes (one edge: low-pass, else high-pass; two: band-stop, else band-pass; and so on), `scale`
/// to normalize the passband gain to exactly 1.
pub fn firwin(numtaps: usize, cutoff: &[f64], window: WindowSpec, pass_zero: bool, scale: bool, fs: f64) -> Result<Vec<f64>, DesignError> {
    let nyq = fs / 2.0;
    if cutoff.is_empty() {
        return Err(DesignError::Invalid("at least one cutoff frequency must be given".into()));
    }
    let cut: Vec<f64> = cutoff.iter().map(|c| c / nyq).collect();
    if cut.iter().any(|&c| c <= 0.0 || c >= 1.0) {
        return Err(DesignError::Invalid(format!("cutoffs must be between 0 and the Nyquist frequency {nyq} Hz")));
    }
    if cut.windows(2).any(|w| w[1] <= w[0]) {
        return Err(DesignError::Invalid("cutoffs must strictly increase".into()));
    }
    let pass_nyquist = (cut.len() % 2 == 1) ^ pass_zero;
    if pass_nyquist && numtaps.is_multiple_of(2) {
        return Err(DesignError::Invalid("a filter with an even number of taps must have zero response at the Nyquist frequency".into()));
    }
    let edges: Vec<f64> = std::iter::once(0.0).filter(|_| pass_zero).chain(cut).chain(std::iter::once(1.0).filter(|_| pass_nyquist)).collect();
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let m: Vec<f64> = (0..numtaps).map(|i| i as f64 - alpha).collect();
    let mut h = vec![0.0; numtaps];
    for band in edges.chunks(2) {
        let (left, right) = (band[0], band[1]);
        for (hi, &mi) in h.iter_mut().zip(&m) {
            *hi += right * sinc(right * mi) - left * sinc(left * mi);
        }
    }
    let win = get_window(window, numtaps, false);
    for (hi, wi) in h.iter_mut().zip(&win) {
        *hi *= wi;
    }
    if scale {
        let (left, right) = (edges[0], edges[1]);
        let f = if left == 0.0 {
            0.0
        } else if right == 1.0 {
            1.0
        } else {
            0.5 * (left + right)
        };
        let s: f64 = h.iter().zip(&m).map(|(hi, mi)| hi * (PI * mi * f).cos()).sum();
        for hi in &mut h {
            *hi /= s;
        }
    }
    Ok(h)
}

/// A FIR by frequency sampling (`scipy.signal.firwin2`): the gain is piecewise linear through
/// `(freq, gain)` (Hz, from 0 to Nyquist; a frequency may repeat once for a step), sampled on
/// `nfreqs` points (default `1 + 2^ceil(log2 numtaps)`), inverse transformed and windowed.
/// `antisymmetric` gives type III / IV filters (90° phase).
pub fn firwin2(numtaps: usize, freq: &[f64], gain: &[f64], nfreqs: Option<usize>, window: Option<WindowSpec>, antisymmetric: bool, fs: f64) -> Result<Vec<f64>, DesignError> {
    let nyq = fs / 2.0;
    let bad = |m: &str| Err(DesignError::Invalid(m.into()));
    if freq.len() != gain.len() {
        return bad("freq and gain must have the same length");
    }
    if numtaps < 3 {
        return bad("numtaps must be at least 3");
    }
    if freq.len() < 2 || freq[0] != 0.0 || freq[freq.len() - 1] != nyq {
        return bad("freq must start at 0 and end at the Nyquist frequency");
    }
    let d: Vec<f64> = freq.windows(2).map(|w| w[1] - w[0]).collect();
    if d.iter().any(|&x| x < 0.0) {
        return bad("freq must be nondecreasing");
    }
    if d.windows(2).any(|w| w[0] + w[1] == 0.0) {
        return bad("a frequency must not occur more than twice");
    }
    if freq[1] == 0.0 || freq[freq.len() - 2] == nyq {
        return bad("0 and the Nyquist frequency must not repeat");
    }
    let ftype = match (antisymmetric, numtaps % 2) {
        (false, 1) => 1,
        (false, _) => 2,
        (true, 1) => 3,
        (true, _) => 4,
    };
    let last = gain[gain.len() - 1];
    if (ftype == 2 && last != 0.0) || (ftype == 3 && (gain[0] != 0.0 || last != 0.0)) || (ftype == 4 && gain[0] != 0.0) {
        return bad("this filter type needs zero gain at 0 Hz and/or the Nyquist frequency");
    }
    let nfreqs = nfreqs.unwrap_or(1 + (numtaps as f64).log2().ceil().exp2() as usize);
    if numtaps >= nfreqs {
        return bad("nfreqs must be larger than numtaps");
    }
    // nudge repeated frequencies apart so interpolation sees a step
    let mut freq = freq.to_vec();
    let eps = f64::EPSILON * nyq;
    for k in 0..freq.len() - 1 {
        if freq[k] == freq[k + 1] {
            freq[k] -= eps;
            freq[k + 1] += eps;
        }
    }
    let interp = |x: f64| -> f64 {
        let i = freq.partition_point(|&f| f <= x).clamp(1, freq.len() - 1);
        let (f0, f1) = (freq[i - 1], freq[i]);
        if f1 == f0 { gain[i] } else { gain[i - 1] + (gain[i] - gain[i - 1]) * (x - f0) / (f1 - f0) }
    };
    let spectrum: Vec<Complex<f64>> = (0..nfreqs)
        .map(|i| {
            let x = nyq * i as f64 / (nfreqs - 1) as f64;
            let fx = interp(x.min(nyq));
            let mut shift = Complex::cis(-(numtaps as f64 - 1.0) / 2.0 * PI * x / nyq);
            if ftype > 2 {
                shift *= Complex::i();
            }
            shift * fx
        })
        .collect();
    let n = 2 * (nfreqs - 1);
    let mut full = vec![0.0; n];
    RealFft::new(n).inverse(&spectrum, &mut full);
    let win = window.map(|w| get_window(w, numtaps, false)).unwrap_or_else(|| vec![1.0; numtaps]);
    let mut out: Vec<f64> = full[..numtaps].iter().zip(&win).map(|(a, b)| a * b).collect();
    if ftype == 3 {
        out[numtaps / 2] = 0.0;
    }
    Ok(out)
}

/// A linear-phase FIR minimizing the weighted squared error to a piecewise-linear response
/// (`scipy.signal.firls`): `bands` are `(start, end)` pairs in Hz, `desired` the gains at each
/// band's start and end, `weight` one per band (default 1). `numtaps` must be odd.
pub fn firls(numtaps: usize, bands: &[(f64, f64)], desired: &[(f64, f64)], weight: Option<&[f64]>, fs: f64) -> Result<Vec<f64>, DesignError> {
    if numtaps.is_multiple_of(2) || numtaps < 1 {
        return Err(DesignError::Invalid("numtaps must be odd".into()));
    }
    if bands.len() != desired.len() {
        return Err(DesignError::Invalid("one desired pair per band".into()));
    }
    let nyq = fs / 2.0;
    let bands: Vec<(f64, f64)> = bands.iter().map(|&(a, b)| (a / nyq, b / nyq)).collect();
    let ones = vec![1.0; bands.len()];
    let weight = weight.unwrap_or(&ones);
    if weight.len() != bands.len() || weight.iter().any(|&w| w < 0.0) {
        return Err(DesignError::Invalid("one non-negative weight per band".into()));
    }
    let m = (numtaps - 1) / 2;
    // q[n] = Σ_bands w (f1 sinc(f1 n) - f0 sinc(f0 n))
    let q: Vec<f64> = (0..numtaps)
        .map(|n| bands.iter().zip(weight).map(|(&(f0, f1), w)| w * (f1 * sinc(f1 * n as f64) - f0 * sinc(f0 * n as f64))).sum())
        .collect();
    // Q = toeplitz(q[..=m]) + hankel(q[..=m], q[m..])
    let size = m + 1;
    let qmat = crate::signal::NdArray::from_fn(&[size, size], |i| q[i[0].abs_diff(i[1])] + q[i[0] + i[1]]).expect("shape");
    let b: Vec<f64> = (0..size)
        .map(|n| {
            let nf = n as f64;
            bands
                .iter()
                .zip(desired)
                .zip(weight)
                .map(|((&(f0, f1), &(d0, d1)), w)| {
                    let slope = (d1 - d0) / (f1 - f0);
                    let c = d0 - f0 * slope;
                    let term = |f: f64| {
                        let mut t = f * (slope * f + c) * sinc(f * nf);
                        if n == 0 {
                            t -= slope * f * f / 2.0;
                        } else {
                            t += slope * (nf * PI * f).cos() / (PI * nf).powi(2);
                        }
                        t
                    };
                    w * (term(f1) - term(f0))
                })
                .sum()
        })
        .collect();
    let rhs = crate::signal::NdArray::from_vec(b, &[size]).expect("shape");
    let a = crate::linalg::solve(qmat.view(), rhs.view()).map_err(|e| DesignError::Invalid(e.to_string()))?;
    let a = a.as_slice();
    Ok(a[1..].iter().rev().copied().chain(std::iter::once(2.0 * a[0])).chain(a[1..].iter().copied()).collect())
}

/// What [`remez`] designs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemezType {
    /// Symmetric taps: low-pass, high-pass, band-pass, band-stop, multiband.
    Bandpass,
    /// Antisymmetric taps, error weighted by 1/f where the desired gain is non-zero.
    Differentiator,
    /// Antisymmetric taps (a 90° phase shift).
    Hilbert,
}

/// The equiripple (minimax) linear-phase FIR by the Remez exchange algorithm, Parks-McClellan
/// (`scipy.signal.remez`): `bands` are `(start, end)` pairs in Hz at sample rate `fs`, `desired`
/// one gain per band, `weight` one per band (default 1). `maxiter` and `grid_density` are SciPy's
/// (25 and 16 by default there).
#[allow(clippy::too_many_arguments)] // SciPy's signature
pub fn remez(
    numtaps: usize,
    bands: &[(f64, f64)],
    desired: &[f64],
    weight: Option<&[f64]>,
    kind: RemezType,
    maxiter: usize,
    grid_density: usize,
    fs: f64,
) -> Result<Vec<f64>, DesignError> {
    if bands.len() != desired.len() || bands.is_empty() {
        return Err(DesignError::Invalid("one desired gain per band".into()));
    }
    let ones = vec![1.0; bands.len()];
    let weight = weight.unwrap_or(&ones);
    if weight.len() != bands.len() {
        return Err(DesignError::Invalid("one weight per band".into()));
    }
    let bands: Vec<(f64, f64)> = bands.iter().map(|&(a, b)| (a / fs, b / fs)).collect();
    if bands.iter().any(|&(a, b)| a < 0.0 || b > 0.5 || b < a) {
        return Err(DesignError::Invalid("bands must be within 0 and fs/2, each increasing".into()));
    }
    let positive = kind == RemezType::Bandpass;
    let odd = numtaps % 2 == 1;
    let mut r = numtaps / 2;
    if odd && positive {
        r += 1;
    }
    let delf = 0.5 / (grid_density * r) as f64;
    // dense grid
    let (mut grid, mut d, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for (bi, &(lo, hi)) in bands.iter().enumerate() {
        let lowf0 = if bi == 0 && !positive && delf > lo { delf } else { lo };
        let k = ((hi - lowf0) / delf + 0.5) as usize;
        let mut lowf = lowf0;
        for _ in 0..k {
            grid.push(lowf);
            d.push(desired[bi]);
            w.push(weight[bi]);
            lowf += delf;
        }
        if let Some(g) = grid.last_mut() {
            *g = hi;
        }
    }
    let gridsize = grid.len();
    if gridsize < r + 2 {
        return Err(DesignError::Invalid("the frequency grid is too coarse for this many taps".into()));
    }
    if !positive && odd && grid[gridsize - 1] > 0.5 - delf {
        grid[gridsize - 1] = 0.5 - delf;
    }
    if kind == RemezType::Differentiator {
        // a differentiator's gain grows with frequency: desired · f, weighted by 1/f
        for i in 0..gridsize {
            d[i] *= grid[i];
            if d[i] > 0.0001 {
                w[i] /= grid[i];
            }
        }
    }
    // fold the symmetry into the approximation
    for i in 0..gridsize {
        let c = match (positive, odd) {
            (true, true) => 1.0,
            (true, false) => (PI * grid[i]).cos(),
            (false, true) => (2.0 * PI * grid[i]).sin(),
            (false, false) => (PI * grid[i]).sin(),
        };
        d[i] /= c;
        w[i] *= c;
    }
    let mut ext: Vec<usize> = (0..=r).map(|i| i * (gridsize - 1) / r).collect();
    let (mut ad, mut x, mut y) = (vec![0.0; r + 1], vec![0.0; r + 1], vec![0.0; r + 1]);
    let mut e = vec![0.0; gridsize];
    // the cosine of every grid frequency, once
    let xgrid: Vec<f64> = grid.iter().map(|g| (2.0 * PI * g).cos()).collect();
    for _ in 0..maxiter {
        calc_parms(r, &ext, &grid, &d, &w, &mut ad, &mut x, &mut y);
        for i in 0..gridsize {
            e[i] = w[i] * (d[i] - interpolate(xgrid[i], r, &ad, &x, &y));
        }
        search(r, &mut ext, &e).map_err(|m| DesignError::Invalid(format!("remez failed to converge ({m}); try a wider transition band")))?;
        let errs: Vec<f64> = ext.iter().map(|&i| e[i].abs()).collect();
        let (lo, hi) = errs.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        if (hi - lo) / hi < 0.0001 {
            break;
        }
    }
    calc_parms(r, &ext, &grid, &d, &w, &mut ad, &mut x, &mut y);
    // sample the optimal response and unfold the symmetry
    let n = numtaps;
    let taps: Vec<f64> = (0..=n / 2)
        .map(|i| {
            let f = i as f64 / n as f64;
            let c = match (positive, odd) {
                (true, true) => 1.0,
                (true, false) => (PI * f).cos(),
                (false, true) => (2.0 * PI * f).sin(),
                (false, false) => (PI * f).sin(),
            };
            compute_a(f, r, &ad, &x, &y) * c
        })
        .collect();
    Ok(freq_sample(n, &taps, positive))
}

/// Barycentric weights, the deviation, and the interpolated values at the extremal frequencies
/// (Oppenheim & Schafer 7.131-7.133).
#[allow(clippy::too_many_arguments)]
fn calc_parms(r: usize, ext: &[usize], grid: &[f64], d: &[f64], w: &[f64], ad: &mut [f64], x: &mut [f64], y: &mut [f64]) {
    for i in 0..=r {
        x[i] = (2.0 * PI * grid[ext[i]]).cos();
    }
    let ld = (r - 1) / 15 + 1; // skips around to avoid rounding errors
    for i in 0..=r {
        let mut denom = 1.0;
        let xi = x[i];
        for j in 0..ld {
            let mut k = j;
            while k <= r {
                if k != i {
                    denom *= 2.0 * (xi - x[k]);
                }
                k += ld;
            }
        }
        if denom.abs() < 0.00001 {
            denom = 0.00001;
        }
        ad[i] = 1.0 / denom;
    }
    let (mut numer, mut denom, mut sign) = (0.0, 0.0, 1.0);
    for i in 0..=r {
        numer += ad[i] * d[ext[i]];
        denom += sign * ad[i] / w[ext[i]];
        sign = -sign;
    }
    let delta = numer / denom;
    sign = 1.0;
    for i in 0..=r {
        y[i] = d[ext[i]] - sign * delta / w[ext[i]];
        sign = -sign;
    }
}

/// The interpolated response at frequency `freq` (barycentric Lagrange through the r + 1 extremal
/// points, whose values lie on one polynomial of degree r - 1).
fn compute_a(freq: f64, r: usize, ad: &[f64], x: &[f64], y: &[f64]) -> f64 {
    interpolate((2.0 * PI * freq).cos(), r, ad, x, y)
}

/// [`compute_a`] at `xc = cos(2π f)`.
fn interpolate(xc: f64, r: usize, ad: &[f64], x: &[f64], y: &[f64]) -> f64 {
    let (mut numer, mut denom) = (0.0, 0.0);
    for i in 0..=r {
        let c = xc - x[i];
        if c.abs() < 1.0e-7 {
            return y[i];
        }
        let c = ad[i] / c;
        denom += c;
        numer += c * y[i];
    }
    numer / denom
}

/// Finds the r + 1 alternating extremal frequencies of the error.
fn search(r: usize, ext: &mut [usize], e: &[f64]) -> Result<(), &'static str> {
    let n = e.len();
    let mut found = Vec::with_capacity(2 * r);
    if (e[0] > 0.0 && e[0] > e[1]) || (e[0] < 0.0 && e[0] < e[1]) {
        found.push(0);
    }
    for i in 1..n - 1 {
        if (e[i] >= e[i - 1] && e[i] > e[i + 1] && e[i] > 0.0) || (e[i] <= e[i - 1] && e[i] < e[i + 1] && e[i] < 0.0) {
            if found.len() >= 2 * r {
                return Err("too many extremal frequencies");
            }
            found.push(i);
        }
    }
    let j = n - 1;
    if (e[j] > 0.0 && e[j] > e[j - 1]) || (e[j] < 0.0 && e[j] < e[j - 1]) {
        if found.len() >= 2 * r {
            return Err("too many extremal frequencies");
        }
        found.push(j);
    }
    if found.len() < r + 1 {
        return Err("too few extremal frequencies");
    }
    // remove extra extremals, keeping them alternating
    let mut extra = found.len() - (r + 1);
    while extra > 0 {
        let mut up = e[found[0]] > 0.0;
        let mut l = 0;
        let mut alternating = true;
        for jj in 1..found.len() {
            if e[found[jj]].abs() < e[found[l]].abs() {
                l = jj;
            }
            if up && e[found[jj]] < 0.0 {
                up = false;
            } else if !up && e[found[jj]] > 0.0 {
                up = true;
            } else {
                alternating = false;
                break;
            }
        }
        if alternating && extra == 1 {
            let k = found.len();
            l = if e[found[k - 1]].abs() < e[found[0]].abs() { k - 1 } else { 0 };
        }
        found.remove(l);
        extra -= 1;
    }
    ext.copy_from_slice(&found[..=r]);
    Ok(())
}

/// The taps from samples `a` of the amplitude response (frequency sampling). Antisymmetric taps
/// come out with SciPy's sign convention (its sine terms run from the far end).
fn freq_sample(n: usize, a: &[f64], positive: bool) -> Vec<f64> {
    let m = (n as f64 - 1.0) / 2.0;
    (0..n)
        .map(|i| {
            let x = 2.0 * PI * (i as f64 - m) / n as f64;
            let val = if positive {
                let top = if n % 2 == 1 { m as usize } else { n / 2 - 1 };
                a[0] + (1..=top).map(|k| 2.0 * a[k] * (x * k as f64).cos()).sum::<f64>()
            } else if n % 2 == 1 {
                (1..=m as usize).map(|k| 2.0 * a[k] * (x * k as f64).sin()).sum::<f64>()
            } else {
                a[n / 2] * (PI * (i as f64 - m)).sin() + (1..n / 2).map(|k| 2.0 * a[k] * (x * k as f64).sin()).sum::<f64>()
            };
            if positive { val / n as f64 } else { -val / n as f64 }
        })
        .collect()
}
