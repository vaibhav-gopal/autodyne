//! Random processes: sample paths of [`colored_noise`] (power spectrum `1 / f^beta`),
//! [`brownian_motion`], [`geometric_brownian_motion`], [`ornstein_uhlenbeck`], [`arma`] and the
//! event times of a [`poisson_process`]. Each draws from the [`Rng`] it is given, so a seed
//! reproduces the path.

use super::{RandomError, Rng};
use crate::fft::RealFft;
use crate::units::*;

fn finite(name: &str, v: f64) -> Result<f64, RandomError> {
    if v.is_finite() { Ok(v) } else { Err(RandomError::invalid(format!("{name} must be finite, got {v}"))) }
}

fn non_negative(name: &str, v: f64) -> Result<f64, RandomError> {
    if v >= 0.0 && v.is_finite() { Ok(v) } else { Err(RandomError::invalid(format!("{name} must be finite and non-negative, got {v}"))) }
}

/// `n` samples of Gaussian noise whose power spectral density falls as `1 / f^beta`: 0 is white,
/// 1 pink, 2 brown (red), -1 blue, -2 violet; any real exponent works. Made by shaping a white
/// spectrum (independent normal real and imaginary parts, scaled by `f^(-beta/2)`; the lowest
/// frequency `1 / n` stands in for DC) and inverting it, then scaled to unit variance in
/// expectation (Timmer and König's method, as the `colorednoise` package does it).
pub fn colored_noise(rng: &mut Rng, beta: f64, n: usize) -> Result<Vec<f64>, RandomError> {
    finite("beta", beta)?;
    if n < 2 {
        return Ok((0..n).map(|_| rng.standard_normal()).collect());
    }
    let bins = n / 2 + 1;
    // f = k / n; below the lowest nonzero frequency, use it
    let scale: Vec<f64> = (0..bins).map(|k| (k.max(1) as f64 / n as f64).powf(-beta / 2.0)).collect();
    // the expected variance of the result, from the weights of the bins it sums
    let mut weights: Vec<f64> = scale[1..].to_vec();
    if let Some(last) = weights.last_mut() {
        *last *= (1 + n % 2) as f64 / 2.0;
    }
    let sigma = 2.0 * weights.iter().map(|w| w * w).sum::<f64>().sqrt() / n as f64;
    let mut spectrum: Vec<Complex<f64>> = scale.iter().map(|&s| Complex::new(s * rng.standard_normal(), s * rng.standard_normal())).collect();
    // DC (and Nyquist, for even n) are real: their power goes into the real part
    spectrum[0] = Complex::new(spectrum[0].re * std::f64::consts::SQRT_2, 0.0);
    if n.is_multiple_of(2) {
        let last = bins - 1;
        spectrum[last] = Complex::new(spectrum[last].re * std::f64::consts::SQRT_2, 0.0);
    }
    let mut y = vec![0.0; n];
    RealFft::<f64>::new(n).inverse(&spectrum, &mut y);
    y.iter_mut().for_each(|v| *v /= sigma);
    Ok(y)
}

/// `n` samples, `dt` apart, of Brownian motion (a Wiener process) with volatility `sigma`, starting
/// at 0: independent normal steps of variance `sigma² dt`.
pub fn brownian_motion(rng: &mut Rng, n: usize, dt: f64, sigma: f64) -> Result<Vec<f64>, RandomError> {
    let step = (non_negative("dt", dt)? * non_negative("sigma", sigma)?.powi(2)).sqrt();
    let mut x = 0.0;
    Ok((0..n)
        .map(|i| {
            if i > 0 {
                x += step * rng.standard_normal();
            }
            x
        })
        .collect())
}

/// `n` samples, `dt` apart, of geometric Brownian motion `dS = mu S dt + sigma S dW` from `s0`,
/// stepped exactly (`S` times the exponential of a normal increment, so never negative).
pub fn geometric_brownian_motion(rng: &mut Rng, n: usize, dt: f64, mu: f64, sigma: f64, s0: f64) -> Result<Vec<f64>, RandomError> {
    let (dt, sigma) = (non_negative("dt", dt)?, non_negative("sigma", sigma)?);
    let drift = (finite("mu", mu)? - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let mut s = finite("s0", s0)?;
    Ok((0..n)
        .map(|i| {
            if i > 0 {
                s *= (drift + vol * rng.standard_normal()).exp();
            }
            s
        })
        .collect())
}

/// `n` samples, `dt` apart, of the Ornstein-Uhlenbeck process `dx = theta (mu - x) dt + sigma dW`
/// from `x0`, stepped exactly: each step decays towards `mu` by `e^(-theta dt)` and adds normal noise
/// of the matching variance, so any `dt` is exact. Its stationary variance is `sigma² / (2 theta)`.
pub fn ornstein_uhlenbeck(rng: &mut Rng, n: usize, dt: f64, theta: f64, mu: f64, sigma: f64, x0: f64) -> Result<Vec<f64>, RandomError> {
    let (dt, sigma) = (non_negative("dt", dt)?, non_negative("sigma", sigma)?);
    if !(theta > 0.0 && theta.is_finite()) {
        return Err(RandomError::invalid(format!("theta must be positive and finite, got {theta}")));
    }
    let decay = (-theta * dt).exp();
    let noise = sigma * ((1.0 - decay * decay) / (2.0 * theta)).sqrt();
    let (mu, mut x) = (finite("mu", mu)?, finite("x0", x0)?);
    Ok((0..n)
        .map(|i| {
            if i > 0 {
                x = mu + (x - mu) * decay + noise * rng.standard_normal();
            }
            x
        })
        .collect())
}

/// `n` samples of an ARMA process driven by normal noise of standard deviation `sigma`
/// (`statsmodels.tsa.arima_process.arma_generate_sample`): `ar` and `ma` are the lag polynomials
/// with their leading coefficient (`ar = [1, -phi_1, ...]`, `ma = [1, theta_1, ...]`), so
/// `ar(L) y = ma(L) e`. The first `burnin` samples are generated and dropped, letting the process
/// forget its zero start.
pub fn arma(rng: &mut Rng, ar: &[f64], ma: &[f64], n: usize, sigma: f64, burnin: usize) -> Result<Vec<f64>, RandomError> {
    let a0 = *ar.first().ok_or_else(|| RandomError::invalid("ar needs its leading coefficient"))?;
    if a0 == 0.0 || ma.is_empty() {
        return Err(RandomError::invalid("ar[0] must be nonzero and ma non-empty"));
    }
    let sigma = non_negative("sigma", sigma)?;
    let total = n + burnin;
    let e: Vec<f64> = (0..total).map(|_| sigma * rng.standard_normal()).collect();
    let mut y = vec![0.0; total];
    for t in 0..total {
        let mut v: f64 = ma.iter().enumerate().take(t + 1).map(|(j, &b)| b * e[t - j]).sum();
        v -= ar.iter().enumerate().skip(1).take(t).map(|(i, &a)| a * y[t - i]).sum::<f64>();
        y[t] = v / a0;
    }
    y.drain(..burnin);
    Ok(y)
}

/// The event times in `[0, duration)` of a Poisson process with `rate` events per unit time
/// (exponential gaps of mean `1 / rate`), in increasing order.
pub fn poisson_process(rng: &mut Rng, rate: f64, duration: f64) -> Result<Vec<f64>, RandomError> {
    let (rate, duration) = (non_negative("rate", rate)?, non_negative("duration", duration)?);
    let mut times = Vec::new();
    if rate == 0.0 {
        return Ok(times);
    }
    let mut t = rng.standard_exponential() / rate;
    while t < duration {
        times.push(t);
        t += rng.standard_exponential() / rate;
    }
    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        (mean, x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n)
    }

    /// The least-squares slope of log power against log frequency, from averaged periodograms.
    fn spectral_slope(beta: f64) -> f64 {
        let (n, runs) = (4096, 40);
        let mut rng = Rng::new(3);
        let mut power = vec![0.0; n / 2 + 1];
        let mut fft = RealFft::<f64>::new(n);
        let mut spec = vec![Complex::zero(); n / 2 + 1];
        for _ in 0..runs {
            let x = colored_noise(&mut rng, beta, n).unwrap();
            fft.forward(&x, &mut spec);
            for (p, z) in power.iter_mut().zip(&spec) {
                *p += z.norm_sqr();
            }
        }
        let pts: Vec<(f64, f64)> = (4..n / 2).map(|k| ((k as f64).ln(), power[k].ln())).collect();
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64, pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64);
        pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>()
    }

    #[test]
    fn colored_noise_has_its_spectral_slope_and_unit_variance() {
        for beta in [0.0, 1.0, 2.0, -1.0] {
            let slope = spectral_slope(beta);
            assert!((slope + beta).abs() < 0.05, "beta {beta}: slope {slope}");
        }
        let mut rng = Rng::new(4);
        let var: f64 = (0..50).map(|_| moments(&colored_noise(&mut rng, 1.0, 2048).unwrap()).1).sum::<f64>() / 50.0;
        assert!((var - 1.0).abs() < 0.15, "pink variance {var}");
    }

    #[test]
    fn diffusions_have_their_moments() {
        let mut rng = Rng::new(5);
        // Brownian motion: Var W(t) = sigma² t
        let ends: Vec<f64> = (0..4000).map(|_| *brownian_motion(&mut rng, 101, 0.01, 2.0).unwrap().last().unwrap()).collect();
        let (mean, var) = moments(&ends);
        assert!(mean.abs() < 0.1 && (var - 4.0).abs() < 0.3, "W(1): {mean} {var}");
        // Ornstein-Uhlenbeck: stationary mean mu, variance sigma² / (2 theta)
        let x = ornstein_uhlenbeck(&mut rng, 200_000, 0.1, 0.5, 3.0, 1.0, 3.0).unwrap();
        let (mean, var) = moments(&x);
        assert!((mean - 3.0).abs() < 0.05 && (var - 1.0).abs() < 0.08, "OU: {mean} {var}");
        // geometric Brownian motion: E[S(t)] = s0 e^(mu t)
        let ends: Vec<f64> = (0..20_000).map(|_| *geometric_brownian_motion(&mut rng, 11, 0.1, 0.05, 0.2, 100.0).unwrap().last().unwrap()).collect();
        let (mean, _) = moments(&ends);
        assert!((mean - 100.0 * 0.05f64.exp()).abs() < 0.5, "GBM: {mean}");
        assert!(ends.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn arma_and_poisson_events() {
        let mut rng = Rng::new(6);
        // AR(1) with phi = 0.7: lag-1 autocorrelation 0.7, variance 1 / (1 - 0.49)
        let y = arma(&mut rng, &[1.0, -0.7], &[1.0], 100_000, 1.0, 100).unwrap();
        let (mean, var) = moments(&y);
        let lag1 = y.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f64>() / (y.len() as f64 * var);
        assert!((lag1 - 0.7).abs() < 0.01 && (var - 1.0 / 0.51).abs() < 0.05, "AR(1): {lag1} {var}");
        // MA(1): lag-1 autocorrelation theta / (1 + theta²)
        let y = arma(&mut rng, &[1.0], &[1.0, 0.5], 100_000, 1.0, 0).unwrap();
        let (mean, var) = moments(&y);
        let lag1 = y.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f64>() / (y.len() as f64 * var);
        assert!((lag1 - 0.4).abs() < 0.01, "MA(1): {lag1}");
        let t = poisson_process(&mut rng, 50.0, 200.0).unwrap();
        assert!((t.len() as f64 - 10_000.0).abs() < 400.0 && t.windows(2).all(|w| w[0] < w[1]) && t.iter().all(|&v| (0.0..200.0).contains(&v)));
        assert!(arma(&mut rng, &[0.0], &[1.0], 10, 1.0, 0).is_err());
        assert!(ornstein_uhlenbeck(&mut rng, 10, 0.1, 0.0, 0.0, 1.0, 0.0).is_err());
    }
}
