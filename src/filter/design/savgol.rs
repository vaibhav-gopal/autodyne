//! Savitzky-Golay smoothing and differentiation filters (`scipy.signal.savgol_coeffs`): the FIR
//! that fits a polynomial to each window by least squares and evaluates it (or a derivative) at one
//! point.

use crate::filter::FilterError;
use crate::linalg::lstsq;
use crate::signal::NdArray;

/// The coefficients of a Savitzky-Golay filter for convolution (`scipy.signal.savgol_coeffs` with
/// `use="conv"`): a degree-`polyorder` polynomial fitted by least squares to `window_length`
/// samples `delta` apart, then its `deriv`-th derivative evaluated at position `pos` in the window
/// (`None`: the centre, `(window_length - 1) / 2`). Convolving a signal with them (centred) smooths
/// it, or differentiates it when `deriv > 0`; derivatives above `polyorder` are all zeros.
///
/// Errors unless `polyorder < window_length` and `0 <= pos < window_length`.
///
/// ```
/// use autodyne::filter::design::savgol_coeffs;
///
/// // the classic 5-point quadratic smoother: (-3, 12, 17, 12, -3) / 35
/// let c = savgol_coeffs(5, 2, 0, 1.0, None).unwrap();
/// assert!(c.iter().zip([-3.0, 12.0, 17.0, 12.0, -3.0]).all(|(c, w)| (c - w / 35.0).abs() < 1e-12));
/// ```
pub fn savgol_coeffs(window_length: usize, polyorder: usize, deriv: usize, delta: f64, pos: Option<f64>) -> Result<Vec<f64>, FilterError> {
    if polyorder >= window_length {
        return Err(FilterError::invalid(format!("polyorder ({polyorder}) must be less than window_length ({window_length})")));
    }
    let pos = pos.unwrap_or((window_length - 1) as f64 / 2.0);
    if !(0.0..window_length as f64).contains(&pos) {
        return Err(FilterError::invalid(format!("pos ({pos}) must be in [0, window_length)")));
    }
    if delta == 0.0 || !delta.is_finite() {
        return Err(FilterError::invalid("delta must be finite and nonzero"));
    }
    if deriv > polyorder {
        return Ok(vec![0.0; window_length]);
    }
    let cols = polyorder + 1;
    // A[o][i] = t_i^o at the window's positions relative to `pos`, reversed for convolution
    let t = |i: usize| (window_length - 1 - i) as f64 - pos;
    let a = NdArray::from_fn(&[cols, window_length], |ix| t(ix[1]).powi(ix[0] as i32)).expect("cols x window");
    // the fitted polynomial's deriv-th derivative at t = 0 is deriv! times its coefficient deriv
    let factorial: f64 = (1..=deriv).map(|k| k as f64).product();
    let mut y = vec![0.0; cols];
    y[deriv] = factorial / delta.powi(deriv as i32);
    let rhs = NdArray::from_vec(y, &[cols]).expect("a vector");
    // underdetermined: the minimum-norm solution
    Ok(lstsq(a.view(), rhs.view())?.solution.into_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn coefficients_match_scipy() {
        // scipy.signal.savgol_coeffs(5, 2), (7, 3, deriv=1, delta=0.5), (6, 2)
        let c = savgol_coeffs(5, 2, 0, 1.0, None).unwrap();
        assert!(close(&c, &[-0.0857142857142858, 0.3428571428571427, 0.4857142857142856, 0.34285714285714275, -0.0857142857142858], 1e-13));
        // the 7-point cubic derivative (22, -67, -58, 0, 58, 67, -22) / 252, reversed, over delta
        let c = savgol_coeffs(7, 3, 1, 0.5, None).unwrap();
        assert!(close(&c, &[-22.0 / 126.0, 67.0 / 126.0, 58.0 / 126.0, 0.0, -58.0 / 126.0, -67.0 / 126.0, 22.0 / 126.0], 1e-13));
        let c = savgol_coeffs(6, 2, 0, 1.0, None).unwrap();
        assert!(close(&c, &[-0.09375, 0.21875, 0.375, 0.375, 0.21875, -0.09375], 1e-13));
        assert_eq!(savgol_coeffs(5, 2, 3, 1.0, None).unwrap(), vec![0.0; 5]);
        assert!(savgol_coeffs(3, 3, 0, 1.0, None).is_err());
        assert!(savgol_coeffs(5, 2, 0, 1.0, Some(5.0)).is_err());
    }

    #[test]
    fn smoothing_preserves_polynomials_up_to_its_order() {
        let c = savgol_coeffs(9, 3, 0, 1.0, None).unwrap();
        // centred convolution of a cubic reproduces it
        let p = |t: f64| 0.2 * t * t * t - t * t + 3.0;
        let at = 10.0;
        let y: f64 = c.iter().enumerate().map(|(j, &cj)| cj * p(at + 4.0 - j as f64)).sum();
        assert!((y - p(at)).abs() < 1e-9);
        // the first derivative filter differentiates it
        let d = savgol_coeffs(9, 3, 1, 1.0, None).unwrap();
        let dy: f64 = d.iter().enumerate().map(|(j, &cj)| cj * p(at + 4.0 - j as f64)).sum();
        assert!((dy - (0.6 * at * at - 2.0 * at)).abs() < 1e-9);
    }
}
