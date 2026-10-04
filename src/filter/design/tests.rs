use super::*;
use crate::filter::FilterError;
use crate::systems::sosfreqz;

#[test]
fn digital_designs_hit_their_specifications() {
    let fs = 8_000.0;
    let db = |sos: &[[f64; 6]], f: f64| 20.0 * sosfreqz(sos, &[f], fs)[0].norm().log10();
    // Butterworth: -3 dB at the cutoff
    let sos = butter(6, Band::Lowpass(1_000.0), Design::Digital { fs }).unwrap().to_sos().unwrap();
    assert!((db(&sos, 1_000.0) + 3.0103).abs() < 1e-3);
    // Chebyshev I: the ripple at the edge
    let sos = cheby1(5, 1.0, Band::Highpass(1_000.0), Design::Digital { fs }).unwrap().to_sos().unwrap();
    assert!((db(&sos, 1_000.0) + 1.0).abs() < 1e-6);
    // Chebyshev II: the attenuation from the edge on
    let sos = cheby2(6, 40.0, Band::Lowpass(1_000.0), Design::Digital { fs }).unwrap().to_sos().unwrap();
    assert!((db(&sos, 1_000.0) + 40.0).abs() < 1e-6);
    assert!(db(&sos, 2_000.0) <= -40.0 + 1e-6);
    // elliptic: ripple in the passband, attenuation in the stopband
    let sos = ellip(4, 0.5, 60.0, Band::Bandpass(500.0, 1_500.0), Design::Digital { fs }).unwrap().to_sos().unwrap();
    assert!((db(&sos, 500.0) + 0.5).abs() < 1e-6 && (db(&sos, 1_500.0) + 0.5).abs() < 1e-6);
    assert!(db(&sos, 100.0) < -59.0 && db(&sos, 3_000.0) < -59.0);
    assert!(matches!(butter(2, Band::Lowpass(5_000.0), Design::Digital { fs }), Err(FilterError::Invalid(_))));
}

#[test]
fn fir_designs_have_their_gains() {
    let fs = 8_000.0;
    let taps = firwin(101, &[1_000.0, 2_000.0], crate::spectral::WindowSpec::Hamming, false, true, fs).unwrap();
    let gain = |f: f64| crate::systems::freqz(&taps, &[1.0], &[f], fs)[0].norm();
    assert!((gain(1_500.0) - 1.0).abs() < 1e-3);
    assert!(gain(200.0) < 0.01 && gain(3_000.0) < 0.01);
    // the equiripple low-pass: symmetric taps
    let eq = remez(41, &[(0.0, 1_000.0), (1_400.0, 4_000.0)], &[1.0, 0.0], None, RemezType::Bandpass, 25, 16, fs).unwrap();
    assert!(eq.iter().zip(eq.iter().rev()).all(|(a, b)| (a - b).abs() < 1e-12));
    let (n, beta) = kaiserord(60.0, 0.05).unwrap();
    assert!(n > 70 && beta > 5.0);
}
