//! autodyne against SciPy: reference outputs from `tests/scipy/gen_fixtures.py` (SciPy 1.18),
//! committed as `tests/scipy/fixtures.json`.

#![cfg(feature = "faer")]

use std::sync::OnceLock;

use autodyne::filter::design::{besselap, firls, firwin, firwin2, iirfilter, kaiserord, remez, Band, BesselNorm, Design, IirKind, RemezType};
use autodyne::filter::{filtfilt, lfilter, lfilter_with_state, lfilter_zi, sosfilt, sosfilt_with_state, sosfilt_zi, sosfiltfilt, Pad};
use autodyne::linalg::expm;
use autodyne::signal::NdArray;
use autodyne::spectral::{coherence, csd, get_window, istft, periodogram, spectrogram, stft, welch, Average, Boundary, Detrend, IstftOptions, Scaling, Segments, SpectrogramMode, StftOptions, WindowSpec};
use autodyne::systems::{self, bilinear, freqs, freqz, group_delay, sosfreqz, ss2tf, tf2ss, tf2zpk, zpk2sos, Domain, Method, Pairing, StateSpace, C64};
use serde_json::Value;

fn fixtures() -> &'static Value {
    static F: OnceLock<Value> = OnceLock::new();
    F.get_or_init(|| serde_json::from_str(include_str!("scipy/fixtures.json")).expect("valid JSON"))
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

/// A nested array of numbers, flattened row-major, with its shape.
fn nd(v: &Value) -> (Vec<f64>, Vec<usize>) {
    fn walk(v: &Value, out: &mut Vec<f64>, shape: &mut Vec<usize>, depth: usize) {
        match v {
            Value::Array(items) => {
                if shape.len() == depth {
                    shape.push(items.len());
                }
                for item in items {
                    walk(item, out, shape, depth + 1);
                }
            }
            x => out.push(x.as_f64().expect("number")),
        }
    }
    let (mut out, mut shape) = (Vec::new(), Vec::new());
    walk(v, &mut out, &mut shape, 0);
    (out, shape)
}

fn array(v: &Value) -> NdArray<f64> {
    let (data, shape) = nd(v);
    NdArray::from_vec(data, &shape).unwrap()
}

fn complexes(v: &Value) -> Vec<C64> {
    v.as_array().unwrap().iter().map(|p| C64::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect()
}

fn sos_of(v: &Value) -> Vec<[f64; 6]> {
    v.as_array().unwrap().iter().map(|row| f64s(row).try_into().unwrap()).collect()
}

fn assert_close(name: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length {} vs {}", got.len(), want.len());
    let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * scale, "{name}[{i}]: {g} vs {w} (scale {scale})");
    }
}

/// Compares root sets regardless of order.
fn assert_same_roots(name: &str, got: &[C64], want: &[C64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: {} roots vs {}", got.len(), want.len());
    let mut left: Vec<C64> = want.to_vec();
    for g in got {
        let (i, d) = left.iter().enumerate().map(|(i, w)| (i, (*g - *w).norm())).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert!(d <= tol * (1.0 + g.norm()), "{name}: root {g:?} has no match (nearest off by {d})");
        left.remove(i);
    }
}

fn band(case: &Value) -> Band {
    let e = f64s(&case["edges"]);
    match case["band"].as_str().unwrap() {
        "lowpass" => Band::Lowpass(e[0]),
        "highpass" => Band::Highpass(e[0]),
        "bandpass" => Band::Bandpass(e[0], e[1]),
        _ => Band::Bandstop(e[0], e[1]),
    }
}

#[test]
fn iir_designs_match_scipy() {
    for case in fixtures()["iir"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let kind = match case["kind"].as_str().unwrap() {
            "butter" => IirKind::Butterworth,
            "cheby1" => IirKind::Chebyshev1 { rp: p["rp"].as_f64().unwrap() },
            "cheby2" => IirKind::Chebyshev2 { rs: p["rs"].as_f64().unwrap() },
            "ellip" => IirKind::Elliptic { rp: p["rp"].as_f64().unwrap(), rs: p["rs"].as_f64().unwrap() },
            _ => IirKind::Bessel {
                norm: match p["norm"].as_str().unwrap() {
                    "phase" => BesselNorm::Phase,
                    "mag" => BesselNorm::Mag,
                    _ => BesselNorm::Delay,
                },
            },
        };
        let order = case["order"].as_u64().unwrap() as usize;
        let zpk = iirfilter(order, band(case), kind, Design::Digital { fs: case["fs"].as_f64().unwrap() }).unwrap();
        assert_same_roots(&format!("{name} zeros"), &zpk.zeros, &complexes(&case["zeros"]), 1e-8);
        assert_same_roots(&format!("{name} poles"), &zpk.poles, &complexes(&case["poles"]), 1e-9);
        assert!((zpk.gain - case["gain"].as_f64().unwrap()).abs() <= 1e-9 * case["gain"].as_f64().unwrap().abs(), "{name} gain");
        let sos = zpk.to_sos().unwrap();
        let want = sos_of(&case["sos"]);
        assert_eq!(sos.len(), want.len(), "{name}: sections");
        for (i, (g, w)) in sos.iter().zip(&want).enumerate() {
            assert_close(&format!("{name} section {i}"), g, w, 1e-8);
        }
    }
}

#[test]
fn analog_prototypes_match_scipy() {
    use autodyne::filter::design::{butter, cheby1, cheby2, ellip};
    let tau = std::f64::consts::TAU;
    let designs = [
        butter(4, Band::Lowpass(tau * 1000.0), Design::Analog),
        cheby1(3, 1.0, Band::Lowpass(tau * 500.0), Design::Analog),
        ellip(4, 0.5, 40.0, Band::Lowpass(100.0), Design::Analog),
        cheby2(4, 40.0, Band::Bandpass(100.0, 400.0), Design::Analog),
    ];
    for (case, zpk) in fixtures()["analog"].as_array().unwrap().iter().zip(designs) {
        let name = case["name"].as_str().unwrap();
        let zpk = zpk.unwrap();
        assert_same_roots(&format!("{name} zeros"), &zpk.zeros, &complexes(&case["zeros"]), 1e-9);
        assert_same_roots(&format!("{name} poles"), &zpk.poles, &complexes(&case["poles"]), 1e-9);
        let k = case["gain"].as_f64().unwrap();
        assert!((zpk.gain - k).abs() <= 1e-9 * k.abs(), "{name} gain {} vs {k}", zpk.gain);
    }
    // the delay-normalized Bessel prototype has group delay 1: the denominator's constant term
    let (_, p, k) = besselap(3, BesselNorm::Delay).unwrap();
    assert_eq!(p.len(), 3);
    assert!((k - 15.0).abs() < 1e-9);
}

fn window_spec(v: &Value) -> WindowSpec {
    match v {
        Value::String(s) => s.parse().unwrap(),
        Value::Array(parts) => {
            let x = parts.last().unwrap().as_f64().unwrap();
            match parts[0].as_str().unwrap() {
                "kaiser" => WindowSpec::Kaiser { beta: x },
                "gaussian" => WindowSpec::Gaussian { std: x },
                "tukey" => WindowSpec::Tukey { alpha: x },
                "exponential" => WindowSpec::Exponential { tau: x },
                "chebwin" => WindowSpec::Chebwin { attenuation: x },
                other => panic!("unknown window {other}"),
            }
        }
        _ => panic!("bad window spec"),
    }
}

#[test]
fn windows_match_scipy() {
    for case in fixtures()["windows"].as_array().unwrap() {
        let spec = window_spec(&case["spec"]);
        let n = case["n"].as_u64().unwrap() as usize;
        let periodic = case["periodic"].as_bool().unwrap();
        assert_close(&format!("{spec:?} n={n} periodic={periodic}"), &get_window(spec, n, periodic), &f64s(&case["w"]), 1e-12);
    }
}

#[test]
fn fir_designs_match_scipy() {
    let fx = fixtures();
    for case in fx["firwin"].as_array().unwrap() {
        let taps = firwin(
            case["numtaps"].as_u64().unwrap() as usize,
            &f64s(&case["cutoff"]),
            window_spec(&case["window"]),
            case["pass_zero"].as_bool().unwrap(),
            case["scale"].as_bool().unwrap(),
            case["fs"].as_f64().unwrap(),
        )
        .unwrap();
        assert_close("firwin", &taps, &f64s(&case["taps"]), 1e-12);
    }
    for case in fx["firwin2"].as_array().unwrap() {
        let taps = firwin2(
            case["numtaps"].as_u64().unwrap() as usize,
            &f64s(&case["freq"]),
            &f64s(&case["gain"]),
            None,
            Some(WindowSpec::Hamming),
            case["antisymmetric"].as_bool().unwrap(),
            case["fs"].as_f64().unwrap(),
        )
        .unwrap();
        assert_close("firwin2", &taps, &f64s(&case["taps"]), 1e-12);
    }
    let pairs = |v: &Value| v.as_array().unwrap().iter().map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect::<Vec<_>>();
    for case in fx["firls"].as_array().unwrap() {
        let weight = case["weight"].as_array().map(|_| f64s(&case["weight"]));
        let taps = firls(case["numtaps"].as_u64().unwrap() as usize, &pairs(&case["bands"]), &pairs(&case["desired"]), weight.as_deref(), case["fs"].as_f64().unwrap()).unwrap();
        assert_close("firls", &taps, &f64s(&case["taps"]), 1e-9);
    }
    for case in fx["remez"].as_array().unwrap() {
        let weight = case["weight"].as_array().map(|_| f64s(&case["weight"]));
        let kind = match case["type"].as_str().unwrap() {
            "bandpass" => RemezType::Bandpass,
            "hilbert" => RemezType::Hilbert,
            _ => RemezType::Differentiator,
        };
        let taps = remez(case["numtaps"].as_u64().unwrap() as usize, &pairs(&case["bands"]), &f64s(&case["desired"]), weight.as_deref(), kind, 25, 16, case["fs"].as_f64().unwrap())
            .unwrap_or_else(|e| panic!("remez {kind:?} {}: {e}", case["numtaps"]));
        assert_close(&format!("remez {kind:?}"), &taps, &f64s(&case["taps"]), 1e-9);
    }
    for case in fx["kaiserord"].as_array().unwrap() {
        let (n, beta) = kaiserord(case["ripple"].as_f64().unwrap(), case["width"].as_f64().unwrap()).unwrap();
        assert_eq!(n as u64, case["numtaps"].as_u64().unwrap());
        assert!((beta - case["beta"].as_f64().unwrap()).abs() < 1e-12);
    }
}

#[test]
fn conversions_match_scipy() {
    let cv = &fixtures()["convert"];
    let (z, p, k) = tf2zpk(&f64s(&cv["b"]), &f64s(&cv["a"])).unwrap();
    assert_same_roots("tf2zpk zeros", &z, &complexes(&cv["zeros"]), 1e-10);
    assert_same_roots("tf2zpk poles", &p, &complexes(&cv["poles"]), 1e-10);
    assert!((k - cv["gain"].as_f64().unwrap()).abs() < 1e-12);

    let t = &cv["tf2ss"];
    let (a, b, c, d) = tf2ss(&f64s(&t["num"]), &f64s(&t["den"])).unwrap();
    for (name, got, want) in [("A", &a, &t["A"]), ("B", &b, &t["B"]), ("C", &c, &t["C"]), ("D", &d, &t["D"])] {
        assert_close(&format!("tf2ss {name}"), got.as_slice(), &nd(want).0, 1e-12);
    }

    let sys = StateSpace::new(
        NdArray::from_vec(vec![0.0, 1.0, -2.0, -3.0], &[2, 2]).unwrap(),
        NdArray::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]).unwrap(),
        NdArray::from_vec(vec![1.0, 0.0, 1.0, 1.0], &[2, 2]).unwrap(),
        NdArray::from_vec(vec![0.0, 0.0, 0.5, 0.0], &[2, 2]).unwrap(),
        Domain::Continuous,
    )
    .unwrap();
    let (nums, den) = ss2tf(&sys, 0).unwrap();
    assert_close("ss2tf den", &den, &f64s(&cv["ss2tf"]["den"]), 1e-12);
    assert_close("ss2tf num", &nums.concat(), &nd(&cv["ss2tf"]["num"]).0, 1e-12);

    let s = &cv["zpk2sos"];
    let (zs, ps) = (complexes(&s["zeros"]), complexes(&s["poles"]));
    let digital = Domain::Discrete { dt: 1.0 };
    for (pairing, key) in [(Pairing::Nearest, "nearest"), (Pairing::KeepOdd, "keep_odd")] {
        let got = zpk2sos(&zs, &ps, 2.0, pairing, digital).unwrap();
        assert_close(&format!("zpk2sos {key}"), &got.concat(), &nd(&s[key]).0, 1e-12);
    }
    let minimal = zpk2sos(&[C64::new(-3.0, 0.0)], &[C64::new(-1.0, 2.0), C64::new(-1.0, -2.0), C64::new(-4.0, 0.0)], 1.5, Pairing::Minimal, Domain::Continuous).unwrap();
    assert_close("zpk2sos minimal", &minimal.concat(), &nd(&s["minimal"]).0, 1e-12);
}

#[test]
fn frequency_responses_match_scipy() {
    let f = &fixtures()["freq"];
    let flat = |h: &[C64]| h.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>();
    let want = |key: &str| flat(&complexes(&f[key]));
    let freqs_hz = f64s(&f["freqs"]);
    assert_close("freqz", &flat(&freqz(&f64s(&f["b"]), &f64s(&f["a"]), &freqs_hz, 8000.0)), &want("h"), 1e-12);
    assert_close("sosfreqz", &flat(&sosfreqz(&sos_of(&f["sos"]), &freqs_hz, 8000.0)), &want("hs"), 1e-10);
    assert_close("group_delay", &group_delay(&f64s(&f["b"]), &f64s(&f["a"]), &f64s(&f["gd_freqs"]), 8000.0), &f64s(&f["gd"]), 1e-9);
    assert_close("freqs", &flat(&freqs(&[1.0], &[1.0, 2.0, 1.0], &f64s(&f["w"]))), &want("ha"), 1e-12);
}

#[test]
fn filtering_matches_scipy() {
    let f = &fixtures()["filter"];
    let x = array(&f["x"]);
    let (b, a, sos) = (f64s(&f["b"]), f64s(&f["a"]), sos_of(&f["sos"]));
    let xt = x.view().transpose().to_owned();
    assert_close("lfilter", lfilter(&b, &a, x.view(), 1).unwrap().as_slice(), &nd(&f["lfilter"]).0, 1e-12);
    assert_close("lfilter axis 0", lfilter(&b, &a, xt.view(), 0).unwrap().as_slice(), &nd(&f["lfilter_axis0"]).0, 1e-12);
    let zi = lfilter_zi(&b, &a).unwrap();
    assert_close("lfilter_zi", &zi, &f64s(&f["lfilter_zi"]), 1e-12);
    let zi0 = NdArray::from_fn(&[3, zi.len()], |i| zi[i[1]] * x.as_slice()[i[0] * 200]).unwrap();
    let (y, zf) = lfilter_with_state(&b, &a, x.view(), 1, zi0.view()).unwrap();
    assert_close("lfilter with state", y.as_slice(), &nd(&f["lfilter_state_y"]).0, 1e-12);
    assert_close("lfilter final state", zf.as_slice(), &nd(&f["lfilter_state_zf"]).0, 1e-12);

    assert_close("sosfilt", sosfilt(&sos, x.view(), 1).unwrap().as_slice(), &nd(&f["sosfilt"]).0, 1e-12);
    let szi = sosfilt_zi(&sos).unwrap();
    assert_close("sosfilt_zi", &szi.concat(), &nd(&f["sosfilt_zi"]).0, 1e-12);
    // zi: [sections, 2, channels] for x.T (time on axis 0)
    let szi0 = NdArray::from_fn(&[sos.len(), 2, 3], |i| szi[i[0]][i[1]] * x.as_slice()[i[2] * 200]).unwrap();
    let (y, zf) = sosfilt_with_state(&sos, xt.view(), 0, szi0.view()).unwrap();
    assert_close("sosfilt with state", y.as_slice(), &nd(&f["sosfilt_state_y"]).0, 1e-12);
    assert_close("sosfilt final state", zf.as_slice(), &nd(&f["sosfilt_state_zf"]).0, 1e-12);

    assert_close("filtfilt", filtfilt(&b, &a, x.view(), 1, Pad::Odd(None)).unwrap().as_slice(), &nd(&f["filtfilt"]).0, 1e-10);
    assert_close("filtfilt even", filtfilt(&b, &a, x.view(), 1, Pad::Even(Some(20))).unwrap().as_slice(), &nd(&f["filtfilt_even"]).0, 1e-10);
    assert_close("filtfilt none", filtfilt(&b, &a, x.view(), 1, Pad::None).unwrap().as_slice(), &nd(&f["filtfilt_none"]).0, 1e-10);
    assert_close("sosfiltfilt", sosfiltfilt(&sos, x.view(), 1, Pad::Odd(None)).unwrap().as_slice(), &nd(&f["sosfiltfilt"]).0, 1e-10);
    assert_close("sosfiltfilt constant", sosfiltfilt(&sos, x.view(), 1, Pad::Constant(None)).unwrap().as_slice(), &nd(&f["sosfiltfilt_const"]).0, 1e-10);
}

#[test]
fn systems_match_scipy() {
    let s = &fixtures()["systems"];
    let sys = StateSpace::new(array(&s["A"]), array(&s["B"]), array(&s["C"]), array(&s["D"]), Domain::Continuous).unwrap();
    for (method, key) in [(Method::Zoh, "zoh"), (Method::Bilinear, "bilinear"), (Method::Euler, "euler"), (Method::BackwardDiff, "backward_diff")] {
        let d = sys.discretize(0.1, method).unwrap();
        let want = &s["discretized"][key];
        for (name, got, w) in [("A", &d.a, &want["A"]), ("B", &d.b, &want["B"]), ("C", &d.c, &want["C"]), ("D", &d.d, &want["D"])] {
            assert_close(&format!("{key} {name}"), got.as_slice(), &nd(w).0, 1e-12);
        }
    }
    let m = array(&s["M"]);
    assert_close("expm", expm(m.view()).unwrap().as_slice(), &nd(&s["expm"]).0, 1e-12);
    assert_close("expm (scaled)", expm(m.map(|&x| x * 10.0).view()).unwrap().as_slice(), &nd(&s["expm_big"]).0, 1e-9);
    assert_close("step", sys.step(50, Some(0.1)).unwrap().as_slice(), &f64s(&s["step"]), 1e-10);
    assert_close("impulse", sys.impulse(50, Some(0.1)).unwrap().as_slice(), &f64s(&s["impulse"]), 1e-10);
    let dsys = sys.discretize(0.1, Method::Zoh).unwrap();
    let u = NdArray::from_vec(f64s(&s["u"]), &[40]).unwrap();
    let (y, x) = dsys.simulate(&u, None).unwrap();
    assert_close("dlsim y", y.as_slice(), &nd(&s["dlsim_y"]).0, 1e-12);
    assert_close("dlsim x", x.as_slice(), &nd(&s["dlsim_x"]).0, 1e-12);
    let (bb, ab) = bilinear(&[1.0, 2.0], &[1.0, 3.0, 2.0], 10.0).unwrap();
    assert_close("bilinear b", &bb, &f64s(&s["bilinear_b"]), 1e-12);
    assert_close("bilinear a", &ab, &f64s(&s["bilinear_a"]), 1e-12);
    let _ = systems::freq_grid(4, false, 8000.0);
}

#[test]
fn signal_tools_match_scipy() {
    use autodyne::filter::{savgol_filter, SavgolMode};
    use autodyne::resample::{resample_poly, upfirdn};
    use autodyne::signal::{convolve_with, correlate_with, find_peaks, Bounds, ConvMethod, ConvMode, PeakOptions};
    use autodyne::spectral::hilbert;

    let s = &fixtures()["signal"];
    let (a, k) = (f64s(&s["a"]), f64s(&s["k"]));
    for (mode, key) in [(ConvMode::Full, "full"), (ConvMode::Same, "same"), (ConvMode::Valid, "valid")] {
        for method in [ConvMethod::Direct, ConvMethod::Fft, ConvMethod::OverlapAdd] {
            let name = format!("convolve {key} {method:?}");
            assert_close(&name, &convolve_with(&a, &k, mode, method), &f64s(&s[format!("convolve_{key}")]), 1e-12);
            assert_close(&format!("{name}, swapped"), &convolve_with(&k, &a, mode, method), &f64s(&s[format!("convolve_swapped_{key}")]), 1e-12);
            assert_close(&format!("correlate {key} {method:?}"), &correlate_with(&a, &k, mode, method), &f64s(&s[format!("correlate_{key}")]), 1e-12);
        }
    }

    let x = array(&s["x2"]);
    for (w, p, d) in [(11, 3, 0), (10, 2, 1), (7, 4, 2)] {
        for (mode, key) in [(SavgolMode::Interp, "interp"), (SavgolMode::Mirror, "mirror"), (SavgolMode::Nearest, "nearest"), (SavgolMode::Constant(1.5), "constant"), (SavgolMode::Wrap, "wrap")] {
            let got = savgol_filter(x.view(), w, p, d, 0.5, 1, mode).unwrap();
            assert_close(&format!("savgol {w} {p} {d} {key}"), got.as_slice(), &nd(&s["savgol"][format!("{w}_{p}_{d}_{key}")]).0, 1e-10);
        }
    }
    let xt = x.view().transpose().to_owned();
    assert_close("savgol axis 0", savgol_filter(xt.view(), 9, 2, 0, 1.0, 0, SavgolMode::Interp).unwrap().as_slice(), &nd(&s["savgol_axis0"]).0, 1e-10);

    let flat = |v: &[autodyne::units::Complex<f64>]| v.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>();
    assert_close("hilbert", &flat(hilbert(x.view(), 1, None).unwrap().as_slice()), &flat(&complexes(&s["hilbert"])), 1e-12);
    assert_close("hilbert n=97", &flat(hilbert(x.view(), 1, Some(97)).unwrap().as_slice()), &flat(&complexes(&s["hilbert_n"])), 1e-12);
    assert_close("hilbert n=131", &flat(hilbert(x.view(), 1, Some(131)).unwrap().as_slice()), &flat(&complexes(&s["hilbert_n_odd"])), 1e-12);

    let taps = f64s(&s["upfirdn_taps"]);
    for (u, d) in [(1, 1), (3, 2), (2, 7), (5, 1), (1, 4)] {
        let got = upfirdn(&taps, x.view(), u, d, 1).unwrap();
        let (want, shape) = nd(&s["upfirdn"][format!("{u}_{d}")]);
        assert_eq!(got.shape(), shape.as_slice(), "upfirdn {u}/{d} shape");
        assert_close(&format!("upfirdn {u}/{d}"), got.as_slice(), &want, 1e-12);
    }
    for (u, d) in [(3, 2), (2, 3), (1, 4), (160, 147), (7, 1)] {
        let got = resample_poly(x.view(), u, d, 1, WindowSpec::Kaiser { beta: 5.0 }).unwrap();
        let (want, shape) = nd(&s["resample_poly"][format!("{u}_{d}")]);
        assert_eq!(got.shape(), shape.as_slice(), "resample_poly {u}/{d} shape");
        assert_close(&format!("resample_poly {u}/{d}"), got.as_slice(), &want, 1e-12);
    }
    let got = resample_poly(x.view(), 5, 3, 1, WindowSpec::Kaiser { beta: 8.0 }).unwrap();
    assert_close("resample_poly kaiser 8", got.as_slice(), &nd(&s["resample_poly_kaiser8"]).0, 1e-12);

    let xp = f64s(&s["xp"]);
    let bounds = |v: &Value| match v {
        Value::Array(p) => Bounds { min: p[0].as_f64(), max: p[1].as_f64() },
        v => Bounds::at_least(v.as_f64().unwrap()),
    };
    for (name, case) in s["peaks"].as_object().unwrap() {
        let kw = &case["kw"];
        let options = PeakOptions {
            height: kw.get("height").map(bounds),
            threshold: kw.get("threshold").map(bounds),
            distance: kw.get("distance").and_then(Value::as_f64),
            prominence: kw.get("prominence").map(bounds),
            width: kw.get("width").map(bounds),
            wlen: kw.get("wlen").and_then(Value::as_u64).map(|w| w as usize),
            rel_height: kw.get("rel_height").and_then(Value::as_f64).unwrap_or(0.5),
            plateau_size: kw.get("plateau_size").map(bounds),
        };
        let p = find_peaks(&xp, &options).unwrap();
        let as_f64 = |v: &[usize]| v.iter().map(|&i| i as f64).collect::<Vec<_>>();
        assert_eq!(as_f64(&p.indices), f64s(&case["indices"]), "peaks {name}");
        let props = &case["props"];
        let check = |key: &str, got: Option<Vec<f64>>| match (props.get(key), got) {
            (Some(want), Some(got)) => assert_close(&format!("peaks {name} {key}"), &got, &f64s(want), 1e-12),
            (None, None) => {}
            (want, got) => panic!("peaks {name} {key}: SciPy has {want:?}, autodyne {got:?}"),
        };
        check("plateau_sizes", p.plateaus.as_ref().map(|t| as_f64(&t.sizes)));
        check("left_edges", p.plateaus.as_ref().map(|t| as_f64(&t.left_edges)));
        check("right_edges", p.plateaus.as_ref().map(|t| as_f64(&t.right_edges)));
        check("peak_heights", p.peak_heights.clone());
        check("left_thresholds", p.left_thresholds.clone());
        check("right_thresholds", p.right_thresholds.clone());
        check("prominences", p.prominences.as_ref().map(|q| q.prominences.clone()));
        check("left_bases", p.prominences.as_ref().map(|q| as_f64(&q.left_bases)));
        check("right_bases", p.prominences.as_ref().map(|q| as_f64(&q.right_bases)));
        check("widths", p.widths.as_ref().map(|w| w.widths.clone()));
        check("width_heights", p.widths.as_ref().map(|w| w.width_heights.clone()));
        check("left_ips", p.widths.as_ref().map(|w| w.left_ips.clone()));
        check("right_ips", p.widths.as_ref().map(|w| w.right_ips.clone()));
    }
}

#[test]
fn statistics_match_numpy_scipy_and_statsmodels() {
    use autodyne::stats::*;

    let s = &fixtures()["stats"];
    let x = array(&s["x"]);
    let q = f64s(&s["q"]);
    for (method, key) in [(QuantileMethod::Linear, "linear"), (QuantileMethod::Lower, "lower"), (QuantileMethod::Higher, "higher"), (QuantileMethod::Nearest, "nearest"), (QuantileMethod::Midpoint, "midpoint")] {
        // NumPy puts the quantiles first; ours replace the axis: compare transposed
        let got = quantile_axis(x.view(), &q, 1, method, true).unwrap();
        let got = got.view().transpose().to_owned();
        assert_close(&format!("quantile {key}"), got.as_slice(), &nd(&s["quantiles"][key]).0, 1e-14);
    }
    assert_close("median", median_axis(x.view(), 1).unwrap().as_slice(), &f64s(&s["median"]), 1e-15);
    let even = x.view().slice_axis(1, 0..100).unwrap();
    assert_close("median, even length", median_axis(even, 1).unwrap().as_slice(), &f64s(&s["median_even"]), 1e-15);
    assert_close("skew", skew_axis(x.view(), 1, true).unwrap().as_slice(), &f64s(&s["skew"]), 1e-12);
    assert_close("skew unbiased", skew_axis(x.view(), 1, false).unwrap().as_slice(), &f64s(&s["skew_unbiased"]), 1e-12);
    assert_close("kurtosis", kurtosis_axis(x.view(), 1, true, true).unwrap().as_slice(), &f64s(&s["kurtosis"]), 1e-12);
    assert_close("kurtosis Pearson unbiased", kurtosis_axis(x.view(), 1, false, false).unwrap().as_slice(), &f64s(&s["kurtosis_pearson_unbiased"]), 1e-12);
    assert_close("moment 4", moment_axis(x.view(), 4, 1).unwrap().as_slice(), &f64s(&s["moment4"]), 1e-12);
    assert_close("zscore", zscore_axis(x.view(), 1, 1).unwrap().as_slice(), &nd(&s["zscore"]).0, 1e-12);
    assert_close("cov", cov(x.view(), true, 1).unwrap().as_slice(), &nd(&s["cov"]).0, 1e-12);
    let xt = x.view().transpose().to_owned();
    assert_close("cov of columns, ddof 0", cov(xt.view(), false, 0).unwrap().as_slice(), &nd(&s["cov_ddof0_cols"]).0, 1e-12);
    assert_close("corrcoef", corrcoef(x.view(), true).unwrap().as_slice(), &nd(&s["corrcoef"]).0, 1e-12);

    let rows: Vec<Vec<f64>> = (0..3).map(|i| x.as_slice()[i * 101..(i + 1) * 101].to_vec()).collect();
    let h = &s["histogram"];
    for (key, bins) in [
        ("10", Bins::Count(10)),
        ("auto", Bins::Rule(BinRule::Auto)),
        ("fd", Bins::Rule(BinRule::Fd)),
        ("sturges", Bins::Rule(BinRule::Sturges)),
        ("scott", Bins::Rule(BinRule::Scott)),
        ("rice", Bins::Rule(BinRule::Rice)),
        ("sqrt", Bins::Rule(BinRule::Sqrt)),
    ] {
        let (counts, edges) = histogram(&rows[2], &bins, None, None, false).unwrap();
        assert_close(&format!("histogram {key} edges"), &edges, &f64s(&h[key]["e"]), 1e-14);
        assert_eq!(counts, f64s(&h[key]["h"]), "histogram {key}");
    }
    let (counts, _) = histogram(&rows[1], &Bins::Edges(vec![2.0, 4.0, 4.5, 7.0]), None, None, false).unwrap();
    assert_eq!(counts, f64s(&h["edges"]["h"]));
    let weights: Vec<f64> = rows[1].iter().map(|v| v.abs()).collect();
    let (dens, _) = histogram(&rows[0], &Bins::Count(8), Some((-1.0, 1.0)), Some(&weights), true).unwrap();
    assert_close("weighted density", &dens, &f64s(&h["weighted_density"]["h"]), 1e-13);

    let y = f64s(&s["ts"]);
    assert_close("acovf", &acovf(&y, 12, false, true).unwrap(), &f64s(&s["acovf"]), 1e-12);
    assert_close("acovf adjusted", &acovf(&y, 12, true, true).unwrap(), &f64s(&s["acovf_adjusted"]), 1e-12);
    assert_close("acf", &acf(&y, 12, false).unwrap(), &f64s(&s["acf"]), 1e-12);
    assert_close("pacf yw", &pacf(&y, 10, YuleWalker::Adjusted).unwrap(), &f64s(&s["pacf_yw"]), 1e-10);
    assert_close("pacf ywm", &pacf(&y, 10, YuleWalker::Mle).unwrap(), &f64s(&s["pacf_ywm"]), 1e-10);
    for (method, key) in [(YuleWalker::Adjusted, "yule_walker"), (YuleWalker::Mle, "yule_walker_mle")] {
        let fit = yule_walker(&y, 3, method).unwrap();
        assert_close(key, &fit.ar, &f64s(&s[key]["ar"]), 1e-10);
        assert!((fit.sigma2.sqrt() - s[key]["sigma"].as_f64().unwrap()).abs() < 1e-10, "{key} sigma");
    }
    let fit = burg(&y, 4).unwrap();
    assert_close("burg", &fit.ar, &f64s(&s["burg"]["ar"]), 1e-10);
    assert!((fit.sigma2 - s["burg"]["sigma2"].as_f64().unwrap()).abs() < 1e-10 * fit.sigma2, "burg sigma2 {} vs {}", fit.sigma2, s["burg"]["sigma2"]);
}

#[test]
fn matrix_equations_match_scipy() {
    use autodyne::linalg::{eigvals, matmul, schur, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are, solve_discrete_lyapunov, solve_sylvester};

    for (key, case) in fixtures()["equations"].as_object().unwrap() {
        let m = |k: &str| array(&case[k]);
        let (a, b, q, r) = (m("a"), m("b"), m("q"), m("r"));
        // Schur: a = z t zᵀ with z orthogonal, t quasi-triangular, the same eigenvalues
        let (t, z) = schur(a.view()).unwrap();
        let n = a.shape()[0];
        let back = matmul(matmul(z.view(), t.view()).unwrap().view(), z.view().transpose()).unwrap();
        assert_close(&format!("{key}: z t zᵀ"), back.as_slice(), a.as_slice(), 1e-12);
        let ztz = matmul(z.view().transpose(), z.view()).unwrap();
        let eye: Vec<f64> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
        assert_close(&format!("{key}: zᵀ z"), ztz.as_slice(), &eye, 1e-13);
        for i in 0..n {
            for j in 0..i.saturating_sub(1) {
                assert_eq!(t.as_slice()[i * n + j], 0.0, "{key}: t[{i}][{j}]");
            }
        }
        let ev: Vec<C64> = eigvals(t.view()).unwrap();
        assert_same_roots(&format!("{key}: eigenvalues"), &ev, &eigvals(a.view()).unwrap(), 1e-10);

        assert_close(&format!("{key}: sylvester"), solve_sylvester(a.view(), m("bb").view(), m("c").view()).unwrap().as_slice(), &nd(&case["sylvester"]).0, 1e-10);
        let neg_q = q.map(|v| -v);
        assert_close(&format!("{key}: lyapunov"), solve_continuous_lyapunov(m("stable").view(), neg_q.view()).unwrap().as_slice(), &nd(&case["lyapunov"]).0, 1e-10);
        // a general (not symmetric) right-hand side, a not stable
        let general = NdArray::from_vec(nd(&case["lyapunov_general"]).0, &[n, n]).unwrap();
        let mm = {
            // recover q from the SciPy solution: q = a x + x aᵀ
            let ax = matmul(a.view(), general.view()).unwrap();
            let xa = matmul(general.view(), a.view().transpose()).unwrap();
            NdArray::from_vec(ax.as_slice().iter().zip(xa.as_slice()).map(|(p, q)| p + q).collect(), &[n, n]).unwrap()
        };
        assert_close(&format!("{key}: lyapunov, general"), solve_continuous_lyapunov(a.view(), mm.view()).unwrap().as_slice(), general.as_slice(), 1e-9);
        assert_close(&format!("{key}: discrete lyapunov"), solve_discrete_lyapunov(m("dstable").view(), q.view()).unwrap().as_slice(), &nd(&case["discrete_lyapunov"]).0, 1e-10);
        // Riccati: our residual must be tiny and no worse than SciPy's; the values must agree where
        // the problem is well conditioned (at n = 20 the random a is far from stable, x reaches
        // 1e11, and SciPy's own residual is only 2e-6 relative)
        let care = solve_continuous_are(a.view(), b.view(), q.view(), r.view()).unwrap();
        let dare = solve_discrete_are(a.view(), b.view(), q.view(), r.view()).unwrap();
        let scipy_care = NdArray::from_vec(nd(&case["care"]).0, &[n, n]).unwrap();
        let scipy_dare = NdArray::from_vec(nd(&case["dare"]).0, &[n, n]).unwrap();
        for (name, residual, ours, theirs) in [("care", care_residual as fn(&_, &_, &_, &_, &_) -> f64, &care, &scipy_care), ("dare", dare_residual, &dare, &scipy_dare)] {
            let (mine, scipy) = (residual(&a, &b, &q, &r, ours), residual(&a, &b, &q, &r, theirs));
            assert!(mine < 1e-9 && mine <= scipy.max(1e-14) * 2.0, "{key}: {name} relative residual {mine:e} (SciPy {scipy:e})");
            if n <= 6 {
                assert_close(&format!("{key}: {name}"), ours.as_slice(), theirs.as_slice(), 1e-10);
            }
        }
    }
}

#[test]
fn control_analysis_matches_python_control() {
    use autodyne::systems::{damp, dlqr, lqr, stability_margins, TransferFunction};

    let ctl = &fixtures()["control"];
    for key in ["continuous", "discrete"] {
        let s = &ctl[key];
        let dt = s["dt"].as_f64().unwrap();
        let domain = if dt > 0.0 { Domain::Discrete { dt } } else { Domain::Continuous };
        let sys = StateSpace::new(array(&s["A"]), array(&s["B"]), array(&s["C"]), array(&s["D"]), domain).unwrap();
        assert_close(&format!("{key} ctrb"), sys.ctrb().unwrap().as_slice(), &nd(&s["ctrb"]).0, 1e-12);
        assert_close(&format!("{key} obsv"), sys.obsv().unwrap().as_slice(), &nd(&s["obsv"]).0, 1e-12);
        assert!(sys.is_controllable().unwrap() && sys.is_observable().unwrap());
        assert_close(&format!("{key} controllability Gramian"), sys.controllability_gramian().unwrap().as_slice(), &nd(&s["wc"]).0, 1e-10);
        assert_close(&format!("{key} observability Gramian"), sys.observability_gramian().unwrap().as_slice(), &nd(&s["wo"]).0, 1e-10);
        assert_close(&format!("{key} dcgain"), sys.dcgain().unwrap().as_slice(), &nd(&s["dcgain"]).0, 1e-10);
        // damping, matched pole by pole
        let want_poles = complexes(&s["poles"]);
        let (wn, zeta) = (f64s(&s["wn"]), f64s(&s["zeta"]));
        for (p, w, z) in damp(&sys.poles().unwrap(), domain) {
            let i = want_poles.iter().enumerate().min_by(|a, b| (*a.1 - p).norm().total_cmp(&(*b.1 - p).norm())).unwrap().0;
            assert!((w - wn[i]).abs() < 1e-9 * wn[i] && (z - zeta[i]).abs() < 1e-9, "{key} damp: {p:?} {w} {z} vs {} {}", wn[i], zeta[i]);
        }
        let (q, r) = (array(&s["Q"]), array(&s["R"]));
        let reg = if dt > 0.0 { dlqr(&sys.a, &sys.b, &q, &r) } else { lqr(&sys.a, &sys.b, &q, &r) }.unwrap();
        assert_close(&format!("{key} lqr K"), reg.k.as_slice(), &nd(&s["lqr_k"]).0, 1e-9);
        assert_close(&format!("{key} lqr S"), reg.s.as_slice(), &nd(&s["lqr_s"]).0, 1e-9);
        assert_same_roots(&format!("{key} lqr poles"), &reg.poles, &complexes(&s["lqr_poles"]), 1e-9);
    }
    for m in ctl["margins"].as_array().unwrap() {
        let dt = m["dt"].as_f64().unwrap();
        let domain = if dt > 0.0 { Domain::Discrete { dt } } else { Domain::Continuous };
        let tf = TransferFunction::new(f64s(&m["num"]), f64s(&m["den"]), domain);
        let got = stability_margins(&tf).unwrap();
        let name = format!("{:?} / {:?}", m["num"], m["den"]);
        assert!((got.gain_margin - m["gm"].as_f64().unwrap()).abs() < 1e-9 * got.gain_margin, "{name}: gm {} vs {}", got.gain_margin, m["gm"]);
        assert!((got.phase_margin - m["pm"].as_f64().unwrap()).abs() < 1e-8, "{name}: pm {} vs {}", got.phase_margin, m["pm"]);
        assert!((got.phase_crossover - m["wpc"].as_f64().unwrap()).abs() < 1e-9 * got.phase_crossover, "{name}: wpc {} vs {}", got.phase_crossover, m["wpc"]);
        assert!((got.gain_crossover - m["wgc"].as_f64().unwrap()).abs() < 1e-9 * got.gain_crossover, "{name}: wgc {} vs {}", got.gain_crossover, m["wgc"]);
    }
}

#[test]
fn initial_value_problems_match_solve_ivp() {
    use autodyne::ode::{solve_ivp, OdeMethod, OdeOptions};

    let lotka = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = 1.5 * y[0] - y[0] * y[1];
        dy[1] = -3.0 * y[1] + y[0] * y[1];
    };
    let ivp = &fixtures()["ivp"];
    for (method, key) in [(OdeMethod::Rk45, "RK45"), (OdeMethod::Rk23, "RK23")] {
        let case = &ivp[key];
        let options = || OdeOptions { method, rtol: 1e-6, atol: 1e-9, ..Default::default() };
        // the same steps as SciPy: its initial step, error norm and step factors
        let sol = solve_ivp(lotka, (0.0, 15.0), &[10.0, 5.0], options()).unwrap();
        assert_eq!(sol.nfev as u64, case["nfev"].as_u64().unwrap(), "{key}: function evaluations");
        assert_close(&format!("{key} step times"), &sol.t, &f64s(&case["t"]), 1e-12);
        assert_close(&format!("{key} states"), &sol.y.concat(), &nd(&case["y"]).0, 1e-10);
        let te = f64s(&case["t_eval"]);
        let dense = solve_ivp(lotka, (0.0, 15.0), &[10.0, 5.0], OdeOptions { t_eval: Some(te.clone()), ..options() }).unwrap();
        assert_eq!(dense.t, te);
        assert_close(&format!("{key} dense output"), &dense.y.concat(), &nd(&case["y_eval"]).0, 1e-10);
    }
    // the stiff method against a tight reference
    let stiff = solve_ivp(lotka, (0.0, 15.0), &[10.0, 5.0], OdeOptions { method: OdeMethod::Rosenbrock23, rtol: 1e-8, atol: 1e-10, ..Default::default() }).unwrap();
    assert_close("Rosenbrock23 end", stiff.y.last().unwrap(), &f64s(&ivp["reference_end"]), 1e-4);
}

type M = NdArray<f64>;

fn mm(x: &M, y: &M) -> M {
    autodyne::linalg::matmul(x.view(), y.view()).unwrap()
}

fn tr(x: &M) -> M {
    x.view().transpose().to_owned()
}

fn combine(parts: &[(f64, &M)]) -> M {
    let shape = parts[0].1.shape().to_vec();
    let n: usize = shape.iter().product();
    NdArray::from_vec((0..n).map(|i| parts.iter().map(|(c, m)| c * m.as_slice()[i]).sum()).collect(), &shape).unwrap()
}

fn rel_norm(res: &M, x: &M) -> f64 {
    let norm = |m: &M| m.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
    norm(res) / norm(x)
}

/// `‖aᵀ x + x a - x b r⁻¹ bᵀ x + q‖ / ‖x‖`.
fn care_residual(a: &M, b: &M, q: &M, r: &M, x: &M) -> f64 {
    let g = mm(&mm(b, &autodyne::linalg::inv(r.view()).unwrap()), &tr(b));
    rel_norm(&combine(&[(1.0, &mm(&tr(a), x)), (1.0, &mm(x, a)), (-1.0, &mm(&mm(x, &g), x)), (1.0, q)]), x)
}

/// `‖aᵀ x a - x - aᵀ x b (r + bᵀ x b)⁻¹ bᵀ x a + q‖ / ‖x‖`.
fn dare_residual(a: &M, b: &M, q: &M, r: &M, x: &M) -> f64 {
    let xa = mm(x, a);
    let s = autodyne::linalg::inv(combine(&[(1.0, r), (1.0, &mm(&mm(&tr(b), x), b))]).view()).unwrap();
    let k = mm(&s, &mm(&tr(b), &xa));
    rel_norm(&combine(&[(1.0, &mm(&tr(a), &xa)), (-1.0, x), (-1.0, &mm(&mm(&mm(&tr(a), x), b), &k)), (1.0, q)]), x)
}

#[test]
fn spectral_estimates_match_scipy() {
    let s = &fixtures()["spectral"];
    let (x, y) = (array(&s["x"]), array(&s["y"]));
    let fs = 1000.0;
    let flat = |v: &[autodyne::units::Complex<f64>]| v.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>();
    let check = |name: &str, f: &[f64], p: &[f64], key: &str, field: &str| {
        assert_close(&format!("{name} freqs"), f, &f64s(&s[key]["f"]), 1e-12);
        assert_close(name, p, &nd(&s[key][field]).0, 1e-10);
    };

    let (f, p) = welch(x.view(), fs, 1, &Segments::welch().nperseg(128), Average::Mean).unwrap();
    check("welch", &f, p.as_slice(), "welch", "p");
    let seg = Segments::welch().nperseg(100).noverlap(30).nfft(256).detrend(Detrend::Linear).scaling(Scaling::Spectrum);
    let (f, p) = welch(x.view(), fs, 1, &seg, Average::Median).unwrap();
    check("welch median", &f, p.as_slice(), "welch_median", "p");
    let xt = x.view().transpose().to_owned();
    let seg = Segments::welch().nperseg(64).window(WindowSpec::Kaiser { beta: 5.0 }).onesided(false);
    let (f, p) = welch(xt.view(), fs, 0, &seg, Average::Mean).unwrap();
    check("welch two-sided, axis 0", &f, p.as_slice(), "welch_twosided_axis0", "p");

    let (f, p) = periodogram(x.view(), fs, 1, WindowSpec::Boxcar, None, Detrend::Constant, Scaling::Density).unwrap();
    check("periodogram", &f, p.as_slice(), "periodogram", "p");
    let (f, p) = periodogram(x.view(), fs, 1, WindowSpec::Hann, Some(800), Detrend::Constant, Scaling::Spectrum).unwrap();
    check("periodogram truncated", &f, p.as_slice(), "periodogram_short", "p");

    let (f, p) = csd(x.view(), y.view(), fs, 1, &Segments::welch().nperseg(128), Average::Mean).unwrap();
    assert_close("csd freqs", &f, &f64s(&s["csd"]["f"]), 1e-12);
    assert_close("csd", &flat(p.as_slice()), &flat(&complexes(&s["csd"]["p"])), 1e-10);
    let (_, cxy) = coherence(x.view(), y.view(), fs, 1, &Segments::welch().nperseg(128)).unwrap();
    assert_close("coherence", cxy.as_slice(), &nd(&s["coherence"]["c"]).0, 1e-10);

    for (mode, key) in [(SpectrogramMode::Psd, "psd"), (SpectrogramMode::Magnitude, "magnitude"), (SpectrogramMode::Angle, "angle"), (SpectrogramMode::Phase, "phase")] {
        let (f, t, sxx) = spectrogram(x.view(), fs, 1, &Segments::spectrogram().nperseg(64), mode).unwrap();
        let k = format!("spectrogram_{key}");
        assert_close(&format!("{k} freqs"), &f, &f64s(&s[&k]["f"]), 1e-12);
        assert_close(&format!("{k} times"), &t, &f64s(&s[&k]["t"]), 1e-12);
        assert_close(&k, sxx.as_slice(), &nd(&s[&k]["s"]).0, 1e-9);
    }

    let (f, t, z) = stft(x.view(), fs, 1, &Segments::stft().nperseg(64), StftOptions::default()).unwrap();
    assert_eq!(z.shape(), f64s(&s["stft"]["shape"]).iter().map(|&d| d as usize).collect::<Vec<_>>().as_slice());
    assert_close("stft freqs", &f, &f64s(&s["stft"]["f"]), 1e-12);
    assert_close("stft times", &t, &f64s(&s["stft"]["t"]), 1e-12);
    assert_close("stft", &flat(z.as_slice()), &flat(&complexes(&s["stft"]["z"])), 1e-10);
    let seg = Segments::stft().nperseg(50).noverlap(20).onesided(false);
    let (f, t, z2) = stft(x.view(), fs, 1, &seg, StftOptions { boundary: Boundary::Odd, padded: false }).unwrap();
    assert_eq!(z2.shape(), f64s(&s["stft_odd_twosided"]["shape"]).iter().map(|&d| d as usize).collect::<Vec<_>>().as_slice());
    assert_close("stft two-sided freqs", &f, &f64s(&s["stft_odd_twosided"]["f"]), 1e-12);
    assert_close("stft two-sided times", &t, &f64s(&s["stft_odd_twosided"]["t"]), 1e-12);
    assert_close("stft two-sided", &flat(z2.as_slice()), &flat(&complexes(&s["stft_odd_twosided"]["z"])), 1e-10);

    let (t, back) = istft(z.view(), fs, IstftOptions { nperseg: Some(64), ..Default::default() }).unwrap();
    // SciPy builds the times from the first axis (the channels here); ours follow the signal
    assert_eq!(t.len(), back.shape()[1]);
    assert!((t[1] - 1.0 / fs).abs() < 1e-15);
    assert_close("istft", back.as_slice(), &nd(&s["istft"]["x"]).0, 1e-10);
    // the round trip recovers the signal
    let n = x.shape()[1];
    for ch in 0..2 {
        let a = &back.as_slice()[ch * back.shape()[1]..ch * back.shape()[1] + n];
        assert_close("round trip", a, &x.as_slice()[ch * n..(ch + 1) * n], 1e-10);
    }
}
