//! autodyne against SciPy: reference outputs from `tests/scipy/gen_fixtures.py` (SciPy 1.18),
//! committed as `tests/scipy/fixtures.json`.

#![cfg(feature = "faer")]

use std::sync::OnceLock;

use autodyne::filter::design::{besselap, firls, firwin, firwin2, iirfilter, kaiserord, remez, Band, BesselNorm, Design, IirKind, RemezType};
use autodyne::filter::{filtfilt, lfilter, lfilter_with_state, lfilter_zi, sosfilt, sosfilt_with_state, sosfilt_zi, sosfiltfilt, Pad};
use autodyne::linalg::expm;
use autodyne::signal::NdArray;
use autodyne::spectral::{get_window, WindowSpec};
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
