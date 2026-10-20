//! autodyne's pYIN (`analysis::PitchTracker`) against librosa's `pyin`: reference outputs from
//! `tests/librosa/gen_pyin.py` (librosa 1.0), committed as `tests/librosa/pyin.json`.

use autodyne::analysis::PitchTracker;
use serde_json::Value;

fn cents(a: f64, b: f64) -> f64 {
    1_200.0 * (a / b).log2()
}

#[test]
fn pyin_matches_librosa() {
    let fx: Value = serde_json::from_str(include_str!("librosa/pyin.json")).expect("valid JSON");
    let num = |k: &str| fx[k].as_f64().expect("number");
    let (sr, fmin, fmax) = (num("sr"), num("fmin"), num("fmax"));
    let (frame, hop) = (num("frame_length") as usize, num("hop_length") as usize);
    for sig in fx["signals"].as_array().expect("signals") {
        let name = sig["name"].as_str().expect("name");
        let y: Vec<f64> = sig["pcm"].as_array().expect("pcm").iter().map(|v| v.as_f64().expect("pcm") / 32768.0).collect();
        let f0: Vec<Option<f64>> = sig["f0"].as_array().expect("f0").iter().map(Value::as_f64).collect();
        let prob: Vec<f64> = sig["voiced_prob"].as_array().expect("voiced_prob").iter().map(|v| v.as_f64().expect("prob")).collect();

        let track = PitchTracker::new(fmin, fmax, sr).with_frame(frame, hop).track(&y);
        assert_eq!(track.len(), f0.len(), "{name}: frame count");

        // the observation stage alone: the voiced probability of every frame
        let worst_prob = track.iter().zip(&prob).map(|(v, p)| (v.probability - p).abs()).fold(0.0, f64::max);
        // the decoded track: voicing, and pitch where both are voiced (librosa reports the bin's
        // centre, 10 cents apart; ours is the candidate's own frequency in that bin)
        let voicing_differs = track.iter().zip(&f0).filter(|(v, f)| v.frequency.is_some() != f.is_some()).count();
        let pitch_off = track.iter().zip(&f0).filter_map(|(v, f)| Some(cents(v.frequency?, (*f)?).abs())).fold(0.0, f64::max);
        println!("{name}: {} frames, voiced prob within {worst_prob:e}, voicing differs on {voicing_differs}, pitch within {pitch_off:.2} cents", f0.len());
        assert!(worst_prob < 1e-9, "{name}: voiced probability off by {worst_prob}");
        assert_eq!(voicing_differs, 0, "{name}: voicing differs on {voicing_differs} frames");
        assert!(pitch_off <= 5.0 + 1e-9, "{name}: pitch {pitch_off} cents from librosa's bin");
    }
}

/// The fixtures' signals end to end, 10 times over (55.5 s of audio at 16 kHz): best of 5 for one
/// offline pass. `cargo test --release --test librosa_parity -- --ignored --nocapture`; time
/// librosa.pyin on the same samples and parameters (after a warm-up call for numba) to compare.
#[test]
#[ignore]
fn timing() {
    let fx: Value = serde_json::from_str(include_str!("librosa/pyin.json")).expect("valid JSON");
    let mut y = Vec::new();
    for _ in 0..10 {
        for sig in fx["signals"].as_array().expect("signals") {
            y.extend(sig["pcm"].as_array().expect("pcm").iter().map(|v| v.as_f64().expect("pcm") / 32768.0));
        }
    }
    let t = PitchTracker::new(65.0, 1000.0, 16_000.0).with_frame(1024, 128);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let start = std::time::Instant::now();
        std::hint::black_box(t.track(&y));
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!("TIMING pyin {:.1} s of audio in {:.1} ms ({:.0}x real time)", y.len() as f64 / 16_000.0, best * 1e3, y.len() as f64 / 16_000.0 / best);
}
