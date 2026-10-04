//! The smallest autodyne program: a source through a processor, a block at a time, measured.
//!
//!     cargo run --example simple
//!
//! A 440 Hz sine and a 9 kHz one are mixed, a Butterworth low-pass at 1 kHz removes the high one,
//! and the levels before and after are printed.

use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::Sine;
use autodyne::prelude::*;

fn main() {
    let fs = 48_000.0f32;
    let mix = || Sine::new(440.0, fs).mix(Sine::new(9_000.0, fs));

    let mut dry = vec![0.0f32; 4_800];
    mix().fill(&mut dry);

    // the same mix, low-passed as it is generated
    let mut wet = vec![0.0f32; 4_800];
    mix().through(Biquad::lowpass(1_000.0, BUTTERWORTH_Q as f32, fs)).fill(&mut wet);

    let db = |x: &[f32]| x.rms_db().unwrap_or(f32::NEG_INFINITY);
    println!("mix:        {:6.2} dBFS rms", db(&dry));
    println!("low-passed: {:6.2} dBFS rms (the 440 Hz sine alone is {:.2})", db(&wet[480..]), 20.0 * (0.5f32).sqrt().log10());
}
