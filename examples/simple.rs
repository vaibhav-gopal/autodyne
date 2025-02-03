use dsp_lib::numbers::real::*;
use dsp_lib::numbers::complex::*;
use dsp_lib::numbers::*;
use dsp_lib::{complex, re, im};

fn main() {
    let a: f32 = 64.0;
    let b: f32 = 32.0;
    
    let result = a + b;
    let comp = Complex::new(a + b, a + b);
    let comp2 = Complex::new(comp.re + comp.im, comp.im - comp.re);
    let result2 = re!(32f32) + im!(64f32);

    ()
}