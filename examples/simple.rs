use dsp_lib::{
    complex, 
    re, 
    im, 
    buffer::*,
    units::{
        *,
        complex::*
    }
};

fn main() {
    let bruh = 32f32;
    let comp = complex!(32f32, 38f32);
    let res = bruh * comp.re;
    
    println!("{}", res);
    ()
}