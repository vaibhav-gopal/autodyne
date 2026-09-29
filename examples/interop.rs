//! Moving signals across boundaries: tensors, raw memory and byte streams.
//!
//!     cargo run --release --example interop               # writes to target/examples-out/
//!     cargo run --release --example interop -- <out_dir>
//!
//! 1. An ML-style `[batch, channel, time]` tensor (`NdArray`) is filtered along time for every
//!    (batch, channel) lane, then measured per lane with the `Signal` traits.
//! 2. Its memory is exported as bytes (`as_bytes`, no copy) and viewed back as a typed tensor the way
//!    another runtime would hand it over (`DynView`, no copy).
//! 3. One lane is streamed to a 16-bit PCM file (`SampleWriter`), read back (`SampleReader`) and
//!    compared with the original.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use autodyne::dynamic::{DynArray, DynView};
use autodyne::filter::{Biquad, BUTTERWORTH_Q};
use autodyne::osc::{Noise, Sine};
use autodyne::units::DType;
use autodyne::signal::{
    Axis, NdArray, SampleEncoding, SampleReader, SampleWriter, Signal, SignalRead, SignalWrite, Source,
};

const FS: f32 = 16_000.0;
const BATCH: usize = 3;
const CHANNELS: usize = 2;
const TIME: usize = 16_000; // one second per lane

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "target/examples-out".into());
    std::fs::create_dir_all(&out_dir)?;

    // 1. A batch of noisy tones: clip b has a (500 * (b + 1)) Hz tone, channel c at 0.35 * (c + 1),
    //    keeping peaks under full scale so the 16-bit round trip in step 3 doesn't clip.
    let mut tensor = NdArray::<f32>::zeros(&[BATCH, CHANNELS, TIME])?.with_labels(&[Axis::Batch, Axis::Channel, Axis::Time])?;
    let time = tensor.axis_of(Axis::Time).unwrap();
    let mut lane_index = 0;
    tensor.for_each_lane(time, |lane| {
        let (b, c) = (lane_index / CHANNELS, lane_index % CHANNELS);
        Sine::new(500.0 * (b + 1) as f32, FS).with_amplitude(0.35 * (c + 1) as f32).mix(Noise::new(lane_index as u64).scaled(0.2)).fill(lane);
        lane_index += 1;
    })?;
    let noisy_rms: Vec<f32> = tensor.view().lanes(time)?.iter().map(|l| l.as_slice().unwrap().rms().unwrap()).collect();

    // every lane is an independent clip: filter each with a fresh 2 kHz low-pass
    tensor.for_each_lane(time, |lane| Biquad::lowpass(2_000.0, FS, BUTTERWORTH_Q as f32).process(lane))?;
    println!("filtered {} lanes of a {:?} tensor along its {:?} axis", noisy_rms.len(), tensor.shape(), Axis::Time);
    for (i, lane) in tensor.view().lanes(time)?.iter().enumerate() {
        let x = lane.as_slice().unwrap();
        println!("  batch {} channel {}: rms {:.3} -> {:.3}, peak {:.3}", i / CHANNELS, i % CHANNELS, noisy_rms[i], x.rms().unwrap(), x.peak());
    }

    // 2. Hand the memory to "another runtime" and take it back, both without copying.
    let shared = DynArray::from_array(tensor);
    let bytes = shared.as_bytes();
    let view = DynView::new(bytes, DType::F32, &[BATCH, CHANNELS, TIME])?;
    let typed = view.typed::<f32>()?;
    let original = shared.as_array::<f32>()?;
    assert_eq!(typed.as_slice().unwrap().as_ptr(), original.as_slice().as_ptr(), "zero-copy round trip");
    println!("\nshared {} bytes as a {} {:?} view: same memory, no copy", bytes.len(), view.dtype(), view.shape());

    // 3. Stream batch 2 / channel 1 to 16-bit PCM and back.
    let lane = typed.index_axis(0, 2)?.index_axis(0, 1)?; // [time]
    let samples = lane.as_slice().unwrap();
    let path = Path::new(&out_dir).join("interop_lane.pcm");
    let encoding = SampleEncoding::little(DType::I16);
    let mut writer = SampleWriter::new(BufWriter::new(File::create(&path)?), encoding)?;
    for block in samples.chunks(512) {
        writer.write_all_samples(block)?;
    }
    writer.flush_samples()?;
    drop(writer);

    let mut reader = SampleReader::<_, f32>::new(BufReader::new(File::open(&path)?), encoding)?;
    let mut back = Vec::new();
    reader.read_to_end_samples(&mut back)?;
    let error_db = back.distance(samples)?.log10() * 20.0 - samples.norm_l2().log10() * 20.0;
    println!(
        "\nwrote {} and read {} samples back as i16 PCM: {} bytes on disk, error {:.1} dB below the signal",
        samples.len(),
        back.len(),
        std::fs::metadata(&path)?.len(),
        -error_db,
    );
    Ok(())
}
