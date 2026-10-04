//! WAV files: read and write RIFF/WAVE audio in memory, including the sampler metadata (root key
//! and loop points) of the `smpl` chunk.
//!
//! Reads 8/16/24/32-bit integer PCM and 32/64-bit float, plain or `WAVE_FORMAT_EXTENSIBLE`; writes
//! 16/24-bit PCM or 32-bit float. No dependencies; loading is meant for setup time, not the audio
//! thread (it allocates).
//!
//! ```
//! use autodyne::wav::{self, WavFormat};
//!
//! let audio = wav::Wav { sample_rate: 44_100, channels: vec![vec![0.0f32, 0.5, -0.5]], ..Default::default() };
//! let bytes = wav::write(&audio, WavFormat::Pcm16);
//! let back: wav::Wav<f32> = wav::read(&bytes).unwrap();
//! assert_eq!(back.channels[0].len(), 3);
//! ```

use crate::units::*;
use thiserror::Error;

/// Audio and metadata from (or for) a WAV file.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav<T> {
    /// frames per second
    pub sample_rate: u32,
    /// planar samples, one Vec per channel, in [-1, 1]
    pub channels: Vec<Vec<T>>,
    /// the `smpl` chunk's MIDI unity note plus its pitch fraction, in semitones
    pub root_key: Option<f64>,
    /// the `smpl` chunk's loops
    pub loops: Vec<WavLoop>,
}

impl<T> Default for Wav<T> {
    fn default() -> Self {
        Self { sample_rate: 48_000, channels: Vec::new(), root_key: None, loops: Vec::new() }
    }
}

impl<T> Wav<T> {
    /// Frames (samples per channel).
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

/// A loop from a `smpl` chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavLoop {
    /// first frame of the loop
    pub start: usize,
    /// one past the last frame (the file stores the last frame inclusively)
    pub end: usize,
    /// how the loop plays
    pub kind: WavLoopKind,
}

/// How a `smpl` loop plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavLoopKind {
    /// start to end, repeatedly
    Forward,
    /// back and forth between start and end
    PingPong,
    /// end to start, repeatedly
    Backward,
}

/// Errors reading WAV files.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WavError {
    /// The file isn't RIFF/WAVE.
    #[error("not a RIFF/WAVE file")]
    NotWave,
    /// The file ends inside a chunk or header.
    #[error("the file ends inside a chunk or header")]
    Truncated,
    /// No `fmt ` chunk before the audio.
    #[error("no `fmt ` chunk before the audio data")]
    MissingFormat,
    /// No `data` chunk.
    #[error("no `data` chunk")]
    MissingData,
    /// A sample format this reader doesn't decode.
    #[error("unsupported sample format (format tag {tag:#06x}, {bits} bits)")]
    Unsupported {
        /// The `fmt ` chunk's format tag.
        tag: u16,
        /// Bits per sample.
        bits: u16,
    },
}

/// Sample encodings [`write()`] produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WavFormat {
    /// 16-bit integer PCM.
    Pcm16,
    /// 24-bit integer PCM.
    Pcm24,
    /// 32-bit IEEE float.
    Float32,
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, WavError> {
    b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or(WavError::Truncated)
}
fn u32_at(b: &[u8], at: usize) -> Result<u32, WavError> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or(WavError::Truncated)
}

#[derive(Clone, Copy)]
struct Format {
    float: bool,
    channels: usize,
    sample_rate: u32,
    bits: u16,
    block_align: usize,
}

/// Parses a WAV file held in memory.
pub fn read<T: Float>(bytes: &[u8]) -> Result<Wav<T>, WavError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotWave);
    }
    let mut format = None;
    let mut data = None;
    let mut root_key = None;
    let mut loops = Vec::new();
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4)? as usize;
        let body_start = at + 8;
        // a truncated final data chunk is common (streams that never patched the size): take what is there
        let body = &bytes[body_start..(body_start + size).min(bytes.len())];
        match id {
            b"fmt " => {
                let tag = u16_at(body, 0)?;
                let bits = u16_at(body, 14)?;
                let tag = if tag == 0xFFFE { u16_at(body, 24)? } else { tag }; // extensible: the subformat GUID starts with the tag
                let float = match (tag, bits) {
                    (1, 8 | 16 | 24 | 32) => false,
                    (3, 32 | 64) => true,
                    _ => return Err(WavError::Unsupported { tag, bits }),
                };
                let channels = u16_at(body, 2)? as usize;
                let block_align = u16_at(body, 12)? as usize;
                if channels == 0 || block_align < channels * bits as usize / 8 {
                    return Err(WavError::Unsupported { tag, bits });
                }
                format = Some(Format { float, channels, sample_rate: u32_at(body, 4)?, bits, block_align });
            }
            b"data" => data = Some(body),
            b"smpl" => {
                let unity = u32_at(body, 12)?;
                let fraction = u32_at(body, 16)?;
                root_key = Some(unity as f64 + fraction as f64 / 4_294_967_296.0);
                let count = u32_at(body, 28)? as usize;
                for i in 0..count {
                    let base = 36 + i * 24;
                    let kind = match u32_at(body, base + 4)? {
                        1 => WavLoopKind::PingPong,
                        2 => WavLoopKind::Backward,
                        _ => WavLoopKind::Forward,
                    };
                    let (start, last) = (u32_at(body, base + 8)? as usize, u32_at(body, base + 12)? as usize);
                    if last >= start {
                        loops.push(WavLoop { start, end: last + 1, kind });
                    }
                }
            }
            _ => {}
        }
        at = body_start + size + (size & 1); // chunks are padded to even sizes
    }
    let format = format.ok_or(WavError::MissingFormat)?;
    let data = data.ok_or(WavError::MissingData)?;
    let frames = data.len() / format.block_align;
    let width = format.bits as usize / 8;
    let mut channels = vec![Vec::with_capacity(frames); format.channels];
    for frame in data.chunks_exact(format.block_align) {
        for (c, out) in channels.iter_mut().enumerate() {
            let s = &frame[c * width..(c + 1) * width];
            let value = match (format.float, format.bits) {
                (false, 8) => (s[0] as f64 - 128.0) / 128.0,
                (false, 16) => i16::from_le_bytes([s[0], s[1]]) as f64 / 32_768.0,
                (false, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f64 / 8_388_608.0,
                (false, _) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0,
                (true, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
                (true, _) => f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
            };
            out.push(T::_lit(value));
        }
    }
    let loops = loops.into_iter().filter(|l| l.end <= frames).collect();
    Ok(Wav { sample_rate: format.sample_rate, channels, root_key, loops })
}

/// Encodes `wav` as a WAV file (with a `smpl` chunk when it has a root key or loops). Integer
/// formats scale by 2^(bits-1), the inverse of [`read()`], and clip at full scale. Panics if the channels differ in length.
pub fn write<T: Float>(wav: &Wav<T>, format: WavFormat) -> Vec<u8> {
    let frames = wav.frames();
    assert!(wav.channels.iter().all(|c| c.len() == frames), "every channel needs the same length");
    let channels = wav.channels.len();
    let (tag, bits): (u16, u16) = match format {
        WavFormat::Pcm16 => (1, 16),
        WavFormat::Pcm24 => (1, 24),
        WavFormat::Float32 => (3, 32),
    };
    let block_align = channels * bits as usize / 8;
    let data_len = frames * block_align;
    let has_smpl = wav.root_key.is_some() || !wav.loops.is_empty();
    let smpl_len = 36 + 24 * wav.loops.len();
    let total = 4 + (8 + 16) + (8 + data_len + (data_len & 1)) + if has_smpl { 8 + smpl_len } else { 0 };
    let mut out = Vec::with_capacity(total + 8);
    let u16le = |out: &mut Vec<u8>, v: u16| out.extend_from_slice(&v.to_le_bytes());
    let u32le = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
    out.extend_from_slice(b"RIFF");
    u32le(&mut out, total as u32);
    out.extend_from_slice(b"WAVEfmt ");
    u32le(&mut out, 16);
    u16le(&mut out, tag);
    u16le(&mut out, channels as u16);
    u32le(&mut out, wav.sample_rate);
    u32le(&mut out, wav.sample_rate * block_align as u32);
    u16le(&mut out, block_align as u16);
    u16le(&mut out, bits);
    out.extend_from_slice(b"data");
    u32le(&mut out, data_len as u32);
    for i in 0..frames {
        for channel in &wav.channels {
            let x = channel[i].to_f64().unwrap_or(0.0);
            match format {
                WavFormat::Pcm16 => out.extend_from_slice(&((x * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16).to_le_bytes()),
                WavFormat::Pcm24 => out.extend_from_slice(&((x * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32).to_le_bytes()[..3]),
                WavFormat::Float32 => out.extend_from_slice(&(x as f32).to_le_bytes()),
            }
        }
    }
    if data_len & 1 == 1 {
        out.push(0);
    }
    if has_smpl {
        out.extend_from_slice(b"smpl");
        u32le(&mut out, smpl_len as u32);
        let root = wav.root_key.unwrap_or(60.0);
        let unity = root.floor();
        u32le(&mut out, 0); // manufacturer
        u32le(&mut out, 0); // product
        u32le(&mut out, (1e9 / wav.sample_rate as f64).round() as u32); // sample period, ns
        u32le(&mut out, unity as u32);
        u32le(&mut out, ((root - unity) * 4_294_967_296.0) as u32);
        u32le(&mut out, 0); // SMPTE format
        u32le(&mut out, 0); // SMPTE offset
        u32le(&mut out, wav.loops.len() as u32);
        u32le(&mut out, 0); // sampler data
        for (i, l) in wav.loops.iter().enumerate() {
            u32le(&mut out, i as u32);
            u32le(&mut out, match l.kind {
                WavLoopKind::Forward => 0,
                WavLoopKind::PingPong => 1,
                WavLoopKind::Backward => 2,
            });
            u32le(&mut out, l.start as u32);
            u32le(&mut out, l.end.saturating_sub(1) as u32);
            u32le(&mut out, 0); // fraction
            u32le(&mut out, 0); // play count: forever
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize, channels: usize) -> Vec<Vec<f64>> {
        (0..channels).map(|c| (0..frames).map(|i| 0.9 * ((i as f64 * 0.01) + c as f64).sin()).collect()).collect()
    }

    #[test]
    fn round_trips_every_format_with_metadata() {
        for (format, tolerance) in [(WavFormat::Pcm16, 1.0 / 32_000.0), (WavFormat::Pcm24, 1.0 / 8_000_000.0), (WavFormat::Float32, 1e-7)] {
            let wav = Wav {
                sample_rate: 44_100,
                channels: tone(1_001, 2),
                root_key: Some(57.25),
                loops: vec![WavLoop { start: 100, end: 900, kind: WavLoopKind::Forward }, WavLoop { start: 5, end: 6, kind: WavLoopKind::PingPong }],
            };
            let bytes = write(&wav, format);
            let back: Wav<f64> = read(&bytes).unwrap();
            assert_eq!((back.sample_rate, back.frames(), back.channels.len()), (44_100, 1_001, 2));
            assert_eq!(back.loops, wav.loops);
            assert!((back.root_key.unwrap() - 57.25).abs() < 1e-6);
            let err = back.channels.iter().flatten().zip(wav.channels.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < tolerance, "{format:?}: {err}");
        }
    }

    #[test]
    fn reads_other_encodings_and_skips_unknown_chunks() {
        // build a 2-frame stereo file by hand: an odd-sized unknown chunk, then extensible 32-bit int
        let mut b = b"RIFF\0\0\0\0WAVE".to_vec();
        b.extend_from_slice(b"junk\x03\0\0\0abc\0");
        b.extend_from_slice(b"fmt \x28\0\0\0");
        b.extend_from_slice(&0xFFFEu16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&48_000u32.to_le_bytes());
        b.extend_from_slice(&(48_000u32 * 8).to_le_bytes());
        b.extend_from_slice(&8u16.to_le_bytes());
        b.extend_from_slice(&32u16.to_le_bytes());
        b.extend_from_slice(&22u16.to_le_bytes()); // extension size
        b.extend_from_slice(&32u16.to_le_bytes()); // valid bits
        b.extend_from_slice(&3u32.to_le_bytes()); // channel mask
        b.extend_from_slice(&1u16.to_le_bytes()); // subformat: PCM
        b.extend_from_slice(&[0; 14]);
        b.extend_from_slice(b"data\x10\0\0\0");
        for v in [i32::MAX, i32::MIN, 0, 1 << 30] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let wav: Wav<f32> = read(&b).unwrap();
        assert_eq!(wav.channels, vec![vec![1.0, 0.0], vec![-1.0, 0.5]]);
        assert_eq!(wav.root_key, None);

        // 8-bit unsigned, mono
        let mut b = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x44\xac\0\0\x44\xac\0\0\x01\0\x08\0data\x03\0\0\0".to_vec();
        b.extend_from_slice(&[128, 255, 0, 0]);
        let wav: Wav<f64> = read(&b).unwrap();
        assert_eq!(wav.channels, vec![vec![0.0, 127.0 / 128.0, -1.0]]);
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        assert_eq!(read::<f32>(b"RIFX\0\0\0\0WAVE"), Err(WavError::NotWave));
        assert_eq!(read::<f32>(b"RIFF\0\0\0\0WAVEdata\0\0\0\0"), Err(WavError::MissingFormat));
        let adpcm = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x02\0\x01\0\x44\xac\0\0\x44\xac\0\0\x01\0\x04\0";
        assert_eq!(read::<f32>(adpcm), Err(WavError::Unsupported { tag: 2, bits: 4 }));
        let no_data = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x44\xac\0\0\x88\x58\x01\0\x02\0\x10\0";
        assert_eq!(read::<f32>(no_data), Err(WavError::MissingData));
        assert_eq!(read::<f32>(b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0"), Err(WavError::Truncated));
    }
}
