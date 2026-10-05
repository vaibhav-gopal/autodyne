//! Signal streams: signals read, written and seeked a block of samples at a time, for data that is
//! generated or arrives at runtime (generators, files, sockets, device callbacks, other processes).
//!
//! The traits mirror `std::io::{Read, Write, Seek}` but move samples, not bytes, and count in samples:
//! - [`SignalRead`] / [`SignalWrite`] / [`SignalSeek`], and [`SignalStream`] = read + seek.
//! - [`SignalCursor`]: an in-memory signal as a stream.
//! - [`SourceStream`]: any `Source` as an endless (or length-limited) stream.
//! - [`SampleReader`] / [`SampleWriter`]: the byte boundary. They wrap any `std::io::Read` / `Write`
//!   (files, sockets, pipes, FFI buffers) and convert with a [`SampleEncoding`], so signal code never
//!   handles bytes, partial samples or endianness.

use crate::alloc_prelude::*;
use std::io::{self, Read, Seek, SeekFrom, Write};
use core::marker::PhantomData;

use super::{Endian, Source};
use crate::processor::Processor;
use crate::units::*;

// TRAITS ==========================================================================================

/// A source of samples read in blocks. `Ok(0)` means the stream has ended (for a non-empty `out`).
pub trait SignalRead {
    /// The sample type.
    type Sample: Copy;

    /// Reads up to `out.len()` samples into `out` and returns how many were read.
    fn read_samples(&mut self, out: &mut [Self::Sample]) -> io::Result<usize>;

    /// Fills `out` completely, or fails with `UnexpectedEof` if the stream ends first.
    fn read_exact_samples(&mut self, mut out: &mut [Self::Sample]) -> io::Result<()> {
        while !out.is_empty() {
            match self.read_samples(out)? {
                0 => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "signal stream ended early")),
                n => out = &mut out[n..],
            }
        }
        Ok(())
    }

    /// Reads until the stream ends, appending to `out`; returns the number of samples read.
    /// Never returns for an endless stream (e.g. an unlimited `SourceStream`).
    fn read_to_end_samples(&mut self, out: &mut Vec<Self::Sample>) -> io::Result<usize>
    where
        Self::Sample: Default,
    {
        let start = out.len();
        let mut block = [Self::Sample::default(); 1024];
        loop {
            match self.read_samples(&mut block)? {
                0 => return Ok(out.len() - start),
                n => out.extend_from_slice(&block[..n]),
            }
        }
    }

    /// Pumps samples into `writer`, `block` at a time, until this stream ends; returns the count.
    fn copy_to<W: SignalWrite<Sample = Self::Sample> + ?Sized>(&mut self, writer: &mut W, block: usize) -> io::Result<u64>
    where
        Self::Sample: Default,
    {
        let mut buf = vec![Self::Sample::default(); block.max(1)];
        let mut total = 0;
        loop {
            match self.read_samples(&mut buf)? {
                0 => return Ok(total),
                n => {
                    writer.write_all_samples(&buf[..n])?;
                    total += n as u64;
                }
            }
        }
    }

    /// Runs every block read through a processor (or a tuple chain of them).
    fn through<P>(self, processor: P) -> ProcessedStream<Self, P>
    where
        Self: Sized,
        Self::Sample: Float,
        P: Processor<Self::Sample>,
    {
        ProcessedStream { stream: self, processor }
    }
}

/// A sink of samples written in blocks.
pub trait SignalWrite {
    /// The sample type.
    type Sample: Copy;

    /// Writes up to `samples.len()` samples and returns how many were accepted.
    fn write_samples(&mut self, samples: &[Self::Sample]) -> io::Result<usize>;

    /// Pushes any buffered samples to their destination.
    fn flush_samples(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Writes all of `samples`, or fails with `WriteZero` if the sink stops accepting them.
    fn write_all_samples(&mut self, mut samples: &[Self::Sample]) -> io::Result<()> {
        while !samples.is_empty() {
            match self.write_samples(samples)? {
                0 => return Err(io::Error::new(io::ErrorKind::WriteZero, "signal sink accepted no samples")),
                n => samples = &samples[n..],
            }
        }
        Ok(())
    }
}

/// Random access in a stream, with positions counted in samples.
pub trait SignalSeek {
    /// Moves to `pos` and returns the new position from the start.
    fn seek_samples(&mut self, pos: SeekFrom) -> io::Result<u64>;

    /// The current position from the start, in samples.
    fn sample_position(&mut self) -> io::Result<u64> {
        self.seek_samples(SeekFrom::Current(0))
    }
}

/// A readable, seekable signal stream (a file of samples, an in-memory signal, ...).
pub trait SignalStream: SignalRead + SignalSeek {}

impl<S: SignalRead + SignalSeek + ?Sized> SignalStream for S {}

impl<R: SignalRead + ?Sized> SignalRead for &mut R {
    type Sample = R::Sample;
    fn read_samples(&mut self, out: &mut [R::Sample]) -> io::Result<usize> {
        (**self).read_samples(out)
    }
}

impl<W: SignalWrite + ?Sized> SignalWrite for &mut W {
    type Sample = W::Sample;
    fn write_samples(&mut self, samples: &[W::Sample]) -> io::Result<usize> {
        (**self).write_samples(samples)
    }
    fn flush_samples(&mut self) -> io::Result<()> {
        (**self).flush_samples()
    }
}

/// Resolves a sample-based `SeekFrom` against a stream of `len` samples at `current`.
fn resolve_seek(pos: SeekFrom, current: u64, len: u64) -> io::Result<u64> {
    let target = match pos {
        SeekFrom::Start(n) => Some(n),
        SeekFrom::End(d) => len.checked_add_signed(d),
        SeekFrom::Current(d) => current.checked_add_signed(d),
    };
    target.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek to a negative sample position"))
}

// IN-MEMORY AND GENERATED STREAMS =================================================================

/// An in-memory signal as a stream, like `std::io::Cursor` but in samples.
/// Writing past the end of a `Vec` grows it; a borrowed slice stops at its end.
#[derive(Debug, Clone)]
pub struct SignalCursor<C, T> {
    inner: C,
    pos: usize,
    _sample: PhantomData<T>,
}

impl<C, T> SignalCursor<C, T> {
    /// A cursor at the start of `inner`.
    pub fn new(inner: C) -> Self {
        Self { inner, pos: 0, _sample: PhantomData }
    }
    /// The underlying signal.
    pub fn get_ref(&self) -> &C {
        &self.inner
    }
    /// Gives back the underlying signal.
    pub fn into_inner(self) -> C {
        self.inner
    }
}

impl<T: Copy, C: AsRef<[T]>> SignalRead for SignalCursor<C, T> {
    type Sample = T;
    fn read_samples(&mut self, out: &mut [T]) -> io::Result<usize> {
        let data = self.inner.as_ref();
        let n = out.len().min(data.len().saturating_sub(self.pos));
        out[..n].copy_from_slice(&data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl<T: Copy, C: AsRef<[T]>> SignalSeek for SignalCursor<C, T> {
    fn seek_samples(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target = resolve_seek(pos, self.pos as u64, self.inner.as_ref().len() as u64)?;
        self.pos = usize::try_from(target).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "position too large"))?;
        Ok(target)
    }
}

impl<T: Copy + Default> SignalWrite for SignalCursor<Vec<T>, T> {
    type Sample = T;
    fn write_samples(&mut self, samples: &[T]) -> io::Result<usize> {
        let end = self.pos + samples.len();
        if self.inner.len() < end {
            // like io::Cursor: a gap left by seeking past the end is filled with defaults (zeros)
            self.inner.resize(end, T::default());
        }
        self.inner[self.pos..end].copy_from_slice(samples);
        self.pos = end;
        Ok(samples.len())
    }
}

impl<T: Copy> SignalWrite for SignalCursor<&mut [T], T> {
    type Sample = T;
    fn write_samples(&mut self, samples: &[T]) -> io::Result<usize> {
        let n = samples.len().min(self.inner.len().saturating_sub(self.pos));
        self.inner[self.pos..self.pos + n].copy_from_slice(&samples[..n]);
        self.pos += n;
        Ok(n)
    }
}

/// A `Source` read as a stream; endless unless created with a length (`Source::stream_for`).
#[derive(Debug, Clone)]
pub struct SourceStream<S> {
    source: S,
    remaining: Option<u64>,
}

impl<S> SourceStream<S> {
    /// `source` as a stream of `length` samples (`None`: endless).
    pub fn new(source: S, length: Option<u64>) -> Self {
        Self { source, remaining: length }
    }
    /// The source, e.g. to change its parameters while streaming.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

impl<S: Source> SignalRead for SourceStream<S> {
    type Sample = S::Sample;
    fn read_samples(&mut self, out: &mut [S::Sample]) -> io::Result<usize> {
        let n = match self.remaining {
            Some(r) => out.len().min(usize::try_from(r).unwrap_or(usize::MAX)),
            None => out.len(),
        };
        self.source.fill(&mut out[..n]);
        if let Some(r) = &mut self.remaining {
            *r -= n as u64;
        }
        Ok(n)
    }
}

/// See [`SignalRead::through`].
#[derive(Debug, Clone)]
pub struct ProcessedStream<R, P> {
    stream: R,
    processor: P,
}

impl<R, P> ProcessedStream<R, P> {
    /// The processor, e.g. to change its parameters while streaming.
    pub fn processor_mut(&mut self) -> &mut P {
        &mut self.processor
    }
    /// Gives back the stream and the processor.
    pub fn into_inner(self) -> (R, P) {
        (self.stream, self.processor)
    }
}

impl<R: SignalRead, P: Processor<R::Sample>> SignalRead for ProcessedStream<R, P>
where
    R::Sample: Float,
{
    type Sample = R::Sample;
    fn read_samples(&mut self, out: &mut [R::Sample]) -> io::Result<usize> {
        let n = self.stream.read_samples(out)?;
        self.processor.process(&mut out[..n]);
        Ok(n)
    }
}

// BYTE BOUNDARY ===================================================================================

/// How samples are laid out as bytes: element type plus byte order.
///
/// Conversions to and from float samples follow audio conventions: signed integers map to [-1, 1)
/// (i16 `16384` is 0.5), unsigned integers are offset binary (u8 `128` is 0.0, as in WAV files),
/// and out-of-range values are clipped when encoding. Values pass through f64, so 64-bit integers keep
/// 53 bits of precision. Complex types are not stream encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleEncoding {
    /// The element type of each sample.
    pub dtype: DType,
    /// The byte order.
    pub endian: Endian,
}

impl SampleEncoding {
    /// An encoding of `dtype` in `endian` byte order.
    pub const fn new(dtype: DType, endian: Endian) -> Self {
        Self { dtype, endian }
    }
    /// Little-endian `dtype`.
    pub const fn little(dtype: DType) -> Self {
        Self::new(dtype, Endian::Little)
    }
    /// Big-endian `dtype`.
    pub const fn big(dtype: DType) -> Self {
        Self::new(dtype, Endian::Big)
    }
    /// Bytes per encoded sample.
    pub const fn bytes_per_sample(&self) -> usize {
        self.dtype.size_bytes()
    }

    fn check(&self) -> io::Result<()> {
        if self.dtype.is_complex() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "complex types are not supported as sample encodings"));
        }
        Ok(())
    }

    /// Integer scale: 2^(bits - 1).
    fn int_scale(&self) -> f64 {
        (1u64 << (8 * self.dtype.size_bytes() - 1)) as f64
    }

    /// Decodes one sample from exactly `bytes_per_sample()` bytes.
    pub fn decode<T: Float>(&self, bytes: &[u8]) -> T {
        debug_assert_eq!(bytes.len(), self.bytes_per_sample());
        // assemble as a big-endian u64 of the sample's width
        let mut raw = 0u64;
        let mut push = |b: u8| raw = (raw << 8) | b as u64;
        match self.endian {
            Endian::Big => bytes.iter().for_each(|&b| push(b)),
            Endian::Little => bytes.iter().rev().for_each(|&b| push(b)),
        }
        let bits = 8 * bytes.len() as u32;
        let value = match self.dtype {
            DType::F32 => f32::from_bits(raw as u32) as f64,
            DType::F64 => f64::from_bits(raw),
            d if d.is_signed() => {
                let shift = 64 - bits; // sign-extend from `bits` wide
                (((raw << shift) as i64) >> shift) as f64 / self.int_scale()
            }
            _ => (raw as f64 - self.int_scale()) / self.int_scale(),
        };
        T::_lit(value)
    }

    /// Encodes one sample into exactly `bytes_per_sample()` bytes.
    pub fn encode<T: Float>(&self, sample: T, out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.bytes_per_sample());
        let x = sample.to_f64().unwrap_or(0.0);
        let bits = 8 * out.len() as u32;
        let raw: u64 = match self.dtype {
            DType::F32 => (x as f32).to_bits() as u64,
            DType::F64 => x.to_bits(),
            d if d.is_signed() => {
                let scale = self.int_scale();
                let v = (x * scale).round().clamp(-scale, scale - 1.0) as i64;
                (v as u64) & (u64::MAX >> (64 - bits))
            }
            _ => {
                let scale = self.int_scale();
                ((x * scale).round().clamp(-scale, scale - 1.0) + scale) as u64
            }
        };
        let width = out.len();
        for (i, b) in out.iter_mut().enumerate() {
            let byte = (raw >> (8 * (width - 1 - i))) as u8; // big-endian order
            *b = byte;
        }
        if self.endian == Endian::Little {
            out.reverse();
        }
    }
}

/// Reads float samples from any byte source (`File`, `TcpStream`, `&[u8]`, ...).
/// Handles reads that end mid-sample by carrying the partial bytes into the next read.
#[derive(Debug)]
pub struct SampleReader<R, T> {
    inner: R,
    encoding: SampleEncoding,
    bytes: Vec<u8>,
    /// bytes of an incomplete sample left over from the last read
    carry: usize,
    _sample: PhantomData<T>,
}

impl<R: Read, T: Float> SampleReader<R, T> {
    /// Fails with `InvalidInput` for complex encodings.
    pub fn new(inner: R, encoding: SampleEncoding) -> io::Result<Self> {
        encoding.check()?;
        Ok(Self { inner, encoding, bytes: Vec::new(), carry: 0, _sample: PhantomData })
    }
    /// How samples are decoded.
    pub fn encoding(&self) -> SampleEncoding {
        self.encoding
    }
    /// The byte source.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }
    /// Gives back the byte source (any incomplete sample still buffered is dropped).
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read, T: Float> SignalRead for SampleReader<R, T> {
    type Sample = T;
    fn read_samples(&mut self, out: &mut [T]) -> io::Result<usize> {
        let size = self.encoding.bytes_per_sample();
        let want = out.len() * size;
        if want == 0 {
            return Ok(0);
        }
        if self.bytes.len() < want {
            self.bytes.resize(want, 0);
        }
        loop {
            let n = self.inner.read(&mut self.bytes[self.carry..want])?;
            let have = self.carry + n;
            let samples = have / size;
            if samples > 0 {
                for (o, chunk) in out.iter_mut().zip(self.bytes[..samples * size].chunks_exact(size)) {
                    *o = self.encoding.decode(chunk);
                }
                self.bytes.copy_within(samples * size..have, 0);
                self.carry = have - samples * size;
                return Ok(samples);
            }
            self.carry = have;
            if n == 0 {
                return if self.carry == 0 {
                    Ok(0)
                } else {
                    Err(io::Error::new(io::ErrorKind::UnexpectedEof, "byte stream ended in the middle of a sample"))
                };
            }
        }
    }
}

impl<R: Read + Seek, T: Float> SignalSeek for SampleReader<R, T> {
    fn seek_samples(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let size = self.encoding.bytes_per_sample() as u64;
        // position in whole samples, accounting for carried bytes already read from `inner`
        let current = (self.inner.stream_position()? - self.carry as u64) / size;
        let end = self.inner.seek(SeekFrom::End(0))? / size;
        let target = resolve_seek(pos, current, end)?;
        self.inner.seek(SeekFrom::Start(target * size))?;
        self.carry = 0;
        Ok(target)
    }
}

/// Writes float samples to any byte sink with the given encoding.
#[derive(Debug)]
pub struct SampleWriter<W: Write, T> {
    inner: W,
    encoding: SampleEncoding,
    bytes: Vec<u8>,
    _sample: PhantomData<T>,
}

impl<W: Write, T: Float> SampleWriter<W, T> {
    /// Fails with `InvalidInput` for complex encodings.
    pub fn new(inner: W, encoding: SampleEncoding) -> io::Result<Self> {
        encoding.check()?;
        Ok(Self { inner, encoding, bytes: Vec::new(), _sample: PhantomData })
    }
    /// How samples are encoded.
    pub fn encoding(&self) -> SampleEncoding {
        self.encoding
    }
    /// The byte sink.
    pub fn get_ref(&self) -> &W {
        &self.inner
    }
    /// Flushes and returns the byte sink.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write, T: Float> SignalWrite for SampleWriter<W, T> {
    type Sample = T;
    fn write_samples(&mut self, samples: &[T]) -> io::Result<usize> {
        let size = self.encoding.bytes_per_sample();
        self.bytes.resize(samples.len() * size, 0);
        for (chunk, &s) in self.bytes.chunks_exact_mut(size).zip(samples) {
            self.encoding.encode(s, chunk);
        }
        // all or nothing, so a count of accepted samples never splits one
        self.inner.write_all(&self.bytes)?;
        Ok(samples.len())
    }
    fn flush_samples(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::osc::Sine;
    use crate::signal::Signal;

    /// A byte reader that returns at most `chunk` bytes per read, to exercise partial samples.
    struct Trickle<'a> {
        data: &'a [u8],
        chunk: usize,
    }

    impl Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.chunk).min(self.data.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    #[test]
    fn known_byte_values() {
        let i16le = SampleEncoding::little(DType::I16);
        assert_eq!(i16le.decode::<f64>(&[0x00, 0x40]), 0.5);
        assert_eq!(i16le.decode::<f64>(&[0x00, 0x80]), -1.0);
        let u8e = SampleEncoding::little(DType::U8);
        assert_eq!(u8e.decode::<f64>(&[128]), 0.0);
        assert_eq!(u8e.decode::<f64>(&[0]), -1.0);
        let i24be = SampleEncoding::big(DType::I24);
        assert_eq!(i24be.decode::<f64>(&[0xC0, 0x00, 0x00]), -0.5);

        let mut b = [0u8; 2];
        i16le.encode(2.0f64, &mut b); // clipped
        assert_eq!(b, 32767i16.to_le_bytes());
        i16le.encode(-2.0f64, &mut b);
        assert_eq!(b, (-32768i16).to_le_bytes());
    }

    #[test]
    fn every_real_encoding_roundtrips() {
        let values = [0.0, 0.5, -0.5, 0.25, -1.0, 0.999];
        for dtype in DType::ALL.into_iter().filter(|d| !d.is_complex()) {
            for endian in [Endian::Little, Endian::Big] {
                let e = SampleEncoding::new(dtype, endian);
                // one integer step, but no finer than f64 can resolve (64-bit integers pass through f64)
                let tol = if dtype.is_float() { 1e-7 } else { (1.0 / e.int_scale()).max(1e-15) };
                let mut bytes = vec![0u8; e.bytes_per_sample()];
                for v in values {
                    e.encode(v, &mut bytes);
                    let back: f64 = e.decode(&bytes);
                    assert!((back - v).abs() <= tol, "{dtype} {endian:?}: {v} -> {back}");
                }
            }
        }
    }

    #[test]
    fn writer_then_reader_through_bytes_with_partial_reads() {
        let signal: Vec<f64> = Sine::new(440.0, 48_000.0).take(1_000).collect();
        for dtype in [DType::I16, DType::I24, DType::F32, DType::U8] {
            let enc = SampleEncoding::little(dtype);
            let mut w = SampleWriter::new(Vec::new(), enc).unwrap();
            w.write_all_samples(&signal).unwrap();
            let bytes = w.into_inner().unwrap();
            assert_eq!(bytes.len(), 1_000 * dtype.size_bytes());

            // 5 bytes per read never lines up with 2/3/4-byte samples
            let mut r = SampleReader::<_, f64>::new(Trickle { data: &bytes, chunk: 5 }, enc).unwrap();
            let mut back = Vec::new();
            r.read_to_end_samples(&mut back).unwrap();
            assert_eq!(back.len(), 1_000);
            let err = back.distance(&signal).unwrap() / signal.norm_l2();
            assert!(err < 0.02, "{dtype}: relative error {err}");
        }
    }

    #[test]
    fn truncated_byte_stream_is_an_error() {
        let mut r = SampleReader::<_, f32>::new(&[1u8, 2, 3][..], SampleEncoding::little(DType::I16)).unwrap();
        let mut out = [0.0f32; 4];
        assert_eq!(r.read_samples(&mut out).unwrap(), 1);
        assert_eq!(r.read_samples(&mut out).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        assert!(SampleReader::<_, f32>::new(&[][..], SampleEncoding::little(DType::ComplexF32)).is_err());
    }

    #[test]
    fn sample_reader_seeks_in_samples() {
        let enc = SampleEncoding::little(DType::F32);
        let mut w = SampleWriter::new(io::Cursor::new(Vec::new()), enc).unwrap();
        w.write_all_samples(&[0.0f32, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut r = SampleReader::<_, f32>::new(w.into_inner().unwrap(), enc).unwrap();
        assert_eq!(r.seek_samples(SeekFrom::Start(3)).unwrap(), 3);
        let mut one = [0.0f32];
        r.read_exact_samples(&mut one).unwrap();
        assert_eq!(one, [3.0]);
        assert_eq!(r.sample_position().unwrap(), 4);
        assert_eq!(r.seek_samples(SeekFrom::End(-5)).unwrap(), 0);
        assert!(r.seek_samples(SeekFrom::Current(-1)).is_err());
    }

    #[test]
    fn cursor_reads_writes_and_seeks() {
        let mut c = SignalCursor::new(vec![1.0f64, 2.0, 3.0]);
        c.seek_samples(SeekFrom::End(-1)).unwrap();
        c.write_all_samples(&[9.0, 10.0]).unwrap(); // overwrites the last, then grows
        assert_eq!(c.get_ref(), &vec![1.0, 2.0, 9.0, 10.0]);
        c.seek_samples(SeekFrom::Start(1)).unwrap();
        let mut two = [0.0; 2];
        c.read_exact_samples(&mut two).unwrap();
        assert_eq!(two, [2.0, 9.0]);

        let mut fixed = [0.0f64; 2];
        let mut s = SignalCursor::new(&mut fixed[..]);
        assert_eq!(s.write_samples(&[1.0, 2.0, 3.0]).unwrap(), 2);
        assert!(s.write_all_samples(&[4.0]).is_err());
    }

    #[test]
    fn generated_stream_processed_and_pumped() {
        let fs = 48_000.0;
        let mut stream = Sine::new(8_000.0, fs).stream_for(4_800).through(Biquad::lowpass(500.0, BUTTERWORTH_Q, fs));
        let mut sink = SignalCursor::new(Vec::new());
        assert_eq!(stream.copy_to(&mut sink, 256).unwrap(), 4_800);
        let out = sink.into_inner();
        assert_eq!(out.len(), 4_800);
        assert!(out[2_400..].rms().unwrap() < 0.01, "8 kHz is far above the 500 Hz cutoff");
    }
}
