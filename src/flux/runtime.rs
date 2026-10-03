//! Running programs: the [`Backend`] / [`Executable`] traits, and the plumbing the backends share
//! (`.npy` files in a temporary directory, finding tools).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::graph::FluxFloat;
use super::hlo::Program;
use super::FluxError;
use crate::signal::NdArray;
use crate::units::DType;

/// A compiler and runtime for [`Program`]s (StableHLO).
pub trait Backend {
    /// A short name for messages (`"iree"`, `"xla"`).
    fn name(&self) -> &'static str;
    /// Compiles `program`.
    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError>;
    /// The longest FFT this backend compiles, if limited: write programs for it with
    /// [`Emit::for_backend`](super::Emit::for_backend), which builds longer FFTs from shorter ones.
    fn max_fft(&self) -> Option<usize> {
        None
    }
}

/// A host array going into a program: `f32` or `f64` (the program's element type).
#[derive(Clone, Copy, Debug)]
pub enum HostRef<'a> {
    F32(&'a NdArray<f32>),
    F64(&'a NdArray<f64>),
}

/// A host array coming out of a program.
#[derive(Clone, Debug, PartialEq)]
pub enum HostArray {
    F32(NdArray<f32>),
    F64(NdArray<f64>),
}

impl HostRef<'_> {
    pub fn shape(&self) -> &[usize] {
        match self {
            HostRef::F32(a) => a.shape(),
            HostRef::F64(a) => a.shape(),
        }
    }
    pub fn dtype(&self) -> DType {
        match self {
            HostRef::F32(_) => DType::F32,
            HostRef::F64(_) => DType::F64,
        }
    }
    pub fn to_owned(self) -> HostArray {
        match self {
            HostRef::F32(a) => HostArray::F32(a.clone()),
            HostRef::F64(a) => HostArray::F64(a.clone()),
        }
    }
}

impl HostArray {
    pub fn as_ref(&self) -> HostRef<'_> {
        match self {
            HostArray::F32(a) => HostRef::F32(a),
            HostArray::F64(a) => HostRef::F64(a),
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            HostArray::F32(a) => a.shape(),
            HostArray::F64(a) => a.shape(),
        }
    }
    pub fn dtype(&self) -> DType {
        self.as_ref().dtype()
    }
}

/// A compiled program. Its methods move [`HostRef`] / [`HostArray`] data, whose element type must
/// be the program's; [`ExecutableExt`] adds the typed calls (`run`, `upload`, `download` on
/// `NdArray<f32>` or `NdArray<f64>`) that code normally uses.
///
/// [`run`](ExecutableExt::run) copies its inputs to the device and its outputs back on every
/// call. To keep data on the device across calls (a fitting loop's signal and target, say),
/// [`upload`](ExecutableExt::upload) it once and call [`run_resident`](Self::run_resident), which
/// leaves its outputs on the device too; [`download`](ExecutableExt::download) what the host
/// needs. Backends without device memory of their own (IREE's command-line tools) keep resident
/// arrays on the host, so the same code runs everywhere.
///
/// ```no_run
/// use autodyne::flux::{scalar, vector, Backend, ExecutableExt, Pjrt, Program};
/// # fn fit(pjrt: &Pjrt, program: &Program, xs: &[f32], target: &[f32]) -> Result<(), autodyne::flux::FluxError> {
/// let exe = pjrt.compile(program)?;
/// // once: the signal, the target and the initial state
/// let (xs, target, s0) = (exe.upload(&vector(xs))?, exe.upload(&vector(target))?, exe.upload(&scalar(0.0))?);
/// let mut cutoff = 500.0;
/// for _ in 0..100 {
///     let p = exe.upload(&scalar(cutoff))?; // small: the parameters change every step
///     let out = exe.run_resident(&[&p, &xs, &target, &s0])?;
///     cutoff -= 1e3 * exe.download::<f32>(&out[1])?.as_slice()[0]; // only the gradient comes back
/// }
/// # Ok(())
/// # }
/// ```
pub trait Executable {
    /// The program it runs (its text may be empty for a loaded module).
    fn program(&self) -> &Program;

    /// Runs `@main` on `inputs` (one array per program input, of its shape); returns the outputs.
    fn run_host(&self, inputs: &[HostRef<'_>]) -> Result<Vec<HostArray>, FluxError>;

    /// Copies `a` into this executable's memory, for [`run_resident`](Self::run_resident).
    fn upload_host(&self, a: HostRef<'_>) -> Result<DeviceArray, FluxError> {
        Ok(DeviceArray::host(a.to_owned()))
    }

    /// Runs `@main` on arrays already in this executable's memory; the outputs stay there.
    fn run_resident(&self, inputs: &[&DeviceArray]) -> Result<Vec<DeviceArray>, FluxError> {
        let host = inputs.iter().map(|a| self.download_host(a)).collect::<Result<Vec<_>, _>>()?;
        Ok(self.run_host(&host.iter().map(HostArray::as_ref).collect::<Vec<_>>())?.into_iter().map(DeviceArray::host).collect())
    }

    /// Copies a resident array back to the host.
    fn download_host(&self, a: &DeviceArray) -> Result<HostArray, FluxError> {
        match &a.data {
            Resident::Host(x) => Ok((**x).clone()),
            Resident::Device(_) => Err(FluxError::Shape("this array is held by another backend's device".into())),
        }
    }
}

/// Typed calls on any [`Executable`]: `NdArray<f32>` for f32 programs, `NdArray<f64>` for f64.
pub trait ExecutableExt: Executable {
    /// Runs `@main` on `inputs` (one array per program input, of its shape); returns the outputs.
    fn run<T: FluxFloat>(&self, inputs: &[NdArray<T>]) -> Result<Vec<NdArray<T>>, FluxError> {
        let inputs: Vec<HostRef<'_>> = inputs.iter().map(T::host_ref).collect();
        self.run_host(&inputs)?.into_iter().map(T::from_host).collect()
    }
    /// Copies `a` into this executable's memory, for [`run_resident`](Executable::run_resident).
    fn upload<T: FluxFloat>(&self, a: &NdArray<T>) -> Result<DeviceArray, FluxError> {
        self.upload_host(T::host_ref(a))
    }
    /// Copies a resident array back to the host.
    fn download<T: FluxFloat>(&self, a: &DeviceArray) -> Result<NdArray<T>, FluxError> {
        T::from_host(self.download_host(a)?)
    }
}

impl<E: Executable + ?Sized> ExecutableExt for E {}

/// An array held in an [`Executable`]'s memory: on its device for backends with device memory
/// (PJRT, the XLA server), on the host otherwise. Freed when dropped.
pub struct DeviceArray {
    shape: Vec<usize>,
    dtype: DType,
    pub(crate) data: Resident,
}

pub(crate) enum Resident {
    Host(Box<HostArray>),
    /// A backend's own handle (freeing the memory when dropped).
    Device(Box<dyn std::any::Any + Send + Sync>),
}

impl DeviceArray {
    pub(crate) fn host(a: HostArray) -> DeviceArray {
        DeviceArray { shape: a.shape().to_vec(), dtype: a.dtype(), data: Resident::Host(Box::new(a)) }
    }
    pub(crate) fn device(shape: Vec<usize>, dtype: DType, handle: Box<dyn std::any::Any + Send + Sync>) -> DeviceArray {
        DeviceArray { shape, dtype, data: Resident::Device(handle) }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Whether the data is in device memory (rather than held on the host for a backend without any).
    pub fn is_on_device(&self) -> bool {
        matches!(self.data, Resident::Device(_))
    }
}

impl std::fmt::Debug for DeviceArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceArray({:?} {:?}, {})", self.shape, self.dtype, if self.is_on_device() { "device" } else { "host" })
    }
}

/// Checks arguments (shape and element type each) against the program's inputs.
pub(crate) fn check_args<'a>(program: &Program, args: impl ExactSizeIterator<Item = (&'a [usize], DType)>) -> Result<(), FluxError> {
    if args.len() != program.inputs.len() {
        return Err(FluxError::Shape(format!("the program takes {} inputs, got {}", program.inputs.len(), args.len())));
    }
    for (k, ((shape, dtype), s)) in args.zip(&program.inputs).enumerate() {
        if shape != s.as_slice() {
            return Err(FluxError::Shape(format!("input {k} should be {s:?}, got {shape:?}")));
        }
        if dtype != program.dtype {
            return Err(FluxError::Shape(format!("input {k} should be {:?}, got {dtype:?}", program.dtype)));
        }
    }
    Ok(())
}

/// Checks host inputs against the program's inputs.
pub(crate) fn check_inputs(program: &Program, inputs: &[HostRef<'_>]) -> Result<(), FluxError> {
    check_args(program, inputs.iter().map(|x| (x.shape(), x.dtype())))
}

/// Checks resident inputs against the program's inputs.
pub(crate) fn check_resident(program: &Program, inputs: &[&DeviceArray]) -> Result<(), FluxError> {
    check_args(program, inputs.iter().map(|x| (x.shape(), x.dtype())))
}

/// Checks `outputs` against the program's outputs.
pub(crate) fn check_outputs(program: &Program, outputs: &[HostArray]) -> Result<(), FluxError> {
    let shapes: Vec<&[usize]> = outputs.iter().map(|o| o.shape()).collect();
    let expected: Vec<&[usize]> = program.outputs.iter().map(Vec::as_slice).collect();
    if shapes != expected {
        return Err(FluxError::Shape(format!("the program should return {expected:?}, got {shapes:?}")));
    }
    if let Some(o) = outputs.iter().find(|o| o.dtype() != program.dtype) {
        return Err(FluxError::Shape(format!("the program should return {:?}, got {:?}", program.dtype, o.dtype())));
    }
    Ok(())
}
/// Finds `name` (plus the platform's executable suffix) in `dir` if given, else on `PATH`.
pub(crate) fn find_tool(dir: Option<PathBuf>, name: &str) -> Option<PathBuf> {
    let dirs: Vec<PathBuf> = match dir {
        Some(dir) => vec![dir],
        None => std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default(),
    };
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    dirs.iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// A directory removed when dropped.
#[derive(Debug)]
pub(crate) struct TempDir {
    pub path: PathBuf,
}

impl TempDir {
    pub fn new() -> std::io::Result<TempDir> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("autodyne-flux-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }

    /// A file name unique within this directory.
    pub fn file(&self, stem: &str, ext: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        self.path.join(format!("{stem}-{}.{ext}", NEXT.fetch_add(1, Ordering::Relaxed)))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Files removed when dropped.
pub(crate) struct Scratch(pub Vec<PathBuf>);

impl Drop for Scratch {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Minimal `.npy` (versions 1-3) for little-endian, C-order f32 and f64 arrays.
pub(crate) mod npy {
    use super::*;

    pub fn save(path: &Path, a: HostRef<'_>) -> Result<(), FluxError> {
        std::fs::write(path, write(a))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<HostArray, FluxError> {
        read(&std::fs::read(path)?)
    }

    pub fn write(a: HostRef<'_>) -> Vec<u8> {
        let shape = match a.shape() {
            [] => "()".to_string(),
            [n] => format!("({n},)"),
            dims => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let descr = if a.dtype() == DType::F64 { "<f8" } else { "<f4" };
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        // magic (6) + version (2) + length (2) + header + '\n', padded to a multiple of 64
        let total = (10 + header.len() + 1).div_ceil(64) * 64;
        header.extend(std::iter::repeat_n(' ', total - 10 - header.len() - 1));
        header.push('\n');
        let mut out = Vec::with_capacity(total + 8 * a.shape().iter().product::<usize>());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        match a {
            HostRef::F32(a) => a.as_slice().iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            HostRef::F64(a) => a.as_slice().iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        }
        out
    }

    pub fn read(bytes: &[u8]) -> Result<HostArray, FluxError> {
        let bad = |why: &str| FluxError::Npy(why.to_string());
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(bad("not an .npy file"));
        }
        let (len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 if bytes.len() >= 12 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
            _ => return Err(bad("unsupported .npy version")),
        };
        let header = std::str::from_utf8(bytes.get(start..start + len).ok_or_else(|| bad("truncated header"))?).map_err(|_| bad("header is not text"))?;
        let wide = if header.contains("'descr': '<f8'") {
            true
        } else if header.contains("'descr': '<f4'") {
            false
        } else {
            return Err(bad("only little-endian f32 and f64 arrays are supported"));
        };
        if header.contains("'fortran_order': True") {
            return Err(bad("Fortran-order arrays are not supported"));
        }
        let dims = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| bad("no shape"))?;
        let shape = dims
            .split(',')
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .map(|d| d.parse::<usize>().map_err(|_| bad("bad shape")))
            .collect::<Result<Vec<_>, _>>()?;
        let body = &bytes[start + len..];
        let count = shape.iter().product::<usize>();
        if body.len() != count * if wide { 8 } else { 4 } {
            return Err(bad("data length does not match the shape"));
        }
        if wide {
            let data: Vec<f64> = body.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect();
            NdArray::from_vec(data, &shape).map(HostArray::F64).map_err(|e| bad(&e.to_string()))
        } else {
            let data: Vec<f32> = body.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            NdArray::from_vec(data, &shape).map(HostArray::F32).map_err(|e| bad(&e.to_string()))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn round_trip() {
            let arrays = [
                HostArray::F32(NdArray::from_vec(vec![0.5f32], &[]).unwrap()),
                HostArray::F32(NdArray::from_vec(vec![1.0, -2.0, 3.5], &[3]).unwrap()),
                HostArray::F64(NdArray::from_vec((0..6).map(|v| v as f64 / 3.0).collect(), &[2, 3]).unwrap()),
            ];
            for a in arrays {
                let bytes = write(a.as_ref());
                assert_eq!(read(&bytes).unwrap(), a);
            }
        }
    }
}