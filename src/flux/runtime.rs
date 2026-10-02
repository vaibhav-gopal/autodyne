//! Running programs: the [`Backend`] / [`Executable`] traits, and the plumbing the backends share
//! (`.npy` files in a temporary directory, finding tools).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::hlo::Program;
use super::FluxError;
use crate::signal::NdArray;

/// A compiler and runtime for [`Program`]s (StableHLO).
pub trait Backend {
    /// A short name for messages (`"iree"`, `"xla"`).
    fn name(&self) -> &'static str;
    /// Compiles `program`.
    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError>;
}

/// A compiled program.
pub trait Executable {
    /// Runs `@main` on `inputs` (one array per program input, of its shape); returns the outputs.
    fn run(&self, inputs: &[NdArray<f32>]) -> Result<Vec<NdArray<f32>>, FluxError>;
}

/// Checks `inputs` against the program's input shapes.
pub(crate) fn check_inputs(program: &Program, inputs: &[NdArray<f32>]) -> Result<(), FluxError> {
    if inputs.len() != program.inputs.len() {
        return Err(FluxError::Shape(format!("the program takes {} inputs, got {}", program.inputs.len(), inputs.len())));
    }
    for (k, (x, s)) in inputs.iter().zip(&program.inputs).enumerate() {
        if x.shape() != s.as_slice() {
            return Err(FluxError::Shape(format!("input {k} should be {s:?}, got {:?}", x.shape())));
        }
    }
    Ok(())
}

/// Checks `outputs` against the program's output shapes.
pub(crate) fn check_outputs(program: &Program, outputs: &[NdArray<f32>]) -> Result<(), FluxError> {
    let shapes: Vec<&[usize]> = outputs.iter().map(|o| o.shape()).collect();
    let expected: Vec<&[usize]> = program.outputs.iter().map(Vec::as_slice).collect();
    if shapes != expected {
        return Err(FluxError::Shape(format!("the program should return {expected:?}, got {shapes:?}")));
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

/// Minimal `.npy` (versions 1-3) for little-endian, C-order f32 arrays.
pub(crate) mod npy {
    use super::*;

    pub fn save(path: &Path, a: &NdArray<f32>) -> Result<(), FluxError> {
        std::fs::write(path, write(a))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<NdArray<f32>, FluxError> {
        read(&std::fs::read(path)?)
    }

    pub fn write(a: &NdArray<f32>) -> Vec<u8> {
        let shape = match a.shape() {
            [] => "()".to_string(),
            [n] => format!("({n},)"),
            dims => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        // magic (6) + version (2) + length (2) + header + '\n', padded to a multiple of 64
        let total = (10 + header.len() + 1).div_ceil(64) * 64;
        header.extend(std::iter::repeat_n(' ', total - 10 - header.len() - 1));
        header.push('\n');
        let mut out = Vec::with_capacity(total + 4 * a.len());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in a.as_slice() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn read(bytes: &[u8]) -> Result<NdArray<f32>, FluxError> {
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
        if !header.contains("'descr': '<f4'") {
            return Err(bad("only little-endian f32 arrays are supported"));
        }
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
        let data: Vec<f32> = bytes[start + len..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        if data.len() != shape.iter().product::<usize>() {
            return Err(bad("data length does not match the shape"));
        }
        NdArray::from_vec(data, &shape).map_err(|e| bad(&e.to_string()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn round_trip() {
            let arrays = [
                NdArray::from_vec(vec![0.5f32], &[]).unwrap(),
                NdArray::from_vec(vec![1.0, -2.0, 3.5], &[3]).unwrap(),
                NdArray::from_vec((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap(),
            ];
            for a in arrays {
                let bytes = write(&a);
                assert_eq!((bytes.len() - 4 * a.len()) % 64, 0);
                assert_eq!(read(&bytes).unwrap(), a);
            }
        }
    }
}
