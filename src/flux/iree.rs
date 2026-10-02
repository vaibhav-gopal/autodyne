//! Runs emitted programs with IREE's command-line tools (`iree-compile`, `iree-run-module`).
//!
//! The tools are external and found at run time: in `AUTODYNE_IREE_DIR` if set, else on `PATH`
//! (`pip install iree-base-compiler iree-base-runtime` provides both). Arrays cross as `.npy` files
//! in a temporary directory. Compiles for the host CPU (`llvm-cpu`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::FluxError;

/// The located IREE tools.
#[derive(Clone, Debug)]
pub struct Iree {
    compiler: PathBuf,
    runner: PathBuf,
}

impl Iree {
    /// Finds `iree-compile` and `iree-run-module` in `AUTODYNE_IREE_DIR`, else on `PATH`.
    pub fn find() -> Option<Iree> {
        let dirs: Vec<PathBuf> = match std::env::var_os("AUTODYNE_IREE_DIR") {
            Some(dir) => vec![PathBuf::from(dir)],
            None => std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default(),
        };
        let find = |name: &str| {
            let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
            dirs.iter().map(|d| d.join(&file)).find(|p| p.is_file())
        };
        Some(Iree { compiler: find("iree-compile")?, runner: find("iree-run-module")? })
    }

    /// Compiles a StableHLO module for the host CPU.
    pub fn compile(&self, mlir: &str) -> Result<Module, FluxError> {
        let dir = TempDir::new()?;
        let source = dir.path.join("module.mlir");
        let vmfb = dir.path.join("module.vmfb");
        std::fs::write(&source, mlir)?;
        let mut cmd = Command::new(&self.compiler);
        cmd.arg(&source)
            .args([
                "--iree-input-type=stablehlo",
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
                "-o",
            ])
            .arg(&vmfb);
        run(cmd, "iree-compile")?;
        Ok(Module { runner: self.runner.clone(), vmfb, dir })
    }
}

/// A compiled module, ready to call.
#[derive(Debug)]
pub struct Module {
    runner: PathBuf,
    vmfb: PathBuf,
    dir: TempDir,
}

/// An f32 array crossing to or from IREE.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn scalar(v: f32) -> Tensor {
        Tensor { shape: Vec::new(), data: vec![v] }
    }
    pub fn vector(v: &[f32]) -> Tensor {
        Tensor { shape: vec![v.len()], data: v.to_vec() }
    }
}

impl Module {
    /// Calls `function` with `inputs`, expecting `outputs` results.
    pub fn call(&self, function: &str, inputs: &[Tensor], outputs: usize) -> Result<Vec<Tensor>, FluxError> {
        static CALL: AtomicUsize = AtomicUsize::new(0);
        let call = CALL.fetch_add(1, Ordering::Relaxed);
        let file = |kind: &str, k: usize| self.dir.path.join(format!("{call}-{kind}{k}.npy"));
        let mut cmd = Command::new(&self.runner);
        cmd.arg("--device=local-task").arg(arg("--module=", &self.vmfb)).arg(format!("--function={function}"));
        for (k, t) in inputs.iter().enumerate() {
            let path = file("in", k);
            std::fs::write(&path, npy::write(t))?;
            cmd.arg(arg("--input=@", &path));
        }
        for k in 0..outputs {
            cmd.arg(arg("--output=@", &file("out", k)));
        }
        run(cmd, "iree-run-module")?;
        let results = (0..outputs).map(|k| npy::read(&std::fs::read(file("out", k))?)).collect();
        for k in 0..inputs.len() {
            let _ = std::fs::remove_file(file("in", k));
        }
        for k in 0..outputs {
            let _ = std::fs::remove_file(file("out", k));
        }
        results
    }
}

fn arg(flag: &str, path: &Path) -> std::ffi::OsString {
    let mut s = std::ffi::OsString::from(flag);
    s.push(path);
    s
}

fn run(mut cmd: Command, tool: &'static str) -> Result<(), FluxError> {
    let out = cmd.output()?;
    if out.status.success() {
        Ok(())
    } else {
        Err(FluxError::Tool { tool, message: String::from_utf8_lossy(&out.stderr).trim().to_string() })
    }
}

/// A directory removed when dropped.
#[derive(Debug)]
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> std::io::Result<TempDir> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("autodyne-flux-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Minimal `.npy` (version 1) for little-endian f32 arrays.
mod npy {
    use super::{FluxError, Tensor};

    pub fn write(t: &Tensor) -> Vec<u8> {
        let shape = match t.shape.as_slice() {
            [] => "()".to_string(),
            [n] => format!("({n},)"),
            dims => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        // magic (6) + version (2) + length (2) + header + '\n', padded to a multiple of 64
        let total = (10 + header.len() + 1).div_ceil(64) * 64;
        header.extend(std::iter::repeat_n(' ', total - 10 - header.len() - 1));
        header.push('\n');
        let mut out = Vec::with_capacity(total + 4 * t.data.len());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn read(bytes: &[u8]) -> Result<Tensor, FluxError> {
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
        let count: usize = shape.iter().product();
        let data = bytes[start + len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        if data.len() != count {
            return Err(bad("data length does not match the shape"));
        }
        Ok(Tensor { shape, data })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn round_trip() {
            for t in [Tensor::scalar(0.5), Tensor::vector(&[1.0, -2.0, 3.5]), Tensor { shape: vec![2, 3], data: (0..6).map(|v| v as f32).collect() }] {
                let bytes = write(&t);
                assert_eq!((bytes.len() - 4 * t.data.len()) % 64, 0);
                assert_eq!(read(&bytes).unwrap(), t);
            }
        }
    }
}
