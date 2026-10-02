//! IREE through its command-line tools (`iree-compile`, `iree-run-module`).
//!
//! The tools are found at run time: in `AUTODYNE_IREE_DIR` if set, else on `PATH`
//! (`pip install iree-base-compiler iree-base-runtime` provides both). Arrays cross as `.npy` files.
//! Compiles for the host CPU (`llvm-cpu`); each run starts `iree-run-module` (tens of milliseconds).

use std::path::PathBuf;
use std::process::Command;

use super::hlo::Program;
use super::runtime::{check_inputs, check_outputs, find_tool, npy, Backend, Executable, Scratch, TempDir};
use super::FluxError;
use crate::signal::NdArray;

/// The located IREE tools.
#[derive(Clone, Debug)]
pub struct Iree {
    compiler: PathBuf,
    runner: PathBuf,
}

impl Iree {
    /// Finds `iree-compile` and `iree-run-module` in `AUTODYNE_IREE_DIR`, else on `PATH`.
    pub fn find() -> Option<Iree> {
        let dir = std::env::var_os("AUTODYNE_IREE_DIR").map(PathBuf::from);
        Some(Iree { compiler: find_tool(dir.clone(), "iree-compile")?, runner: find_tool(dir, "iree-run-module")? })
    }
}

impl Backend for Iree {
    fn name(&self) -> &'static str {
        "iree"
    }

    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError> {
        let dir = TempDir::new()?;
        let source = dir.path.join("module.mlir");
        let vmfb = dir.path.join("module.vmfb");
        std::fs::write(&source, &program.text)?;
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
        Ok(Box::new(Module { runner: self.runner.clone(), vmfb, dir, program: program.clone() }))
    }
}

/// A compiled IREE module.
#[derive(Debug)]
struct Module {
    runner: PathBuf,
    vmfb: PathBuf,
    dir: TempDir,
    program: Program,
}

impl Executable for Module {
    fn run(&self, inputs: &[NdArray<f32>]) -> Result<Vec<NdArray<f32>>, FluxError> {
        check_inputs(&self.program, inputs)?;
        let ins: Vec<PathBuf> = inputs.iter().map(|_| self.dir.file("in", "npy")).collect();
        let outs: Vec<PathBuf> = self.program.outputs.iter().map(|_| self.dir.file("out", "npy")).collect();
        let _cleanup = Scratch(ins.iter().chain(&outs).cloned().collect());
        let mut cmd = Command::new(&self.runner);
        cmd.arg("--device=local-task").arg(flag("--module=", &self.vmfb)).arg("--function=main");
        for (x, path) in inputs.iter().zip(&ins) {
            npy::save(path, x)?;
            cmd.arg(flag("--input=@", path));
        }
        for path in &outs {
            cmd.arg(flag("--output=@", path));
        }
        run(cmd, "iree-run-module")?;
        let results = outs.iter().map(|p| npy::load(p)).collect::<Result<Vec<_>, _>>()?;
        check_outputs(&self.program, &results)?;
        Ok(results)
    }
}

fn flag(flag: &str, path: &std::path::Path) -> std::ffi::OsString {
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
