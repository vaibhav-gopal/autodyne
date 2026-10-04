//! IREE through its command-line tools (`iree-compile`, `iree-run-module`).
//!
//! The tools are found at run time: in `AUTODYNE_IREE_DIR` if set, else on `PATH`
//! (`pip install iree-base-compiler iree-base-runtime` provides both). Arrays cross as `.npy` files.
//! Each run starts `iree-run-module` (tens of milliseconds). On GPUs, IREE drives a scan's loop
//! from the host, one dispatch per step: long scans of a single channel run far faster on the CPU,
//! and GPUs pay off when each step is wide (many channels, or frames). f64 programs need a target
//! with f64 maths: IREE 3.11 compiles f64 transcendentals (`exp`, `sin`, ...) for CUDA but not for
//! the CPU or Vulkan.
//!
//! An [`IreeTarget`] picks the hardware: the host CPU (the default), Vulkan, CUDA, ROCm or Metal.
//! Compiled modules (`.vmfb`) can be saved with [`Iree::compile_to`] and run later, or elsewhere,
//! with [`Iree::load`]: IREE's ahead-of-time deployment path.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::hlo::Program;
use super::runtime::{check_inputs, check_outputs, find_tool, npy, Backend, Executable, HostArray, HostRef, Scratch, TempDir};
use super::FluxError;

/// The hardware IREE compiles for and runs on.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum IreeTarget {
    /// The host CPU (`llvm-cpu`, tuned for this machine; run multithreaded on `local-task`).
    #[default]
    Cpu,
    /// Any Vulkan GPU (`vulkan-spirv`). `target` names the architecture (`"ampere"`, `"rdna3"`,
    /// `"valhall4"`, ...) to use its features; `None` is IREE's portable baseline, which lacks the
    /// 64-bit integers IREE's FFTs need (programs with FFTs need a named architecture). IREE 3.11
    /// also fails to compile FFTs of 128 points or more for Vulkan: emit programs for it with
    /// `Emit::for_backend`, which builds them from shorter ones.
    Vulkan {
        /// The GPU architecture, or `None` for IREE's baseline.
        target: Option<String>,
    },
    /// NVIDIA GPUs through CUDA. `target` is an architecture this IREE knows (`"sm_80"`,
    /// `"ampere"`, ...); newer GPUs run it too, the driver compiling the embedded PTX for them.
    Cuda {
        /// The GPU architecture.
        target: String,
    },
    /// AMD GPUs through ROCm / HIP; `target` is the chip (`"gfx1100"`, ...).
    Rocm {
        /// The chip.
        target: String,
    },
    /// Apple GPUs (`metal-spirv`).
    Metal,
    /// Anything else: `iree-compile` flags (after `--iree-input-type=stablehlo`) and the
    /// `iree-run-module` device.
    Custom {
        /// `iree-compile` flags.
        flags: Vec<String>,
        /// The `iree-run-module` device.
        device: String,
    },
}

impl IreeTarget {
    /// Vulkan with IREE's portable baseline.
    pub fn vulkan() -> Self {
        IreeTarget::Vulkan { target: None }
    }
    /// CUDA for `sm_80` (Ampere) PTX, which every later NVIDIA GPU also runs.
    pub fn cuda() -> Self {
        IreeTarget::Cuda { target: "sm_80".into() }
    }

    fn flags(&self) -> Vec<String> {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        match self {
            IreeTarget::Cpu => s(&["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=host"]),
            IreeTarget::Vulkan { target } => {
                let mut f = s(&["--iree-hal-target-device=vulkan"]);
                f.extend(target.iter().map(|t| format!("--iree-vulkan-target={t}")));
                f
            }
            IreeTarget::Cuda { target } => vec!["--iree-hal-target-device=cuda".into(), format!("--iree-cuda-target={target}")],
            IreeTarget::Rocm { target } => vec!["--iree-hal-target-device=hip".into(), format!("--iree-rocm-target={target}")],
            IreeTarget::Metal => s(&["--iree-hal-target-device=metal"]),
            IreeTarget::Custom { flags, .. } => flags.clone(),
        }
    }

    /// `iree-run-module` flags for this target's driver.
    fn run_flags(&self) -> Vec<String> {
        match self {
            // stream-ordered allocations crash loops on some drivers (seen with IREE 3.11 on a
            // GeForce RTX 5070 Ti); synchronous allocation costs little
            IreeTarget::Cuda { .. } => vec!["--cuda_async_allocations=false".into()],
            _ => Vec::new(),
        }
    }

    /// The runtime device (`iree-run-module --device=`).
    fn device(&self) -> &str {
        match self {
            IreeTarget::Cpu => "local-task",
            IreeTarget::Vulkan { .. } => "vulkan",
            IreeTarget::Cuda { .. } => "cuda",
            IreeTarget::Rocm { .. } => "hip",
            IreeTarget::Metal => "metal",
            IreeTarget::Custom { device, .. } => device,
        }
    }
}

/// The located IREE tools, and the target they compile for.
#[derive(Clone, Debug)]
pub struct Iree {
    compiler: PathBuf,
    runner: PathBuf,
    target: IreeTarget,
}

impl Iree {
    /// Finds `iree-compile` and `iree-run-module` in `AUTODYNE_IREE_DIR`, else on `PATH`; targets
    /// the host CPU.
    pub fn find() -> Option<Iree> {
        let dir = std::env::var_os("AUTODYNE_IREE_DIR").map(PathBuf::from);
        Some(Iree { compiler: find_tool(dir.clone(), "iree-compile")?, runner: find_tool(dir, "iree-run-module")?, target: IreeTarget::Cpu })
    }

    /// The same tools for another target.
    pub fn with_target(self, target: IreeTarget) -> Iree {
        Iree { target, ..self }
    }

    /// What the programs are compiled for.
    pub fn target(&self) -> &IreeTarget {
        &self.target
    }

    /// Compiles `program` into an IREE module file (`.vmfb`) for this target, to run later with
    /// [`load`](Self::load) (here, or on another machine with the IREE runtime).
    pub fn compile_to(&self, program: &Program, vmfb: &Path) -> Result<(), FluxError> {
        let dir = TempDir::new()?;
        let source = dir.path.join("module.mlir");
        std::fs::write(&source, &program.text)?;
        let mut cmd = Command::new(&self.compiler);
        cmd.arg(&source).arg("--iree-input-type=stablehlo").args(self.target.flags());
        if program.dtype == crate::units::DType::F64 {
            // IREE narrows f64 to f32 unless told not to
            cmd.arg("--iree-input-demote-f64-to-f32=false");
        }
        cmd.arg("-o").arg(vmfb);
        run(cmd, "iree-compile")
    }

    /// A module saved by [`compile_to`](Self::compile_to), run on this target's device. `signature`
    /// gives its input and output shapes and element type (the text is not used, and may be empty).
    pub fn load(&self, vmfb: &Path, signature: &Program) -> Result<Box<dyn Executable>, FluxError> {
        if !vmfb.is_file() {
            return Err(FluxError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("no module at {}", vmfb.display()))));
        }
        let program = Program { text: String::new(), ..signature.clone() };
        Ok(Box::new(Module { runner: self.runner.clone(), target: self.target.clone(), vmfb: vmfb.to_path_buf(), dir: TempDir::new()?, program }))
    }
}

impl Backend for Iree {
    fn name(&self) -> &'static str {
        match self.target {
            IreeTarget::Cpu => "iree",
            IreeTarget::Vulkan { .. } => "iree-vulkan",
            IreeTarget::Cuda { .. } => "iree-cuda",
            IreeTarget::Rocm { .. } => "iree-rocm",
            IreeTarget::Metal => "iree-metal",
            IreeTarget::Custom { .. } => "iree-custom",
        }
    }

    /// IREE 3.11's Vulkan backend fails on FFTs of 128 points or more.
    fn max_fft(&self) -> Option<usize> {
        matches!(self.target, IreeTarget::Vulkan { .. }).then_some(64)
    }

    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError> {
        let dir = TempDir::new()?;
        let vmfb = dir.path.join("module.vmfb");
        self.compile_to(program, &vmfb)?;
        Ok(Box::new(Module { runner: self.runner.clone(), target: self.target.clone(), vmfb, dir, program: program.clone() }))
    }
}

/// A compiled IREE module.
#[derive(Debug)]
struct Module {
    runner: PathBuf,
    target: IreeTarget,
    vmfb: PathBuf,
    /// scratch space for the arrays (and the module, when compiled here)
    dir: TempDir,
    program: Program,
}

impl Executable for Module {
    fn program(&self) -> &Program {
        &self.program
    }

    fn run_host(&self, inputs: &[HostRef<'_>]) -> Result<Vec<HostArray>, FluxError> {
        check_inputs(&self.program, inputs)?;
        let ins: Vec<PathBuf> = inputs.iter().map(|_| self.dir.file("in", "npy")).collect();
        let outs: Vec<PathBuf> = self.program.outputs.iter().map(|_| self.dir.file("out", "npy")).collect();
        let _cleanup = Scratch(ins.iter().chain(&outs).cloned().collect());
        let mut cmd = Command::new(&self.runner);
        cmd.arg(format!("--device={}", self.target.device())).args(self.target.run_flags()).arg(flag("--module=", &self.vmfb)).arg("--function=main");
        for (x, path) in inputs.iter().zip(&ins) {
            npy::save(path, *x)?;
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
        // some failures are reported on stdout only
        let message = [&out.stderr, &out.stdout].map(|b| String::from_utf8_lossy(b).trim().to_string()).into_iter().filter(|m| !m.is_empty()).collect::<Vec<_>>().join("\n");
        Err(FluxError::Tool { tool, message: format!("{message} ({})", out.status) })
    }
}
