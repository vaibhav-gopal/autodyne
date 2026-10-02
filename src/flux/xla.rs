//! XLA through PJRT, driven by JAX's client (`jaxlib`) in a Python process that stays up.
//!
//! The process is started once (importing JAX takes seconds) and serves compile and run requests
//! over its stdin / stdout; arrays cross as `.npy` files. Python is found at run time:
//! `AUTODYNE_XLA_PYTHON` if set, else `python3` / `python` on `PATH`, with `jax` installed
//! (`pip install jax`). The platform defaults to the CPU; set `AUTODYNE_XLA_PLATFORM` (`cuda`,
//! `rocm`, `tpu`) when that jaxlib plugin is installed.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use super::hlo::Program;
use super::runtime::{check_inputs, check_outputs, find_tool, npy, Backend, Executable, Scratch, TempDir};
use super::FluxError;
use crate::signal::NdArray;

/// The request loop run by Python: one tab-separated request per line, one reply per line.
const SERVER: &str = r#"
import os, sys
try:
    import numpy as np
    import jax, jax.extend
    backend = jax.extend.backend.get_backend(os.environ.get("AUTODYNE_XLA_PLATFORM", "cpu"))
    device = backend.local_devices()[0]
except Exception as e:
    print("error\t" + repr(e).replace("\n", " ").replace("\t", " "), flush=True)
    sys.exit(1)
print(f"ready\t{backend.platform} {device.device_kind} (jax {jax.__version__})", flush=True)
executables = []
for line in sys.stdin:
    request = line.rstrip("\n").split("\t")
    try:
        if request[0] == "compile":
            with open(request[1], encoding="utf-8") as f:
                executables.append(backend.compile_and_load(f.read(), [device]))
            print(f"ok\t{len(executables) - 1}", flush=True)
        elif request[0] == "run":
            exe, n = executables[int(request[1])], int(request[2])
            args = [jax.device_put(np.load(p), device) for p in request[3:3 + n]]
            outs = exe.execute_sharded(args).disassemble_into_single_device_arrays()
            for path, out in zip(request[3 + n:], outs):
                np.save(path, np.asarray(out[0]))
            print(f"ok\t{len(outs)}", flush=True)
        else:
            print("error\tunknown request", flush=True)
    except Exception as e:
        print("error\t" + repr(e).replace("\n", " ").replace("\t", " "), flush=True)
"#;

/// A running XLA server (a Python process with JAX).
pub struct Xla {
    server: Arc<Server>,
    description: String,
}

struct Server {
    io: Mutex<(Child, ChildStdin, BufReader<ChildStdout>)>,
    dir: TempDir,
}

impl Xla {
    /// Starts the server; `None` if Python or JAX is missing (the reason is in the error from
    /// [`start`](Self::start)).
    pub fn find() -> Option<Xla> {
        Xla::start().ok()
    }

    /// Starts the server.
    pub fn start() -> Result<Xla, FluxError> {
        let python = match std::env::var_os("AUTODYNE_XLA_PYTHON") {
            Some(p) => PathBuf::from(p),
            None => find_tool(None, "python3")
                .or_else(|| find_tool(None, "python"))
                .ok_or_else(|| FluxError::Tool { tool: "python", message: "no python3 / python on PATH (or set AUTODYNE_XLA_PYTHON)".into() })?,
        };
        let dir = TempDir::new()?;
        let script = dir.path.join("xla_server.py");
        std::fs::write(&script, SERVER)?;
        let mut child = Command::new(&python).arg("-u").arg(&script).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
        let stdin = child.stdin.take().expect("piped");
        let mut stdout = BufReader::new(child.stdout.take().expect("piped"));
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        let description = match line.trim_end().split_once('\t') {
            Some(("ready", d)) => d.to_string(),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                let message = line.trim_end().strip_prefix("error\t").unwrap_or("the server did not start").to_string();
                return Err(FluxError::Tool { tool: "xla", message });
            }
        };
        Ok(Xla { server: Arc::new(Server { io: Mutex::new((child, stdin, stdout)), dir }), description })
    }

    /// The platform and device the server runs on, e.g. `cpu cpu (jax 0.11.2)`.
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl std::fmt::Debug for Xla {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Xla({})", self.description)
    }
}

impl Server {
    /// Sends one request; returns the reply's payload.
    fn request(&self, fields: &[String]) -> Result<String, FluxError> {
        let mut io = self.io.lock().unwrap_or_else(|e| e.into_inner());
        let (_, stdin, stdout) = &mut *io;
        writeln!(stdin, "{}", fields.join("\t"))?;
        stdin.flush()?;
        let mut line = String::new();
        if stdout.read_line(&mut line)? == 0 {
            return Err(FluxError::Tool { tool: "xla", message: "the server exited".into() });
        }
        match line.trim_end().split_once('\t') {
            Some(("ok", payload)) => Ok(payload.to_string()),
            Some(("error", message)) => Err(FluxError::Tool { tool: "xla", message: message.to_string() }),
            _ => Err(FluxError::Tool { tool: "xla", message: format!("unexpected reply {line:?}") }),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let io = self.io.get_mut().unwrap_or_else(|e| e.into_inner());
        let _ = io.0.kill();
        let _ = io.0.wait();
    }
}

fn path(p: &std::path::Path) -> Result<String, FluxError> {
    let s = p.to_str().ok_or_else(|| FluxError::Tool { tool: "xla", message: format!("non-UTF-8 path {p:?}") })?;
    if s.contains(['\t', '\n']) {
        return Err(FluxError::Tool { tool: "xla", message: format!("unusable path {p:?}") });
    }
    Ok(s.to_string())
}

impl Backend for Xla {
    fn name(&self) -> &'static str {
        "xla"
    }

    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError> {
        let source = self.server.dir.file("module", "mlir");
        let _cleanup = Scratch(vec![source.clone()]);
        std::fs::write(&source, &program.text)?;
        let id = self.server.request(&["compile".into(), path(&source)?])?;
        Ok(Box::new(Compiled { server: self.server.clone(), id, program: program.clone() }))
    }
}

/// A program compiled and loaded by the server.
struct Compiled {
    server: Arc<Server>,
    id: String,
    program: Program,
}

impl Executable for Compiled {
    fn run(&self, inputs: &[NdArray<f32>]) -> Result<Vec<NdArray<f32>>, FluxError> {
        check_inputs(&self.program, inputs)?;
        let ins: Vec<PathBuf> = inputs.iter().map(|_| self.server.dir.file("in", "npy")).collect();
        let outs: Vec<PathBuf> = self.program.outputs.iter().map(|_| self.server.dir.file("out", "npy")).collect();
        let _cleanup = Scratch(ins.iter().chain(&outs).cloned().collect());
        let mut request = vec!["run".to_string(), self.id.clone(), inputs.len().to_string()];
        for (x, p) in inputs.iter().zip(&ins) {
            npy::save(p, x)?;
            request.push(path(p)?);
        }
        for p in &outs {
            request.push(path(p)?);
        }
        self.server.request(&request)?;
        let results = outs.iter().map(|p| npy::load(p)).collect::<Result<Vec<_>, _>>()?;
        check_outputs(&self.program, &results)?;
        Ok(results)
    }
}
