//! XLA (or any other PJRT plugin) loaded directly through the PJRT C API: no Python, no linking at
//! build time. The plugin is a shared library (`libpjrt_cpu.so`, `pjrt_c_api_gpu_plugin.so`, ...)
//! opened at run time; flux compiles StableHLO text with it and runs the executable on its first
//! device.
//!
//! The plugin is found in `AUTODYNE_PJRT_PLUGIN` (a path). Prebuilt plugins exist for Linux and
//! macOS (CPU, CUDA, ROCm: e.g. jaxlib's GPU plugins, or the builds at github.com/zml/pjrt-artifacts);
//! on Windows use [`Xla`](super::Xla) instead.
//!
//! The bindings follow `xla/pjrt/c/pjrt_c_api.h` (API 0.x). Every call passes an argument struct
//! whose `struct_size` tells the plugin which fields exist; the function table only grows at its
//! end, so the entries used here sit at fixed positions.

use std::ffi::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::Arc;

use super::hlo::Program;
use super::runtime::{check_inputs, check_resident, Backend, DeviceArray, Executable, Resident};
use super::FluxError;
use crate::signal::NdArray;

// opaque plugin objects
type Error = c_void;
type Event = c_void;
type Client = c_void;
type Device = c_void;
type LoadedExecutable = c_void;
type Buffer = c_void;

/// Every PJRT function: one argument struct in, an error (null on success) out.
type Fn = unsafe extern "C" fn(*mut c_void) -> *mut Error;

/// Positions in the function table (after the version header).
mod index {
    pub const ERROR_DESTROY: usize = 0;
    pub const ERROR_MESSAGE: usize = 1;
    pub const PLUGIN_INITIALIZE: usize = 3;
    pub const EVENT_DESTROY: usize = 5;
    pub const EVENT_AWAIT: usize = 8;
    pub const CLIENT_CREATE: usize = 10;
    pub const CLIENT_DESTROY: usize = 11;
    pub const CLIENT_PLATFORM_NAME: usize = 12;
    pub const CLIENT_ADDRESSABLE_DEVICES: usize = 16;
    pub const CLIENT_COMPILE: usize = 20;
    pub const CLIENT_BUFFER_FROM_HOST_BUFFER: usize = 22;
    pub const LOADED_EXECUTABLE_DESTROY: usize = 50;
    pub const LOADED_EXECUTABLE_EXECUTE: usize = 55;
    pub const BUFFER_DESTROY: usize = 58;
    pub const BUFFER_TO_HOST_BUFFER: usize = 70;
    /// The highest position used.
    pub const LAST: usize = 70;
}

#[repr(C)]
struct ApiVersion {
    struct_size: usize,
    extension_start: *mut c_void,
    major: i32,
    minor: i32,
}

/// The head of `PJRT_Api`; the function pointers follow it.
#[repr(C)]
struct ApiHeader {
    struct_size: usize,
    extension_start: *mut c_void,
    version: ApiVersion,
}

const BUFFER_TYPE_F32: i32 = 11;
const HOST_BUFFER_IMMUTABLE_ONLY_DURING_CALL: i32 = 0;

/// Declares an argument struct (`struct_size` and `extension_start` first, then the fields) with a
/// constructor that zeroes it and sets its size.
macro_rules! args {
    ($Name:ident { $($field:ident: $T:ty),* $(,)? }) => {
        #[repr(C)]
        struct $Name {
            struct_size: usize,
            extension_start: *mut c_void,
            $($field: $T),*
        }
        impl $Name {
            fn new() -> Self {
                // SAFETY: every field is an integer, a bool or a raw pointer, for which zero is valid
                let mut a: Self = unsafe { std::mem::zeroed() };
                a.struct_size = std::mem::size_of::<Self>();
                a
            }
        }
    };
}

args!(ErrorDestroyArgs { error: *mut Error });
args!(ErrorMessageArgs { error: *const Error, message: *const c_char, message_size: usize });
args!(PluginInitializeArgs {});
args!(EventDestroyArgs { event: *mut Event });
args!(EventAwaitArgs { event: *mut Event });
args!(ClientCreateArgs {
    create_options: *const c_void,
    num_options: usize,
    kv_get_callback: *const c_void,
    kv_get_user_arg: *mut c_void,
    kv_put_callback: *const c_void,
    kv_put_user_arg: *mut c_void,
    client: *mut Client,
    kv_try_get_callback: *const c_void,
    kv_try_get_user_arg: *mut c_void,
});
args!(ClientDestroyArgs { client: *mut Client });
args!(ClientPlatformNameArgs { client: *mut Client, platform_name: *const c_char, platform_name_size: usize });
args!(ClientAddressableDevicesArgs { client: *mut Client, devices: *const *mut Device, num_devices: usize });
args!(ProgramArgs { code: *const c_char, code_size: usize, format: *const c_char, format_size: usize });
args!(ClientCompileArgs {
    client: *mut Client,
    program: *const ProgramArgs,
    compile_options: *const c_char,
    compile_options_size: usize,
    executable: *mut LoadedExecutable,
});
args!(BufferFromHostArgs {
    client: *mut Client,
    data: *const c_void,
    element_type: i32,
    dims: *const i64,
    num_dims: usize,
    byte_strides: *const i64,
    num_byte_strides: usize,
    semantics: i32,
    device: *mut Device,
    memory: *mut c_void,
    device_layout: *mut c_void,
    done_with_host_buffer: *mut Event,
    buffer: *mut Buffer,
});
args!(LoadedExecutableDestroyArgs { executable: *mut LoadedExecutable });
args!(ExecuteOptions {
    send_callbacks: *mut c_void,
    recv_callbacks: *mut c_void,
    num_send_ops: usize,
    num_recv_ops: usize,
    launch_id: i32,
    non_donatable_input_indices: *const i64,
    num_non_donatable_input_indices: usize,
    context: *mut c_void,
    call_location: *const c_char,
    num_tasks: usize,
    task_ids: *mut i32,
    incarnation_ids: *mut i64,
    multi_slice_config: *mut c_void,
    use_major_to_minor_data_layout_for_callbacks: bool,
    hlo_output_callbacks: *mut c_void,
    num_hlo_output_callbacks: usize,
    custom_options: *const c_void,
    num_custom_options: usize,
});
args!(ExecuteArgs {
    executable: *mut LoadedExecutable,
    options: *mut ExecuteOptions,
    argument_lists: *const *const *mut Buffer,
    num_devices: usize,
    num_args: usize,
    output_lists: *const *mut *mut Buffer,
    device_complete_events: *mut *mut Event,
    execute_device: *mut Device,
});
args!(BufferDestroyArgs { buffer: *mut Buffer });
args!(BufferToHostArgs { src: *mut Buffer, host_layout: *mut c_void, dst: *mut c_void, dst_size: usize, event: *mut Event });

/// `xla.CompileOptionsProto { executable_build_options { num_replicas: 1, num_partitions: 1 } }`,
/// serialized.
const COMPILE_OPTIONS: &[u8] = &[0x1A, 0x04, 0x20, 0x01, 0x28, 0x01];

/// A loaded plugin with a client on its first device.
struct Plugin {
    functions: *const Fn,
    client: *mut Client,
    device: *mut Device,
    platform: String,
    // last, so the library is unloaded after the client is destroyed
    _library: libloading::Library,
}

// SAFETY: PJRT clients, executables and buffers are thread-safe by contract (the C API is used from
// many threads by JAX); the raw pointers are only handles into the plugin.
unsafe impl Send for Plugin {}
unsafe impl Sync for Plugin {}

impl Plugin {
    fn function(&self, i: usize) -> Fn {
        // SAFETY: `load` checked the table holds at least `index::LAST + 1` entries
        unsafe { *self.functions.add(i) }
    }

    /// Calls entry `i` with `args`, turning a returned error into a `FluxError`.
    fn call<A>(&self, i: usize, args: &mut A) -> Result<(), FluxError> {
        // SAFETY: `args` is the argument struct for entry `i` (the callers pair them)
        let error = unsafe { self.function(i)((args as *mut A).cast()) };
        if error.is_null() {
            return Ok(());
        }
        let mut message = ErrorMessageArgs::new();
        message.error = error;
        // SAFETY: a valid error from the plugin
        unsafe { self.function(index::ERROR_MESSAGE)((&mut message as *mut ErrorMessageArgs).cast()) };
        let text = if message.message.is_null() {
            "unknown error".to_string()
        } else {
            // SAFETY: the plugin returns `message_size` bytes valid until the error is destroyed
            String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(message.message.cast::<u8>(), message.message_size) }).into_owned()
        };
        let mut destroy = ErrorDestroyArgs::new();
        destroy.error = error;
        // SAFETY: destroying the error once
        unsafe { self.function(index::ERROR_DESTROY)((&mut destroy as *mut ErrorDestroyArgs).cast()) };
        Err(FluxError::Tool { tool: "pjrt", message: text })
    }

    /// Waits for `event` (if any) and destroys it.
    fn await_event(&self, event: *mut Event) -> Result<(), FluxError> {
        if event.is_null() {
            return Ok(());
        }
        let mut wait = EventAwaitArgs::new();
        wait.event = event;
        let waited = self.call(index::EVENT_AWAIT, &mut wait);
        let mut destroy = EventDestroyArgs::new();
        destroy.event = event;
        let destroyed = self.call(index::EVENT_DESTROY, &mut destroy);
        waited.and(destroyed)
    }

    fn destroy_buffer(&self, buffer: *mut Buffer) {
        if !buffer.is_null() {
            let mut args = BufferDestroyArgs::new();
            args.buffer = buffer;
            let _ = self.call(index::BUFFER_DESTROY, &mut args);
        }
    }
}

impl Drop for Plugin {
    fn drop(&mut self) {
        let mut args = ClientDestroyArgs::new();
        args.client = self.client;
        let _ = self.call(index::CLIENT_DESTROY, &mut args);
    }
}

/// A PJRT plugin (XLA on CPU or GPU, or any other PJRT backend) loaded in-process.
pub struct Pjrt {
    plugin: Arc<Plugin>,
}

impl std::fmt::Debug for Pjrt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pjrt({})", self.plugin.platform)
    }
}

impl Pjrt {
    /// Loads the plugin named by `AUTODYNE_PJRT_PLUGIN`; `None` if it is unset or fails to load
    /// (the reason is in the error from [`load`](Self::load)).
    pub fn find() -> Option<Pjrt> {
        let path = PathBuf::from(std::env::var_os("AUTODYNE_PJRT_PLUGIN")?);
        Pjrt::load(&path).ok()
    }

    /// Loads a PJRT plugin library and creates a client on its first device.
    pub fn load(path: &Path) -> Result<Pjrt, FluxError> {
        let tool_error = |message: String| FluxError::Tool { tool: "pjrt", message };
        // SAFETY: loading a PJRT plugin runs its initializers, which is what the caller asks for
        let library = unsafe { libloading::Library::new(path) }.map_err(|e| tool_error(format!("cannot load {}: {e}", path.display())))?;
        // SAFETY: `GetPjrtApi` has this signature in every PJRT plugin
        let api = unsafe {
            let get: libloading::Symbol<unsafe extern "C" fn() -> *const ApiHeader> =
                library.get(b"GetPjrtApi\0").map_err(|e| tool_error(format!("{} is not a PJRT plugin: {e}", path.display())))?;
            get()
        };
        if api.is_null() {
            return Err(tool_error("GetPjrtApi returned null".into()));
        }
        // SAFETY: the plugin returns a pointer to its static API table
        let header = unsafe { &*api };
        if header.version.major != 0 {
            return Err(tool_error(format!("unsupported PJRT API version {}.{}", header.version.major, header.version.minor)));
        }
        let head = std::mem::size_of::<ApiHeader>();
        if header.struct_size < head + (index::LAST + 1) * std::mem::size_of::<Fn>() {
            return Err(tool_error(format!("PJRT API table too small ({} bytes)", header.struct_size)));
        }
        // SAFETY: the function pointers start right after the header (checked large enough above)
        let functions = unsafe { api.cast::<u8>().add(head).cast::<Fn>() };
        let mut plugin = Plugin { functions, client: null_mut(), device: null_mut(), platform: String::new(), _library: library };

        plugin.call(index::PLUGIN_INITIALIZE, &mut PluginInitializeArgs::new())?;
        let mut create = ClientCreateArgs::new();
        plugin.call(index::CLIENT_CREATE, &mut create)?;
        plugin.client = create.client;

        let mut devices = ClientAddressableDevicesArgs::new();
        devices.client = plugin.client;
        plugin.call(index::CLIENT_ADDRESSABLE_DEVICES, &mut devices)?;
        if devices.num_devices == 0 {
            return Err(tool_error("the plugin has no addressable device".into()));
        }
        // SAFETY: the plugin returned `num_devices` (>= 1) device pointers
        plugin.device = unsafe { *devices.devices };

        let mut name = ClientPlatformNameArgs::new();
        name.client = plugin.client;
        plugin.call(index::CLIENT_PLATFORM_NAME, &mut name)?;
        // SAFETY: `platform_name_size` bytes owned by the client
        plugin.platform = String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(name.platform_name.cast::<u8>(), name.platform_name_size) }).into_owned();
        plugin.platform = format!("{} (PJRT API 0.{})", plugin.platform, header.version.minor);
        Ok(Pjrt { plugin: Arc::new(plugin) })
    }

    /// The platform, e.g. `cpu (PJRT API 0.116)`.
    pub fn description(&self) -> &str {
        &self.plugin.platform
    }
}

impl Backend for Pjrt {
    fn name(&self) -> &'static str {
        "pjrt"
    }

    fn compile(&self, program: &Program) -> Result<Box<dyn Executable>, FluxError> {
        let format = "mlir";
        let mut code = ProgramArgs::new();
        code.code = program.text.as_ptr().cast();
        code.code_size = program.text.len();
        code.format = format.as_ptr().cast();
        code.format_size = format.len();
        let mut args = ClientCompileArgs::new();
        args.client = self.plugin.client;
        args.program = &code;
        args.compile_options = COMPILE_OPTIONS.as_ptr().cast();
        args.compile_options_size = COMPILE_OPTIONS.len();
        self.plugin.call(index::CLIENT_COMPILE, &mut args)?;
        Ok(Box::new(Compiled { plugin: self.plugin.clone(), executable: args.executable, program: program.clone() }))
    }
}

/// An executable loaded on the plugin's device.
struct Compiled {
    plugin: Arc<Plugin>,
    executable: *mut LoadedExecutable,
    program: Program,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        let mut args = LoadedExecutableDestroyArgs::new();
        args.executable = self.executable;
        let _ = self.plugin.call(index::LOADED_EXECUTABLE_DESTROY, &mut args);
    }
}

/// A buffer on the plugin's device, destroyed when dropped (the handle inside a [`DeviceArray`]).
struct DeviceBuffer {
    plugin: Arc<Plugin>,
    buffer: *mut Buffer,
}

// SAFETY: PJRT buffers are thread-safe handles (see `Plugin`)
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.plugin.destroy_buffer(self.buffer);
    }
}

impl Compiled {
    /// The PJRT buffer of `a`, which must live on this executable's plugin.
    fn buffer_of(&self, a: &DeviceArray) -> Result<*mut Buffer, FluxError> {
        match &a.data {
            Resident::Device(handle) => match handle.downcast_ref::<DeviceBuffer>() {
                Some(b) if Arc::ptr_eq(&b.plugin, &self.plugin) => Ok(b.buffer),
                _ => Err(FluxError::Shape("this array is held by another backend".into())),
            },
            Resident::Host(_) => Err(FluxError::Shape("this array was not uploaded to this backend".into())),
        }
    }
}

impl Executable for Compiled {
    fn run(&self, inputs: &[NdArray<f32>]) -> Result<Vec<NdArray<f32>>, FluxError> {
        check_inputs(&self.program, inputs)?;
        let resident = inputs.iter().map(|x| self.upload(x)).collect::<Result<Vec<_>, _>>()?;
        let outputs = self.run_resident(&resident.iter().collect::<Vec<_>>())?;
        outputs.iter().map(|o| self.download(o)).collect()
    }

    fn upload(&self, a: &NdArray<f32>) -> Result<DeviceArray, FluxError> {
        let plugin = &*self.plugin;
        // the data is copied during the call
        let dims: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
        let mut args = BufferFromHostArgs::new();
        args.client = plugin.client;
        args.data = a.as_slice().as_ptr().cast();
        args.element_type = BUFFER_TYPE_F32;
        args.dims = dims.as_ptr();
        args.num_dims = dims.len();
        args.semantics = HOST_BUFFER_IMMUTABLE_ONLY_DURING_CALL;
        args.device = plugin.device;
        plugin.call(index::CLIENT_BUFFER_FROM_HOST_BUFFER, &mut args)?;
        let buffer = DeviceBuffer { plugin: self.plugin.clone(), buffer: args.buffer };
        plugin.await_event(args.done_with_host_buffer)?;
        Ok(DeviceArray::device(a.shape().to_vec(), Box::new(buffer)))
    }

    fn run_resident(&self, inputs: &[&DeviceArray]) -> Result<Vec<DeviceArray>, FluxError> {
        check_resident(&self.program, inputs)?;
        let plugin = &*self.plugin;
        let arguments = inputs.iter().map(|a| self.buffer_of(a)).collect::<Result<Vec<_>, _>>()?;

        // execute on one device
        let mut outputs: Vec<*mut Buffer> = vec![null_mut(); self.program.outputs.len()];
        let argument_list: *const *mut Buffer = arguments.as_ptr();
        let output_list: *mut *mut Buffer = outputs.as_mut_ptr();
        let mut done: *mut Event = null_mut();
        let mut options = ExecuteOptions::new();
        let mut args = ExecuteArgs::new();
        args.executable = self.executable;
        args.options = &mut options;
        args.argument_lists = &argument_list;
        args.num_devices = 1;
        args.num_args = arguments.len();
        args.output_lists = &output_list;
        args.device_complete_events = &mut done;
        plugin.call(index::LOADED_EXECUTABLE_EXECUTE, &mut args)?;
        // own the outputs before anything can fail
        let outputs: Vec<DeviceArray> = outputs
            .into_iter()
            .zip(&self.program.outputs)
            .map(|(buffer, shape)| DeviceArray::device(shape.clone(), Box::new(DeviceBuffer { plugin: self.plugin.clone(), buffer })))
            .collect();
        plugin.await_event(done)?;
        Ok(outputs)
    }

    fn download(&self, a: &DeviceArray) -> Result<NdArray<f32>, FluxError> {
        let plugin = &*self.plugin;
        let buffer = self.buffer_of(a)?;
        let mut data = vec![0.0f32; a.shape().iter().product()];
        let mut args = BufferToHostArgs::new();
        args.src = buffer;
        args.dst = data.as_mut_ptr().cast();
        args.dst_size = std::mem::size_of_val(data.as_slice());
        plugin.call(index::BUFFER_TO_HOST_BUFFER, &mut args)?;
        plugin.await_event(args.event)?;
        NdArray::from_vec(data, a.shape()).map_err(|e| FluxError::Shape(e.to_string()))
    }
}