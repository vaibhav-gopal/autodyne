/// stored here are iterator adapters
/// primary purpose is to represent signals in buffers (or iterators which can procedurally generate samples or continuously fetch them)
/// then use iterator adapters as the primary way of applying functions to them (lazily!)
/// and only execute the functions at the very end all at once

// private module declarations
mod dft;

// pub use for export

