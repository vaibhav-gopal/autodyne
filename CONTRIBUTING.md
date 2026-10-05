# Contributing to autodyne

Thanks for helping. Bug reports, fixes, new processors, tests, benchmarks and documentation are all welcome.

## Before you start

- For anything larger than a fix, open an issue first so we can agree on the design.
- Contributions come in on the terms in [LICENSE.md](LICENSE.md#contributing): you license your contribution under
  MIT No Attribution (MIT-0), so it can ship under every license autodyne is offered under; you sign off each commit
  under the Developer Certificate of Origin 1.1 (`git commit -s`, with the name and email of the commit's author);
  and you grant a patent license. You keep the copyright in your work.
- Only submit work you wrote yourself, or third-party code whose license allows it. Say where such code comes from
  and under which license. Third-party code and dependencies must not block any of autodyne's three licenses:
  permissive (MIT, Apache-2.0, BSD, ISC, Zlib, ...) or MPL-2.0, never GPL, LGPL or AGPL. `tend check` reports one
  that would.

## Setup

- Rust stable, pinned in `rust-toolchain.toml` (rustup installs it on first use).
- Linux: ALSA headers for the `live` example (`libasound2-dev` on Debian/Ubuntu).
- Plugins: `cargo xtask bundle -p autodyne-reverb -p autodyne-synth --release`.

## What CI checks

Run these before pushing; a pull request must pass all of them:

```sh
cargo test --workspace --all-targets
cargo test --workspace --doc
cargo test -p autodyne --features jit,gpu --lib -- flux:: gpu::
cargo clippy --workspace --all-targets --features autodyne/ndarray,autodyne/flux,autodyne/jit,autodyne/gpu -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude xtask --no-deps --features autodyne/flux,autodyne/jit,autodyne/gpu
cargo deny --workspace check advisories bans sources   # RustSec, bans, sources (install: cargo install cargo-deny)
tend check --strict   # the design, licenses against LICENSE.md's three, sign-offs (not yet in CI)
```

The Python bindings (`bindings/python`) and the comparison benchmarks (`bench/rust`) are workspaces of their
own: after changing a public API, build them too (`cargo check` in each).

## Code guidelines

- **Real time.** Processing (`process`, `render`, `next_sample`, ...) must never allocate, lock or block.
  Allocate in constructors (or a plugin's `initialize`). `tests/no_alloc.rs` enforces this for the processors it
  covers; add new processors to it.
- **Tests against known answers.** Check a processor against something independent: a closed-form signal, a
  published frequency response, a slow reference implementation, a measured quantity (RT60, SNR, pitch). "It
  runs" is not a test.
- **Parameters.** User-facing settings go through `Parameterized` (ids, ranges, units, defaults), so hosts,
  presets and the plugin bridge see them.
- **Layering.** A module uses only modules below it (the order of the README's module table); each processor
  module implements `Processor` and `Parameterized` for its own types (`<module>/params.rs`). No cycles.
- **Errors.** Operations that can fail on their inputs (shapes, designs, files, external tools) return `Result`
  with their module's error type (`NdError`, `LinalgError`, `FilterError`, `SpectralError`, ...): an enum deriving
  `Debug, Clone, PartialEq, Eq, Error` (`Copy` when it can), every variant documented, an `Invalid(String)`
  variant built with `XxxError::invalid(message)` for bad arguments, and transparent `#[from]` variants wrapping
  the lower layers' errors (never flattened to strings). Processors clamp settings into range instead of failing;
  a constructor may panic only on a contract violation its docs state ("Panics if ..."). Internal invariants use
  `expect("why it holds")`, not bare `unwrap()`.
- **Documentation.** Every public item has a doc comment (`#![warn(missing_docs)]`): units and ranges for
  settings, what each variant or field means.
- **Tests.** Unit tests sit at the end of their file in `#[cfg(test)] mod tests` (a module's `tests.rs` when they
  are long), named for what they check (`biquad_lowpass_is_3db_down_at_cutoff`), using the shared helpers in
  `src/testing.rs` (`assert_close`, deterministic `noise_*` / `random_*` data) rather than new copies.
  Cross-module behaviour goes in `tests/`. GPU tests return early when `gpu::available()` is false.
- **Benchmarks.** `bench/dsp.rs` (criterion) times autodyne alone; comparisons with other libraries live under
  `bench/<suite>/`, check that every library agrees before timing, print a Markdown table and take `--out FILE`
  (see [`bench/README.md`](bench/README.md)). Wrap criterion inputs in `black_box`.
- **Match the surrounding code.** Naming, comment density and doc style as in the module you are changing. Put
  math in doc comments inside backticks.
- **New dependencies** need a reason; they must pass `cargo deny` and `tend check`. Their license notices are
  regenerated automatically on main (`cargo xtask notices` does it locally, with `cargo install cargo-about`).

## Commits and pull requests

- Small, focused commits with messages that explain why, not only what; signed off (`git commit -s`).
- Describe in the pull request what changed, how you tested it, and any performance numbers (`cargo bench`) for
  DSP changes.
