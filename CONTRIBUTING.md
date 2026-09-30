# Contributing to autodyne

Thanks for helping. Bug reports, fixes, new processors, tests, benchmarks and documentation are all welcome.

## Before you start

- For anything larger than a fix, open an issue first so we can agree on the design.
- Pull requests need a one-time signature of the [Contributor License Agreement](CLA.md). The CLA Assistant bot
  comments on your first pull request with instructions: you sign by replying with a single comment. autodyne is
  GPLv3 and also licensed commercially; the CLA gives the maintainer the rights to do both with your contribution,
  while you keep the copyright in your work.
- Only submit work you wrote yourself, or third-party code whose license allows it. Say where such code comes
  from and under which license (see section 5 of the CLA). Third-party code must be permissively licensed (MIT,
  Apache-2.0, BSD, ISC, Zlib, ...): no GPL, LGPL or AGPL code or dependencies.

## Setup

- Stable Rust (see CI for the version in use).
- Linux: ALSA headers for the `live` example (`libasound2-dev` on Debian/Ubuntu).
- Plugins: `cargo xtask bundle -p autodyne-reverb -p autodyne-synth --release`.

## What CI checks

Run these before pushing; a pull request must pass all of them:

```sh
cargo test --workspace --all-targets
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude xtask --no-deps
cargo deny --workspace check          # licenses, advisories, sources (install: cargo install cargo-deny)
```

## Code guidelines

- **Real time.** Processing (`process`, `render`, `next_sample`, ...) must never allocate, lock or block.
  Allocate in constructors (or a plugin's `initialize`). `tests/no_alloc.rs` enforces this for the processors it
  covers; add new processors to it.
- **Tests against known answers.** Check a processor against something independent: a closed-form signal, a
  published frequency response, a slow reference implementation, a measured quantity (RT60, SNR, pitch). "It
  runs" is not a test.
- **Parameters.** User-facing settings go through `Parameterized` (ids, ranges, units, defaults), so hosts,
  presets and the plugin bridge see them.
- **Match the surrounding code.** Naming, comment density and doc style as in the module you are changing. Put
  math in doc comments inside backticks.
- **New dependencies** need a reason; they must pass `cargo deny`.

## Commits and pull requests

- Small, focused commits with messages that explain why, not only what.
- Describe in the pull request what changed, how you tested it, and any performance numbers (`cargo bench`) for
  DSP changes.
