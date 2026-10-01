//! Plugin build tasks.
//!
//! - `cargo xtask bundle <plugin> [--release]`, `cargo xtask bundle -p <a> -p <b> [--release]`:
//!   builds plugin bundles (.clap, .vst3) in `target/bundled`, each with its third-party notices
//!   (next to the bundle, and inside bundles that are directories).
//! - `cargo xtask notices`: regenerates every plugin's `THIRD-PARTY-LICENSES.html` from its
//!   dependency graph (needs `cargo install cargo-about`; CI does this on every change to main).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nice_plug_xtask::Result;

const PLUGINS_DIR: &str = "plugins";
const NOTICES: &str = "THIRD-PARTY-LICENSES.html";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("notices") => {
            nice_plug_xtask::chdir_workspace_root()?;
            notices()
        }
        Some("bundle" | "bundle-universal") => {
            nice_plug_xtask::main_with_args("cargo xtask", args.clone())?;
            // nice_plug_xtask has moved to the workspace root
            bundle_notices(&bundled_packages(&args[1..]))
        }
        _ => nice_plug_xtask::main_with_args("cargo xtask", args),
    }
}

/// The packages a `bundle` command builds: every `-p <name>`, or else the first argument.
fn bundled_packages(args: &[String]) -> Vec<String> {
    if args.first().map(String::as_str) == Some("-p") {
        args.windows(2).filter(|w| w[0] == "-p").map(|w| w[1].clone()).collect()
    } else {
        args.first().cloned().into_iter().collect()
    }
}

/// Plugin crates: the directories under `plugins/` that build a plugin library (cdylib).
fn plugin_dirs() -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(PLUGINS_DIR)? {
        let dir = entry?.path();
        let manifest = fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
        if manifest.contains("\"cdylib\"") {
            dirs.push(dir);
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// Regenerates each plugin's notices with cargo-about (policy and template: about.toml, about.hbs).
fn notices() -> Result<()> {
    for dir in plugin_dirs()? {
        let output = dir.join(NOTICES);
        let status = Command::new("cargo")
            .args(["about", "generate", "--all-features", "--locked", "--fail", "-c", "about.toml", "-m"])
            .arg(dir.join("Cargo.toml"))
            .arg("-o")
            .arg(&output)
            .arg("about.hbs")
            .status();
        match status {
            Ok(status) if status.success() => println!("Wrote '{}'", output.display()),
            Ok(status) => anyhow_bail(format!("cargo about failed for '{}' ({status})", dir.display()))?,
            Err(e) => anyhow_bail(format!("could not run cargo about ({e}); install it with `cargo install cargo-about`"))?,
        }
    }
    Ok(())
}

/// Copies each bundled plugin's notices next to its bundles, and into those that are directories
/// (`.vst3` everywhere, `.clap` on macOS) under `Contents/Resources`.
fn bundle_notices(packages: &[String]) -> Result<()> {
    let bundled = std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from).join("bundled");
    for package in packages {
        let source = Path::new(PLUGINS_DIR).join(package).join(NOTICES);
        if !source.exists() {
            anyhow_bail(format!("'{}' is missing: run `cargo xtask notices`", source.display()))?;
        }
        fs::copy(&source, bundled.join(format!("{package}-{NOTICES}")))?;
        for extension in ["clap", "vst3"] {
            let bundle = bundled.join(format!("{package}.{extension}"));
            if bundle.is_dir() {
                let resources = bundle.join("Contents").join("Resources");
                fs::create_dir_all(&resources)?;
                fs::copy(&source, resources.join(NOTICES))?;
            }
        }
        println!("Added third-party notices to the '{package}' bundles");
    }
    Ok(())
}

fn anyhow_bail(message: String) -> Result<()> {
    Err(std::io::Error::other(message).into())
}
