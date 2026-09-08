//! Vendors the host's prebuilt `libpdfium` and stamps build provenance.
//!
//! Fetches the right prebuilt `libpdfium` for the host's target triple from
//! [bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries),
//! caches it under `$OUT_DIR/pdfium/`, and exposes the absolute path of
//! the dylib to the runtime via `cargo:rustc-env=KERNELGUY_PDFIUM_LIB=...`.
//!
//! The PDF tool (`src/tool/pdf.rs`) reads that env via `env!()` at compile
//! time and calls `Pdfium::bind_to_library(...)` at runtime — no
//! third-party companion crate, no runtime download, no
//! `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` setup.
//!
//! ## Versioning
//!
//! `PDFIUM_VERSION` is pinned so builds are reproducible. Bump it when we
//! want to upgrade the bundled `libpdfium`.
//!
//! ## Caching
//!
//! Cache lives in `$OUT_DIR/pdfium/`. `cargo clean` wipes it (forcing a
//! re-download next build); a normal incremental build skips the
//! download because we early-out as soon as the dylib exists.
//!
//! ## Tooling
//!
//! We shell out to `curl` and `tar` instead of pulling pure-Rust HTTP +
//! gunzip + tar crates, because (a) both tools are universally
//! available on the platform we care about (Linux), and (b) it
//! keeps `[build-dependencies]` empty.
//!
//! ## Error handling
//!
//! `main` returns `Result` rather than panicking: Cargo renders a returned `Err`
//! as a plain build error, where a panic buries it in a backtrace.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Pinned pdfium-binaries release tag. Update to upgrade `libpdfium`.
const PDFIUM_VERSION: &str = "chromium/7763";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    // Map (target_os, target_arch) → (pdfium-binaries archive name,
    // path inside the archive to the dynamic library).
    //
    // Names follow bblanchon/pdfium-binaries' release naming
    // (`pdfium-<os>-<arch>.tgz`); their layout puts the dylib under
    // `lib/` on Unix and `bin/` on Windows.
    let (archive, lib_subpath) = match (target_os.as_str(), target_arch.as_str()) {
        ("linux", "x86_64") => ("pdfium-linux-x64.tgz", "lib/libpdfium.so"),
        ("linux", "aarch64") => ("pdfium-linux-arm64.tgz", "lib/libpdfium.so"),
        (os, arch) => {
            return Err(format!("build.rs: no libpdfium mapping for target {os}/{arch}; add one in build.rs").into());
        }
    };

    let out_dir: PathBuf = env::var_os("OUT_DIR").ok_or("build.rs: OUT_DIR unset")?.into();
    let extract_dir = out_dir.join("pdfium");
    let lib_path = extract_dir.join(lib_subpath);

    if !lib_path.exists() {
        fs::create_dir_all(&extract_dir)?;

        let url = format!(
            "https://github.com/bblanchon/pdfium-binaries/releases/download/{}/{}",
            PDFIUM_VERSION.replace('/', "%2F"),
            archive,
        );
        let archive_path = out_dir.join(archive);

        eprintln!("build.rs: fetching libpdfium ({PDFIUM_VERSION}) for {target_os}/{target_arch}...");
        let curl = Command::new("curl")
            .args(["--fail", "--location", "--silent", "--show-error", "--output"])
            .arg(&archive_path)
            .arg(&url)
            .status()
            .map_err(|e| format!("build.rs: spawn curl (is curl installed?): {e}"))?;
        if !curl.success() {
            return Err(format!("build.rs: curl failed for {url}").into());
        }

        let tar = Command::new("tar")
            .args(["-xzf"])
            .arg(&archive_path)
            .arg("-C")
            .arg(&extract_dir)
            .status()
            .map_err(|e| format!("build.rs: spawn tar (is tar installed?): {e}"))?;
        if !tar.success() {
            return Err(format!("build.rs: tar failed for {}", archive_path.display()).into());
        }

        // Drop the archive — we only needed it long enough to extract.
        drop(fs::remove_file(&archive_path));

        if !lib_path.exists() {
            return Err(format!(
                "build.rs: expected libpdfium at {} after extraction (archive layout changed?)",
                lib_path.display()
            )
            .into());
        }
    }

    // Resolve to a canonical absolute path so the runtime `env!()` lookup
    // doesn't trip over symlinks or relative components.
    let lib_path = fs::canonicalize(&lib_path).unwrap_or(lib_path);
    println!("cargo:rustc-env=KERNELGUY_PDFIUM_LIB={}", lib_path.display());

    // ── Build provenance ─────────────────────────────────────────────────────
    // Stamp the binary with the git revision so a run's startup banner reveals
    // exactly which commit produced it. (Freshness — "is this binary stale?" —
    // is reported separately at runtime from the executable's mtime.) This
    // exists because a 10-hour run was once launched against a stale
    // `target/release` binary, silently exercising old code.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rustc-env=KERNELGUY_GIT={}", git_revision());

    Ok(())
}

/// `<short-sha>`, suffixed with how the tree diverges from it: `-dirty` when
/// TRACKED files differ, `-unknown` when `git status` itself fails. Plain
/// `"unknown"` when git cannot report a sha at all.
///
/// Untracked files are deliberately excluded (`--untracked-files=no`). They do not
/// reach the binary, and counting them pinned the flag permanently on any host that
/// keeps scratch in the tree — draining it of meaning exactly where runs get
/// launched. The trade-off accepted: an untracked `src/*.rs` reached by a tracked
/// `mod` would change the binary without showing up here.
///
/// The flag only refreshes when `.git/HEAD`/`.git/index` change, so it can lag pure
/// working-tree edits — the runtime mtime banner is the authoritative freshness
/// signal.
fn git_revision() -> String {
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(sha) = sha else {
        return "unknown".to_string();
    };
    // Checking the exit status matters: a git that runs and FAILS writes nothing to
    // stdout, which is indistinguishable from a clean tree. Reporting a confident
    // `-clean` sha there would be a lie, so it degrades to `-unknown` instead.
    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success());
    let Some(status) = status else {
        return format!("{sha}-unknown");
    };
    if status.stdout.is_empty() {
        sha
    } else {
        format!("{sha}-dirty")
    }
}
