//! Small filesystem utilities shared across the orchestrator.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::domain::types::SolutionFiles;

/// Lock `m`, recovering from poisoning instead of panicking.
///
/// Every mutex here guards a plain accumulator (cache, pending-set, ledger) with
/// no invariant a mid-update panic could break, so propagating the poison would
/// only turn one thread's failure into a second, unrelated panic. Replaces
/// `m.lock().unwrap()`, which is a real panic path.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Durably write `bytes` to `path`, crash-safely.
///
/// Writes to a sibling temp file, fsyncs it, then atomically renames over the
/// target, and fsyncs the directory. Crash-safe: a concurrent or post-crash
/// reader sees either the old contents or the new, never a torn write. Used for
/// the resume journal, the manifest, sidecars, and committed solution-fileset
/// snapshots.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] if the parent directory cannot be
/// created, the temp file cannot be written or fsynced, or the rename fails.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("out");
    let tmp = parent.join(format!(".{name}.tmp"));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Replace the contents of `root` with `files` (keyed by relative path), each written atomically.
///
/// Any existing tree under `root` is removed first, so the result is *exactly*
/// `files` with no stale entries from a prior candidate.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] if the existing tree under `root` cannot
/// be removed, `root` cannot be created, or any file's atomic write fails.
pub fn write_file_tree(root: &Path, files: &SolutionFiles) -> io::Result<()> {
    if root.exists() {
        fs::remove_dir_all(root)?;
    }
    fs::create_dir_all(root)?;
    for (rel, content) in files {
        atomic_write(&root.join(rel), content.as_bytes())?;
    }
    Ok(())
}

/// Truncate a string to at most `max` chars, appending an ellipsis if cut.
#[must_use]
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}\u{2026}")
    }
}
