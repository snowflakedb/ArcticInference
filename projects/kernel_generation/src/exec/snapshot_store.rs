//! Workspace snapshot types and the canonical workspace layout.

use std::collections::BTreeMap;
use std::path::Path;

// ── Canonical workspace layout ────────────────────────────────────────────────
// The single source of truth for the top-level directory names inside the agent's
// workspace. The exclusion predicate, the sandbox mount setup, and the tools all
// reference THESE — so adding/renaming a mount or scratch dir can't leave the
// snapshot rule silently stale (the bug this replaces: `solution` was spelled in
// three places and `_trusted`/`checkout` had a const in one module but a bare
// literal in the predicate).

/// Agent's scored kernel — the tracked source of record.
pub const SOLUTION_DIR: &str = "solution";
/// Read-only mount: the trusted evaluator + `problem.py`.
pub const TRUSTED_DIR: &str = "_trusted";
/// Read-only mount: reference papers + markdown assets.
pub const DOCS_DIR: &str = "docs";
/// Read-only mount: repo-local startup skills.
pub const SKILLS_DIR: &str = "skills";
/// Harness scratch: where the `checkout` tool writes a recovered node's workspace.
/// Excluded so a snapshot never recursively contains a copy of another node.
pub const CHECKOUT_DIR: &str = "checkout";

/// Top-level dirs that are NOT the agent's source: read-only input mounts + harness
/// scratch + build-output dirs. Excluded by first path segment. (`docs`/`_trusted`/
/// `skills` can't hold agent data anyway — they're read-only binds — so excluding
/// them is always safe; `target` is a build-output tree, never source.)
const NON_SOURCE_TOPLEVEL: [&str; 8] = [
    DOCS_DIR,
    "trusted",
    TRUSTED_DIR,
    SKILLS_DIR,
    ".git",
    ".kg",
    CHECKOUT_DIR,
    "target",
];

/// Runtime/compile cache dir names, excluded at ANY depth (they nest, e.g.
/// `.cache/torch_extensions/…`). Shared with the sandbox's cache-env setup.
pub const CACHE_SEGMENTS: [&str; 5] = ["__pycache__", ".cache", "torch_extensions", "nv_compute_cache", ".nv"];

/// Derived binaries excluded by extension — they rebuild on eval and bloated the git
/// object store, so there's no reason to snapshot them.
///
/// Profiler outputs (`.nsys-rep`/`.ncu-rep`/`.sqlite`) are deliberately NOT here: the
/// agent may want to compare against past profiler runs, so they carry across episodes
/// like any other artifact. This is only safe because the workspace restore path is
/// bytes-exact ([`WorkspaceFiles`] = `Vec<u8>`, read via `run_git_bytes`) — do not add
/// a lossy `String` decode back into that path.
const BINARY_EXT: [&str; 5] = [".pyc", ".pyo", ".so", ".o", ".d"];

/// Full-workspace file contents keyed by workspace-relative path.
///
/// Values are raw **bytes** — the workspace can hold binary artifacts (profiler
/// dumps, etc.), and a restore must reproduce them exactly. (The narrower
/// solution-source path uses [`crate::domain::types::SolutionFiles`] = `String`;
/// that tree is text source with binaries excluded, and is consumed by the
/// diversity/hashing code as strings.)
pub type WorkspaceFiles = BTreeMap<String, Vec<u8>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceSnapshot {
    pub files: BTreeMap<String, Vec<u8>>,
    pub sha256: String,
}

impl WorkspaceSnapshot {
    /// # Errors
    ///
    /// Returns a message if walking `root` fails (`read_dir`/`file_type` I/O), if a
    /// discovered path is somehow not under `root` (`strip_prefix`), if a file
    /// cannot be read, or if a path/content length does not fit in `u64` (the width
    /// the digest is framed with).
    pub fn from_dir(root: &Path) -> Result<Self, String> {
        use sha2::Digest as _;
        let mut paths = Vec::new();
        if root.is_dir() {
            walk_files(root, &mut paths).map_err(|e| e.to_string())?;
        }
        paths.sort();
        let mut files = BTreeMap::new();
        let mut hasher = sha2::Sha256::new();
        for path in paths {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            if should_exclude_workspace_path(&rel) {
                continue;
            }
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            // Length-prefixed framing, unchanged: `u64::try_from` produces the same
            // little-endian bytes the `as u64` cast did, without the silent wrap.
            let rel_len = u64::try_from(rel.len()).map_err(|e| format!("path length exceeds u64: {e}"))?;
            hasher.update(rel_len.to_le_bytes());
            hasher.update(rel.as_bytes());
            let bytes_len = u64::try_from(bytes.len()).map_err(|e| format!("file length exceeds u64: {e}"))?;
            hasher.update(bytes_len.to_le_bytes());
            hasher.update(&bytes);
            files.insert(rel, bytes);
        }
        Ok(Self {
            files,
            sha256: crate::domain::types::sha256_hex(&hasher.finalize()),
        })
    }
}

/// The single canonical "not part of the source workspace snapshot" rule.
///
/// Shared by every path that decides what to snapshot, restore, digest, or read as
/// source:
/// `WorkspaceSnapshot::from_dir`, the git snapshot load/restore (`sandbox_manager`),
/// the change-detection digest (`orchestrator::workspace_digest`), and the solution
/// read (`Sandbox::read_tree`). Keeping ONE predicate stops the three copies from
/// drifting (they previously disagreed on `_trusted` and on binary extensions).
///
/// The rule is "track everything the agent authors; exclude only what isn't its
/// source." Three buckets, so an agent-created dir/file of ANY name carries
/// reliably across episodes and only genuinely-non-source paths are dropped:
/// 1. **Read-only input mounts** by top-level segment (`docs`, `_trusted`/`trusted`,
///    `skills`) — the agent physically can't write under a read-only bind, so
///    excluding these never loses agent data; they'd just bloat every snapshot.
/// 2. **Harness-owned scratch** by top-level segment (`.git`, `.kg`, and
///    `checkout` — the dir the `checkout` tool fills with a *copy of another
///    node's workspace*; tracking it would make snapshots recursively contain
///    other snapshots).
/// 3. **Runtime/compile caches** by ANY segment (they nest, e.g.
///    `.cache/torch_extensions/…`) and **derived binaries** by extension — never
///    source (they rebuild on eval); they were bloating the git object store.
#[must_use]
pub fn should_exclude_workspace_path(rel: &str) -> bool {
    let first = rel.split('/').next().unwrap_or("");
    // Read-only mounts + harness-owned scratch (not the agent's source).
    if NON_SOURCE_TOPLEVEL.contains(&first) {
        return true;
    }
    // Caches at any depth (top-level `__pycache__` included here).
    if rel.split('/').any(|c| CACHE_SEGMENTS.contains(&c)) {
        return true;
    }
    BINARY_EXT.iter().any(|ext| rel.ends_with(ext))
}

fn walk_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk_files(&path, out)?;
        } else if ft.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_exclusion_keeps_source_drops_derived() {
        // Agent-authored source is kept — including an arbitrary agent-created dir
        // (any name carries; the rule is a denylist, not an allowlist).
        for keep in [
            "solution.py",
            "solution/attn_kernel.cu",
            "notes/SOLVED_GOTCHAS.md",
            "artifacts/test_mma.cu",
            "DOC_INDEX.md",            // ".md" must NOT be caught by the ".d" rule
            "experiments/v3/probe.cu", // arbitrary agent dir → carries
            "artifacts/d5.nsys-rep",   // profiler outputs carry (bytes-exact restore)
            "artifacts/prof.ncu-rep",
            "artifacts/d5.sqlite",
        ] {
            assert!(!should_exclude_workspace_path(keep), "should keep {keep}");
        }
        // RO mounts + harness scratch + build dirs (top-level), caches (any segment), binaries (ext).
        for drop in [
            "docs/nvidia/ptx.md",
            "_trusted/evaluate.py",
            "skills/foo/skill.md", // RO mount — excluded for parity with docs/_trusted
            ".git/config",
            ".kg/meta.json",
            "checkout/solution/solution.py", // recovered-node scratch, never re-snapshotted
            "target/debug/incremental/x",    // build-output tree, non-binary content also dropped
            "artifacts/t.o",
            "kernel.cuda.o.d",
            "solution/ext.so",
            ".cache/torch_extensions/ext/ext.so",
            "foo/__pycache__/x.pyc",
        ] {
            assert!(should_exclude_workspace_path(drop), "should drop {drop}");
        }
    }

    #[test]
    fn checkout_dir_is_excluded_everywhere() {
        // The recovered-node scratch dir must never be snapshotted or keyed.
        assert!(should_exclude_workspace_path("checkout/solution/solution.py"));
        assert!(should_exclude_workspace_path("checkout/artifacts/tc.cu"));
    }

    #[test]
    fn from_dir_excludes_binaries_and_mounts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let write = |rel: &str, bytes: &[u8]| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, bytes).unwrap();
        };
        write("solution/solution.py", b"class Solution: pass\n");
        write("solution/attn_kernel.cu", b"// kernel\n");
        write("artifacts/test.cu", b"// source kept\n");
        write("artifacts/test.o", b"\x7fELF binary");
        write("artifacts/prof.nsys-rep", b"\x00\x01");
        write("docs/paper.md", b"# mounted, excluded\n");
        write(".cache/torch_extensions/ext/ext.so", b"\x7fELF");

        let snap = WorkspaceSnapshot::from_dir(root).unwrap();
        let keys: std::collections::BTreeSet<&str> = snap.files.keys().map(std::string::String::as_str).collect();
        assert!(keys.contains("solution/solution.py"));
        assert!(keys.contains("solution/attn_kernel.cu"));
        assert!(keys.contains("artifacts/test.cu"));
        assert!(
            !keys.contains("artifacts/test.o"),
            "build binary must not be snapshotted"
        );
        assert!(
            keys.contains("artifacts/prof.nsys-rep"),
            "profiler outputs carry across episodes (restore is bytes-exact)"
        );
        assert!(!keys.iter().any(|k| k.starts_with("docs/")), "mounts excluded");
        assert!(!keys.iter().any(|k| k.contains("torch_extensions")), "caches excluded");
    }
}
