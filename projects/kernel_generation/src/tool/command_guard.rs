//! Lightweight policy checks for agent shell commands.

/// Reject commands that inspect vendor/framework implementation source trees
/// instead of the mounted task documentation.
///
/// Toolchains and profilers under
/// these prefixes remain usable through normal PATH lookup; the guard only
/// catches explicit source-tree spelunking in the command text.
///
/// # Errors
///
/// Returns `Err` with the offending prefix when the command text contains one of
/// the forbidden vendor source-tree paths (matched case-insensitively anywhere in
/// the command, so quoting or flag position cannot evade it).
pub fn reject_vendor_source_spelunking(command: &str) -> Result<(), String> {
    let lowered = command.to_ascii_lowercase();
    let forbidden = [
        "/opt/pytorch/pytorch/third_party/cutlass",
        "/opt/pytorch/third_party/cutlass",
        "/third_party/cutlass",
        "site-packages/cutlass",
        "python/site-packages/cutlass",
    ];

    if let Some(hit) = forbidden.iter().find(|needle| lowered.contains(**needle)) {
        return Err(format!(
            "command rejected: `{hit}` is a vendor/framework source tree, not the mounted docs. \
             Ground architecture decisions in `docs/` first; use vendor libraries only as black-box baselines, profiling targets, or disassembly outputs."
        ));
    }

    Ok(())
}

/// In-sandbox scratch dirs that are *meant* to be ephemeral: `TMPDIR` points
/// `/tmp` at a per-sandbox scratch directory holding torch/nvcc build caches.
/// Writing here is expected and correct, so these prefixes never trigger the
/// persistence warning below.
const EPHEMERAL_TMP_CACHE_PREFIXES: [&str; 3] = ["/tmp/.cache", "/tmp/torch_extensions", "/tmp/nv_compute_cache"];

/// Commands that write a file where they are pointed (copy/move/tee) or profilers
/// whose whole purpose is to emit a report. See step 2 of [`writes_persistent_tmp`].
const WRITE_CMDS: [&str; 8] = ["tee", "cp", "mv", "rsync", "install", "ncu", "nsys", "nvprof"];

/// Warn (do NOT block) when a shell command writes *persistent-looking* artifacts
/// under `/tmp`.
///
/// `/tmp` is a per-sandbox scratch dir that is NOT captured by the
/// workspace snapshot, so anything left there is silently lost when the next
/// episode re-materializes the workspace from a snapshot. This is the observed
/// "lost `/tmp` state" failure: kernel backups / `ncu` reports saved to `/tmp`
/// did not survive a sandbox reset and had to be hand-reconstructed from text.
///
/// Returns a note to append to the tool output when a write to `/tmp` is
/// detected. `/tmp` stays fully writable — genuinely throwaway scratch and build
/// caches are unaffected; only the mistaken persistence expectation is corrected.
#[must_use]
pub fn ephemeral_tmp_write_warning(command: &str) -> Option<String> {
    writes_persistent_tmp(command).then(|| {
        "note: this command writes under /tmp, which is per-sandbox scratch and is NOT saved with \
         the workspace — artifacts left there (kernel backups, ncu/nsys reports, notes) are lost \
         when the next episode re-materializes the workspace. Write anything you need to keep under \
         the workspace instead (e.g. solution/, or a workspace-relative scratch dir like artifacts/)."
            .to_string()
    })
}

/// The `/tmp` path a token names, with surrounding quotes stripped; `None` if
/// the token is not a `/tmp` path. `/tmpfoo` is not a match (must be exactly the
/// root or have a `/` after it).
fn tmp_path_token(tok: &str) -> Option<&str> {
    let t = tok.trim_matches(|c| c == '"' || c == '\'');
    for root in ["/tmp"] {
        if t == root || t.strip_prefix(root).is_some_and(|rest| rest.starts_with('/')) {
            return Some(t);
        }
    }
    None
}

/// Whether a `/tmp` path is one of the known ephemeral build-cache dirs (writing
/// there is expected, so it must not trigger the warning).
fn is_ephemeral_cache(path: &str) -> bool {
    EPHEMERAL_TMP_CACHE_PREFIXES
        .iter()
        .any(|p| path == *p || path.strip_prefix(p).is_some_and(|rest| rest.starts_with('/')))
}

/// A token that names a persistent (non-cache) `/tmp` path.
fn persistent_tmp_token(tok: &str) -> bool {
    tmp_path_token(tok).is_some_and(|p| !is_ephemeral_cache(p))
}

fn writes_persistent_tmp(command: &str) -> bool {
    // 1. Output redirection into /tmp: `> /tmp/x`, `>>/tmp/x`, `2>/tmp/x`, ...
    if redirects_into_persistent_tmp(command) {
        return true;
    }

    // Token view for command/flag heuristics. Split on whitespace AND the shell
    // separators that bound words, plus `=` so `--export=/tmp/x` splits cleanly.
    let toks: Vec<&str> = command
        .split(|c: char| c.is_whitespace() || matches!(c, ';' | '|' | '&' | '(' | ')' | '='))
        .filter(|s| !s.is_empty())
        .collect();

    // No persistent /tmp path mentioned at all ⇒ nothing to warn about. (Pure
    // reads like `cat /tmp/x` fall through the checks below and never warn.)
    if !toks.iter().any(|t| persistent_tmp_token(t)) {
        return false;
    }

    // 2. A write-capable command (copy/move/tee) or a profiler — whose whole
    //    purpose is to emit a report you want to keep — touches a /tmp path.
    if toks.iter().any(|t| WRITE_CMDS.contains(t)) {
        return true;
    }

    // 3. Working directory change into /tmp (agent treating it as a scratch home).
    toks.windows(2)
        .any(|w| matches!(w, ["cd" | "pushd", target] if persistent_tmp_token(target)))
}

/// Scan for a redirection (`>`/`>>`, any leading fd) whose target is a persistent
/// `/tmp` path. Char-based so glued (`>/tmp/x`) and spaced (`> /tmp/x`) forms are
/// handled uniformly; fd-dup redirections (`2>&1`, `>&2`) are skipped.
fn redirects_into_persistent_tmp(command: &str) -> bool {
    let b = command.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b.get(i) != Some(&b'>') {
            i = i.saturating_add(1);
            continue;
        }
        // Consume consecutive '>' (`>>`).
        let mut k = i.saturating_add(1);
        while b.get(k) == Some(&b'>') {
            k = k.saturating_add(1);
        }
        // Skip whitespace between the operator and its target.
        while matches!(b.get(k), Some(&(b' ' | b'\t'))) {
            k = k.saturating_add(1);
        }
        // `>&...` / `>&2` is an fd dup, not a file target.
        if b.get(k) == Some(&b'&') {
            i = k.saturating_add(1);
            continue;
        }
        // Read the target token up to the next word boundary. `is_word_boundary`
        // only matches ASCII, so this scan runs *through* whole multi-byte
        // sequences: both `start` and `k` land on an ASCII byte or on `b.len()`,
        // i.e. always on a char boundary, so `get` here always yields `Some`.
        let start = k;
        while b.get(k).is_some_and(|&c| !is_word_boundary(c)) {
            k = k.saturating_add(1);
        }
        if start < k && command.get(start..k).is_some_and(persistent_tmp_token) {
            return true;
        }
        i = k.max(i.saturating_add(1));
    }
    false
}

const fn is_word_boundary(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t' | b'\n' | b'\r' | b';' | b'|' | b'&' | b'<' | b'>' | b'(' | b')'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cutlass_source_tree_reads() {
        let err = reject_vendor_source_spelunking(
            "sed -n '1,80p' /opt/pytorch/pytorch/third_party/cutlass/include/cute/arch/foo.hpp",
        )
        .unwrap_err();
        assert!(err.contains("vendor/framework source tree"), "{err}");
    }

    #[test]
    fn allows_docs_and_cuda_tools() {
        reject_vendor_source_spelunking("grep -R architecture docs/nvidia && nvdisasm solution/kernel.cubin").unwrap();
    }

    #[test]
    fn warns_on_persistent_tmp_writes() {
        // Redirection (spaced + glued + fd-prefixed), tee, cp/mv, profiler output,
        // and cd-into-/tmp are all persistence hazards.
        for cmd in [
            "cat solution/kernel.cu > /tmp/backup.cu",
            "echo notes >>/tmp/notes.md",
            "ncu --set full -o /tmp/report.ncu-rep python _trusted/run_kernel.py 0",
            "nsys profile --export=/tmp/prof python x.py",
            "cp solution/kernel.cu /tmp/kernel_backup.cu",
            "mv results.json /tmp/keep.json",
            "python bench.py 2>&1 | tee /tmp/out.txt",
            "cd /tmp && ls",
            "python x.py 2> /tmp/err.log",
        ] {
            assert!(
                ephemeral_tmp_write_warning(cmd).is_some(),
                "expected a /tmp persistence warning for: {cmd}"
            );
        }
    }

    #[test]
    fn does_not_warn_on_reads_or_ephemeral_cache_or_workspace() {
        // Pure reads of /tmp, removals, ephemeral build-cache writes, fd dups, and
        // workspace-only writes must stay silent (no false-positive nagging).
        for cmd in [
            "cat /tmp/scratch.txt",
            "ls -la /tmp",
            "rm -f /tmp/old.cu",
            "grep -R foo /tmp",
            "python train.py 2>&1 | tee run.log",
            "cat solution/kernel.cu > artifacts/backup.cu",
            "echo hi > /tmp/torch_extensions/note",
            "nvcc solution/kernel.cu -o /tmp/run && /tmp/run",
        ] {
            assert!(ephemeral_tmp_write_warning(cmd).is_none(), "must not warn for: {cmd}");
        }
    }
}
