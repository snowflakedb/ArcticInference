//! `bash` — run a shell command inside the sandboxed workspace.
//!
//! Thin wrapper over [`Sandbox::run_streaming`]: validates the timeout cap,
//! streams each output line to the terminal as it arrives (so the user can
//! see progress mid-call), and packages the merged stdout+stderr + exit
//! code into the format the agent expects.
//!
//! Mirrors the Python reference:
//! ```ignore
//! exit code: 0
//! stdout+stderr:
//! <body>
//! ```
//! Timeouts return `err: process timed out after Ns` instead of an exit
//! code, which lets the model recover gracefully (it's still `Ok(...)` —
//! the harness only marks `Err` as a tool error).

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::tool::command_guard::{ephemeral_tmp_write_warning, reject_vendor_source_spelunking};
use crate::tool::truncate::clamp_with_spill;
use crate::tool::{Tool, ToolOutput};
use crate::ui::{DIM, RESET};

/// Hard cap on per-call timeout, matching the Python reference. Long
/// builds should chunk into smaller invocations rather than block the
/// agent for >10 minutes per turn.
const MAX_TIMEOUT_SECS: u32 = 10 * 60;

#[derive(Deserialize, JsonSchema)]
pub struct BashArgs {
    /// The shell command to run, executed as `sh -c <cmd>` from the
    /// workspace root (e.g. `python main.py` or
    /// `nvcc kernel.cu -o run && ./run`). You already start in the
    /// workspace root, so use workspace-relative paths (`solution/solution.py`,
    /// `docs/foo.md`, `_trusted/run_kernel.py`) and avoid `cd`. If you need an
    /// absolute base, use the `$WORKSPACE` env var.
    pub cmd: String,
    /// Maximum seconds the command may run before the entire process tree
    /// is killed. 20s is a good default for fast iteration; the hard
    /// ceiling is 600 (10 minutes) — beyond that, split the work.
    pub timeout: u32,
}

pub struct Bash {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Bash {
    type Args = BashArgs;
    const NAME: &'static str = "bash";
    const DESCRIPTION: &'static str = "Run a shell command (`sh -c <cmd>`) from the workspace root — a LOCAL, CPU-only shell with \
         NO GPU. Instant and unqueued: use it freely to edit, build/compile (nvcc and torch-extension builds \
         run here), read, grep, and inspect. It CANNOT run anything on the device; to execute on the GPU use \
         `gpu_job` (your own profiling/benchmarks) or `evaluate` (the official score). Your cwd is already the \
         workspace, so use workspace-relative paths (`solution/solution.py`, `docs/foo.md`) and don't `cd`; \
         there is no fixed `/workspace` dir on every platform — use the `$WORKSPACE` env var for an absolute \
         base. Output is streamed; stdout and stderr are merged.";

    async fn call(&self, args: BashArgs) -> ToolOutput {
        if args.timeout > MAX_TIMEOUT_SECS {
            return Err(format!(
                "timeout has to be less than {} seconds (10 minutes); got {}",
                MAX_TIMEOUT_SECS, args.timeout
            ));
        }
        reject_vendor_source_spelunking(&args.cmd)?;

        let timeout = Duration::from_secs(u64::from(args.timeout));

        // bash is CPU-only (gpu=false below: no /dev/nvidia* is bound), so it takes
        // NO GPU lease — builds/edits run concurrently with a GPU measurement without
        // perturbing it. Matches this tool's own "instant and unqueued" contract.
        let result = self
            .sandbox
            .run_streaming(&["sh", "-c", &args.cmd], Some(timeout), false, None, |line| {
                // Stream each line to the terminal as it arrives. Matches
                // the Python reference's `│ <line>` look so transcripts
                // stay visually consistent across the two implementations.
                let mut stdout = io::stdout().lock();
                let _ = writeln!(stdout, "  {DIM}│ {line}{RESET}");
                let _ = stdout.flush();
            })
            .await
            .map_err(|e| format!("spawn error: {e}"))?;

        let raw = if result.stdout.is_empty() {
            "<no output>".to_string()
        } else {
            result.stdout
        };
        // Tail-clamp to 2000 lines / 50 KiB; if anything was dropped, the full
        // output is spilled to a temp file and cited by a suffix notice. `?`
        // only fires on a spill I/O failure (temp dir unwritable).
        let body = clamp_with_spill(&raw, "kg-bash")?;

        // Non-blocking nudge: persistent artifacts written to /tmp are lost on the
        // next sandbox reset (see `command_guard`); appended after the real output.
        let tmp_note = ephemeral_tmp_write_warning(&args.cmd)
            .map(|w| format!("\n\n{w}"))
            .unwrap_or_default();

        if result.timed_out {
            Ok(format!(
                "err: process timed out after {} seconds\nstdout+stderr:\n{body}{tmp_note}",
                args.timeout
            )
            .into())
        } else {
            Ok(format!("exit code: {}\nstdout+stderr:\n{body}{tmp_note}", result.returncode).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;

    fn make_bash() -> impl Tool {
        let sb = Arc::new(Sandbox::new().expect("sandbox"));
        Bash { sandbox: sb }
    }

    #[tokio::test]
    async fn bash_runs_simple_command() {
        let tool = make_bash();
        let result = tool
            .call_json(json!({ "cmd": "echo hello world", "timeout": 5 }))
            .await
            .unwrap()
            .as_text();
        assert!(result.contains("exit code: 0"), "{result}");
        assert!(result.contains("hello world"), "{result}");
    }

    #[tokio::test]
    async fn bash_reports_timeout_with_partial_output() {
        let tool = make_bash();
        // 3s rather than 1s. The `MARKER_PRE` assertion needs the child to spawn,
        // echo, and have that line read back off the pipe *before* the deadline
        // fires. At 1s that margin was thinner than process-spawn variance under a
        // loaded suite: this failed ~2 in 7 full-suite runs while passing 6/6 in
        // isolation. The sleep only has to outlast the deadline, so widening the
        // deadline costs a couple of seconds and buys determinism.
        let result = tool
            .call_json(json!({
                "cmd": "echo MARKER_PRE; sleep 30; echo MARKER_POST",
                "timeout": 3
            }))
            .await
            .unwrap()
            .as_text();
        assert!(result.contains("timed out after 3 seconds"), "{result}");
        assert!(
            result.contains("MARKER_PRE"),
            "should have streamed pre-sleep output: {result}"
        );
        assert!(
            !result.contains("MARKER_POST"),
            "post-sleep echo must not run after timeout: {result}"
        );
    }

    #[tokio::test]
    async fn bash_rejects_timeout_above_cap() {
        let tool = make_bash();
        let err = tool
            .call_json(json!({ "cmd": "true", "timeout": 1000 }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("less than 600 seconds"), "{err}");
    }

    #[tokio::test]
    async fn bash_rejects_vendor_source_tree_spelunking() {
        let tool = make_bash();
        let err = tool
            .call_json(json!({
                "cmd": "cat /opt/pytorch/pytorch/third_party/cutlass/include/cute/arch/example.hpp",
                "timeout": 5,
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vendor/framework source tree"), "{err}");
    }

    #[tokio::test]
    async fn bash_propagates_nonzero_exit() {
        let tool = make_bash();
        let result = tool
            .call_json(json!({ "cmd": "exit 7", "timeout": 5 }))
            .await
            .unwrap()
            .as_text();
        assert!(result.contains("exit code: 7"), "{result}");
    }

    /// End-to-end with `mount_docs`: bash sees `agent_ressources/docs/` as
    /// `docs/` in its cwd and can `cat` the upstream content.
    #[tokio::test]
    async fn bash_can_read_mounted_docs_via_relative_path() {
        let upstream = tempfile::TempDir::new().unwrap();
        std::fs::write(upstream.path().join("note.md"), "MARKER_LINE\nbody\n").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.mount_docs(upstream.path()).unwrap();
        let tool = Bash { sandbox: Arc::new(sb) };

        let result = tool
            .call_json(json!({ "cmd": "ls docs && head -1 docs/note.md", "timeout": 5 }))
            .await
            .unwrap()
            .as_text();
        assert!(result.contains("exit code: 0"), "{result}");
        assert!(result.contains("note.md"), "{result}");
        assert!(result.contains("MARKER_LINE"), "{result}");
    }
}
