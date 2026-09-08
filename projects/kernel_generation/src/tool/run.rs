//! `gpu_job` — agent-driven, UNTRUSTED execution for profiling and debugging.
//!
//! Runs a command on the GPU and BLOCKS until it finishes, returning its exit
//! code and (spilled) output inline — the same synchronous shape as `evaluate`
//! and `bash`. Like `evaluate`, it acquires a [`GpuPool`] device lease
//! before running so it never shares a card with a concurrent measurement. Unlike
//! `evaluate`, its output is UNTRUSTED and never influences the committed score.

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::queue::{GpuPool, warn_if_devices_still_held};
use crate::exec::sandbox::Sandbox;
use crate::tool::command_guard::{ephemeral_tmp_write_warning, reject_vendor_source_spelunking};
use crate::tool::truncate::{MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, clamp_with_spill, truncate_tail};
use crate::tool::{Tool, ToolOutput};
use crate::ui::{DIM, RESET};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 10 * 60;

#[derive(Deserialize, JsonSchema)]
pub struct RunArgs {
    /// Shell command to run via `sh -c` from the workspace root, e.g.
    /// `python3 _trusted/run_kernel.py 0` under a profiler, or an ad-hoc debug
    /// script. Acquires the shared execution queue first (may wait for the device).
    pub command: String,
    /// Max seconds before the whole process tree is killed. Defaults to 120;
    /// hard ceiling 600.
    #[serde(default)]
    pub timeout: Option<u64>,
}

pub struct GpuJob {
    pub sandbox: Arc<Sandbox>,
    pub queue: GpuPool,
}

impl Tool for GpuJob {
    type Args = RunArgs;
    const NAME: &'static str = "gpu_job";
    const DESCRIPTION: &'static str = "Run YOUR OWN command on the GPU (profiler like ncu/nsys, a debug script, a \
         micro-benchmark) and BLOCK until it finishes, returning its exit code and output inline. \
         The GPUs are a shared, leased resource (one job per card at a time), so this waits for a free \
         device before running; files the job writes \
         (e.g. `ncu -o prof.rep`) stay in the workspace so you can read them back with `bash`/`read`. \
         Output is UNTRUSTED and never affects your score; use `evaluate` for the official score. \
         `bash` is local and CPU-only — the GPU is reachable only via this tool and `evaluate`. \
         Typical use: profile `_trusted/run_kernel.py <config_index>` to find the bottleneck, \
         then edit the solution with `bash`/`write`.";

    async fn call(&self, args: RunArgs) -> ToolOutput {
        let timeout = args.timeout.unwrap_or(DEFAULT_TIMEOUT_SECS).min(MAX_TIMEOUT_SECS);
        reject_vendor_source_spelunking(&args.command)?;

        // Take a device lease so this untrusted job never shares a card with a
        // concurrent measurement; held across the subprocess, released on drop.
        let lease = self
            .queue
            .acquire_any()
            .await
            .map_err(|e| format!("infrastructure error: {e}"))?;
        let result = self
            .sandbox
            .run_streaming(
                &["sh", "-c", &args.command],
                Some(Duration::from_secs(timeout)),
                true,
                Some(lease.devices()),
                |line| {
                    // Stream each line to the terminal as it arrives so the user
                    // can watch the profiler/benchmark progress mid-call.
                    let mut stdout = io::stdout().lock();
                    let _ = writeln!(stdout, "  {DIM}\u{2502} {line}{RESET}");
                    let _ = stdout.flush();
                },
            )
            .await
            .map_err(|e| format!("infrastructure error: spawn failed: {e}"))?;
        // Checked while the lease is still held, so the slot is not handed to the next
        // caller before we know whether its devices came back. An agent benchmark that
        // deadlocks on the GPU is the likeliest source of a stuck device.
        if result.timed_out {
            warn_if_devices_still_held(lease.devices(), "gpu_job").await;
        }
        drop(lease);

        // Profiler logs are expensive to reproduce, so spill the full output and
        // let the path ride in the returned text. On a spill I/O failure (rare)
        // fall back to a bounded tail so the result never carries unbounded output.
        let output = clamp_with_spill(&result.stdout, "kg-gpu-job").unwrap_or_else(|e| {
            format!(
                "{}\n\n[full-output spill failed: {e}]",
                truncate_tail(&result.stdout, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES).content
            )
        });
        // Same /tmp-persistence nudge as `bash`: a job that saved a report/backup
        // to /tmp will lose it on the next sandbox reset.
        let output = match ephemeral_tmp_write_warning(&args.command) {
            Some(w) => format!("{output}\n\n{w}"),
            None => output,
        };

        if result.timed_out {
            Ok(format!("[gpu_job] timed out after {timeout}s\nstdout+stderr (tail):\n{output}").into())
        } else {
            Ok(format!(
                "[gpu_job] exit code: {}\nstdout+stderr (tail):\n{output}",
                result.returncode
            )
            .into())
        }
    }
}
