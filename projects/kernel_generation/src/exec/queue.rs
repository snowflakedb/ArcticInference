//! GPU execution pool.
//!
//! The machine has one or more GPUs (probed via [`crate::env::hardware::gpu_count`]).
//! Every sandbox subprocess that touches a device — `evaluate`, `gpu_job` — must
//! hold *its* device exclusively while it runs, so a timing measurement is never
//! perturbed by a co-tenant and two jobs never OOM each other on the same card.
//! [`GpuPool::acquire_any`] waits for the first free device and returns a lease
//! pinned to it.
//!
//! B200s are effectively clock-locked, so any device is as good as any other for
//! timing; the pool is **symmetric** and the only invariant is *at most one
//! holder per device at a time*. CPU-only work (`bash`, file ops, lineage tools)
//! takes no lease — it touches no device and must never block behind a benchmark.
//!
//! With `p>1` agent episodes overlapping their model-latency time, this pool is
//! what lets their GPU evals actually run concurrently across the cards: a size-N
//! pool runs up to N measured subprocesses at once, one per device. A size-1 pool
//! degenerates to the old whole-machine serial lease, so the single-GPU path is
//! unchanged.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::future::select_all;
use tokio::sync::{Mutex, OwnedMutexGuard};

/// A cheaply-cloneable handle to the GPU pool. Clones share the same underlying
/// per-device locks, so any number of tool instances can hold a handle while
/// still coordinating exclusive device access.
#[derive(Clone)]
pub struct GpuPool {
    /// One lock per *slot*, where a slot is the fixed group of devices one job gets.
    ///
    /// Every job in a run needs the same number of devices -- a run evaluates one problem
    /// and its `NUM_GPUS` is fixed -- so the grouping is decided once at construction. That
    /// makes acquiring a single lock, which is why there is no multi-device acquire here to
    /// deadlock: two waiters can never each hold half of what the other needs.
    slots: Vec<Arc<Mutex<()>>>,
    /// `groups[i]` is the device indices slot `i` owns. Parallel to `slots`.
    groups: Vec<Vec<usize>>,
    /// How long a caller waits to acquire *a* device before giving up. Distinct
    /// from the per-command *execution* timeout, which the sandbox enforces once
    /// the command is actually running.
    acquire_timeout: Duration,
    /// Cumulative milliseconds a device lease has been *held* (a card busy with a
    /// measured subprocess), summed across all devices. Wait time before
    /// acquisition is excluded. With N>1 devices this can exceed wall-clock — it
    /// is total device-seconds, the `queue_busy_secs` the manifest reports.
    busy_ms: Arc<AtomicU64>,
}

/// Report, loudly, if a timed-out command left a CUDA context on its leased devices.
///
/// The timeout is expected to release the cards. When it does not, the slot is
/// poisoned rather than merely slow: a process still holding SMs makes every later
/// lease on those devices measure against a busy GPU, or block on a sync that never
/// completes — one such leftover silently invalidated hours of a run. Operator-facing
/// because nothing the agent does can fix it.
///
/// Detection only. The cause it was written for is fixed at the source (the sandbox
/// timeout sweeps the whole mount namespace, not just the process group), so this is the
/// check that the sweep worked, and cover for a cause we have not seen.
///
/// **Polls before reporting.** `SIGKILL` is asynchronous and the driver takes time to
/// tear a CUDA context down, so a process that is already dying still appears to hold
/// its device for a short while. Reporting on the first look calls every healthy
/// timeout a compromised device — measured: 8 alarms in a run that then went on to
/// score its best result and ended with the GPUs clear. A warning that is wrong is
/// worse than none, because it trains the reader to skip it.
pub async fn warn_if_devices_still_held(devices: &[usize], what: &str) {
    let mut held = Vec::new();
    for attempt in 0..DEVICE_RELEASE_ATTEMPTS {
        held = crate::env::hardware::pids_holding_devices(devices);
        if held.is_empty() {
            return;
        }
        // Not after the last look: sleeping then would only delay the report.
        if attempt < DEVICE_RELEASE_ATTEMPTS.saturating_sub(1) {
            tokio::time::sleep(DEVICE_RELEASE_POLL).await;
        }
    }
    eprintln!(
        "kernelguy: WARNING — {what} timed out on device(s) {devices:?} and pid(s) {held:?} still \
         held a CUDA context there {DEVICE_RELEASE_ATTEMPTS} checks later. Those devices are \
         compromised for the rest of the run: kill the pid(s) or restart, or every later \
         evaluation leased onto them is untrustworthy."
    );
}

/// How long to let a killed process release its device before calling it stuck.
///
/// Generous against observed teardown (well under a second) and still short enough that
/// a genuinely stuck device is reported while the timeout that caused it is on screen.
const DEVICE_RELEASE_POLL: std::time::Duration = std::time::Duration::from_millis(400);
const DEVICE_RELEASE_ATTEMPTS: u32 = 5;

/// Held for the duration of a sandbox subprocess.
///
/// Dropping it releases the
/// slot for the next waiter and records how long it was held. Carries the
/// [`devices`](ExecLease::devices) the caller must confine the subprocess to.
pub struct ExecLease {
    _guard: OwnedMutexGuard<()>,
    devices: Vec<usize>,
    started: Instant,
    busy_ms: Arc<AtomicU64>,
}

impl ExecLease {
    /// The device indices this lease reserves, ascending. Pass them to the sandbox,
    /// which binds only those cards' device nodes so the subprocess cannot open any
    /// other -- a multi-rank job then addresses them as `0..n-1` whichever slot it got,
    /// because the cards it can see renumber from 0.
    ///
    /// Deliberately not `CUDA_VISIBLE_DEVICES`: that is the operator's inbound knob and
    /// an env var inside a process the solution controls, so it cannot bound anything.
    #[must_use]
    pub fn devices(&self) -> &[usize] {
        &self.devices
    }
}

impl Drop for ExecLease {
    fn drop(&mut self) {
        // `u64::MAX` ms is ~5.8e8 years, so the saturation arm is unreachable; it
        // replaces an `as u64` that would have silently wrapped.
        let held = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.busy_ms.fetch_add(held, Ordering::Relaxed);
    }
}

/// Why acquiring a device failed.
#[derive(Debug)]
pub enum ExecQueueError {
    /// Waited longer than `acquire_timeout` for any device (the whole pool is
    /// saturated with other work). Distinct from a command timing out while it
    /// runs.
    AcquireTimeout(Duration),
}

impl std::fmt::Display for ExecQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AcquireTimeout(d) => {
                write!(f, "timed out after {}s waiting for a free GPU", d.as_secs())
            }
        }
    }
}

impl std::error::Error for ExecQueueError {}

impl GpuPool {
    /// Build a pool over `devices` GPUs handing out `per_job` of them at a time.
    ///
    /// `per_job` is the problem's `NUM_GPUS`, queried once before the run starts
    /// (`evaluate.py --describe`), because it is the same for every job in the run. The pool
    /// holds `devices / per_job` slots, so concurrency is that quotient: four cards run four
    /// single-GPU jobs at once, or exactly one TP4 job.
    ///
    /// A remainder is never leased. With 4 devices and `per_job = 3` there is one slot and
    /// device 3 idles for the whole run -- deliberate, because a job needing 3 cannot use it,
    /// and handing out a short group would hang the job's collectives instead of failing.
    #[must_use]
    pub fn new(devices: usize, per_job: usize, acquire_timeout: Duration) -> Self {
        let total = devices.max(1);
        let per = per_job.clamp(1, total);
        // Checked throughout: `arithmetic_side_effects` and `indexing_slicing` are denied,
        // and a silent wrap here would hand out an empty or overlapping device group.
        let n_slots = total.checked_div(per).unwrap_or(1).max(1);
        let groups: Vec<Vec<usize>> = (0..n_slots)
            .map(|s| {
                let lo = s.saturating_mul(per);
                (lo..lo.saturating_add(per)).collect()
            })
            .collect();
        Self {
            slots: (0..groups.len()).map(|_| Arc::new(Mutex::new(()))).collect(),
            groups,
            acquire_timeout,
            busy_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    /// How many jobs can hold a lease at once -- `devices / per_job`.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.groups.len()
    }

    /// Acquire the first free slot, waiting up to `acquire_timeout`. Hold the
    /// returned [`ExecLease`] across the subprocess `await`; drop it to release.
    /// The hold duration is added to [`busy_secs`](Self::busy_secs).
    ///
    /// # Errors
    ///
    /// [`ExecQueueError::AcquireTimeout`] if no slot became free within
    /// `acquire_timeout` — i.e. every card is busy with other measured work.
    pub async fn acquire_any(&self) -> Result<ExecLease, ExecQueueError> {
        // Race one `lock_owned` future per device; the first to complete wins and
        // the rest are dropped (they never acquired). `select_all` needs a
        // non-empty vec, which `new` guarantees.
        let locks = self
            .slots
            .iter()
            .map(|d| Box::pin(d.clone().lock_owned()))
            .collect::<Vec<_>>();
        match tokio::time::timeout(self.acquire_timeout, select_all(locks)).await {
            Ok((guard, idx, _rest)) => Ok(self.lease(guard, idx)),
            Err(_) => Err(ExecQueueError::AcquireTimeout(self.acquire_timeout)),
        }
    }

    fn lease(&self, guard: OwnedMutexGuard<()>, slot: usize) -> ExecLease {
        ExecLease {
            _guard: guard,
            // `slot` comes from `select_all` over `self.slots`, which is parallel to
            // `groups`, so this is always present; defaulting keeps `indexing_slicing` happy
            // without an unwrap that could ever fire.
            devices: self.groups.get(slot).cloned().unwrap_or_default(),
            started: Instant::now(),
            busy_ms: self.busy_ms.clone(),
        }
    }

    /// Total device-seconds held across the whole run — the sum over devices of
    /// the time each spent running a serialized (measured) subprocess. With N>1
    /// devices this can exceed wall-clock.
    #[must_use]
    pub fn busy_secs(&self) -> f64 {
        crate::domain::convert::u64_to_f64_lossy(self.busy_ms.load(Ordering::Relaxed)) / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[tokio::test]
    async fn one_holder_per_device_but_n_concurrent() {
        // A 2-device pool must allow exactly 2 concurrent holders, never 3.
        let q = GpuPool::new(2, 1, Duration::from_secs(5));
        let in_critical = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let q = q.clone();
            let in_critical = in_critical.clone();
            let max_seen = max_seen.clone();
            handles.push(tokio::spawn(async move {
                let _lease = q.acquire_any().await.expect("acquire");
                let now = in_critical.fetch_add(1, Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                in_critical.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            2,
            "a 2-device pool must run at most 2 measured subprocesses at once"
        );
    }

    #[tokio::test]
    async fn acquire_any_returns_distinct_live_devices() {
        let q = GpuPool::new(2, 1, Duration::from_secs(5));
        let a = q.acquire_any().await.expect("first");
        let b = q.acquire_any().await.expect("second");
        // Both devices held ⇒ they must be the two distinct indices {0,1}.
        assert_ne!(a.devices(), b.devices(), "two live leases must pin different devices");
        assert!(a.devices().iter().all(|&d| d < 2) && b.devices().len() == 1);
        // Explicit drops at the (already-implicit) end of the leases' lifetime: both
        // must stay live *across* the assertions above, which is the point of the
        // test, so they cannot be released any earlier than this.
        drop(a);
        drop(b);
    }

    #[test]
    fn slots_group_devices_by_job_size() {
        // Four cards, one GPU per job: four independent slots.
        let single = GpuPool::new(4, 1, Duration::from_secs(1));
        assert_eq!(single.slot_count(), 4);

        // Four cards, four per job: exactly one slot owning every device, so a TP4 job gets
        // all of them and nothing else runs beside it.
        let tp4 = GpuPool::new(4, 4, Duration::from_secs(1));
        assert_eq!(tp4.slot_count(), 1);
        assert_eq!(tp4.groups.first().map(Vec::as_slice), Some(&[0, 1, 2, 3][..]));

        // Two per job over four: two slots, disjoint and contiguous.
        let tp2 = GpuPool::new(4, 2, Duration::from_secs(1));
        assert_eq!(tp2.slot_count(), 2);
        assert_eq!(tp2.groups.first().map(Vec::as_slice), Some(&[0, 1][..]));
        assert_eq!(tp2.groups.get(1).map(Vec::as_slice), Some(&[2, 3][..]));

        // A remainder is dropped rather than handed out short: 3 per job over 4 leaves one
        // slot and idles device 3, because a short group would hang the job's collectives.
        let tp3 = GpuPool::new(4, 3, Duration::from_secs(1));
        assert_eq!(tp3.slot_count(), 1);
        assert_eq!(tp3.groups.first().map(Vec::as_slice), Some(&[0, 1, 2][..]));

        // Asking for more than exists is clamped to everything, never zero slots.
        let over = GpuPool::new(2, 8, Duration::from_secs(1));
        assert_eq!(over.slot_count(), 1);
        assert_eq!(over.groups.first().map(Vec::as_slice), Some(&[0, 1][..]));
    }

    #[tokio::test]
    async fn busy_secs_accrues_only_while_held() {
        let q = GpuPool::new(2, 1, Duration::from_secs(5));
        // `busy_secs` is exactly 0.0 before any lease is taken (0 ms / 1000.0);
        // spelled as an epsilon compare because `==` on floats is denied.
        assert!(q.busy_secs().abs() < f64::EPSILON, "a fresh pool has no busy time");
        {
            let _lease = q.acquire_any().await.expect("acquire");
            tokio::time::sleep(Duration::from_millis(30)).await;
        } // drop records the hold
        assert!(
            q.busy_secs() >= 0.02,
            "expected >=20ms of busy time, got {}s",
            q.busy_secs()
        );
    }
}
