//! One-shot hardware / GPU identity probe.
//!
//! At run start the harness shells out to `nvidia-smi` and parses a short,
//! **normalized** `key: value` block describing the *actual* machine. That
//! block is injected into the agent's kickoff and into the supervisor's review
//! context so both reason about the real GPU (part, core count, memory) instead
//! of a hardcoded assumption.
//!
//! It runs in the harness process (outside the agent's `bwrap` jail), so it has
//! the access it needs. Strictly best-effort: any failure yields a short
//! fallback note rather than aborting the run.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Probe the host GPU once and return a compact, normalized `key: value`
/// identity block. Never panics; returns a fallback note if no probe succeeds.
#[must_use]
pub fn probe_hardware() -> String {
    // CUDA boxes (Linux): `nvidia-smi`. Tried first because it cleanly fails
    // (missing binary) off-NVIDIA. The queried fields are the ones that change
    // the kernel — part, compute capability (arch), VRAM, driver.
    let base = [
        "--query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.memory",
        "--format=csv,noheader",
    ];
    // Describe the GPU the run will actually use rather than whichever card
    // `nvidia-smi` lists first. `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so
    // on a mixed-GPU host the first row can be a different part with a
    // different arch and VRAM than the device the agent times against — and
    // this block is what the kickoff and supervisor reason about.
    let selected = visible_device_uuids().first().cloned();
    let csv = selected
        .as_deref()
        .and_then(|uuid| {
            let mut query = base.to_vec();
            query.extend(["-i", uuid]);
            run_capture("nvidia-smi", &query)
        })
        // A mask naming a device `nvidia-smi -i` cannot select must not sink the
        // whole probe: that would drop the `backend: cuda` line, and
        // `should_validate_nvidia_tools` keys off it — so a typo'd mask would
        // silently skip NVIDIA startup validation instead of failing loudly.
        .or_else(|| run_capture("nvidia-smi", &base));
    if let Some(csv) = csv {
        let mut block = parse_nvidia(&csv);
        if let Some(os) = linux_os() {
            let _ = write!(block, "\nos: {os}");
        }
        return block;
    }
    "(GPU identity probe unavailable — `nvidia-smi` did not succeed)".to_string()
}

/// Size of the [`crate::exec::queue`] device pool.
///
/// Number of GPUs the harness may use — the size of the [`crate::exec::queue`]
/// device pool. This is the count of devices left after applying
/// `CUDA_VISIBLE_DEVICES` (see [`visible_device_uuids`]), not the physical device
/// count: a pool sized to the physical count would hand out leases for devices
/// the CUDA runtime cannot see, and every eval on such a lease fails.
///
/// Returns `1` when the probe is unavailable or selects nothing (no NVIDIA
/// driver, a parse miss, or an empty mask), so the pool always has at least one
/// device and the single-GPU path behaves exactly as before.
#[must_use]
pub fn gpu_count() -> usize {
    visible_device_uuids().len().max(1)
}

/// UUIDs of the GPUs the harness may use, in CUDA's post-mask order — so slot
/// `i` of the device pool is `visible_device_uuids()[i]`.
///
/// Probed once and cached; device topology does not change mid-run, and this is
/// consulted per sandboxed command.
///
/// Why UUIDs rather than indices: `nvidia-smi` enumerates in PCI-bus order while
/// the CUDA runtime defaults to fastest-first, so the two disagree about which
/// card is "device 0". A UUID names one physical device unambiguously under
/// either ordering, and a single-UUID `CUDA_VISIBLE_DEVICES` composes with an
/// outer mask instead of reinterpreting an index against the full device list.
pub fn visible_device_uuids() -> &'static [String] {
    static CACHE: OnceLock<Vec<String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let listing = run_capture("nvidia-smi", &["-L"]).unwrap_or_default();
        let mask = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        resolve_visible_devices(&parse_gpu_listing(&listing), mask.as_deref())
    })
}

/// Device minor numbers of the GPUs the harness may use, parallel to
/// [`visible_device_uuids`]: `visible_device_minors()[i]` is the minor of the
/// device pool slot `i` names, i.e. it owns `/dev/nvidia<minor>`.
///
/// This is what lets the sandbox confine a job by *binding only its device
/// nodes* rather than by setting `CUDA_VISIBLE_DEVICES`, which a solution can
/// simply overwrite. `None` for a device whose minor cannot be resolved; the
/// caller must then fall back rather than bind nothing.
pub fn visible_device_minors() -> &'static [Option<u32>] {
    static CACHE: OnceLock<Vec<Option<u32>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let map = read_uuid_minors();
        visible_device_uuids()
            .iter()
            .map(|uuid| map.iter().find(|(u, _)| u == uuid).map(|&(_, m)| m))
            .collect()
    })
}

/// The `/dev` nodes a sandbox needs in order to drive GPUs, split by whether
/// they belong to one card or to all of them.
///
/// This is the whole NVIDIA-facing input the sandbox layer takes: it binds these
/// paths and knows nothing else about the driver.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GpuDeviceNodes {
    /// Bound on every GPU-enabled run. Driver control and fabric interfaces —
    /// every CUDA process needs them whichever card it drives, so withholding
    /// one fails `cuInit` or breaks `NVLink` rather than restricting access.
    pub shared: Vec<PathBuf>,
    /// `per_device[i]` is the node owned by device pool slot `i`, parallel to
    /// [`visible_device_uuids`], so a lease on index `i` binds `per_device[i]`.
    ///
    /// **Empty means "no per-device confinement available"**, in which case the
    /// numbered nodes appear in `shared` instead and every run sees every card.
    /// It is all-or-nothing on purpose: dropping just the unresolvable entries
    /// would shift the indices and silently bind the wrong card.
    pub per_device: Vec<PathBuf>,
}

/// The device nodes this host exposes, resolved once.
///
/// Cached like its neighbours — the device topology does not change mid-run, and
/// this is consulted per sandbox creation.
pub fn gpu_device_nodes() -> &'static GpuDeviceNodes {
    static CACHE: OnceLock<GpuDeviceNodes> = OnceLock::new();
    CACHE.get_or_init(|| {
        let names: Vec<String> = std::fs::read_dir("/dev")
            .map(|read| {
                read.flatten()
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| n.starts_with("nvidia"))
                    .collect()
            })
            .unwrap_or_default();
        classify(&names, visible_device_minors(), |p| Path::new(p).exists())
    })
}

/// Split discovered `/dev` entry names into shared and per-device nodes.
///
/// Pure, so the rules below are testable without a GPU. `exists` decides whether
/// a mapped node is really present — [`read_uuid_minors`] enumerates every
/// *physical* GPU including cards absent from `/dev`, so the mapping can name a
/// node that does not exist, and emitting it would make `bwrap` fail the whole
/// command rather than degrade.
///
/// Two rules, both load-bearing:
///
/// 1. **Per-device nodes come from `minors`, never from the `/dev` listing.**
///    `minors` already has the operator's `CUDA_VISIBLE_DEVICES` mask applied;
///    trusting `/dev` instead would hand a sandbox cards the operator excluded.
/// 2. **Binding is the default.** A node is per-device only when its suffix is a
///    bare integer. Every other name — known control interfaces today, whatever
///    NVIDIA ships tomorrow — lands in `shared`, because withholding an
///    unrecognised interface breaks the cards the job legitimately owns.
fn classify(names: &[String], minors: &[Option<u32>], exists: impl Fn(&str) -> bool) -> GpuDeviceNodes {
    let mut shared: Vec<PathBuf> = names
        .iter()
        .filter(|n| n.strip_prefix("nvidia").is_none_or(|s| s.parse::<u32>().is_err()))
        .map(|n| PathBuf::from(format!("/dev/{n}")))
        .collect();
    shared.sort();
    shared.dedup();

    let per_device: Option<Vec<PathBuf>> = minors
        .iter()
        .map(|m| {
            let path = format!("/dev/nvidia{}", (*m)?);
            exists(&path).then(|| PathBuf::from(path))
        })
        .collect();

    match per_device {
        Some(per_device) if !per_device.is_empty() => GpuDeviceNodes { shared, per_device },
        // No usable mapping: fall back to every numbered node this host has, so a
        // job still gets its GPUs. Confinement is lost, not silently partial.
        _ => {
            let mut numbered: Vec<PathBuf> = names
                .iter()
                .filter(|n| n.strip_prefix("nvidia").is_some_and(|s| s.parse::<u32>().is_ok()))
                .map(|n| PathBuf::from(format!("/dev/{n}")))
                .collect();
            numbered.append(&mut shared);
            numbered.sort();
            numbered.dedup();
            GpuDeviceNodes {
                shared: numbered,
                per_device: Vec::new(),
            }
        }
    }
}

/// Host pids still holding a CUDA context on the pool devices `devices` names.
///
/// Called after a sandboxed GPU command times out, to confirm the kill actually
/// released the cards. Anything left is a process the timeout could not reach, and it
/// poisons the slot: it holds SMs, so every later lease on those devices measures
/// against a busy GPU or blocks on a sync that never completes. Silence is the
/// expected result — a non-empty answer means the sweep missed something.
///
/// Matched by device UUID rather than `nvidia-smi`'s index, for the same reason
/// [`visible_device_uuids`] is: the two orderings need not agree. Best-effort — an
/// unavailable or unparsable `nvidia-smi` yields nothing rather than a false alarm.
#[must_use]
pub fn pids_holding_devices(devices: &[usize]) -> Vec<u32> {
    let uuids = visible_device_uuids();
    let wanted: Vec<&str> = devices
        .iter()
        .filter_map(|&i| uuids.get(i).map(String::as_str))
        .collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    let Some(listing) = run_capture(
        "nvidia-smi",
        &["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
    ) else {
        return Vec::new();
    };
    listing
        .lines()
        .filter_map(|line| {
            let (uuid, pid) = line.split_once(',')?;
            let uuid = uuid.trim();
            wanted.contains(&uuid).then(|| pid.trim().parse().ok())?
        })
        .collect()
}

/// `(uuid, minor)` for every GPU the driver knows, from
/// `/proc/driver/nvidia/gpus/*/information`.
///
/// The authoritative uuid -> `/dev/nvidia<minor>` mapping. Deliberately NOT
/// `nvidia-smi`'s index: that is PCI-bus order, and while it coincides with the
/// minor on a typical host, nothing guarantees it — which is the same reason
/// [`visible_device_uuids`] keys on UUIDs.
///
/// Note this lists every *physical* GPU, including cards absent from `/dev`
/// (a container given a subset of the host's nodes still sees them all here),
/// so it is a mapping table only, never a device count.
fn read_uuid_minors() -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let Ok(dirs) = std::fs::read_dir("/proc/driver/nvidia/gpus") else {
        return out;
    };
    for dir in dirs.flatten() {
        let Ok(text) = std::fs::read_to_string(dir.path().join("information")) else {
            continue;
        };
        let mut uuid = None;
        let mut minor = None;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("GPU UUID:") {
                uuid = Some(v.trim().to_string());
            } else if let Some(v) = line.strip_prefix("Device Minor:") {
                minor = v.trim().parse().ok();
            }
        }
        if let (Some(u), Some(m)) = (uuid, minor) {
            out.push((u, m));
        }
    }
    out
}

/// One physical device as listed by `nvidia-smi -L`.
#[derive(Debug, PartialEq, Eq)]
struct GpuEntry {
    index: usize,
    uuid: String,
}

/// Parse `nvidia-smi -L` lines of the form
/// `GPU <index>: <name> (UUID: <uuid>)`. Lines without both an index and a UUID
/// are skipped, so MIG sub-device lines never inflate the device list.
fn parse_gpu_listing(listing: &str) -> Vec<GpuEntry> {
    listing
        .lines()
        .map(str::trim_start)
        .filter_map(|line| {
            let rest = line.strip_prefix("GPU ")?;
            let (index, rest) = rest.split_once(':')?;
            let index = index.trim().parse().ok()?;
            let uuid = rest.split_once("(UUID:")?.1.split_once(')')?.0.trim().to_string();
            (!uuid.is_empty()).then_some(GpuEntry { index, uuid })
        })
        .collect()
}

/// Apply a `CUDA_VISIBLE_DEVICES` value to a physical device listing, yielding
/// the selected devices' UUIDs in mask order.
///
/// An **empty** mask selects nothing, matching CUDA. That is indistinguishable here
/// from a host with no driver, so a caller that treats "no visible devices" as
/// permissive must check whether the variable was set — see
/// `SandboxManager::spawn`, which still forwards an empty mask as a ceiling.
///
/// Mirrors the CUDA runtime's own rules: entries are comma-separated and may be
/// either indices or (possibly abbreviated) `GPU-`/`MIG-` UUIDs; enumeration
/// stops at the first entry that names no device, so a bad entry hides
/// everything after it. An unset mask selects every device; an empty mask
/// selects none.
fn resolve_visible_devices(devices: &[GpuEntry], mask: Option<&str>) -> Vec<String> {
    let Some(mask) = mask else {
        return devices.iter().map(|d| d.uuid.clone()).collect();
    };
    let mut out = Vec::new();
    for token in mask.split(',').map(str::trim) {
        let found = token.parse::<usize>().map_or_else(
            // CUDA accepts a UUID prefix, so match on that rather than equality.
            |_| {
                (!token.is_empty())
                    .then(|| devices.iter().find(|d| d.uuid.starts_with(token)))
                    .flatten()
            },
            |i| devices.iter().find(|d| d.index == i),
        );
        match found {
            Some(d) => out.push(d.uuid.clone()),
            None => break,
        }
    }
    out
}

/// Run `cmd args...`, returning trimmed stdout on a clean, non-empty exit, else
/// `None` (missing binary, non-zero exit, or empty output).
fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Parse `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,
/// clocks.max.sm,clocks.max.memory --format=csv,noheader` (one CSV row per GPU)
/// into a normalized block. Unsupported/empty columns are dropped.
fn parse_nvidia(csv: &str) -> String {
    let rows: Vec<&str> = csv.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let cols: Vec<&str> = rows.first().copied().unwrap_or("").split(',').map(str::trim).collect();
    let mut out = vec!["backend: cuda (NVIDIA)".to_string()];
    push_nvidia_field(&mut out, &cols, "gpu_model", 0);
    push_nvidia_field(&mut out, &cols, "compute_capability", 1);
    if let Some(arch) = cols
        .get(1)
        .and_then(|cc| cuda_arch_from_compute_cap(cc, cols.first().copied()))
    {
        out.push(format!("cuda_arch: {arch}"));
    }
    push_nvidia_field(&mut out, &cols, "vram", 2);
    push_nvidia_field(&mut out, &cols, "gpu_clock_max", 4);
    push_nvidia_field(&mut out, &cols, "mem_clock_max", 5);
    push_nvidia_field(&mut out, &cols, "driver", 3);
    if rows.len() > 1 {
        out.push(format!("gpu_count: {}", rows.len()));
    }
    out.join("\n")
}

fn push_nvidia_field(out: &mut Vec<String>, cols: &[&str], key: &str, index: usize) {
    if let Some(value) = cols.get(index).map(|s| s.trim())
        && !value.is_empty()
        && !value.contains("Not Supported")
        && !value.contains("N/A")
    {
        out.push(format!("{key}: {value}"));
    }
}

fn cuda_arch_from_compute_cap(compute_cap: &str, gpu_model: Option<&str>) -> Option<String> {
    let mut parts = compute_cap.trim().split('.');
    let major: u32 = parts.next()?.trim().parse().ok()?;
    let minor: u32 = parts.next().unwrap_or("0").trim().parse().ok()?;
    let suffix = if major >= 10 && gpu_model.is_some_and(is_blackwell_a_variant) {
        "a"
    } else {
        ""
    };
    Some(format!("sm_{major}{minor}{suffix}"))
}

fn is_blackwell_a_variant(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("b200") || m.contains("b300") || m.contains("gb200") || m.contains("gb300") || m.contains("blackwell")
}

/// Linux distro / kernel string: `PRETTY_NAME` from `/etc/os-release`, falling
/// back to `uname -sr`.
fn linux_os() -> Option<String> {
    if let Ok(text) = std::fs::read_to_string("/etc/os-release")
        && let Some(name) = text
            .lines()
            .find_map(|l| l.strip_prefix("PRETTY_NAME="))
            .map(|v| v.trim().trim_matches('"').to_string())
            .filter(|v| !v.is_empty())
    {
        return Some(name);
    }
    run_capture("uname", &["-sr"])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `/dev/nvidia*` name a 4-GPU host presents, in `read_dir` order
    /// (i.e. unsorted) so the classifier's own sort is exercised.
    fn dev_names() -> Vec<String> {
        [
            "nvidia-modeset",
            "nvidia3",
            "nvidiactl",
            "nvidia0",
            "nvidia-uvm",
            "nvidia1",
            "nvidia-uvm-tools",
            "nvidia2",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    #[test]
    fn per_device_nodes_come_from_the_mask_not_from_dev() {
        // The operator's `CUDA_VISIBLE_DEVICES` is already applied to `minors`, so a
        // two-device mask must yield exactly two per-device nodes even though /dev
        // lists four. Trusting the listing instead would hand a job cards the
        // operator excluded.
        let got = classify(&dev_names(), &[Some(2), Some(3)], |_| true);
        assert_eq!(
            got.per_device,
            [PathBuf::from("/dev/nvidia2"), PathBuf::from("/dev/nvidia3")]
        );
        // ... and the excluded cards are not reachable via the shared set either.
        assert!(
            !got.shared
                .iter()
                .any(|p| p.ends_with("nvidia0") || p.ends_with("nvidia1")),
            "{got:?}"
        );
    }

    #[test]
    fn per_device_is_index_aligned_with_the_pool() {
        // A lease names pool slot `i`, so `per_device[i]` must be that slot's card.
        let got = classify(&dev_names(), &[Some(3), Some(0)], |_| true);
        assert_eq!(got.per_device[0], PathBuf::from("/dev/nvidia3"));
        assert_eq!(got.per_device[1], PathBuf::from("/dev/nvidia0"));
    }

    #[test]
    fn an_unresolvable_minor_disables_confinement_rather_than_shifting_indices() {
        // Dropping just the `None` would leave per_device[1] == /dev/nvidia2, so a
        // lease on slot 1 would bind slot 2's card. All-or-nothing instead: no
        // per-device set, and every numbered node moves to shared.
        let got = classify(&dev_names(), &[Some(0), None, Some(2)], |_| true);
        assert!(got.per_device.is_empty(), "{got:?}");
        for n in ["nvidia0", "nvidia1", "nvidia2", "nvidia3"] {
            assert!(
                got.shared.iter().any(|p| p.ends_with(n)),
                "{n} must still be reachable: {got:?}"
            );
        }
    }

    #[test]
    fn a_mapped_node_that_does_not_exist_degrades_instead_of_being_emitted() {
        // `read_uuid_minors` enumerates physical GPUs including cards absent from
        // /dev, so the mapping can name a missing node. Emitting it would make
        // bwrap fail the whole command.
        let got = classify(&dev_names(), &[Some(0), Some(9)], |p| p != "/dev/nvidia9");
        assert!(got.per_device.is_empty(), "{got:?}");
    }

    #[test]
    fn binding_is_the_default_only_numbered_nodes_are_per_device() {
        // The rule is the parse, not a list: control and fabric interfaces stay
        // shared, including names this code has never heard of, because
        // withholding one breaks the cards the job legitimately owns.
        let mut names = dev_names();
        for future in [
            "nvidia-nvswitch0",
            "nvidia-nvlink",
            "nvidia-caps",
            "nvidia-whatever-ships-next",
        ] {
            names.push(future.to_string());
        }
        let got = classify(&names, &[Some(0)], |_| true);
        for shared in [
            "nvidiactl",
            "nvidia-uvm",
            "nvidia-uvm-tools",
            "nvidia-modeset",
            "nvidia-caps",
            "nvidia-nvswitch0",
            "nvidia-nvlink",
            "nvidia-whatever-ships-next",
        ] {
            assert!(
                got.shared.iter().any(|p| p.ends_with(shared)),
                "{shared} must be shared: {got:?}"
            );
        }
        // A two-digit node is parsed, not prefix-matched.
        let two = classify(&["nvidia10".to_string()], &[Some(1)], |_| true);
        assert!(two.shared.is_empty(), "nvidia10 is a device node, not shared: {two:?}");
    }

    #[test]
    fn resolved_nodes_agree_with_this_host() {
        // The pure tests above pin the rules; this one checks the rules were applied to
        // the real driver, since a lease is only as good as the node it names. A no-op
        // off a GPU box, where `per_device` is empty by construction.
        let nodes = gpu_device_nodes();
        if nodes.per_device.is_empty() {
            return;
        }
        assert_eq!(
            nodes.per_device.len(),
            gpu_count(),
            "one node per pool slot, or lease indices do not address what they think"
        );
        for (slot, path) in nodes.per_device.iter().enumerate() {
            assert!(path.exists(), "slot {slot} names a nonexistent node {path:?}");
            let expected = visible_device_minors()
                .get(slot)
                .copied()
                .flatten()
                .map(|m| format!("/dev/nvidia{m}"));
            assert_eq!(
                Some(path.to_string_lossy().into_owned()),
                expected,
                "slot {slot} maps to the wrong minor"
            );
        }
        let mut seen = nodes.per_device.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), nodes.per_device.len(), "two slots share a node: {nodes:?}");
        // A per-device node must never also be shared, or confinement leaks.
        for path in &nodes.per_device {
            assert!(!nodes.shared.contains(path), "{path:?} is both leased and shared");
        }
    }

    #[test]
    fn parse_nvidia_produces_normalized_fields() {
        let csv = "NVIDIA H100 80GB HBM3, 9.0, 81559 MiB, 560.35.03, 1980 MHz, 2619 MHz";
        let got = parse_nvidia(csv);
        assert!(got.contains("backend: cuda (NVIDIA)"), "{got}");
        assert!(got.contains("gpu_model: NVIDIA H100 80GB HBM3"), "{got}");
        assert!(got.contains("compute_capability: 9.0"), "{got}");
        assert!(got.contains("cuda_arch: sm_90"), "{got}");
        assert!(got.contains("vram: 81559 MiB"), "{got}");
        assert!(got.contains("gpu_clock_max: 1980 MHz"), "{got}");
        assert!(got.contains("mem_clock_max: 2619 MHz"), "{got}");
        assert!(got.contains("driver: 560.35.03"), "{got}");
        assert!(!got.contains("gpu_count"), "single GPU omits count: {got}");
    }

    #[test]
    fn parse_nvidia_drops_unsupported_columns() {
        let csv = "Tesla K80, 3.7, 11441 MiB, 470.00, [Not Supported], [Not Supported]";
        let got = parse_nvidia(csv);
        assert!(got.contains("gpu_model: Tesla K80"), "{got}");
        assert!(!got.contains("gpu_clock_max"), "drops unsupported clock: {got}");
        assert!(!got.contains("mem_clock_max"), "drops unsupported clock: {got}");
    }

    #[test]
    fn parse_nvidia_counts_multiple_gpus() {
        let row = "NVIDIA H100, 9.0, 81559 MiB, 560.35.03, 1980 MHz, 2619 MHz";
        assert!(parse_nvidia(&format!("{row}\n{row}")).contains("gpu_count: 2"));
    }

    #[test]
    fn parse_nvidia_marks_blackwell_a_arch() {
        let row = "NVIDIA B200, 10.0, 183000 MiB, 570.00, 1900 MHz, 4000 MHz";
        let got = parse_nvidia(row);
        assert!(got.contains("compute_capability: 10.0"), "{got}");
        assert!(got.contains("cuda_arch: sm_100a"), "{got}");
    }

    const LISTING: &str = "\
GPU 0: NVIDIA RTX PRO 2000 Blackwell (UUID: GPU-47bb2b52-8441)
GPU 1: NVIDIA GB300 (UUID: GPU-52acce9d-137b)";

    #[test]
    fn parses_nvidia_smi_l_devices() {
        let got = parse_gpu_listing(LISTING);
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].index, 1);
        assert_eq!(got[1].uuid, "GPU-52acce9d-137b");
        assert_eq!(parse_gpu_listing("GPU 0: NVIDIA H100 (UUID: GPU-x)").len(), 1);
        // MIG sub-device lines carry no `GPU <n>:` prefix, so they never inflate
        // the physical-device list.
        assert_eq!(
            parse_gpu_listing("GPU 0: NVIDIA H100 (UUID: GPU-x)\n  MIG 1g.10gb Device 0: ...").len(),
            1
        );
        assert_eq!(parse_gpu_listing("").len(), 0);
    }

    #[test]
    fn unset_mask_selects_every_device() {
        let devices = parse_gpu_listing(LISTING);
        assert_eq!(
            resolve_visible_devices(&devices, None),
            vec!["GPU-47bb2b52-8441", "GPU-52acce9d-137b"]
        );
    }

    #[test]
    fn an_empty_mask_selects_nothing_and_is_not_the_same_as_unset() {
        // CUDA reads an empty `CUDA_VISIBLE_DEVICES` as "no GPUs". Both cases yield no
        // devices here, so callers must distinguish set-but-empty from unset: treating
        // the first as permissive hands a job every card in answer to a request for
        // none (`SandboxManager::spawn` forwards it as a ceiling for exactly this).
        let devices = parse_gpu_listing(LISTING);
        assert!(resolve_visible_devices(&devices, Some("")).is_empty());
        assert_eq!(resolve_visible_devices(&devices, None).len(), 2);
    }

    #[test]
    fn mask_selects_by_index_or_uuid_prefix() {
        let devices = parse_gpu_listing(LISTING);
        // An index is resolved to that device's UUID...
        assert_eq!(resolve_visible_devices(&devices, Some("1")), vec!["GPU-52acce9d-137b"]);
        // ...as is an abbreviated UUID, which is what CUDA itself accepts.
        assert_eq!(
            resolve_visible_devices(&devices, Some("GPU-52acce9d")),
            vec!["GPU-52acce9d-137b"]
        );
        // Mask order wins over listing order, so pool slot 0 is the GB300 here.
        assert_eq!(
            resolve_visible_devices(&devices, Some("1,0")),
            vec!["GPU-52acce9d-137b", "GPU-47bb2b52-8441"]
        );
        assert_eq!(
            resolve_visible_devices(&devices, Some(" 1 ")),
            vec!["GPU-52acce9d-137b"]
        );
    }

    #[test]
    fn mask_stops_at_the_first_unknown_device() {
        let devices = parse_gpu_listing(LISTING);
        // CUDA enumerates until an entry names nothing, then stops — so a bad
        // entry hides every device after it rather than being skipped.
        assert_eq!(
            resolve_visible_devices(&devices, Some("0,9")),
            vec!["GPU-47bb2b52-8441"]
        );
        assert!(resolve_visible_devices(&devices, Some("9,0")).is_empty());
        assert!(resolve_visible_devices(&devices, Some("GPU-nope")).is_empty());
        // An empty mask selects nothing; `gpu_count` floors the pool at 1.
        assert!(resolve_visible_devices(&devices, Some("")).is_empty());
        assert_eq!(gpu_count().max(1), gpu_count());
    }

    #[test]
    fn probe_is_best_effort_and_nonempty() {
        // Whatever the test host is, the probe returns a non-empty string (real
        // block or fallback note) and never panics.
        assert!(!probe_hardware().is_empty());
    }
}
