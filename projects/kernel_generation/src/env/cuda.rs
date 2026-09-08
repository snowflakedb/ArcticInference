//! Host CUDA toolchain discovery — where `nvcc` and friends live on this machine.
//!
//! Pure discovery, returned as data. The preflight that *exercises* the toolchain is
//! [`crate::exec::cuda_preflight`]: it needs a sandbox to run in, and this module
//! stays free of `exec`.

use std::collections::BTreeSet;
use std::path::PathBuf;

#[must_use]
pub fn should_validate_nvidia_tools(hardware_summary: &str) -> bool {
    hardware_summary.contains("backend: cuda")
}

/// Host CUDA toolchain paths a sandbox needs, as data.
///
/// Returned rather than pushed into a sandbox, so this module does not import one.
#[derive(Debug, Default)]
pub struct CudaToolchainAccess {
    /// Directories to prepend to the sandbox `PATH` (and expose read-only).
    pub bin_dirs: Vec<PathBuf>,
    /// Toolchain roots to expose read-only.
    pub roots: Vec<PathBuf>,
}

/// Discover the host's CUDA toolchain, for a caller to graft onto a sandbox.
#[must_use]
pub fn cuda_toolchain_access() -> CudaToolchainAccess {
    CudaToolchainAccess {
        bin_dirs: discover_cuda_bin_dirs(),
        roots: discover_cuda_roots(),
    }
}

/// Host CUDA/NVIDIA environment worth forwarding into a GPU-enabled sandbox, as
/// `(key, value)` pairs, skipping anything unset or empty.
///
/// `CUDA_VISIBLE_DEVICES` is deliberately **absent**. It is the operator's
/// inbound knob — consumed by [`crate::env::hardware::visible_device_uuids`] to
/// size the device pool — and not a confinement mechanism, so the harness does
/// not re-emit it. A sandbox is confined by which device nodes it can open, and
/// its cards renumber from 0 inside, which would make a forwarded host value
/// select the wrong devices or none.
#[must_use]
pub fn cuda_env_passthrough() -> Vec<(String, String)> {
    [
        "CUDA_HOME",
        "CUDA_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    ]
    .into_iter()
    .filter_map(|key| {
        let value = std::env::var(key).ok()?;
        (!value.is_empty()).then(|| (key.to_string(), value))
    })
    .collect()
}

fn discover_cuda_bin_dirs() -> Vec<PathBuf> {
    let mut out = BTreeSet::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = nonempty_env_path(var) {
            let bin = root.join("bin");
            if bin.is_dir() {
                out.insert(bin);
            }
        }
    }
    if let Some(root) = nvcc_root_from_path() {
        let bin = root.join("bin");
        if bin.is_dir() {
            out.insert(bin);
        }
    }
    for bin in [
        "/usr/local/cuda/bin",
        "/opt/nvidia/nsight-compute",
        "/opt/nvidia/nsight-compute/target/linux-desktop-glibc_2_11_3-x64",
    ] {
        let p = PathBuf::from(bin);
        if p.is_dir() {
            out.insert(p);
        }
    }
    out.into_iter().collect()
}

fn discover_cuda_roots() -> Vec<PathBuf> {
    let mut out = BTreeSet::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = nonempty_env_path(var)
            && root.exists()
        {
            out.insert(root);
        }
    }
    if let Some(root) = nvcc_root_from_path()
        && root.exists()
    {
        out.insert(root);
    }
    for root in ["/usr/local/cuda", "/opt/nvidia"] {
        let p = PathBuf::from(root);
        if p.exists() {
            out.insert(p);
        }
    }
    out.into_iter().collect()
}

fn nonempty_env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn nvcc_root_from_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let nvcc = dir.join("nvcc");
        if nvcc.is_file()
            && let Some(root) = dir.parent()
        {
            return Some(root.to_path_buf());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_validate_only_cuda_summary() {
        assert!(should_validate_nvidia_tools("backend: cuda (NVIDIA)"));
        assert!(!should_validate_nvidia_tools("GPU identity probe unavailable"));
    }
}
