//! Host environment probing, orthogonal to the search policy: what hardware we're
//! on and where the GPU toolchain is.
//!
//! Nothing here names a sandbox — this module sits *below* `exec` and hands it
//! plain data.
//!
//! - [`hardware`] — one-shot GPU/host identity (`nvidia-smi` / `system_profiler`).
//! - [`cuda`] — host CUDA toolchain discovery, as data for a caller to graft onto
//!   a sandbox. The preflight that *exercises* that toolchain lives in
//!   [`crate::exec::cuda_preflight`], because it needs a sandbox to run in.

pub mod cuda;
pub mod hardware;
