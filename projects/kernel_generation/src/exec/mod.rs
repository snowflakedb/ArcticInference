//! Execution and storage infrastructure shared by the tools and the
//! orchestrator: the serial GPU execution lease, the session-tree/workspace
//! snapshot storage primitives, and the OS-native sandbox.
//!
//! - [`queue`] — the one serial lease every GPU subprocess passes through.
//! - [`turn_tree`] — the pure tree of message-delta/workspace-snapshot nodes.
//! - [`snapshot_store`] — content-addressed workspace snapshots.
//! - [`sandbox`] — `bwrap` workspace jail.
//! - [`sandbox_manager`] — git-backed sandbox minter (owns the snapshot repo + mounts).
//! - [`cuda_preflight`] — checks the sandboxed NVIDIA toolchain before a run leans on it.

pub mod cuda_preflight;
pub mod queue;
pub mod sandbox;
pub mod sandbox_manager;
pub mod snapshot_store;
pub mod turn_tree;
