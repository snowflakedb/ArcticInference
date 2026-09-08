//! Shared data model and foundational support, sitting below every other layer.
//!
//! - [`types`] — on-disk schema (metrics, sidecars, ledger, manifest, hashing).
//! - [`run_state`] — crash-safe journal of run meta (`state.json`).
//! - [`events`] — the per-turn evaluation channel between tools and the loop.
//! - [`util`] — atomic file writes, poison-safe locking, small string helpers.
//! - [`convert`] — the crate's only sanctioned lossy numeric conversions.

pub mod convert;
pub mod events;
pub mod run_state;
pub mod types;
pub mod util;
