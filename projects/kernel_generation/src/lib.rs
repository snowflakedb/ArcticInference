//! `kernelguy` — agentic harness for writing and optimizing GPU kernels.
//!
//! The library is layered bottom-up:
//! - [`domain`] — shared data model: on-disk schema, journal, run-meta state,
//!   the per-turn event channel, and the crate's only sanctioned lossy numeric
//!   conversions. Depends on no other layer.
//! - [`env`](mod@env) — host probing: GPU inventory, device nodes, and where the
//!   CUDA toolchain lives. Hands `exec` plain data and never names a sandbox.
//! - [`exec`] — execution/storage infra: the GPU device pool, session-tree and
//!   workspace-snapshot storage, and the OS-native [`exec::sandbox::Sandbox`]
//!   (wrapping `bwrap`).
//! - [`ai`] — wire protocol: speaking Anthropic / `OpenAI` HTTP
//!   APIs, parsing their streams, and translating both to a unified
//!   [`ai::Event`] enum. Owns the tool *transport* vocabulary
//!   ([`ai::protocol`]) that it serializes; it describes tools as
//!   [`ai::protocol::ToolDescriptor`]s but never runs them.
//! - [`tool`] — the [`tool::Tool`] trait (derived JSON schemas via `schemars`)
//!   and [`tool::ToolBox`], the registry that erases a typed tool into something
//!   callable with raw model JSON.
//! - [`harness`] — shared turn-streaming / retry layer ([`harness::stream`])
//!   driving the AVO orchestrator's agent turns.
//! - [`orchestrator`] — the AVO orchestrator policy tying it all together.
//!
//! "Layered" is the intent, not an invariant: `tool` reaches up into
//! `orchestrator`. Nothing enforces the order.
//!
//! `env` depends on no other layer — it hands `exec` plain data and never names a
//! sandbox, which keeps `exec` → `env` one-way.

pub mod ai;
pub mod domain;
pub mod env;
pub mod exec;
pub mod harness;
pub mod orchestrator;
pub mod tool;
pub(crate) mod ui;
