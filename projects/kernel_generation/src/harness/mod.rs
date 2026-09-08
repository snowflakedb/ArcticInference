//! Agent-running layer around the completion providers.
//!
//! - [`stream`] — turn-streaming, retry, and terminal rendering that the AVO
//!   orchestrator drives its turns through.
//! - [`turn`] — turn-execution mechanics: dispatching a completion's tool calls
//!   and classifying a tool-less completion. Policy-agnostic.
//! - [`skills`] — repo-local `SKILL.md` capability bundles discovered at startup
//!   and mounted read-only into the sandbox for the agent to consult.

pub mod skills;
pub(crate) mod stream;
pub(crate) mod turn;
