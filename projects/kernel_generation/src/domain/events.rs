//! Per-turn event channel between evaluation-producing tools and the loop.
//!
//! Per-turn event channel shared between the tools that produce evaluations and
//! the orchestrator loop that consumes them. Lives in `domain` (not the
//! orchestrator) so tools can record into it without depending on the policy
//! layer above them.

use std::sync::Mutex;

use crate::domain::types::{EvalMetrics, SolutionFiles};
use crate::domain::util::lock;

#[derive(Clone)]
pub struct EvaluationEvent {
    pub metrics: EvalMetrics,
    pub files: SolutionFiles,
}

#[derive(Default)]
pub struct TurnEvents {
    evaluations: Mutex<Vec<EvaluationEvent>>,
}

impl TurnEvents {
    pub fn record_evaluation(&self, metrics: EvalMetrics, files: SolutionFiles) {
        lock(&self.evaluations).push(EvaluationEvent { metrics, files });
    }

    pub fn take_evaluations(&self) -> Vec<EvaluationEvent> {
        std::mem::take(&mut *lock(&self.evaluations))
    }
}
