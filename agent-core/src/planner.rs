use crate::error::AgentError;
use crate::executor::ToolSpec;
use crate::types::{Goal, Plan, Step};

/// Type alias for the planning function. The LLM call lives here — the engine
/// doesn't bundle an LLM, the caller provides this.
pub type PlannerFn =
    Box<dyn Fn(&Goal, &[ToolSpec]) -> Result<Plan, AgentError> + Send + Sync>;

/// Default planner that wraps the entire goal as a single step.
pub fn simple_planner() -> PlannerFn {
    Box::new(|goal: &Goal, _tools: &[ToolSpec]| {
        let step = Step::new(&goal.description);
        Ok(Plan::single_step(&goal.id, &goal.description).with_steps(vec![step]))
    })
}

// Convenience extension for Plan construction.
trait PlanExt {
    fn with_steps(self, steps: Vec<Step>) -> Self;
}

impl PlanExt for Plan {
    fn with_steps(mut self, steps: Vec<Step>) -> Self {
        self.steps = steps;
        self
    }
}
