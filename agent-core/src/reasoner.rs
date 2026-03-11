use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AgentError;
use crate::types::{AgentState, Observation};

/// What the agent decides to do next.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    CallTool { name: String, args: Value },
    Finish { result: Value },
    Replan,
    AskUser { question: String },
}

/// Type alias for the reasoning function. Given current state and the last
/// observation, decide the next action.
pub type ReasonerFn =
    Box<dyn Fn(&AgentState, &Observation) -> Result<Action, AgentError> + Send + Sync>;

/// Default reasoner: executes the tool hinted in the current plan step,
/// passing the goal description as the argument. If no tool hint is present,
/// finishes with the last observation's output.
pub fn default_reasoner() -> ReasonerFn {
    Box::new(|state: &AgentState, observation: &Observation| {
        let step_idx = state.step_index;
        if step_idx >= state.plan.steps.len() {
            return Ok(Action::Finish {
                result: observation.output.clone(),
            });
        }

        let step = &state.plan.steps[step_idx];
        if let Some(ref tool) = step.tool_hint {
            Ok(Action::CallTool {
                name: tool.clone(),
                args: serde_json::json!({ "input": step.description }),
            })
        } else {
            Ok(Action::Finish {
                result: observation.output.clone(),
            })
        }
    })
}
