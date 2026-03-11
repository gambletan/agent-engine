use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AgentError;
use crate::types::Observation;

/// Description of a tool available to the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: Value,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Trait for executing tools. Users implement this to wire up real tools.
pub trait ToolExecutor: Send + Sync {
    fn execute(&self, tool_name: &str, args: Value) -> Result<Observation, AgentError>;
    fn available_tools(&self) -> Vec<ToolSpec>;
}
