use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("planning failed: {0}")]
    Planning(String),

    #[error("reasoning failed: {0}")]
    Reasoning(String),

    #[error("tool execution failed: {tool}: {reason}")]
    ToolExecution { tool: String, reason: String },

    #[error("max steps exceeded ({0})")]
    MaxStepsExceeded(usize),

    #[error("timeout after {0}s")]
    Timeout(u64),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}
