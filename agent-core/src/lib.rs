pub mod engine;
pub mod error;
pub mod executor;
pub mod llm;
pub mod memory;
pub mod planner;
pub mod react;
pub mod reasoner;
pub mod reward;
pub mod scheduler;
pub mod types;

#[cfg(test)]
mod tests;

// Re-exports for convenience.
pub use engine::Agent;
pub use error::AgentError;
pub use executor::{ToolExecutor, ToolSpec};
pub use llm::{
    ChatMessage, HttpLlmProvider, LlmConfig, LlmProvider, OllamaProvider, Role, ToolDefinition,
};
pub use memory::WorkingMemory;
pub use planner::PlannerFn;
pub use react::{PlanAndExecuteAgent, ReActAgent};
pub use reasoner::{Action, ReasonerFn};
pub use reward::{
    Outcome, RewardConfig, RewardEngine, RewardScorer, RewardSignal, SimpleRatioScorer,
    StrategyWeight,
};
pub use scheduler::{
    LogNotificationSink, Notification, NotificationSink, Priority, ScheduledTask, Scheduler,
    Trigger,
};
pub use types::{
    AgentConfig, AgentState, ExecutionTrace, Goal, Observation, Plan, Step, StepStatus,
};
