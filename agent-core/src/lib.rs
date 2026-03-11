pub mod engine;
pub mod error;
pub mod executor;
pub mod memory;
pub mod planner;
pub mod reasoner;
pub mod scheduler;
pub mod types;

#[cfg(test)]
mod tests;

// Re-exports for convenience.
pub use engine::Agent;
pub use error::AgentError;
pub use executor::{ToolExecutor, ToolSpec};
pub use memory::WorkingMemory;
pub use planner::PlannerFn;
pub use reasoner::{Action, ReasonerFn};
pub use scheduler::{
    LogNotificationSink, Notification, NotificationSink, Priority, ScheduledTask, Scheduler,
    Trigger,
};
pub use types::{
    AgentConfig, AgentState, ExecutionTrace, Goal, Observation, Plan, Step, StepStatus,
};
