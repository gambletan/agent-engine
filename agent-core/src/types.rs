use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// A user request with optional context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub description: String,
    pub context: Value,
}

impl Goal {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            context: Value::Null,
        }
    }

    pub fn with_context(mut self, context: Value) -> Self {
        self.context = context;
        self
    }
}

/// Status of a plan step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Skipped,
}

/// A single planned action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub description: String,
    pub tool_hint: Option<String>,
    pub status: StepStatus,
    pub result: Option<Value>,
}

impl Step {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            tool_hint: None,
            status: StepStatus::Pending,
            result: None,
        }
    }

    pub fn with_tool_hint(mut self, tool: impl Into<String>) -> Self {
        self.tool_hint = Some(tool.into());
        self
    }
}

/// An ordered list of steps to achieve a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub goal_id: String,
    pub steps: Vec<Step>,
}

impl Plan {
    pub fn new(goal_id: impl Into<String>, steps: Vec<Step>) -> Self {
        Self {
            goal_id: goal_id.into(),
            steps,
        }
    }

    pub fn single_step(goal_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            goal_id: goal_id.into(),
            steps: vec![Step::new(description)],
        }
    }
}

/// Result from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub tool_name: String,
    pub output: Value,
    pub success: bool,
    pub error_message: Option<String>,
}

impl Observation {
    pub fn success(tool_name: impl Into<String>, output: Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            output,
            success: true,
            error_message: None,
        }
    }

    pub fn failure(tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            output: Value::Null,
            success: false,
            error_message: Some(error.into()),
        }
    }
}

/// Current execution state visible to the reasoner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub goal: Goal,
    pub plan: Plan,
    pub step_index: usize,
    pub working_memory: crate::memory::WorkingMemory,
    pub observations: Vec<Observation>,
}

/// A single entry in the execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub timestamp: DateTime<Utc>,
    pub step_index: usize,
    pub action: TraceAction,
    pub observation: Option<Observation>,
}

/// What the agent decided to do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceAction {
    CallTool { name: String, args: Value },
    Finish { result: Value },
    Replan,
    AskUser { question: String },
}

/// Full trace of reasoning and actions for one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub goal: Goal,
    pub plan: Plan,
    pub entries: Vec<TraceEntry>,
    pub final_result: Option<Value>,
    pub success: bool,
    pub total_steps: usize,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
}

/// Configuration for the agent engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub max_retries: usize,
    pub timeout_secs: u64,
    pub planning_enabled: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 20,
            max_retries: 3,
            timeout_secs: 300,
            planning_enabled: true,
        }
    }
}
