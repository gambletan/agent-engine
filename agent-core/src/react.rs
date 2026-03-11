use chrono::Utc;
use log::{debug, info, warn};
use serde_json::Value;

use crate::error::AgentError;
use crate::executor::{ToolExecutor, ToolSpec};
use crate::llm::{ChatMessage, LlmConfig, LlmProvider};
use crate::types::{
    AgentConfig, ExecutionTrace, Goal, Observation, Plan, Step, StepStatus, TraceAction,
    TraceEntry,
};

// ---------------------------------------------------------------------------
// ReActAgent
// ---------------------------------------------------------------------------

/// A ReAct (Reason-Act-Observe) agent driven by an LLM.
pub struct ReActAgent {
    llm: Box<dyn LlmProvider>,
    tools: Box<dyn ToolExecutor>,
    system_prompt: String,
    config: AgentConfig,
}

impl ReActAgent {
    pub fn new(
        llm: Box<dyn LlmProvider>,
        tools: Box<dyn ToolExecutor>,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            tools,
            system_prompt: String::new(),
            config,
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    /// Run the ReAct loop for a given goal.
    pub fn run(&self, goal: &str) -> Result<ExecutionTrace, AgentError> {
        let started_at = Utc::now();
        let available_tools = self.tools.available_tools();
        let tool_descriptions = build_tool_descriptions(&available_tools);

        let system = if self.system_prompt.is_empty() {
            format!(
                "You are an AI agent that solves tasks using available tools.\n\n\
                 Available tools:\n{}\n\n\
                 Respond with JSON in one of these formats:\n\
                 - To use a tool: {{\"thought\": \"...\", \"action\": \"tool_name\", \"action_input\": {{...}}}}\n\
                 - To finish: {{\"thought\": \"...\", \"action\": \"finish\", \"result\": \"...\"}}\n\n\
                 Always respond with valid JSON only.",
                tool_descriptions
            )
        } else {
            format!(
                "{}\n\nAvailable tools:\n{}\n\n\
                 Respond with JSON in one of these formats:\n\
                 - To use a tool: {{\"thought\": \"...\", \"action\": \"tool_name\", \"action_input\": {{...}}}}\n\
                 - To finish: {{\"thought\": \"...\", \"action\": \"finish\", \"result\": \"...\"}}\n\n\
                 Always respond with valid JSON only.",
                self.system_prompt, tool_descriptions
            )
        };

        let mut messages = vec![
            ChatMessage::system(&system),
            ChatMessage::user(format!("Goal: {}", goal)),
        ];

        let mut entries: Vec<TraceEntry> = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        let mut total_steps: usize = 0;

        let llm_config = LlmConfig {
            model: String::new(), // Use provider default.
            temperature: 0.2,
            max_tokens: Some(1024),
            stop_sequences: Vec::new(),
            json_mode: true,
        };

        loop {
            if total_steps >= self.config.max_steps {
                return Err(AgentError::MaxStepsExceeded(self.config.max_steps));
            }

            // Reason: ask the LLM what to do.
            let response = self.llm.chat(&messages, &llm_config)?;
            debug!("LLM response: {}", response);

            let parsed = parse_react_response(&response)?;

            match parsed {
                ReactAction::Finish { thought, result } => {
                    info!("Agent finished: {}", thought);
                    let step = Step::new(&thought);
                    steps.push(step);

                    let result_value = serde_json::json!(result);
                    entries.push(TraceEntry {
                        timestamp: Utc::now(),
                        step_index: total_steps,
                        action: TraceAction::Finish {
                            result: result_value.clone(),
                        },
                        observation: None,
                    });

                    total_steps += 1;
                    let finished_at = Utc::now();
                    let goal_obj = Goal::new(goal);
                    let plan = Plan::new(&goal_obj.id, steps);

                    return Ok(ExecutionTrace {
                        goal: goal_obj,
                        plan,
                        entries,
                        final_result: Some(result_value),
                        success: true,
                        total_steps,
                        started_at,
                        finished_at,
                    });
                }
                ReactAction::ToolCall {
                    thought,
                    action,
                    action_input,
                } => {
                    info!("Agent thought: {} -> calling {}", thought, action);

                    let mut step = Step::new(&thought).with_tool_hint(&action);

                    // Act: execute the tool.
                    let observation = match self.tools.execute(&action, action_input.clone()) {
                        Ok(obs) => obs,
                        Err(e) => {
                            warn!("Tool {} failed: {}", action, e);
                            Observation::failure(&action, e.to_string())
                        }
                    };

                    let entry = TraceEntry {
                        timestamp: Utc::now(),
                        step_index: total_steps,
                        action: TraceAction::CallTool {
                            name: action.clone(),
                            args: action_input,
                        },
                        observation: Some(observation.clone()),
                    };
                    entries.push(entry);

                    if observation.success {
                        step.status = StepStatus::Completed;
                        step.result = Some(observation.output.clone());
                    } else {
                        let msg = observation
                            .error_message
                            .clone()
                            .unwrap_or_else(|| "unknown error".into());
                        step.status = StepStatus::Failed(msg);
                    }
                    steps.push(step);

                    // Observe: feed result back to LLM.
                    messages.push(ChatMessage::assistant(&response));
                    let obs_text = if observation.success {
                        format!("Observation: {}", observation.output)
                    } else {
                        format!(
                            "Observation: Tool failed - {}",
                            observation
                                .error_message
                                .as_deref()
                                .unwrap_or("unknown error")
                        )
                    };
                    messages.push(ChatMessage::user(&obs_text));

                    total_steps += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent
// ---------------------------------------------------------------------------

/// An agent that first decomposes a goal into steps using a planner LLM,
/// then executes each step with a ReAct agent.
pub struct PlanAndExecuteAgent {
    planner_llm: Box<dyn LlmProvider>,
    executor: ReActAgent,
}

impl PlanAndExecuteAgent {
    pub fn new(planner_llm: Box<dyn LlmProvider>, executor: ReActAgent) -> Self {
        Self {
            planner_llm,
            executor,
        }
    }

    /// Run: plan the goal into sub-steps, then execute each with ReAct.
    pub fn run(&self, goal: &str) -> Result<ExecutionTrace, AgentError> {
        let started_at = Utc::now();

        // Step 1: Ask planner to decompose goal into steps.
        let plan_prompt = format!(
            "Decompose the following goal into a list of concrete steps.\n\
             Respond with a JSON array of strings, each string being one step.\n\
             Goal: {}",
            goal
        );

        let config = LlmConfig {
            model: String::new(),
            temperature: 0.3,
            max_tokens: Some(1024),
            stop_sequences: Vec::new(),
            json_mode: true,
        };

        let messages = vec![
            ChatMessage::system(
                "You are a planning assistant. Decompose goals into concrete steps. \
                 Respond with a JSON array of strings.",
            ),
            ChatMessage::user(&plan_prompt),
        ];

        let plan_response = self.planner_llm.chat(&messages, &config)?;
        let sub_steps = parse_plan_steps(&plan_response)?;

        info!("Plan has {} sub-step(s)", sub_steps.len());

        // Step 2: Execute each sub-step with the ReAct agent.
        let mut all_entries: Vec<TraceEntry> = Vec::new();
        let mut all_steps: Vec<Step> = Vec::new();
        let mut total_steps: usize = 0;
        let mut last_result: Option<Value> = None;

        for (i, sub_step) in sub_steps.iter().enumerate() {
            info!("Executing sub-step {}/{}: {}", i + 1, sub_steps.len(), sub_step);

            match self.executor.run(sub_step) {
                Ok(trace) => {
                    total_steps += trace.total_steps;
                    all_entries.extend(trace.entries);

                    let mut step = Step::new(sub_step);
                    step.status = if trace.success {
                        StepStatus::Completed
                    } else {
                        StepStatus::Failed("sub-step failed".to_string())
                    };
                    step.result = trace.final_result.clone();
                    last_result = trace.final_result;
                    all_steps.push(step);
                }
                Err(e) => {
                    warn!("Sub-step {} failed: {}", i, e);
                    let mut step = Step::new(sub_step);
                    step.status = StepStatus::Failed(e.to_string());
                    all_steps.push(step);
                }
            }
        }

        let finished_at = Utc::now();
        let goal_obj = Goal::new(goal);
        let plan = Plan::new(&goal_obj.id, all_steps);

        let success = plan
            .steps
            .iter()
            .all(|s| s.status == StepStatus::Completed);

        Ok(ExecutionTrace {
            goal: goal_obj,
            plan,
            success,
            final_result: last_result,
            total_steps,
            entries: all_entries,
            started_at,
            finished_at,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Internal representation of a parsed ReAct response.
pub enum ReactAction {
    ToolCall {
        thought: String,
        action: String,
        action_input: Value,
    },
    Finish {
        thought: String,
        result: String,
    },
}

/// Build a text description of available tools for the system prompt.
fn build_tool_descriptions(tools: &[ToolSpec]) -> String {
    tools
        .iter()
        .map(|t| {
            format!(
                "- {}: {} (parameters: {})",
                t.name, t.description, t.parameters
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a ReAct JSON response from the LLM, handling markdown code blocks.
pub fn parse_react_response(response: &str) -> Result<ReactAction, AgentError> {
    let json_str = extract_json(response);

    let parsed: Value = serde_json::from_str(json_str).map_err(|e| {
        AgentError::Reasoning(format!(
            "Failed to parse ReAct JSON: {} in response: {}",
            e, response
        ))
    })?;

    let thought = parsed["thought"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let action = parsed["action"]
        .as_str()
        .ok_or_else(|| {
            AgentError::Reasoning(format!(
                "Missing 'action' field in response: {}",
                response
            ))
        })?
        .to_string();

    if action == "finish" {
        let result = parsed["result"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(ReactAction::Finish { thought, result })
    } else {
        let action_input = parsed
            .get("action_input")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));
        Ok(ReactAction::ToolCall {
            thought,
            action,
            action_input,
        })
    }
}

/// Extract JSON from a response that may be wrapped in markdown code blocks.
fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // Handle ```json ... ``` blocks.
    if let Some(start) = trimmed.find("```json") {
        let after_marker = &trimmed[start + 7..];
        if let Some(end) = after_marker.find("```") {
            return after_marker[..end].trim();
        }
    }

    // Handle ``` ... ``` blocks.
    if let Some(start) = trimmed.find("```") {
        let after_marker = &trimmed[start + 3..];
        if let Some(end) = after_marker.find("```") {
            return after_marker[..end].trim();
        }
    }

    trimmed
}

/// Parse the planner response into a list of step strings.
fn parse_plan_steps(response: &str) -> Result<Vec<String>, AgentError> {
    let json_str = extract_json(response);

    let parsed: Value = serde_json::from_str(json_str).map_err(|e| {
        AgentError::Planning(format!(
            "Failed to parse plan JSON: {} in response: {}",
            e, response
        ))
    })?;

    match parsed {
        Value::Array(arr) => {
            let steps: Vec<String> = arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if steps.is_empty() {
                Err(AgentError::Planning(
                    "Plan array was empty or contained no strings".to_string(),
                ))
            } else {
                Ok(steps)
            }
        }
        _ => Err(AgentError::Planning(format!(
            "Expected JSON array of strings, got: {}",
            parsed
        ))),
    }
}
