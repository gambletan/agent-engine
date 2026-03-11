use chrono::Utc;
use log::{debug, info, warn};
use serde_json::Value;

use crate::error::AgentError;
use crate::executor::ToolExecutor;
use crate::memory::WorkingMemory;
use crate::planner::{self, PlannerFn};
use crate::reasoner::{self, Action, ReasonerFn};
use crate::types::{
    AgentConfig, AgentState, ExecutionTrace, Goal, Observation, StepStatus, TraceAction,
    TraceEntry,
};

/// The main agent runtime. Drives the ReAct loop.
pub struct Agent {
    config: AgentConfig,
    executor: Box<dyn ToolExecutor>,
    planner: PlannerFn,
    reasoner: ReasonerFn,
}

impl Agent {
    pub fn new(config: AgentConfig, executor: Box<dyn ToolExecutor>) -> Self {
        Self {
            config,
            executor,
            planner: planner::simple_planner(),
            reasoner: reasoner::default_reasoner(),
        }
    }

    pub fn with_planner(mut self, planner: PlannerFn) -> Self {
        self.planner = planner;
        self
    }

    pub fn with_reasoner(mut self, reasoner: ReasonerFn) -> Self {
        self.reasoner = reasoner;
        self
    }

    /// Run the agent loop for a given goal.
    pub fn run(&self, goal: Goal) -> Result<ExecutionTrace, AgentError> {
        let started_at = Utc::now();
        let tools = self.executor.available_tools();
        let mut entries: Vec<TraceEntry> = Vec::new();
        let mut total_steps: usize = 0;

        info!("Starting agent run for goal: {}", goal.description);

        // --- Planning phase ---
        let plan = if self.config.planning_enabled {
            debug!("Planning with {} available tools", tools.len());
            (self.planner)(&goal, &tools)?
        } else {
            crate::types::Plan::single_step(&goal.id, &goal.description)
        };

        info!("Plan has {} step(s)", plan.steps.len());

        // --- Execution phase ---
        let mut state = AgentState {
            goal: goal.clone(),
            plan: plan.clone(),
            step_index: 0,
            working_memory: WorkingMemory::new(),
            observations: Vec::new(),
        };

        // Seed with a null observation so the reasoner always has something.
        let mut last_observation = Observation::success("_init", Value::Null);
        let mut final_result: Option<Value> = None;

        while state.step_index < state.plan.steps.len() {
            if total_steps >= self.config.max_steps {
                return Err(AgentError::MaxStepsExceeded(self.config.max_steps));
            }

            let step = &state.plan.steps[state.step_index];
            debug!(
                "Step {}: {}",
                state.step_index, step.description
            );

            // Mark step running.
            state.plan.steps[state.step_index].status = StepStatus::Running;

            // --- Reason ---
            let action = (self.reasoner)(&state, &last_observation)?;

            match action {
                Action::CallTool { ref name, ref args } => {
                    debug!("Calling tool: {} with args: {}", name, args);
                    let observation = match self.executor.execute(name, args.clone()) {
                        Ok(obs) => obs,
                        Err(e) => {
                            warn!("Tool {} failed: {}", name, e);
                            Observation::failure(name.clone(), e.to_string())
                        }
                    };

                    let entry = TraceEntry {
                        timestamp: Utc::now(),
                        step_index: state.step_index,
                        action: TraceAction::CallTool {
                            name: name.clone(),
                            args: args.clone(),
                        },
                        observation: Some(observation.clone()),
                    };
                    entries.push(entry);

                    if observation.success {
                        state.plan.steps[state.step_index].status = StepStatus::Completed;
                        state.plan.steps[state.step_index].result =
                            Some(observation.output.clone());
                    } else {
                        let msg = observation
                            .error_message
                            .clone()
                            .unwrap_or_else(|| "unknown error".into());
                        state.plan.steps[state.step_index].status = StepStatus::Failed(msg);
                    }

                    state.observations.push(observation.clone());
                    last_observation = observation;
                    state.step_index += 1;
                }
                Action::Finish { ref result } => {
                    let entry = TraceEntry {
                        timestamp: Utc::now(),
                        step_index: state.step_index,
                        action: TraceAction::Finish {
                            result: result.clone(),
                        },
                        observation: None,
                    };
                    entries.push(entry);
                    state.plan.steps[state.step_index].status = StepStatus::Completed;
                    state.plan.steps[state.step_index].result = Some(result.clone());
                    final_result = Some(result.clone());
                    state.step_index += 1;
                }
                Action::Replan => {
                    let entry = TraceEntry {
                        timestamp: Utc::now(),
                        step_index: state.step_index,
                        action: TraceAction::Replan,
                        observation: None,
                    };
                    entries.push(entry);

                    // Re-plan from current state.
                    let new_plan = (self.planner)(&state.goal, &tools)?;
                    state.plan = new_plan;
                    state.step_index = 0;
                }
                Action::AskUser { ref question } => {
                    let entry = TraceEntry {
                        timestamp: Utc::now(),
                        step_index: state.step_index,
                        action: TraceAction::AskUser {
                            question: question.clone(),
                        },
                        observation: None,
                    };
                    entries.push(entry);
                    // For now, treat as finish — interactive mode not yet supported.
                    final_result = Some(serde_json::json!({ "question": question }));
                    break;
                }
            }

            total_steps += 1;
        }

        let finished_at = Utc::now();
        let success = final_result.is_some()
            || state
                .plan
                .steps
                .iter()
                .all(|s| s.status == StepStatus::Completed);

        // If no explicit Finish, use the last observation output.
        if final_result.is_none() && !state.observations.is_empty() {
            final_result = Some(
                state
                    .observations
                    .last()
                    .unwrap()
                    .output
                    .clone(),
            );
        }

        info!(
            "Agent run complete: {} steps, success={}",
            total_steps, success
        );

        Ok(ExecutionTrace {
            goal,
            plan: state.plan,
            entries,
            final_result,
            success,
            total_steps,
            started_at,
            finished_at,
        })
    }
}
