#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::error::AgentError;
    use crate::executor::{ToolExecutor, ToolSpec};
    use crate::types::{AgentConfig, Goal, Observation, Plan, Step};
    use crate::engine::Agent;
    use crate::reasoner::Action;

    /// A simple executor that echoes the input back.
    struct EchoExecutor;

    impl ToolExecutor for EchoExecutor {
        fn execute(&self, tool_name: &str, args: Value) -> Result<Observation, AgentError> {
            Ok(Observation::success(
                tool_name,
                json!({ "echo": args }),
            ))
        }

        fn available_tools(&self) -> Vec<ToolSpec> {
            vec![ToolSpec::new(
                "echo",
                "Echoes the input",
                json!({ "type": "object" }),
            )]
        }
    }

    /// An executor that always fails.
    struct FailingExecutor;

    impl ToolExecutor for FailingExecutor {
        fn execute(&self, tool_name: &str, _args: Value) -> Result<Observation, AgentError> {
            Err(AgentError::ToolExecution {
                tool: tool_name.to_string(),
                reason: "intentional failure".to_string(),
            })
        }

        fn available_tools(&self) -> Vec<ToolSpec> {
            vec![ToolSpec::new("fail_tool", "Always fails", json!({}))]
        }
    }

    #[test]
    fn test_single_step_execution() {
        let config = AgentConfig {
            max_steps: 10,
            planning_enabled: true,
            ..Default::default()
        };

        let planner = Box::new(|goal: &Goal, _tools: &[ToolSpec]| {
            let step = Step::new(&goal.description).with_tool_hint("echo");
            Ok(Plan::new(&goal.id, vec![step]))
        });

        let agent = Agent::new(config, Box::new(EchoExecutor)).with_planner(planner);

        let goal = Goal::new("Say hello");
        let trace = agent.run(goal).expect("run should succeed");

        assert!(trace.success);
        assert_eq!(trace.total_steps, 1);
        assert!(trace.final_result.is_some());
        assert_eq!(trace.entries.len(), 1);
    }

    #[test]
    fn test_multi_step_execution() {
        let config = AgentConfig {
            max_steps: 10,
            planning_enabled: true,
            ..Default::default()
        };

        let planner = Box::new(|goal: &Goal, _tools: &[ToolSpec]| {
            let steps = vec![
                Step::new("Step 1: gather data").with_tool_hint("echo"),
                Step::new("Step 2: process").with_tool_hint("echo"),
                Step::new("Step 3: summarize").with_tool_hint("echo"),
            ];
            Ok(Plan::new(&goal.id, steps))
        });

        let agent = Agent::new(config, Box::new(EchoExecutor)).with_planner(planner);

        let goal = Goal::new("Multi-step task");
        let trace = agent.run(goal).expect("run should succeed");

        assert!(trace.success);
        assert_eq!(trace.total_steps, 3);
        assert_eq!(trace.entries.len(), 3);
    }

    #[test]
    fn test_max_steps_exceeded() {
        let config = AgentConfig {
            max_steps: 2,
            planning_enabled: true,
            ..Default::default()
        };

        let planner = Box::new(|goal: &Goal, _tools: &[ToolSpec]| {
            let steps = vec![
                Step::new("Step 1").with_tool_hint("echo"),
                Step::new("Step 2").with_tool_hint("echo"),
                Step::new("Step 3").with_tool_hint("echo"),
            ];
            Ok(Plan::new(&goal.id, steps))
        });

        let agent = Agent::new(config, Box::new(EchoExecutor)).with_planner(planner);

        let goal = Goal::new("Too many steps");
        let result = agent.run(goal);

        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::MaxStepsExceeded(n) => assert_eq!(n, 2),
            other => panic!("expected MaxStepsExceeded, got: {:?}", other),
        }
    }

    #[test]
    fn test_tool_execution_failure_handling() {
        let config = AgentConfig {
            max_steps: 10,
            planning_enabled: true,
            ..Default::default()
        };

        let planner = Box::new(|goal: &Goal, _tools: &[ToolSpec]| {
            let step = Step::new("Do something").with_tool_hint("fail_tool");
            Ok(Plan::new(&goal.id, vec![step]))
        });

        // Use a custom reasoner that always calls the tool.
        let reasoner = Box::new(
            |state: &crate::types::AgentState, _obs: &Observation| -> Result<Action, AgentError> {
                let step = &state.plan.steps[state.step_index];
                Ok(Action::CallTool {
                    name: step.tool_hint.clone().unwrap_or_else(|| "fail_tool".into()),
                    args: json!({}),
                })
            },
        );

        let agent = Agent::new(config, Box::new(FailingExecutor))
            .with_planner(planner)
            .with_reasoner(reasoner);

        let goal = Goal::new("This will fail");
        let trace = agent.run(goal).expect("run should complete even on tool failure");

        // The run completes but the step is marked failed.
        assert_eq!(trace.total_steps, 1);
        assert_eq!(trace.entries.len(), 1);

        let entry = &trace.entries[0];
        let obs = entry.observation.as_ref().expect("should have observation");
        assert!(!obs.success);
        assert!(obs.error_message.is_some());
    }
}
