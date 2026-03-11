#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::error::AgentError;
    use crate::executor::{ToolExecutor, ToolSpec};
    use crate::llm::{ChatMessage, LlmConfig, LlmProvider, Role};
    use crate::react::{parse_react_response, PlanAndExecuteAgent, ReActAgent};
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

    // ===================================================================
    // LLM types tests
    // ===================================================================

    #[test]
    fn test_chat_message_construction() {
        let sys = ChatMessage::system("You are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content, "You are helpful");
        assert!(sys.name.is_none());

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, Role::User);

        let asst = ChatMessage::assistant("Hi there");
        assert_eq!(asst.role, Role::Assistant);

        let tool = ChatMessage::tool("result", "my_tool");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.name.as_deref(), Some("my_tool"));
    }

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::default();
        assert_eq!(config.model, "gpt-4");
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert!(config.max_tokens.is_none());
        assert!(config.stop_sequences.is_empty());
        assert!(!config.json_mode);
    }

    // ===================================================================
    // ReAct JSON parsing tests
    // ===================================================================

    #[test]
    fn test_react_parse_tool_call() {
        let response = r#"{"thought": "I need to search", "action": "search", "action_input": {"query": "rust"}}"#;
        let result = parse_react_response(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_react_parse_finish() {
        let response = r#"{"thought": "I have the answer", "action": "finish", "result": "42"}"#;
        let result = parse_react_response(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_react_parse_markdown_code_block() {
        let response = "```json\n{\"thought\": \"thinking\", \"action\": \"finish\", \"result\": \"done\"}\n```";
        let result = parse_react_response(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_react_parse_invalid_json() {
        let response = "this is not json";
        let result = parse_react_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_react_parse_missing_action() {
        let response = r#"{"thought": "hmm"}"#;
        let result = parse_react_response(response);
        assert!(result.is_err());
    }

    // ===================================================================
    // MockLlmProvider
    // ===================================================================

    /// A mock LLM provider that returns canned responses.
    struct MockLlmProvider {
        responses: Vec<String>,
        name: String,
    }

    impl MockLlmProvider {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses,
                name: "mock".to_string(),
            }
        }
    }

    impl LlmProvider for MockLlmProvider {
        fn chat(
            &self,
            messages: &[ChatMessage],
            _config: &LlmConfig,
        ) -> Result<String, AgentError> {
            // Count user messages to determine which response to return.
            // The first user message is the goal, subsequent ones are observations.
            let user_count = messages
                .iter()
                .filter(|m| m.role == Role::User)
                .count();

            // Index: first call = 0, second call = 1, etc.
            let idx = user_count.saturating_sub(1);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                // Default: finish.
                Ok(r#"{"thought": "done", "action": "finish", "result": "fallback"}"#.to_string())
            }
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // ===================================================================
    // ReAct loop tests with mocks
    // ===================================================================

    #[test]
    fn test_react_loop_immediate_finish() {
        let mock_llm = MockLlmProvider::new(vec![
            r#"{"thought": "I know the answer", "action": "finish", "result": "42"}"#.to_string(),
        ]);

        let config = AgentConfig {
            max_steps: 5,
            ..Default::default()
        };

        let agent = ReActAgent::new(
            Box::new(mock_llm),
            Box::new(EchoExecutor),
            config,
        );

        let trace = agent.run("What is 6*7?").expect("should succeed");
        assert!(trace.success);
        assert_eq!(trace.total_steps, 1);
        assert!(trace.final_result.is_some());
    }

    #[test]
    fn test_react_loop_tool_then_finish() {
        let mock_llm = MockLlmProvider::new(vec![
            r#"{"thought": "Let me search", "action": "echo", "action_input": {"query": "test"}}"#.to_string(),
            r#"{"thought": "Got it", "action": "finish", "result": "found the answer"}"#.to_string(),
        ]);

        let config = AgentConfig {
            max_steps: 5,
            ..Default::default()
        };

        let agent = ReActAgent::new(
            Box::new(mock_llm),
            Box::new(EchoExecutor),
            config,
        );

        let trace = agent.run("Find something").expect("should succeed");
        assert!(trace.success);
        assert_eq!(trace.total_steps, 2);
        assert_eq!(trace.entries.len(), 2);

        // First entry should be a tool call.
        match &trace.entries[0].action {
            crate::types::TraceAction::CallTool { name, .. } => {
                assert_eq!(name, "echo");
            }
            other => panic!("expected CallTool, got: {:?}", other),
        }
    }

    #[test]
    fn test_react_max_steps_exceeded() {
        // LLM always wants to call a tool, never finishes.
        let mock_llm = MockLlmProvider::new(vec![
            r#"{"thought": "search", "action": "echo", "action_input": {}}"#.to_string(),
            r#"{"thought": "search more", "action": "echo", "action_input": {}}"#.to_string(),
            r#"{"thought": "search again", "action": "echo", "action_input": {}}"#.to_string(),
            r#"{"thought": "still searching", "action": "echo", "action_input": {}}"#.to_string(),
        ]);

        let config = AgentConfig {
            max_steps: 2,
            ..Default::default()
        };

        let agent = ReActAgent::new(
            Box::new(mock_llm),
            Box::new(EchoExecutor),
            config,
        );

        let result = agent.run("Infinite task");
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::MaxStepsExceeded(n) => assert_eq!(n, 2),
            other => panic!("expected MaxStepsExceeded, got: {:?}", other),
        }
    }

    // ===================================================================
    // PlanAndExecute tests with mocks
    // ===================================================================

    #[test]
    fn test_plan_and_execute_basic() {
        // Planner returns a 2-step plan.
        let planner_llm = MockLlmProvider::new(vec![
            r#"["Step 1: gather data", "Step 2: summarize"]"#.to_string(),
        ]);

        // Executor LLM finishes immediately for each sub-step.
        let executor_llm = MockLlmProvider::new(vec![
            r#"{"thought": "done", "action": "finish", "result": "gathered"}"#.to_string(),
        ]);

        let config = AgentConfig {
            max_steps: 10,
            ..Default::default()
        };

        let react = ReActAgent::new(
            Box::new(executor_llm),
            Box::new(EchoExecutor),
            config,
        );

        let agent = PlanAndExecuteAgent::new(Box::new(planner_llm), react);

        let trace = agent.run("Analyze the data").expect("should succeed");
        assert!(trace.success);
        assert_eq!(trace.plan.steps.len(), 2);
    }
}
