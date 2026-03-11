use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AgentError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Role of a chat message participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
        }
    }

    pub fn tool(content: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: Some(name.into()),
        }
    }
}

/// Configuration for an LLM request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model: String,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub json_mode: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: None,
            stop_sequences: Vec::new(),
            json_mode: false,
        }
    }
}

/// Description of a tool that can be provided to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// Trait for LLM backends. Implementations must be thread-safe.
pub trait LlmProvider: Send + Sync {
    fn chat(&self, messages: &[ChatMessage], config: &LlmConfig) -> Result<String, AgentError>;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// HttpLlmProvider — OpenAI-compatible HTTP backend
// ---------------------------------------------------------------------------

/// Generic HTTP provider that works with any OpenAI-compatible API.
pub struct HttpLlmProvider {
    base_url: String,
    api_key: String,
    default_model: String,
    extra_headers: Vec<(String, String)>,
}

impl HttpLlmProvider {
    pub fn new(base_url: &str, api_key: &str, default_model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            default_model: default_model.to_string(),
            extra_headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((key.into(), value.into()));
        self
    }
}

impl LlmProvider for HttpLlmProvider {
    fn chat(&self, messages: &[ChatMessage], config: &LlmConfig) -> Result<String, AgentError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let model = if config.model.is_empty() {
            &self.default_model
        } else {
            &config.model
        };

        let mut body = serde_json::json!({
            "model": model,
            "temperature": config.temperature,
            "messages": messages,
        });

        if let Some(max_tokens) = config.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if !config.stop_sequences.is_empty() {
            body["stop"] = serde_json::json!(config.stop_sequences);
        }

        if config.json_mode {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        }

        let mut request = ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.api_key));

        for (k, v) in &self.extra_headers {
            request = request.set(k, v);
        }

        let response = request.send_json(&body).map_err(|e| {
            AgentError::Reasoning(format!("HTTP request failed: {}", e))
        })?;

        let resp_body: Value = response.into_json().map_err(|e| {
            AgentError::Reasoning(format!("Failed to parse response JSON: {}", e))
        })?;

        // Extract content from OpenAI-compatible response format.
        resp_body["choices"]
            .get(0)
            .and_then(|c| c["message"]["content"].as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                AgentError::Reasoning(format!(
                    "Unexpected response format: {}",
                    resp_body
                ))
            })
    }

    fn name(&self) -> &str {
        "http"
    }
}

// ---------------------------------------------------------------------------
// OllamaProvider — Optimized for local Ollama
// ---------------------------------------------------------------------------

/// Provider for local Ollama instances using the native API.
pub struct OllamaProvider {
    host: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(host: &str, model: &str) -> Self {
        Self {
            host: host.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    pub fn default_local(model: &str) -> Self {
        Self::new("http://localhost:11434", model)
    }
}

impl LlmProvider for OllamaProvider {
    fn chat(&self, messages: &[ChatMessage], config: &LlmConfig) -> Result<String, AgentError> {
        let url = format!("{}/api/chat", self.host);

        let model = if config.model.is_empty() {
            &self.model
        } else {
            &config.model
        };

        // Build Ollama native message format.
        let ollama_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": ollama_messages,
            "stream": false,
            "options": {
                "temperature": config.temperature,
            },
        });

        if let Some(max_tokens) = config.max_tokens {
            body["options"]["num_predict"] = serde_json::json!(max_tokens);
        }

        if !config.stop_sequences.is_empty() {
            body["options"]["stop"] = serde_json::json!(config.stop_sequences);
        }

        if config.json_mode {
            body["format"] = serde_json::json!("json");
        }

        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| {
                AgentError::Reasoning(format!("Ollama request failed: {}", e))
            })?;

        let resp_body: Value = response.into_json().map_err(|e| {
            AgentError::Reasoning(format!("Failed to parse Ollama response: {}", e))
        })?;

        resp_body["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                AgentError::Reasoning(format!(
                    "Unexpected Ollama response format: {}",
                    resp_body
                ))
            })
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
