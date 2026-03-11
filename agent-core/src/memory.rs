use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Scratchpad memory available during execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkingMemory {
    store: HashMap<String, Value>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.store.get(key)
    }

    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.store.insert(key.into(), value);
    }

    /// Append a value to an array stored at `key`. Creates the array if it doesn't exist.
    pub fn append(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        let arr = self
            .store
            .entry(key)
            .or_insert_with(|| Value::Array(vec![]));
        if let Value::Array(ref mut vec) = arr {
            vec.push(value);
        }
    }

    pub fn clear(&mut self) {
        self.store.clear();
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.store.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}
