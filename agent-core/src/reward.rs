//! Generalized reward engine for strategy evolution.
//!
//! Core concept: Agent takes actions -> gets outcomes -> outcomes are scored ->
//! scores update strategy weights -> future actions improve.
//!
//! The scoring logic is pluggable via the [`RewardScorer`] trait, so callers
//! can define domain-specific reward functions (engagement metrics, task
//! completion rates, latency, etc.) without changing the engine.
//!
//! Weight updates use an exponential moving average (EMA):
//!   `new_weight = alpha * score + (1 - alpha) * old_weight`
//!
//! Strategy selection supports three modes:
//! - Weighted random (higher weight = more likely)
//! - Greedy (always pick highest weight — pure exploitation)
//! - Epsilon-greedy (explore with probability epsilon, exploit otherwise)

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An observed outcome from an agent action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Unique identifier for the task that produced this outcome.
    pub task_id: Uuid,
    /// Name of the action/strategy that was used.
    pub action_name: String,
    /// Arbitrary numeric metrics (domain-specific).
    pub metrics: HashMap<String, f64>,
    /// When the outcome was observed.
    pub timestamp: DateTime<Utc>,
}

/// Configuration for the reward engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// EMA smoothing factor. Higher = new scores have more influence.
    /// Range: 0.0..=1.0. Default: 0.3.
    pub ema_alpha: f32,
    /// Minimum number of outcomes before strategy weights are updated.
    /// Default: 3.
    pub min_samples_for_update: usize,
    /// Multiplicative decay applied to all weights on each `decay_all()` call.
    /// Range: 0.0..=1.0. Default: 0.95.
    pub decay_factor: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.3,
            min_samples_for_update: 3,
            decay_factor: 0.95,
        }
    }
}

/// A tracked strategy with its current weight and usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyWeight {
    pub name: String,
    pub weight: f32,
    pub sample_count: u64,
    pub last_updated: DateTime<Utc>,
}

/// A reward signal derived from scoring an outcome.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RewardSignal {
    /// Score in the range 0.0..=1.0.
    pub score: f32,
    /// Confidence in the score (0.0..=1.0). Reserved for future use.
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Scorer trait + default implementation
// ---------------------------------------------------------------------------

/// Pluggable scoring interface. Implement this to define how outcomes
/// are converted into reward signals.
pub trait RewardScorer: Send + Sync {
    fn score(&self, outcome: &Outcome) -> RewardSignal;
}

/// Default scorer that computes the ratio of a "success" metric to a "total" metric.
///
/// If either metric is missing or total is zero, returns score 0.0.
#[derive(Debug, Clone)]
pub struct SimpleRatioScorer {
    pub success_key: String,
    pub total_key: String,
}

impl SimpleRatioScorer {
    pub fn new(success_key: impl Into<String>, total_key: impl Into<String>) -> Self {
        Self {
            success_key: success_key.into(),
            total_key: total_key.into(),
        }
    }
}

impl RewardScorer for SimpleRatioScorer {
    fn score(&self, outcome: &Outcome) -> RewardSignal {
        let success = outcome.metrics.get(&self.success_key).copied().unwrap_or(0.0);
        let total = outcome.metrics.get(&self.total_key).copied().unwrap_or(0.0);
        let score = if total > 0.0 {
            (success / total).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        RewardSignal {
            score,
            confidence: if total > 0.0 { 1.0 } else { 0.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// RewardEngine
// ---------------------------------------------------------------------------

/// The core reward engine that tracks strategy weights and updates them
/// based on observed outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEngine {
    config: RewardConfig,
    weights: HashMap<String, StrategyWeight>,
    history: Vec<Outcome>,
}

impl RewardEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: RewardConfig) -> Self {
        Self {
            config,
            weights: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Register a new strategy with an initial weight.
    pub fn register_strategy(&mut self, name: &str, initial_weight: f32) {
        self.weights.insert(
            name.to_string(),
            StrategyWeight {
                name: name.to_string(),
                weight: initial_weight,
                sample_count: 0,
                last_updated: Utc::now(),
            },
        );
    }

    /// Record an outcome and update the corresponding strategy's weight.
    ///
    /// The outcome is scored using the provided scorer. If the strategy
    /// has fewer samples than `min_samples_for_update`, the outcome is
    /// recorded but the weight is not updated yet.
    pub fn record_outcome(&mut self, outcome: Outcome, scorer: &dyn RewardScorer) {
        let signal = scorer.score(&outcome);
        let action_name = outcome.action_name.clone();
        self.history.push(outcome);

        if let Some(sw) = self.weights.get_mut(&action_name) {
            sw.sample_count += 1;
            sw.last_updated = Utc::now();

            // Only update weight once we have enough samples
            if sw.sample_count >= self.config.min_samples_for_update as u64 {
                let alpha = self.config.ema_alpha;
                sw.weight = alpha * signal.score + (1.0 - alpha) * sw.weight;
            }
        }
    }

    /// Select a strategy using weighted random sampling.
    /// Higher weight = more likely to be selected.
    /// Returns `None` if no strategies are registered.
    pub fn select_strategy(&self) -> Option<&str> {
        if self.weights.is_empty() {
            return None;
        }

        let total: f32 = self.weights.values().map(|sw| sw.weight.max(0.0)).sum();
        if total <= 0.0 {
            // All weights are zero or negative — pick uniformly
            let keys: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
            let idx = simple_hash_index(keys.len());
            return Some(keys[idx]);
        }

        let threshold = simple_random_f32() * total;
        let mut cumulative = 0.0_f32;
        for sw in self.weights.values() {
            cumulative += sw.weight.max(0.0);
            if cumulative >= threshold {
                return Some(&sw.name);
            }
        }

        // Fallback (floating-point edge case)
        self.weights.values().last().map(|sw| sw.name.as_str())
    }

    /// Always pick the strategy with the highest weight (pure exploitation).
    pub fn select_strategy_greedy(&self) -> Option<&str> {
        self.weights
            .values()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .map(|sw| sw.name.as_str())
    }

    /// Epsilon-greedy selection: with probability `epsilon`, pick a random
    /// strategy (exploration); otherwise pick the highest-weight strategy.
    pub fn select_strategy_epsilon_greedy(&self, epsilon: f32) -> Option<&str> {
        if self.weights.is_empty() {
            return None;
        }

        if simple_random_f32() < epsilon {
            // Explore: pick uniformly at random
            let keys: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
            let idx = simple_hash_index(keys.len());
            Some(keys[idx])
        } else {
            // Exploit: greedy
            self.select_strategy_greedy()
        }
    }

    /// Get a reference to all strategy weights.
    pub fn get_weights(&self) -> &HashMap<String, StrategyWeight> {
        &self.weights
    }

    /// Apply the decay factor to all strategy weights.
    pub fn decay_all(&mut self) {
        let factor = self.config.decay_factor;
        for sw in self.weights.values_mut() {
            sw.weight *= factor;
        }
    }

    /// Serialize engine state to a JSON value.
    pub fn save_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Restore engine state from a JSON value.
    pub fn load_state(&mut self, state: &serde_json::Value) {
        if let Ok(engine) = serde_json::from_value::<RewardEngine>(state.clone()) {
            self.config = engine.config;
            self.weights = engine.weights;
            self.history = engine.history;
        }
    }
}

// ---------------------------------------------------------------------------
// Simple deterministic "random" helpers (no external rand crate needed)
//
// For production use, callers should use `select_strategy_greedy` or
// bring their own RNG. These helpers use a thread-local counter + time
// to produce pseudo-random-ish values sufficient for basic exploration.
// ---------------------------------------------------------------------------

fn simple_random_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Simple xorshift-style mixing
    let mut x = c.wrapping_add(now);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    // Map to [0, 1)
    (x % 10000) as f32 / 10000.0
}

fn simple_hash_index(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let r = simple_random_f32();
    ((r * len as f32) as usize).min(len - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A test scorer that always returns a fixed score.
    struct FixedScorer(f32);

    impl RewardScorer for FixedScorer {
        fn score(&self, _outcome: &Outcome) -> RewardSignal {
            RewardSignal {
                score: self.0,
                confidence: 1.0,
            }
        }
    }

    fn make_outcome(action: &str) -> Outcome {
        Outcome {
            task_id: Uuid::new_v4(),
            action_name: action.to_string(),
            metrics: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_register_and_ema_update() {
        let config = RewardConfig {
            ema_alpha: 0.3,
            min_samples_for_update: 3,
            decay_factor: 0.95,
        };
        let mut engine = RewardEngine::new(config);
        engine.register_strategy("alpha", 0.5);

        let scorer = FixedScorer(1.0);

        // Record 2 outcomes — below min_samples_for_update, weight should not change
        engine.record_outcome(make_outcome("alpha"), &scorer);
        engine.record_outcome(make_outcome("alpha"), &scorer);
        assert_eq!(engine.get_weights()["alpha"].sample_count, 2);
        assert!(
            (engine.get_weights()["alpha"].weight - 0.5).abs() < f32::EPSILON,
            "Weight should not change before min_samples_for_update"
        );

        // 3rd outcome triggers the first EMA update
        engine.record_outcome(make_outcome("alpha"), &scorer);
        // new_weight = 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        let w = engine.get_weights()["alpha"].weight;
        assert!((w - 0.65).abs() < 1e-5, "EMA update incorrect: got {w}");

        // 4th outcome: another EMA step
        engine.record_outcome(make_outcome("alpha"), &scorer);
        // new_weight = 0.3 * 1.0 + 0.7 * 0.65 = 0.755
        let w = engine.get_weights()["alpha"].weight;
        assert!((w - 0.755).abs() < 1e-4, "Second EMA update incorrect: got {w}");
    }

    #[test]
    fn test_select_strategy_greedy() {
        let mut engine = RewardEngine::new(RewardConfig::default());
        engine.register_strategy("low", 0.2);
        engine.register_strategy("high", 0.9);
        engine.register_strategy("mid", 0.5);

        let pick = engine.select_strategy_greedy().unwrap();
        assert_eq!(pick, "high");
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        let mut engine = RewardEngine::new(RewardConfig::default());
        engine.register_strategy("only_one", 1.0);

        // With epsilon=1.0 (always explore) we should still get a valid strategy
        for _ in 0..10 {
            let pick = engine.select_strategy_epsilon_greedy(1.0).unwrap();
            assert_eq!(pick, "only_one"); // only one to pick
        }

        // With epsilon=0.0 (always exploit), always get the highest
        engine.register_strategy("low", 0.1);
        let pick = engine.select_strategy_epsilon_greedy(0.0).unwrap();
        assert_eq!(pick, "only_one");
    }

    #[test]
    fn test_decay_reduces_weights() {
        let config = RewardConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut engine = RewardEngine::new(config);
        engine.register_strategy("a", 1.0);
        engine.register_strategy("b", 0.8);

        engine.decay_all();

        let wa = engine.get_weights()["a"].weight;
        let wb = engine.get_weights()["b"].weight;
        assert!((wa - 0.5).abs() < 1e-5, "Decay incorrect for a: got {wa}");
        assert!((wb - 0.4).abs() < 1e-5, "Decay incorrect for b: got {wb}");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut engine = RewardEngine::new(RewardConfig::default());
        engine.register_strategy("x", 0.7);
        engine.register_strategy("y", 0.3);

        let scorer = FixedScorer(0.8);
        for _ in 0..5 {
            engine.record_outcome(make_outcome("x"), &scorer);
        }

        let state = engine.save_state();

        let mut restored = RewardEngine::new(RewardConfig::default());
        restored.load_state(&state);

        assert_eq!(restored.get_weights().len(), 2);
        assert_eq!(
            restored.get_weights()["x"].sample_count,
            engine.get_weights()["x"].sample_count
        );
        assert!(
            (restored.get_weights()["x"].weight - engine.get_weights()["x"].weight).abs() < 1e-5
        );
        assert_eq!(restored.history.len(), engine.history.len());
    }

    #[test]
    fn test_min_samples_for_update_respected() {
        let config = RewardConfig {
            ema_alpha: 0.5,
            min_samples_for_update: 5,
            decay_factor: 0.95,
        };
        let mut engine = RewardEngine::new(config);
        engine.register_strategy("s", 0.6);

        let scorer = FixedScorer(1.0);

        // Record 4 outcomes — all below min_samples_for_update=5
        for _ in 0..4 {
            engine.record_outcome(make_outcome("s"), &scorer);
        }
        assert_eq!(engine.get_weights()["s"].sample_count, 4);
        assert!(
            (engine.get_weights()["s"].weight - 0.6).abs() < f32::EPSILON,
            "Weight should remain unchanged until min_samples reached"
        );

        // 5th outcome triggers the update
        engine.record_outcome(make_outcome("s"), &scorer);
        // new_weight = 0.5 * 1.0 + 0.5 * 0.6 = 0.8
        let w = engine.get_weights()["s"].weight;
        assert!((w - 0.8).abs() < 1e-5, "Weight should update at min_samples: got {w}");
    }

    #[test]
    fn test_simple_ratio_scorer() {
        let scorer = SimpleRatioScorer::new("successes", "attempts");
        let mut metrics = HashMap::new();
        metrics.insert("successes".to_string(), 7.0);
        metrics.insert("attempts".to_string(), 10.0);

        let outcome = Outcome {
            task_id: Uuid::new_v4(),
            action_name: "test".to_string(),
            metrics,
            timestamp: Utc::now(),
        };

        let signal = scorer.score(&outcome);
        assert!((signal.score - 0.7).abs() < 1e-5);
        assert!((signal.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simple_ratio_scorer_zero_total() {
        let scorer = SimpleRatioScorer::new("successes", "attempts");
        let mut metrics = HashMap::new();
        metrics.insert("successes".to_string(), 5.0);
        metrics.insert("attempts".to_string(), 0.0);

        let outcome = Outcome {
            task_id: Uuid::new_v4(),
            action_name: "test".to_string(),
            metrics,
            timestamp: Utc::now(),
        };

        let signal = scorer.score(&outcome);
        assert!(signal.score.abs() < 1e-5, "Zero total should give zero score");
        assert!(signal.confidence.abs() < 1e-5, "Zero total should give zero confidence");
    }

    #[test]
    fn test_weighted_random_selection_nonzero() {
        // With only one strategy, select_strategy must return it
        let mut engine = RewardEngine::new(RewardConfig::default());
        engine.register_strategy("only", 1.0);

        for _ in 0..20 {
            assert_eq!(engine.select_strategy().unwrap(), "only");
        }
    }

    #[test]
    fn test_select_strategy_empty() {
        let engine = RewardEngine::new(RewardConfig::default());
        assert!(engine.select_strategy().is_none());
        assert!(engine.select_strategy_greedy().is_none());
        assert!(engine.select_strategy_epsilon_greedy(0.5).is_none());
    }
}
