use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AgentError;
use crate::types::Goal;

// ---------------------------------------------------------------------------
// Trigger
// ---------------------------------------------------------------------------

/// Defines when a scheduled task should fire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trigger {
    /// Cron-like schedule (stored as string, parsed by an external crate).
    Cron { expression: String },
    /// Fire every N seconds.
    Interval { seconds: u64 },
    /// Fire once at a specific instant.
    Once { at: DateTime<Utc> },
    /// Fire when a named event is emitted.
    Event { name: String },
}

// ---------------------------------------------------------------------------
// ScheduledTask
// ---------------------------------------------------------------------------

/// A task registered with the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: Uuid,
    pub name: String,
    pub trigger: Trigger,
    pub goal: Goal,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: Option<DateTime<Utc>>,
    pub run_count: u64,
}

// ---------------------------------------------------------------------------
// Notification types
// ---------------------------------------------------------------------------

/// Priority levels for notifications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Urgent,
}

/// A notification emitted by the scheduler or a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: Uuid,
    pub title: String,
    pub body: String,
    pub priority: Priority,
    pub source_task_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

impl Notification {
    pub fn new(title: impl Into<String>, body: impl Into<String>, priority: Priority) -> Self {
        Self {
            id: Uuid::new_v4(),
            title: title.into(),
            body: body.into(),
            priority,
            source_task_id: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_source_task(mut self, task_id: Uuid) -> Self {
        self.source_task_id = Some(task_id);
        self
    }
}

// ---------------------------------------------------------------------------
// NotificationSink
// ---------------------------------------------------------------------------

/// Trait for consuming notifications (logging, email, webhook, etc.).
pub trait NotificationSink: Send + Sync {
    fn send(&self, notification: &Notification) -> Result<(), AgentError>;
}

/// Default sink that writes to `log::info!`.
pub struct LogNotificationSink;

impl NotificationSink for LogNotificationSink {
    fn send(&self, notification: &Notification) -> Result<(), AgentError> {
        log::info!(
            "[{:?}] {}: {}",
            notification.priority,
            notification.title,
            notification.body
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the next run time for an interval trigger.
pub fn next_run_from_interval(last: DateTime<Utc>, seconds: u64) -> DateTime<Utc> {
    last + chrono::Duration::seconds(seconds as i64)
}

/// Compute the initial `next_run` for a trigger at creation time.
fn initial_next_run(trigger: &Trigger, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    match trigger {
        Trigger::Interval { seconds } => Some(now + chrono::Duration::seconds(*seconds as i64)),
        Trigger::Once { at } => Some(*at),
        // Cron parsing deferred to an external crate; start with None.
        Trigger::Cron { .. } => None,
        // Event-based tasks have no time-based schedule.
        Trigger::Event { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Manages a set of scheduled tasks and determines which are due.
pub struct Scheduler {
    tasks: Vec<ScheduledTask>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Register a new task. Returns its unique id.
    pub fn add_task(&mut self, name: &str, trigger: Trigger, goal: Goal) -> Uuid {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let next_run = initial_next_run(&trigger, now);
        self.tasks.push(ScheduledTask {
            id,
            name: name.to_string(),
            trigger,
            goal,
            enabled: true,
            last_run: None,
            next_run,
            run_count: 0,
        });
        id
    }

    /// Remove a task by id. Returns `true` if found.
    pub fn remove_task(&mut self, id: Uuid) -> bool {
        let len_before = self.tasks.len();
        self.tasks.retain(|t| t.id != id);
        self.tasks.len() < len_before
    }

    /// Enable or disable a task.
    pub fn enable_task(&mut self, id: Uuid, enabled: bool) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.enabled = enabled;
        }
    }

    /// List all registered tasks.
    pub fn list_tasks(&self) -> &[ScheduledTask] {
        &self.tasks
    }

    /// Return references to tasks whose `next_run` is at or before `now` and
    /// that are enabled.
    pub fn due_tasks(&self) -> Vec<&ScheduledTask> {
        let now = Utc::now();
        self.tasks
            .iter()
            .filter(|t| t.enabled)
            .filter(|t| matches!(t.next_run, Some(nr) if nr <= now))
            .collect()
    }

    /// Mark a task as completed: update `last_run`, advance `next_run`, and
    /// increment `run_count`.
    pub fn mark_completed(&mut self, id: Uuid) {
        let now = Utc::now();
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.last_run = Some(now);
            task.run_count += 1;
            task.next_run = match &task.trigger {
                Trigger::Interval { seconds } => Some(next_run_from_interval(now, *seconds)),
                Trigger::Once { .. } => None, // one-shot, no next run
                Trigger::Cron { .. } => None,  // deferred to external cron parser
                Trigger::Event { .. } => None, // event-driven, no time-based next
            };
        }
    }

    /// Return all enabled tasks whose trigger is `Event { name }` matching the
    /// given event name.
    pub fn fire_event(&self, event_name: &str) -> Vec<&ScheduledTask> {
        self.tasks
            .iter()
            .filter(|t| t.enabled)
            .filter(|t| matches!(&t.trigger, Trigger::Event { name } if name == event_name))
            .collect()
    }

    /// Convenience: check all tasks and return those that are due (alias for
    /// `due_tasks`).
    pub fn tick(&mut self) -> Vec<&ScheduledTask> {
        self.due_tasks()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(desc: &str) -> Goal {
        Goal::new(desc)
    }

    // -- add / remove -------------------------------------------------------

    #[test]
    fn test_add_and_remove_task() {
        let mut sched = Scheduler::new();
        let id = sched.add_task(
            "heartbeat",
            Trigger::Interval { seconds: 60 },
            make_goal("check health"),
        );

        assert_eq!(sched.list_tasks().len(), 1);
        assert_eq!(sched.list_tasks()[0].name, "heartbeat");

        assert!(sched.remove_task(id));
        assert!(sched.list_tasks().is_empty());

        // Removing again returns false.
        assert!(!sched.remove_task(id));
    }

    // -- interval due detection ---------------------------------------------

    #[test]
    fn test_interval_due_tasks() {
        let mut sched = Scheduler::new();

        // Add a task with a next_run in the past so it's immediately due.
        let id = sched.add_task(
            "poll",
            Trigger::Interval { seconds: 10 },
            make_goal("poll endpoint"),
        );

        // Force next_run into the past.
        sched.tasks.iter_mut().find(|t| t.id == id).unwrap().next_run =
            Some(Utc::now() - chrono::Duration::seconds(5));

        let due = sched.due_tasks();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, id);
    }

    // -- mark_completed updates last_run and next_run -----------------------

    #[test]
    fn test_mark_completed_updates_fields() {
        let mut sched = Scheduler::new();
        let id = sched.add_task(
            "sync",
            Trigger::Interval { seconds: 300 },
            make_goal("sync data"),
        );

        assert!(sched.list_tasks()[0].last_run.is_none());
        assert_eq!(sched.list_tasks()[0].run_count, 0);

        sched.mark_completed(id);

        let task = &sched.list_tasks()[0];
        assert!(task.last_run.is_some());
        assert!(task.next_run.is_some());
        assert_eq!(task.run_count, 1);

        // next_run should be ~300s after last_run.
        let diff = task.next_run.unwrap() - task.last_run.unwrap();
        assert_eq!(diff.num_seconds(), 300);
    }

    // -- event-based triggering ---------------------------------------------

    #[test]
    fn test_fire_event() {
        let mut sched = Scheduler::new();
        sched.add_task(
            "on_deploy",
            Trigger::Event {
                name: "deploy".into(),
            },
            make_goal("run smoke tests"),
        );
        sched.add_task(
            "heartbeat",
            Trigger::Interval { seconds: 60 },
            make_goal("ping"),
        );

        let matched = sched.fire_event("deploy");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "on_deploy");

        // Unknown event returns nothing.
        assert!(sched.fire_event("unknown").is_empty());
    }

    // -- enable / disable ---------------------------------------------------

    #[test]
    fn test_enable_disable_task() {
        let mut sched = Scheduler::new();
        let id = sched.add_task(
            "poll",
            Trigger::Interval { seconds: 10 },
            make_goal("poll"),
        );

        // Force due.
        sched.tasks.iter_mut().find(|t| t.id == id).unwrap().next_run =
            Some(Utc::now() - chrono::Duration::seconds(1));

        assert_eq!(sched.due_tasks().len(), 1);

        sched.enable_task(id, false);
        assert!(sched.due_tasks().is_empty());

        sched.enable_task(id, true);
        assert_eq!(sched.due_tasks().len(), 1);
    }

    // -- once trigger -------------------------------------------------------

    #[test]
    fn test_once_trigger_no_next_run_after_complete() {
        let mut sched = Scheduler::new();
        let id = sched.add_task(
            "one-shot",
            Trigger::Once {
                at: Utc::now() - chrono::Duration::seconds(1),
            },
            make_goal("run once"),
        );

        // Should be due.
        assert_eq!(sched.due_tasks().len(), 1);

        sched.mark_completed(id);

        let task = &sched.list_tasks()[0];
        assert!(task.next_run.is_none());
        assert_eq!(task.run_count, 1);
    }

    // -- notification / sink ------------------------------------------------

    #[test]
    fn test_log_notification_sink() {
        let sink = LogNotificationSink;
        let n = Notification::new("test", "body", Priority::Normal);
        assert!(sink.send(&n).is_ok());
    }

    // -- tick convenience ---------------------------------------------------

    #[test]
    fn test_tick_returns_due_tasks() {
        let mut sched = Scheduler::new();
        let id = sched.add_task(
            "t",
            Trigger::Interval { seconds: 1 },
            make_goal("g"),
        );
        sched.tasks.iter_mut().find(|t| t.id == id).unwrap().next_run =
            Some(Utc::now() - chrono::Duration::seconds(2));

        let due = sched.tick();
        assert_eq!(due.len(), 1);
    }
}
