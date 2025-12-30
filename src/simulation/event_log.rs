//! Event logging system for displaying recent simulation events.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A logged event for display in the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggedEvent {
    /// Timestamp when the event occurred
    pub time: f32,
    /// Human-readable description of the event
    pub description: String,
    /// Color hint for the event (for UI display)
    pub color: EventColor,
}

/// Color categories for events
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EventColor {
    /// Reproduction events (green)
    Reproduction,
    /// Combat/attack events (red)
    Combat,
    /// Energy sharing (blue)
    Sharing,
    /// Death events (gray)
    Death,
    /// Food consumption (yellow)
    Food,
}

/// Event log that tracks recent simulation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    /// Recent events, newest first
    events: VecDeque<LoggedEvent>,
    /// Maximum number of events to keep
    max_events: usize,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new(20)
    }
}

impl EventLog {
    /// Creates a new event log with specified capacity
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
        }
    }

    /// Adds a new event to the log
    pub fn log(&mut self, time: f32, description: String, color: EventColor) {
        self.events.push_front(LoggedEvent {
            time,
            description,
            color,
        });

        // Keep only the most recent events
        while self.events.len() > self.max_events {
            self.events.pop_back();
        }
    }

    /// Returns all events, newest first
    pub fn events(&self) -> &VecDeque<LoggedEvent> {
        &self.events
    }

    /// Clears all events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}
