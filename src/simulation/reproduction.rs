use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::organism;

/// Statistics tracking reproduction strategy effectiveness based on organism deaths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionStats {
    /// Recent asexual deaths: (`child_score` - `parent_score`)
    pub asexual_deltas: VecDeque<f64>,
    /// Recent sexual deaths: (`child_score` - `avg_parent_score`)
    pub sexual_deltas: VecDeque<f64>,
    /// Recent inter-pool deaths: (`child_score` - `avg_parent_score`)
    pub interpool_deltas: VecDeque<f64>,
    /// Maximum number of recent deaths to track
    pub max_history: usize,
}

impl Default for ReproductionStats {
    fn default() -> Self {
        Self {
            asexual_deltas: VecDeque::new(),
            sexual_deltas: VecDeque::new(),
            interpool_deltas: VecDeque::new(),
            max_history: 100, // Track last 100 deaths of each type
        }
    }
}

impl ReproductionStats {
    /// Record a death and update statistics
    pub fn record_death(&mut self, organism: &organism::Organism) {
        // Skip organisms that never reproduced or died too quickly
        if organism.reproduction_method == 0 || organism.age < 0.5 {
            return;
        }

        let score_delta = organism.score as f64 - organism.parent_avg_score;

        match organism.reproduction_method {
            1 => {
                // Asexual
                self.asexual_deltas.push_back(score_delta);
                if self.asexual_deltas.len() > self.max_history {
                    self.asexual_deltas.pop_front();
                }
            }
            2 => {
                // Sexual same-pool
                self.sexual_deltas.push_back(score_delta);
                if self.sexual_deltas.len() > self.max_history {
                    self.sexual_deltas.pop_front();
                }
            }
            3 => {
                // Sexual inter-pool
                self.sexual_deltas.push_back(score_delta);
                if self.sexual_deltas.len() > self.max_history {
                    self.sexual_deltas.pop_front();
                }

                self.interpool_deltas.push_back(score_delta);
                if self.interpool_deltas.len() > self.max_history {
                    self.interpool_deltas.pop_front();
                }
            }
            _ => {}
        }
    }

    /// Average score improvement for asexual reproduction (last N deaths)
    pub fn avg_asexual_delta(&self) -> f64 {
        if self.asexual_deltas.is_empty() {
            0.0
        } else {
            self.asexual_deltas.iter().sum::<f64>() / self.asexual_deltas.len() as f64
        }
    }

    /// Average score improvement for sexual reproduction (last N deaths)
    pub fn avg_sexual_delta(&self) -> f64 {
        if self.sexual_deltas.is_empty() {
            0.0
        } else {
            self.sexual_deltas.iter().sum::<f64>() / self.sexual_deltas.len() as f64
        }
    }

    /// Average score improvement for inter-pool breeding (last N deaths)
    pub fn avg_interpool_delta(&self) -> f64 {
        if self.interpool_deltas.is_empty() {
            0.0
        } else {
            self.interpool_deltas.iter().sum::<f64>() / self.interpool_deltas.len() as f64
        }
    }

    /// Number of asexual deaths tracked
    pub fn asexual_count(&self) -> usize {
        self.asexual_deltas.len()
    }

    /// Number of sexual deaths tracked
    pub fn sexual_count(&self) -> usize {
        self.sexual_deltas.len()
    }

    /// Number of inter-pool deaths tracked
    pub fn interpool_count(&self) -> usize {
        self.interpool_deltas.len()
    }
}
