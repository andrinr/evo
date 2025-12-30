//! Proprioception sense - organism's awareness of its own state.
//!
//! Provides information about the organism's internal state such as
//! memory and energy levels.

use ndarray::Array1;

use super::super::ecosystem::Ecosystem;
use super::super::params::Params;
use super::Organism;
use super::sense::Sense;

/// Proprioception sense for internal state awareness.
///
/// Outputs:
/// - Memory state (all memory cells)
/// - Energy level (normalized)
pub struct Proprioception;

impl Proprioception {
    /// Creates a new proprioception sense.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Proprioception {
    fn default() -> Self {
        Self::new()
    }
}

impl Sense for Proprioception {
    fn sense(
        &self,
        organism: &Organism,
        _ecosystem: &Ecosystem,
        _params: &Params,
        _trees: Option<&super::super::ecosystem::SpatialTrees>,
    ) -> Array1<f32> {
        let memory_size = organism.memory.len();
        let mut proprio_outputs = Array1::zeros(memory_size + 1);

        // Copy memory state
        for (i, &mem_val) in organism.memory.iter().enumerate() {
            proprio_outputs[i] = mem_val;
        }

        // Add energy level (already normalized)
        proprio_outputs[memory_size] = organism.energy;

        proprio_outputs
    }

    fn input_size(&self, params: &Params) -> usize {
        // memory_size + 1 for energy
        params.memory_size + 1
    }

    fn name(&self) -> &'static str {
        "Proprioception"
    }
}
