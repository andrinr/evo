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
/// - Rotation (sin and cos components for continuous encoding)
/// - Position encoding (sin and cos of normalized x and y coordinates)
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
        params: &Params,
        _trees: Option<&super::super::ecosystem::SpatialTrees>,
    ) -> Array1<f32> {
        let memory_size = organism.memory.len();
        // memory + energy + rotation(sin,cos) + position(sin_x, cos_x, sin_y, cos_y) = memory_size + 7
        let mut proprio_outputs = Array1::zeros(memory_size + 7);

        let mut idx = 0;

        // Copy memory state
        for &mem_val in organism.memory.iter() {
            proprio_outputs[idx] = mem_val;
            idx += 1;
        }

        // Add energy level (already normalized)
        proprio_outputs[idx] = organism.energy;
        idx += 1;

        // Add rotation awareness (sin and cos for continuous encoding)
        proprio_outputs[idx] = organism.rot.sin();
        idx += 1;
        proprio_outputs[idx] = organism.rot.cos();
        idx += 1;

        // Add positional encoding using sine and cosine
        // Normalize position to [0, 2π] range for periodic encoding
        let norm_x = (organism.pos[0] / params.box_width) * 2.0 * std::f32::consts::PI;
        let norm_y = (organism.pos[1] / params.box_height) * 2.0 * std::f32::consts::PI;

        proprio_outputs[idx] = norm_x.sin();
        idx += 1;
        proprio_outputs[idx] = norm_x.cos();
        idx += 1;
        proprio_outputs[idx] = norm_y.sin();
        idx += 1;
        proprio_outputs[idx] = norm_y.cos();

        proprio_outputs
    }

    fn input_size(&self, params: &Params) -> usize {
        // memory_size + energy + rotation(sin,cos) + position(sin_x, cos_x, sin_y, cos_y)
        params.memory_size + 7
    }

    fn name(&self) -> &'static str {
        "Proprioception"
    }
}
