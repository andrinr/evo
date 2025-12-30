//! Perception system that combines multiple senses into brain inputs.
//!
//! The perception system manages different sensory modalities and combines
//! their outputs into a single input vector for the organism's brain.

use ndarray::Array1;

use super::super::ecosystem::Ecosystem;
use super::super::params::Params;
use super::Organism;
use super::sense::Sense;

/// Manages multiple senses and combines them into brain inputs.
///
/// The perception system:
/// 1. Queries each sense for its outputs
/// 2. Concatenates all sensory outputs in order
/// 3. Returns a single input vector for the brain
pub struct Perception {
    /// Ordered list of senses that contribute to perception
    senses: Vec<Box<dyn Sense>>,
}

impl Perception {
    /// Creates a new perception system with the given senses.
    ///
    /// # Arguments
    ///
    /// * `senses` - Vector of boxed sense implementations
    ///
    /// # Returns
    ///
    /// A new perception system that will query senses in the order given.
    pub fn new(senses: Vec<Box<dyn Sense>>) -> Self {
        Self { senses }
    }

    /// Process all senses and return combined brain inputs.
    ///
    /// # Arguments
    ///
    /// * `organism` - The organism doing the sensing
    /// * `ecosystem` - The current state of the ecosystem
    /// * `params` - Simulation parameters
    /// * `trees` - Optional pre-built KD-trees for efficient spatial queries
    ///
    /// # Returns
    ///
    /// A 1D array containing all sensory activations concatenated in order.
    pub fn perceive(
        &self,
        organism: &Organism,
        ecosystem: &Ecosystem,
        params: &Params,
        trees: Option<&super::super::ecosystem::SpatialTrees>,
    ) -> Array1<f32> {
        let total_size = self.total_input_size(params);
        let mut combined_inputs = Array1::zeros(total_size);

        let mut offset = 0;
        for sense in &self.senses {
            let sense_outputs = sense.sense(organism, ecosystem, params, trees);
            let sense_size = sense.input_size(params);

            // Copy sense outputs into the combined array
            for (i, &value) in sense_outputs.iter().enumerate() {
                combined_inputs[offset + i] = value;
            }

            offset += sense_size;
        }

        combined_inputs
    }

    /// Returns the total number of brain inputs produced by all senses.
    ///
    /// # Arguments
    ///
    /// * `params` - Simulation parameters
    ///
    /// # Returns
    ///
    /// The sum of input sizes from all senses.
    pub fn total_input_size(&self, params: &Params) -> usize {
        self.senses.iter().map(|s| s.input_size(params)).sum()
    }

    /// Returns a reference to the senses in this perception system.
    pub fn senses(&self) -> &[Box<dyn Sense>] {
        &self.senses
    }
}

impl Default for Perception {
    fn default() -> Self {
        // Default perception includes all available senses
        use super::proprioception::Proprioception;
        use super::scent::Scent;
        use super::vision::Vision;

        Self::new(vec![
            Box::new(Vision::new()),
            Box::new(Scent::new()),
            Box::new(Proprioception::new()),
        ])
    }
}
