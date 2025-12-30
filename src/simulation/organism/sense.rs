//! Abstract sense trait and implementations for organism perception.
//!
//! Senses collect information from the environment and convert it into
//! neural network inputs for the organism's brain.

use ndarray::Array1;

use super::super::ecosystem::{Ecosystem, SpatialTrees};
use super::super::params::Params;
use super::Organism;

/// Trait for different sensory modalities that organisms can use to perceive their environment.
///
/// Each sense processes environmental information and returns a vector of activations
/// that will be fed as inputs to the organism's brain.
pub trait Sense: Sync {
    /// Process sensory information and return neural network inputs.
    ///
    /// # Arguments
    ///
    /// * `organism` - The organism doing the sensing
    /// * `ecosystem` - The current state of the ecosystem
    /// * `params` - Simulation parameters
    /// * `trees` - Optional pre-built KD-trees for efficient spatial queries (if None, builds internally)
    ///
    /// # Returns
    ///
    /// A 1D array of sensory activations to be used as brain inputs.
    fn sense(
        &self,
        organism: &Organism,
        ecosystem: &Ecosystem,
        params: &Params,
        trees: Option<&SpatialTrees>,
    ) -> Array1<f32>;

    /// Returns the number of neural network inputs this sense produces.
    ///
    /// # Returns
    ///
    /// The size of the output array from `sense()`.
    fn input_size(&self, params: &Params) -> usize;

    /// Returns a human-readable name for this sense.
    fn name(&self) -> &str;
}
