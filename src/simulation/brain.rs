//! Neural network implementation for organism brains.
//!
//! Implements a multi-layer perceptron (MLP) with tanh activation and supports
//! genetic algorithm operations (mutation and crossover).

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// A single layer of a multi-layer perceptron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mlp {
    /// Weight matrix (`output_size` × `input_size`).
    pub weights: Array2<f32>,
    /// Bias vector (`output_size`).
    pub biases: Array1<f32>,
}

impl Mlp {
    /// Creates a new layer with random weights and biases.
    pub fn new_random(input_size: usize, output_size: usize, scale: f32) -> Self {
        Self {
            weights: Array2::random((output_size, input_size), Uniform::new(-scale, scale)),
            biases: Array1::random(output_size, Uniform::new(-scale, scale)),
        }
    }

    /// Performs forward pass with tanh activation.
    #[inline]
    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        // SIMD-optimized: dot product uses BLAS when enabled
        let mut output = self.weights.dot(inputs);
        output += &self.biases;

        // In-place tanh for better cache locality
        output.mapv_inplace(f32::tanh);
        output
    }

    /// Mutates weights and biases by adding random noise.
    pub fn mutate(&mut self, mutation_scale: f32) {
        self.weights += &Array2::random(
            self.weights.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.biases += &Array1::random(
            self.biases.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
    }

    /// Creates a new layer by averaging two parent layers.
    pub fn crossover(parent1: &Mlp, parent2: &Mlp) -> Self {
        Self {
            weights: &parent1.weights * 0.5 + &parent2.weights * 0.5,
            biases: &parent1.biases * 0.5 + &parent2.biases * 0.5,
        }
    }
}

/// A multi-layer perceptron neural network.
///
/// Consists of multiple MLP layers with tanh activation between layers.
/// Used as the "brain" that controls organism behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brain {
    /// Ordered layers from input to output.
    pub layers: Vec<Mlp>,
}

impl Brain {
    /// Creates a new brain with random weights.
    pub fn new(layer_sizes: &[usize], scale: f32) -> Self {
        let layers = (0..layer_sizes.len() - 1)
            .map(|i| Mlp::new_random(layer_sizes[i], layer_sizes[i + 1], scale))
            .collect();

        Self { layers }
    }

    /// Runs a forward pass through all layers.
    #[inline]
    pub fn think(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut output = inputs.clone();

        // SIMD-friendly: sequential layer processing
        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    /// Creates a new brain by averaging two parent brains.
    pub fn crossover(parent1: &Brain, parent2: &Brain) -> Self {
        let new_layers = parent1
            .layers
            .iter()
            .zip(&parent2.layers)
            .map(|(layer1, layer2)| Mlp::crossover(layer1, layer2))
            .collect();

        Self { layers: new_layers }
    }

    /// Mutates all layers in the brain.
    pub fn mutate(&mut self, mutation_scale: f32) {
        for layer in &mut self.layers {
            layer.mutate(mutation_scale);
        }
    }

    /// Calculates the Euclidean distance between two brains.
    ///
    /// Returns the sum of squared differences across all weights and biases,
    /// then takes the square root. This measures how different two neural networks are.
    pub fn distance(brain1: &Brain, brain2: &Brain) -> f32 {
        let mut sum_sq = 0.0;

        for (layer1, layer2) in brain1.layers.iter().zip(&brain2.layers) {
            // Sum squared differences for weights
            for (w1, w2) in layer1.weights.iter().zip(layer2.weights.iter()) {
                let diff = w1 - w2;
                sum_sq += diff * diff;
            }

            // Sum squared differences for biases
            for (b1, b2) in layer1.biases.iter().zip(layer2.biases.iter()) {
                let diff = b1 - b2;
                sum_sq += diff * diff;
            }
        }

        sum_sq.sqrt()
    }

    /// Flattens all weights and biases into a single vector.
    ///
    /// Used for PCA visualization and other analyses that need a flat representation.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::new();

        for layer in &self.layers {
            // Add all weights
            flat.extend(layer.weights.iter().copied());
            // Add all biases
            flat.extend(layer.biases.iter().copied());
        }

        flat
    }
}
