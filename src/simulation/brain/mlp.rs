//! Multi-layer perceptron implementation.

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

    /// Creates a new layer by weighted averaging two parent layers.
    pub fn crossover_weighted(parent1: &Mlp, parent2: &Mlp, weight1: f32) -> Self {
        let weight2 = 1.0 - weight1;
        Self {
            weights: &parent1.weights * weight1 + &parent2.weights * weight2,
            biases: &parent1.biases * weight1 + &parent2.biases * weight2,
        }
    }
}
