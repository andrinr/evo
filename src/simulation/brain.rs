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
    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        // Dot product of weights and inputs, plus biases
        let output = self.weights.dot(inputs) + &self.biases;
        // Activation function (tanh)
        output.mapv(f32::tanh)
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
    pub fn think(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut output = inputs.clone();

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
}
