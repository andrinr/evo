//! Neural network implementation for organism brains.
//!
//! Implements both multi-layer perceptron (MLP) and transformer architectures
//! with support for genetic algorithm operations (mutation and crossover).

use ndarray::Array1;
use serde::{Deserialize, Serialize};

pub mod mlp;
pub mod transformer;

pub use mlp::Mlp;
pub use transformer::{AttentionHead, TransformerBlock};

/// Type of neural network architecture to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainType {
    /// Multi-layer perceptron with tanh activation
    MLP,
    /// Transformer with multi-head self-attention
    Transformer,
}

/// Neural network brain that can use either MLP or Transformer architecture.
///
/// Used as the "brain" that controls organism behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Brain {
    /// Multi-layer perceptron with tanh activation.
    MLP {
        /// Ordered layers from input to output.
        layers: Vec<Mlp>,
    },
    /// Transformer with multi-head attention.
    Transformer {
        /// Input embedding layer (maps input to model dimension)
        input_embed: Mlp,
        /// Transformer blocks
        blocks: Vec<TransformerBlock>,
        /// Output projection layer (maps model dimension to output)
        output_proj: Mlp,
    },
}

impl Brain {
    /// Creates a new MLP brain with random weights.
    pub fn new(layer_sizes: &[usize], scale: f32) -> Self {
        let layers = (0..layer_sizes.len() - 1)
            .map(|i| Mlp::new_random(layer_sizes[i], layer_sizes[i + 1], scale))
            .collect();

        Brain::MLP { layers }
    }

    /// Creates a new transformer brain with random weights.
    ///
    /// Parameters:
    /// - `input_size`: Size of input vector
    /// - `output_size`: Size of output vector
    /// - `model_dim`: Hidden dimension for transformer (e.g., 64)
    /// - `num_blocks`: Number of transformer blocks (e.g., 2)
    /// - `num_heads`: Number of attention heads per block (e.g., 4)
    /// - `head_dim`: Dimension per attention head (e.g., 16)
    /// - `ff_dim`: Feed-forward hidden dimension (e.g., 128)
    pub fn new_transformer(
        input_size: usize,
        output_size: usize,
        model_dim: usize,
        num_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        ff_dim: usize,
        scale: f32,
    ) -> Self {
        let input_embed = Mlp::new_random(input_size, model_dim, scale);
        let blocks = (0..num_blocks)
            .map(|_| TransformerBlock::new_random(model_dim, num_heads, head_dim, ff_dim, scale))
            .collect();
        let output_proj = Mlp::new_random(model_dim, output_size, scale);

        Brain::Transformer {
            input_embed,
            blocks,
            output_proj,
        }
    }

    /// Runs a forward pass through the brain.
    #[inline]
    pub fn think(&self, inputs: &Array1<f32>) -> Array1<f32> {
        match self {
            Brain::MLP { layers } => {
                let mut output = inputs.clone();
                for layer in layers {
                    output = layer.forward(&output);
                }
                output
            }
            Brain::Transformer {
                input_embed,
                blocks,
                output_proj,
            } => {
                // Embed input to model dimension
                let mut hidden = input_embed.forward(inputs);

                // Pass through transformer blocks
                for block in blocks {
                    hidden = block.forward(&hidden);
                }

                // Project to output dimension
                output_proj.forward(&hidden)
            }
        }
    }

    /// Creates a new brain by averaging two parent brains.
    /// Both parents must be the same architecture type.
    pub fn crossover(parent1: &Brain, parent2: &Brain) -> Self {
        match (parent1, parent2) {
            (Brain::MLP { layers: l1 }, Brain::MLP { layers: l2 }) => {
                let new_layers = l1
                    .iter()
                    .zip(l2)
                    .map(|(layer1, layer2)| Mlp::crossover(layer1, layer2))
                    .collect();
                Brain::MLP { layers: new_layers }
            }
            (
                Brain::Transformer {
                    input_embed: ie1,
                    blocks: b1,
                    output_proj: op1,
                },
                Brain::Transformer {
                    input_embed: ie2,
                    blocks: b2,
                    output_proj: op2,
                },
            ) => {
                let new_blocks = b1
                    .iter()
                    .zip(b2)
                    .map(|(block1, block2)| TransformerBlock::crossover(block1, block2))
                    .collect();
                Brain::Transformer {
                    input_embed: Mlp::crossover(ie1, ie2),
                    blocks: new_blocks,
                    output_proj: Mlp::crossover(op1, op2),
                }
            }
            _ => {
                // Mismatched types - return clone of parent1
                parent1.clone()
            }
        }
    }

    /// Creates a new brain by weighted averaging two parent brains.
    /// Both parents must be the same architecture type.
    /// weight1 is the weight for parent1, weight2 = 1.0 - weight1 for parent2.
    pub fn crossover_weighted(parent1: &Brain, parent2: &Brain, weight1: f32) -> Self {
        match (parent1, parent2) {
            (Brain::MLP { layers: l1 }, Brain::MLP { layers: l2 }) => {
                let new_layers = l1
                    .iter()
                    .zip(l2)
                    .map(|(layer1, layer2)| Mlp::crossover_weighted(layer1, layer2, weight1))
                    .collect();
                Brain::MLP { layers: new_layers }
            }
            (
                Brain::Transformer {
                    input_embed: ie1,
                    blocks: b1,
                    output_proj: op1,
                },
                Brain::Transformer {
                    input_embed: ie2,
                    blocks: b2,
                    output_proj: op2,
                },
            ) => {
                let new_blocks = b1
                    .iter()
                    .zip(b2)
                    .map(|(block1, block2)| {
                        TransformerBlock::crossover_weighted(block1, block2, weight1)
                    })
                    .collect();
                Brain::Transformer {
                    input_embed: Mlp::crossover_weighted(ie1, ie2, weight1),
                    blocks: new_blocks,
                    output_proj: Mlp::crossover_weighted(op1, op2, weight1),
                }
            }
            _ => {
                // Mismatched types - return clone of parent1
                parent1.clone()
            }
        }
    }

    /// Mutates all parameters in the brain.
    pub fn mutate(&mut self, mutation_scale: f32) {
        match self {
            Brain::MLP { layers } => {
                for layer in layers {
                    layer.mutate(mutation_scale);
                }
            }
            Brain::Transformer {
                input_embed,
                blocks,
                output_proj,
            } => {
                input_embed.mutate(mutation_scale);
                for block in blocks {
                    block.mutate(mutation_scale);
                }
                output_proj.mutate(mutation_scale);
            }
        }
    }

    /// Calculates the Euclidean distance between two brains.
    /// Only works if both brains are the same architecture type.
    pub fn distance(brain1: &Brain, brain2: &Brain) -> f32 {
        match (brain1, brain2) {
            (Brain::MLP { layers: l1 }, Brain::MLP { layers: l2 }) => {
                let mut sum_sq = 0.0;
                for (layer1, layer2) in l1.iter().zip(l2) {
                    for (w1, w2) in layer1.weights.iter().zip(layer2.weights.iter()) {
                        let diff = w1 - w2;
                        sum_sq += diff * diff;
                    }
                    for (b1, b2) in layer1.biases.iter().zip(layer2.biases.iter()) {
                        let diff = b1 - b2;
                        sum_sq += diff * diff;
                    }
                }
                sum_sq.sqrt()
            }
            (
                Brain::Transformer {
                    input_embed: ie1,
                    blocks: b1,
                    output_proj: op1,
                },
                Brain::Transformer {
                    input_embed: ie2,
                    blocks: b2,
                    output_proj: op2,
                },
            ) => {
                let mut sum_sq = 0.0;

                // Input embed distance
                for (w1, w2) in ie1.weights.iter().zip(ie2.weights.iter()) {
                    let diff = w1 - w2;
                    sum_sq += diff * diff;
                }

                // Blocks distance (simplified - just count all parameters)
                for (block1, block2) in b1.iter().zip(b2) {
                    // Attention heads
                    for (head1, head2) in block1.heads.iter().zip(&block2.heads) {
                        for (w1, w2) in head1.w_q.iter().zip(head2.w_q.iter()) {
                            let diff = w1 - w2;
                            sum_sq += diff * diff;
                        }
                        for (w1, w2) in head1.w_k.iter().zip(head2.w_k.iter()) {
                            let diff = w1 - w2;
                            sum_sq += diff * diff;
                        }
                        for (w1, w2) in head1.w_v.iter().zip(head2.w_v.iter()) {
                            let diff = w1 - w2;
                            sum_sq += diff * diff;
                        }
                    }
                }

                // Output proj distance
                for (w1, w2) in op1.weights.iter().zip(op2.weights.iter()) {
                    let diff = w1 - w2;
                    sum_sq += diff * diff;
                }

                sum_sq.sqrt()
            }
            _ => {
                // Different architectures - return large distance
                f32::MAX
            }
        }
    }

    /// Flattens all weights and biases into a single vector.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::new();

        match self {
            Brain::MLP { layers } => {
                for layer in layers {
                    flat.extend(layer.weights.iter().copied());
                    flat.extend(layer.biases.iter().copied());
                }
            }
            Brain::Transformer {
                input_embed,
                blocks,
                output_proj,
            } => {
                flat.extend(input_embed.weights.iter().copied());
                flat.extend(input_embed.biases.iter().copied());

                for block in blocks {
                    for head in &block.heads {
                        flat.extend(head.w_q.iter().copied());
                        flat.extend(head.w_k.iter().copied());
                        flat.extend(head.w_v.iter().copied());
                    }
                    flat.extend(block.w_o.iter().copied());
                }

                flat.extend(output_proj.weights.iter().copied());
                flat.extend(output_proj.biases.iter().copied());
            }
        }

        flat
    }

    /// Returns the type of brain architecture.
    pub fn brain_type(&self) -> BrainType {
        match self {
            Brain::MLP { .. } => BrainType::MLP,
            Brain::Transformer { .. } => BrainType::Transformer,
        }
    }
}
