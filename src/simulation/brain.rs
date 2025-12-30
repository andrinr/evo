//! Neural network implementation for organism brains.
//!
//! Implements both multi-layer perceptron (MLP) and transformer architectures
//! with support for genetic algorithm operations (mutation and crossover).

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Type of neural network architecture to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainType {
    /// Multi-layer perceptron with tanh activation
    MLP,
    /// Transformer with multi-head self-attention
    Transformer,
}

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

/// A single attention head in a transformer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHead {
    /// Query projection weights (`head_dim` × `input_dim`)
    pub w_q: Array2<f32>,
    /// Key projection weights (`head_dim` × `input_dim`)
    pub w_k: Array2<f32>,
    /// Value projection weights (`head_dim` × `input_dim`)
    pub w_v: Array2<f32>,
}

impl AttentionHead {
    /// Creates a new attention head with random weights.
    pub fn new_random(input_dim: usize, head_dim: usize, scale: f32) -> Self {
        Self {
            w_q: Array2::random((head_dim, input_dim), Uniform::new(-scale, scale)),
            w_k: Array2::random((head_dim, input_dim), Uniform::new(-scale, scale)),
            w_v: Array2::random((head_dim, input_dim), Uniform::new(-scale, scale)),
        }
    }

    /// Performs attention on a single input vector.
    #[inline]
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // Q, K, V projections
        let q = self.w_q.dot(input);
        let k = self.w_k.dot(input);
        let v = self.w_v.dot(input);

        // Scaled dot-product attention (self-attention on single vector)
        let scale = (q.len() as f32).sqrt();
        let score = q.dot(&k) / scale;
        let attention = score.tanh(); // Bounded activation

        // Apply attention to value
        &v * attention
    }

    /// Mutates all weights by adding random noise.
    pub fn mutate(&mut self, mutation_scale: f32) {
        self.w_q += &Array2::random(
            self.w_q.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.w_k += &Array2::random(
            self.w_k.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.w_v += &Array2::random(
            self.w_v.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
    }

    /// Creates a new head by averaging two parent heads.
    pub fn crossover(parent1: &AttentionHead, parent2: &AttentionHead) -> Self {
        Self {
            w_q: &parent1.w_q * 0.5 + &parent2.w_q * 0.5,
            w_k: &parent1.w_k * 0.5 + &parent2.w_k * 0.5,
            w_v: &parent1.w_v * 0.5 + &parent2.w_v * 0.5,
        }
    }
}

/// A transformer block with multi-head attention and feed-forward network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBlock {
    /// Attention heads
    pub heads: Vec<AttentionHead>,
    /// Output projection after concatenating heads (`input_dim` × `total_head_outputs`)
    pub w_o: Array2<f32>,
    /// Feed-forward network layer 1 (`ff_dim` × `input_dim`)
    pub ff1: Mlp,
    /// Feed-forward network layer 2 (`input_dim` × `ff_dim`)
    pub ff2: Mlp,
    /// Layer norm gain for attention (pre-normalization)
    pub ln1_gain: Array1<f32>,
    /// Layer norm bias for attention (pre-normalization)
    pub ln1_bias: Array1<f32>,
    /// Layer norm gain for feed-forward (pre-normalization)
    pub ln2_gain: Array1<f32>,
    /// Layer norm bias for feed-forward (pre-normalization)
    pub ln2_bias: Array1<f32>,
}

impl TransformerBlock {
    /// Creates a new transformer block.
    pub fn new_random(
        input_dim: usize,
        num_heads: usize,
        head_dim: usize,
        ff_dim: usize,
        scale: f32,
    ) -> Self {
        let heads: Vec<AttentionHead> = (0..num_heads)
            .map(|_| AttentionHead::new_random(input_dim, head_dim, scale))
            .collect();

        Self {
            heads,
            w_o: Array2::random(
                (input_dim, num_heads * head_dim),
                Uniform::new(-scale, scale),
            ),
            ff1: Mlp::new_random(input_dim, ff_dim, scale),
            ff2: Mlp::new_random(ff_dim, input_dim, scale),
            ln1_gain: Array1::ones(input_dim),
            ln1_bias: Array1::zeros(input_dim),
            ln2_gain: Array1::ones(input_dim),
            ln2_bias: Array1::zeros(input_dim),
        }
    }

    /// Simple layer normalization.
    #[inline]
    fn layer_norm(x: &Array1<f32>, gain: &Array1<f32>, bias: &Array1<f32>) -> Array1<f32> {
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (variance + 1e-5).sqrt();

        ((x - mean) / std) * gain + bias
    }

    /// Forward pass through transformer block.
    #[inline]
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // Multi-head attention with residual
        let normed1 = Self::layer_norm(input, &self.ln1_gain, &self.ln1_bias);

        // Concatenate all head outputs
        let mut head_outputs = Vec::new();
        for head in &self.heads {
            head_outputs.extend(head.forward(&normed1).iter());
        }
        let head_concat = Array1::from_vec(head_outputs);

        // Project concatenated heads back to input dimension
        let attention_out = self.w_o.dot(&head_concat);
        let after_attention = input + &attention_out; // Residual connection

        // Feed-forward network with residual
        let normed2 = Self::layer_norm(&after_attention, &self.ln2_gain, &self.ln2_bias);
        let ff_out1 = self.ff1.forward(&normed2);
        let ff_out2 = self.ff2.forward(&ff_out1);

        &after_attention + &ff_out2 // Residual connection
    }

    /// Mutates all parameters in the block.
    pub fn mutate(&mut self, mutation_scale: f32) {
        for head in &mut self.heads {
            head.mutate(mutation_scale);
        }
        self.w_o += &Array2::random(
            self.w_o.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.ff1.mutate(mutation_scale);
        self.ff2.mutate(mutation_scale);
        self.ln1_gain += &Array1::random(
            self.ln1_gain.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.ln1_bias += &Array1::random(
            self.ln1_bias.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.ln2_gain += &Array1::random(
            self.ln2_gain.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.ln2_bias += &Array1::random(
            self.ln2_bias.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
    }

    /// Creates a new block by averaging two parent blocks.
    pub fn crossover(parent1: &TransformerBlock, parent2: &TransformerBlock) -> Self {
        let new_heads = parent1
            .heads
            .iter()
            .zip(&parent2.heads)
            .map(|(h1, h2)| AttentionHead::crossover(h1, h2))
            .collect();

        Self {
            heads: new_heads,
            w_o: &parent1.w_o * 0.5 + &parent2.w_o * 0.5,
            ff1: Mlp::crossover(&parent1.ff1, &parent2.ff1),
            ff2: Mlp::crossover(&parent1.ff2, &parent2.ff2),
            ln1_gain: &parent1.ln1_gain * 0.5 + &parent2.ln1_gain * 0.5,
            ln1_bias: &parent1.ln1_bias * 0.5 + &parent2.ln1_bias * 0.5,
            ln2_gain: &parent1.ln2_gain * 0.5 + &parent2.ln2_gain * 0.5,
            ln2_bias: &parent1.ln2_bias * 0.5 + &parent2.ln2_bias * 0.5,
        }
    }
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
