//! Transformer architecture implementation.

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use super::Mlp;

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

    /// Creates a new head by weighted averaging two parent heads.
    pub fn crossover_weighted(
        parent1: &AttentionHead,
        parent2: &AttentionHead,
        weight1: f32,
    ) -> Self {
        let weight2 = 1.0 - weight1;
        Self {
            w_q: &parent1.w_q * weight1 + &parent2.w_q * weight2,
            w_k: &parent1.w_k * weight1 + &parent2.w_k * weight2,
            w_v: &parent1.w_v * weight1 + &parent2.w_v * weight2,
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

    /// Creates a new block by weighted averaging two parent blocks.
    pub fn crossover_weighted(
        parent1: &TransformerBlock,
        parent2: &TransformerBlock,
        weight1: f32,
    ) -> Self {
        let weight2 = 1.0 - weight1;
        let new_heads = parent1
            .heads
            .iter()
            .zip(&parent2.heads)
            .map(|(h1, h2)| AttentionHead::crossover_weighted(h1, h2, weight1))
            .collect();

        Self {
            heads: new_heads,
            w_o: &parent1.w_o * weight1 + &parent2.w_o * weight2,
            ff1: Mlp::crossover_weighted(&parent1.ff1, &parent2.ff1, weight1),
            ff2: Mlp::crossover_weighted(&parent1.ff2, &parent2.ff2, weight1),
            ln1_gain: &parent1.ln1_gain * weight1 + &parent2.ln1_gain * weight2,
            ln1_bias: &parent1.ln1_bias * weight1 + &parent2.ln1_bias * weight2,
            ln2_gain: &parent1.ln2_gain * weight1 + &parent2.ln2_gain * weight2,
            ln2_bias: &parent1.ln2_bias * weight1 + &parent2.ln2_bias * weight2,
        }
    }
}
