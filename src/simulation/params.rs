use serde::{Deserialize, Serialize};

use super::brain;

/// Simulation parameters that control ecosystem behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    /// Organism collision radius.
    pub body_radius: f32,
    /// Maximum distance organisms can see.
    pub vision_radius: f32,
    /// Maximum distance organisms can smell food.
    pub scent_radius: f32,
    /// Maximum distance for energy sharing between organisms.
    pub share_radius: f32,
    /// Maximum distance for sexual reproduction between organisms.
    pub reproduction_radius: f32,
    /// Maximum DNA distance for breeding (hard cutoff).
    /// Only organisms with DNA distance less than this value can interbreed.
    /// Typical range: 0.1-0.3 (max periodic distance is ~0.707).
    pub dna_breeding_distance: f32,
    /// DNA mutation rate (standard deviation of Gaussian noise).
    pub dna_mutation_rate: f32,
    /// Energy consumed per second while idle.
    pub idle_energy_rate: f32,
    /// Energy cost per unit of movement.
    pub move_energy_rate: f32,
    /// Movement speed multiplier.
    pub move_multiplier: f32,
    /// Energy cost per unit of rotation.
    pub rot_energy_rate: f32,
    /// Number of vision rays per organism.
    pub num_vision_directions: usize,
    /// Field of view angle in radians.
    pub fov: f32,
    /// Number of signal outputs (RGB color).
    pub signal_size: usize,
    /// Number of memory cells per organism.
    pub memory_size: usize,
    /// Target organism population.
    pub n_organism: usize,
    /// Maximum organism population (hard cap).
    pub max_organism: usize,
    /// Target food item count.
    pub n_food: usize,
    /// Maximum food item count (hard cap).
    pub max_food: usize,
    /// Simulation area width.
    pub box_width: f32,
    /// Simulation area height.
    pub box_height: f32,
    /// Neural network layer dimensions.
    pub layer_sizes: Vec<usize>,
    /// Energy cost multiplier for attacks.
    pub attack_cost_rate: f32,
    /// Damage multiplier for attacks.
    pub attack_damage_rate: f32,
    /// Cooldown duration between attacks (seconds).
    pub attack_cooldown: f32,
    /// Fraction of organism energy converted to corpse food.
    pub corpse_energy_ratio: f32,
    /// Maximum energy an organism can have.
    pub max_energy: f32,
    /// Energy value of spawned food items.
    pub food_energy: f32,
    /// Projectile travel speed.
    pub projectile_speed: f32,
    /// Maximum projectile travel distance.
    pub projectile_range: f32,
    /// Projectile collision radius.
    pub projectile_radius: f32,
    /// Organisms spawned per second when below target population.
    pub organism_spawn_rate: f32,
    /// Food items spawned per second when below target count.
    pub food_spawn_rate: f32,
    /// Maximum lifetime of food in seconds
    pub food_lifetime: f32,
    /// Number of genetic pools (isolated breeding populations).
    /// Organisms can only breed within their pool. Range: 1-10.
    pub num_genetic_pools: usize,
    /// Probability of inter-pool breeding (0.0 = fully isolated, 1.0 = no isolation).
    /// Default: 0.0 (fully isolated pools).
    pub pool_interbreed_prob: f32,
    /// Type of neural network architecture to use for organism brains.
    pub brain_type: brain::BrainType,
    // Transformer-specific parameters (only used when brain_type is Transformer)
    /// Model dimension for transformer (hidden size). Typical: 64-128.
    pub transformer_model_dim: usize,
    /// Number of transformer blocks. Typical: 2-4.
    pub transformer_num_blocks: usize,
    /// Number of attention heads per block. Typical: 4-8.
    pub transformer_num_heads: usize,
    /// Dimension per attention head. Typical: 16-32.
    pub transformer_head_dim: usize,
    /// Feed-forward hidden dimension. Typical: 128-256.
    pub transformer_ff_dim: usize,
    /// Maximum number of deceased organisms to keep in graveyard for breeding selection.
    /// Breeding will select fittest organisms from this graveyard instead of living organisms.
    pub graveyard_size: usize,
    /// Energy multiplier for offspring (offspring gets `parent_energy` * this factor).
    /// Default: 1.2 (20% bonus). Range: 0.5-3.0.
    pub reproduction_energy_multiplier: f32,
}
