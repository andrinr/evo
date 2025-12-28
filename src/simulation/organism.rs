//! Organism behavior, state, and lifecycle management.
//!
//! Organisms have neural network brains, can perceive their environment through vision,
//! and can move, eat, reproduce, and attack.

use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use super::brain;

/// A simulated organism with a neural network brain.
///
/// Organisms can:
/// - Move and rotate based on brain outputs
/// - See other organisms and food within their field of view
/// - Consume food to gain energy
/// - Attack other organisms with projectiles
/// - Reproduce through mutation and crossover
/// - Die when energy reaches zero
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organism {
    /// Unique identifier for this organism.
    pub id: usize,
    /// Time alive in simulation seconds.
    pub age: f32,
    /// Fitness score (incremented when eating food).
    pub score: i32,
    /// Position in 2D space.
    pub pos: Array1<f32>,
    /// Rotation in radians.
    pub rot: f32,
    /// Current energy (dies when <= 0).
    pub energy: f32,
    /// Signal output (RGB color visible to others).
    pub signal: Array1<f32>,
    /// Internal memory state (persists between timesteps).
    pub memory: Array1<f32>,
    /// Neural network that controls behavior.
    pub brain: brain::Brain,
    /// Cooldown before next attack (seconds).
    pub attack_cooldown: f32,
    /// Last brain inputs (for visualization purposes).
    pub last_brain_inputs: Array1<f32>,
    /// Vision ray angles relative to organism's rotation
    pub vision_angles: Array1<f32>,
    /// Vision ray lengths as fraction of max vision radius
    pub vision_lengths: Array1<f32>,
    /// DNA vector for breeding compatibility (2D space)
    pub dna: Array1<f32>,
    /// Genetic pool ID (organisms can only breed within their pool)
    pub pool_id: usize,
}

impl Organism {
    /// Creates a new organism with random position, rotation, and brain weights.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `screen_center` - Center point for calculating random position bounds
    /// * `signal_size` - Number of signal outputs (typically 3 for RGB)
    /// * `memory_size` - Number of memory cells
    /// * `num_vision_directions` - Number of vision rays
    /// * `fov` - Field of view in radians
    /// * `max_vision` - Max length of vision vector
    /// * `layer_sizes` - Neural network layer dimensions
    /// * `pool_id` - Genetic pool ID for breeding isolation
    #[allow(clippy::too_many_arguments)]
    pub fn new_random(
        id: usize,
        screen_center: &Array1<f32>,
        signal_size: usize,
        memory_size: usize,
        num_vision_directions: usize,
        max_vision: f32,
        fov: f32,
        layer_sizes: Vec<usize>,
        pool_id: usize,
    ) -> Self {
        let input_size = layer_sizes[0];

        // Initialize vision angles evenly spread across FOV
        let mut vision_angles = Array1::zeros(num_vision_directions);
        for i in 0..num_vision_directions {
            let angle_offset = if num_vision_directions > 1 {
                (i as f32 / (num_vision_directions - 1) as f32 - 0.5) * fov
            } else {
                0.0
            };
            vision_angles[i] = angle_offset;
        }

        // Initialize vision lengths: center vision is 2x longer than others
        let vision_lengths = Array1::from_elem(num_vision_directions, max_vision);

        Self {
            id,
            age: 0.0,
            score: 0,
            pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
            rot: rand::random::<f32>() * std::f32::consts::PI * 2.,
            energy: 1.0,
            signal: Array1::random(signal_size, Uniform::new(0.0, 1.0)),
            memory: Array1::zeros(memory_size),
            brain: brain::Brain::new(&layer_sizes, 0.1),
            attack_cooldown: 0.0,
            last_brain_inputs: Array1::zeros(input_size),
            vision_angles,
            vision_lengths,
            dna: Array1::random(2, Uniform::new(0.0, 1.0)),
            pool_id,
        }
    }

    /// Checks if the organism is alive.
    ///
    /// # Returns
    ///
    /// `true` if energy > 0, `false` otherwise.
    pub fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    /// Calculates vision ray directions based on evolved vision parameters.
    ///
    /// # Arguments
    ///
    /// * `max_vision_length` - Maximum vision distance
    ///
    /// # Returns
    ///
    /// Vector of vision ray endpoints relative to organism position.
    pub fn get_vision_vectors(&self) -> Vec<Array1<f32>> {
        self.vision_angles
            .iter()
            .zip(self.vision_lengths.iter())
            .map(|(&angle, &length)| {
                let angle_rad = self.rot + angle;
                Array1::from_vec(vec![angle_rad.cos() * length, angle_rad.sin() * length])
            })
            .collect()
    }

    /// Increments the organism's age.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time delta in seconds
    pub fn age_by(&mut self, dt: f32) {
        self.age += dt;
    }

    /// Reduces the organism's energy.
    ///
    /// # Arguments
    ///
    /// * `amount` - Energy to subtract
    pub fn consume_energy(&mut self, amount: f32) {
        self.energy -= amount;
    }

    /// Increases the organism's energy up to a maximum.
    ///
    /// # Arguments
    ///
    /// * `amount` - Energy to add
    /// * `max_energy` - Maximum energy cap
    pub fn gain_energy(&mut self, amount: f32, max_energy: f32) {
        self.energy = (self.energy + amount).min(max_energy);
    }

    /// Kills the organism by setting energy to 0.
    pub fn kill(&mut self) {
        self.energy = 0.0;
    }

    /// Checks if the organism can attack (cooldown expired).
    ///
    /// # Returns
    ///
    /// `true` if attack cooldown <= 0, `false` otherwise.
    pub fn can_attack(&self) -> bool {
        self.attack_cooldown <= 0.0
    }

    /// Resets the attack cooldown timer.
    ///
    /// # Arguments
    ///
    /// * `cooldown_time` - Cooldown duration in seconds
    pub fn reset_attack_cooldown(&mut self, cooldown_time: f32) {
        self.attack_cooldown = cooldown_time;
    }

    /// Decrements the attack cooldown timer.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time delta in seconds
    pub fn update_cooldown(&mut self, dt: f32) {
        if self.attack_cooldown > 0.0 {
            self.attack_cooldown -= dt;
        }
    }
}
