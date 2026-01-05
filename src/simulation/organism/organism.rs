//! Organism behavior, state, and lifecycle management.
//!
//! Organisms have neural network brains, can perceive their environment through vision,
//! and can move, eat, reproduce, and attack.

use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use super::super::brain;
use super::super::locatable::Locatable;
use super::super::params::Params;

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
    /// Velocity in 2D space.
    pub vel: Array1<f32>,
    /// Movement/orientation rotation in radians (direction of movement).
    pub rot: f32,
    /// Vision rotation in radians (direction of vision/perception).
    /// This is now relative to body rotation: `vision_rot` = rot + `neural_output_offset`
    pub vision_rot: f32,
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
    /// Birth generation number (for tracking lineage)
    pub birth_generation: u32,
    /// Reproduction method: 0 = random, 1 = asexual, 2 = sexual same-pool, 3 = sexual inter-pool
    pub reproduction_method: u8,
    /// Parent score(s) at time of birth (for tracking improvement)
    pub parent_avg_score: f64,
    /// Total distance traveled by this organism
    pub distance_traveled: f32,
}

impl Organism {
    /// Creates a shallow clone of the organism without cloning brain weights.
    /// The brain is replaced with an empty dummy brain with minimal memory allocation.
    /// This is used for read-only ecosystem snapshots during parallel processing.
    ///
    /// # Returns
    ///
    /// A clone with all fields except brain. Brain is set to a minimal empty MLP.
    pub fn clone_shallow(&self) -> Self {
        // Create minimal empty brain (just one tiny layer) to satisfy type requirements
        // Use zeros instead of random to avoid any allocation overhead
        let dummy_brain = brain::Brain::MLP { layers: vec![] };

        Self {
            id: self.id,
            age: self.age,
            score: self.score,
            pos: self.pos.clone(),
            vel: self.vel.clone(),
            rot: self.rot,
            vision_rot: self.vision_rot,
            energy: self.energy,
            signal: self.signal.clone(),
            memory: self.memory.clone(),
            brain: dummy_brain, // Dummy brain - never used in parallel queries
            attack_cooldown: self.attack_cooldown,
            last_brain_inputs: self.last_brain_inputs.clone(),
            vision_angles: self.vision_angles.clone(),
            vision_lengths: self.vision_lengths.clone(),
            dna: self.dna.clone(),
            pool_id: self.pool_id,
            birth_generation: self.birth_generation,
            reproduction_method: self.reproduction_method,
            parent_avg_score: self.parent_avg_score,
            distance_traveled: self.distance_traveled,
        }
    }

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
    /// * `layer_sizes` - Neural network layer dimensions (for MLP)
    /// * `pool_id` - Genetic pool ID for breeding isolation
    /// * `params` - Ecosystem parameters (contains brain type and transformer config)
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
        params: &Params,
    ) -> Self {
        let input_size = layer_sizes[0];

        // Initialize vision angles evenly spread across FOV
        let mut vision_angles = Array1::zeros(num_vision_directions);
        let mut vision_lengths = Array1::zeros(num_vision_directions);

        for i in 0..num_vision_directions {
            let angle_offset = if num_vision_directions > 1 {
                (i as f32 / (num_vision_directions - 1) as f32 - 0.5) * fov
            } else {
                0.0
            };
            vision_angles[i] = angle_offset;

            // // Peripheral vision falloff: center rays are longest, edges are shorter
            // // Use cosine falloff: rays at the edge (±fov/2) are 50% as long as center
            // let normalized_angle = angle_offset / (fov / 2.0); // -1 to 1
            // let falloff = (normalized_angle.abs().powi(2) * 0.5).clamp(0.0, 0.5); // 0 at center, 0.5 at edges
            vision_lengths[i] = max_vision;
        }

        // Create brain based on brain type
        let brain = match params.brain_type {
            brain::BrainType::MLP => brain::Brain::new(&layer_sizes, 0.1),
            brain::BrainType::Transformer => {
                let output_size = layer_sizes.last().copied().unwrap_or(40);
                brain::Brain::new_transformer(
                    input_size,
                    output_size,
                    params.transformer_model_dim,
                    params.transformer_num_blocks,
                    params.transformer_num_heads,
                    params.transformer_head_dim,
                    params.transformer_ff_dim,
                    0.1,
                )
            }
        };

        let initial_rot = rand::random::<f32>() * std::f32::consts::PI * 2.;

        Self {
            id,
            age: 0.0,
            score: 0,
            pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
            vel: Array1::zeros(2),
            rot: initial_rot,
            vision_rot: initial_rot, // Start with vision aligned to movement
            energy: 1.0,
            signal: Array1::random(signal_size, Uniform::new(0.0, 1.0)),
            memory: Array1::zeros(memory_size),
            brain,
            attack_cooldown: 0.0,
            last_brain_inputs: Array1::zeros(input_size),
            vision_angles,
            vision_lengths,
            dna: Array1::random(2, Uniform::new(0.0, 1.0)),
            pool_id,
            birth_generation: 0,
            reproduction_method: 0, // random initialization
            parent_avg_score: 0.0,
            distance_traveled: 0.0,
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

    /// Calculates fitness value for breeding selection.
    /// Fitness combines survival time (age), combat success (score), and movement (distance).
    /// This creates evolutionary pressure for longevity, effectiveness, and exploration.
    pub fn fitness(&self) -> f64 {
        // Age component: reward organisms that lived longer
        let age_fitness = self.age as f64;

        // Score component: reward organisms that ate more food (or killed others)
        let score_fitness = self.score as f64;

        // Distance component: reward organisms that moved around more
        let distance_fitness = self.distance_traveled as f64;

        // Combined fitness: weight all components
        // Age: 0.1x, Score: 1.0x, Distance: 0.01x
        // This prioritizes food consumption, with bonuses for survival and exploration
        0.1 * age_fitness + score_fitness + 0.001 * distance_fitness
    }

    /// Calculates vision ray directions based on evolved vision parameters.
    ///
    /// # Returns
    ///
    /// Vector of vision ray endpoints relative to organism position.
    pub fn get_vision_vectors(&self) -> Vec<Array1<f32>> {
        self.vision_angles
            .iter()
            .zip(self.vision_lengths.iter())
            .map(|(&angle, &length)| {
                let angle_rad = self.vision_rot + angle;
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

impl Locatable for Organism {
    fn pos(&self) -> &Array1<f32> {
        &self.pos
    }

    fn pos_mut(&mut self) -> &mut Array1<f32> {
        &mut self.pos
    }

    fn update(&mut self, dt: f32) {
        // Calculate distance moved this frame
        let displacement = &self.vel * dt;
        let distance_this_frame = displacement.mapv(|x| x.powi(2)).sum().sqrt();

        // Update position based on velocity
        self.pos += &displacement;

        // Track total distance traveled
        self.distance_traveled += distance_this_frame;

        // Update age and attack cooldown
        self.age += dt;
        if self.attack_cooldown > 0.0 {
            self.attack_cooldown -= dt;
        }
    }
}
