//! Main ecosystem simulation with parallel organism updates.
//!
//! The ecosystem manages all organisms, food, and projectiles. It handles:
//! - Parallel organism brain updates using rayon
//! - Spatial queries using k-d trees for efficient neighbor finding
//! - Event-driven state updates for thread safety
//! - Organism spawning, reproduction, and evolution

use super::actions;
use super::events;
use super::evolution::EvolutionEngine;
use super::food;
use super::organism;
use super::projectile;
use super::spatial::SpatialIndex;
pub use super::spatial::SpatialTrees;

use super::event_log::EventLog;
use super::geometric_utils::wrap_around_mut;
use super::params::Params;
use super::reproduction::ReproductionStats;
use ndarray::{Array1, s};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The main ecosystem containing all simulation state.
///
/// Manages organisms, food, projectiles, and handles all simulation logic including
/// parallel updates, spatial queries, and evolutionary spawning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ecosystem {
    /// All living organisms in the simulation.
    pub organisms: Vec<organism::Organism>,
    /// Food items (including corpses).
    pub food: Vec<food::Food>,
    /// Active projectiles.
    pub projectiles: Vec<projectile::Projectile>,
    /// Total simulation time elapsed.
    pub time: f32,
    /// Generation counter (incremented with each spawn).
    pub generation: u32,
    /// Statistics about reproduction strategy effectiveness.
    pub reproduction_stats: ReproductionStats,
    /// Evolution engine managing graveyard and organism spawning.
    #[serde(skip)]
    #[serde(default = "default_evolution_engine")]
    evolution_engine: EvolutionEngine,
    /// Active energy sharing interactions (`giver_id`, `receiver_id`, timestamp) for visualization
    #[serde(skip)]
    pub energy_shares: Vec<(usize, usize, f32)>,
    /// Active reproduction intents (`organism1_id`, `organism2_id`, timestamp) for visualization
    #[serde(skip)]
    pub reproduction_intents: Vec<(usize, usize, f32)>,
    /// Event log for displaying recent events in UI
    pub event_log: EventLog,
}

fn default_evolution_engine() -> EvolutionEngine {
    EvolutionEngine::new(1000)
}

impl Ecosystem {
    /// Creates a new ecosystem with random organisms and food.
    pub fn new(params: &Params) -> Self {
        let mut organisms = Vec::with_capacity(params.n_organism);
        let mut food = Vec::with_capacity(params.n_food);

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        for i in 0..params.n_organism {
            // Distribute organisms evenly across genetic pools
            let pool_id = i % params.num_genetic_pools;
            let entity = organism::Organism::new_random(
                i,
                &center,
                params.signal_size,
                params.memory_size,
                params.num_vision_directions,
                params.vision_radius,
                params.fov,
                params.layer_sizes.clone(),
                pool_id,
                params,
            );

            organisms.push(entity);
        }

        for _i in 0..params.n_food {
            let food_item = food::Food::new_random(&center, params.food_energy);
            food.push(food_item);
        }

        Self {
            organisms,
            food,
            projectiles: Vec::new(),
            time: 0.,
            generation: params.n_organism as u32,
            reproduction_stats: ReproductionStats::default(),
            evolution_engine: EvolutionEngine::new(params.graveyard_size),
            energy_shares: Vec::new(),
            reproduction_intents: Vec::new(),
            event_log: EventLog::default(),
        }
    }

    /// Advances the simulation by one timestep with parallel organism updates.
    pub fn step(&mut self, params: &Params, dt: f32) {
        // Build spatial index for efficient neighbor queries
        let spatial_index = SpatialIndex::build(self).expect("Failed to build spatial index");

        self.time += dt;

        // Create perception system for generating brain inputs
        let perception = organism::Perception::default();

        // Get reference to KD-trees for perception system
        let spatial_trees = spatial_index.as_trees();

        // Create Arc wrapper for shared read-only access
        // This does ONE clone before parallelization (unavoidable for now due to perception API)
        let ecosystem_snapshot = Arc::new(self.clone());

        // Parallel phase: collect events from each organism without mutex contention
        // Use larger chunks to reduce cache line bouncing and task switching overhead
        let chunk_size = (self.organisms.len() / rayon::current_num_threads()).max(16);
        let all_events: Vec<events::SimulationEvent> = self
            .organisms
            .par_chunks_mut(chunk_size)
            .flat_map(|chunk| {
                let mut chunk_events = Vec::new();
                for entity in chunk.iter_mut() {
                    // wrap around the screen
                    wrap_around_mut(&mut entity.pos, params.box_width, params.box_height);

                    // Get nearest neighbors for collision and action detection
                    let neighbors_orgs =
                        spatial_index.query_organisms(&entity.pos, params.vision_radius);
                    let neighbor_foods =
                        spatial_index.query_food(&entity.pos, params.vision_radius);

                    // Check for collisions with other organisms
                    for (_, neighbor_id) in &neighbors_orgs {
                        let neighbor_org = &ecosystem_snapshot.organisms[*neighbor_id];
                        if neighbor_org.id == entity.id {
                            continue; // skip self
                        }
                        let org_org_distance =
                            (&entity.pos - &neighbor_org.pos).mapv(f32::abs).sum();
                        if org_org_distance < params.body_radius * 2.0 {
                            entity.kill(); // collision with another organism
                        }
                    }

                    // Generate brain inputs using perception system
                    // Arc dereference is cheap - just a pointer read
                    let brain_inputs = perception.perceive(
                        entity,
                        &ecosystem_snapshot,
                        params,
                        Some(&spatial_trees),
                    );

                    // Store brain inputs for visualization
                    entity.last_brain_inputs.clone_from(&brain_inputs);

                    // Process brain outputs
                    let brain_outputs = entity.brain.think(&brain_inputs);

                    // Update signal and memory from brain outputs
                    entity.signal = brain_outputs.slice(s![..params.signal_size]).to_owned();
                    entity.memory = brain_outputs
                        .slice(s![
                            params.signal_size..params.signal_size + params.memory_size
                        ])
                        .to_owned();

                    // update age, cooldown, and idle energy consumption
                    entity.age_by(dt);
                    entity.update_cooldown(dt);
                    entity.consume_energy(params.idle_energy_rate * dt);

                    // Execute all organism actions and collect events
                    let entity_events = actions::execute_all_actions(
                        entity,
                        &brain_outputs,
                        &neighbors_orgs,
                        &neighbor_foods,
                        &ecosystem_snapshot.organisms,
                        &ecosystem_snapshot.food,
                        params,
                        dt,
                    );
                    chunk_events.extend(entity_events);
                }
                chunk_events
            })
            .collect();

        // Update projectiles and check for collisions using KD tree
        let mut projectile_events = Vec::new();
        for (proj_idx, projectile) in self.projectiles.iter_mut().enumerate() {
            projectile.update(dt);

            // Use spatial index to find nearby organisms within collision range
            let collision_radius = params.body_radius + params.projectile_radius;
            let nearby_organisms = spatial_index.query_organisms(&projectile.pos, collision_radius);

            // Check collision with nearby organisms
            for (_, org_id) in &nearby_organisms {
                let organism = &self.organisms[*org_id];

                if organism.id == projectile.owner_id {
                    continue; // Don't hit self
                }

                let distance = (&projectile.pos - &organism.pos)
                    .mapv(|x| x.powi(2))
                    .sum()
                    .sqrt();

                if distance < collision_radius {
                    projectile_events.push(events::SimulationEvent::ProjectileHit {
                        projectile_idx: proj_idx,
                        target_id: organism.id,
                        damage: projectile.damage,
                        owner_id: projectile.owner_id,
                    });
                    break;
                }
            }
        }

        // Remove expired projectiles
        self.projectiles.retain(|p| !p.is_expired());

        // Generate OrganismDied events for dead organisms
        for organism in &self.organisms {
            if !organism.is_alive() {
                projectile_events.push(events::SimulationEvent::OrganismDied {
                    organism_id: organism.id,
                    pos: organism.pos.clone(),
                });
            }
        }

        // Combine all events and apply them
        let mut combined_events = events::EventQueue::new();
        for event in all_events {
            combined_events.push(event);
        }
        for event in projectile_events {
            combined_events.push(event);
        }

        events::apply_events(self, params, combined_events);

        // Record deaths and add to graveyard before removing organisms
        for organism in &self.organisms {
            if !organism.is_alive() {
                self.evolution_engine
                    .record_death(organism, &mut self.reproduction_stats);
            }
        }

        // Clean up dead organisms and consumed food
        self.organisms.retain(super::organism::Organism::is_alive);
        self.food.retain(|food_item| !food_item.is_consumed());

        // Update food age and remove expired food
        for food_item in &mut self.food {
            food_item.age += dt;
        }

        self.food.retain(|f| f.age < params.food_lifetime);
    }

    /// Spawns new organisms through evolution when population is below target.
    ///
    /// # Arguments
    /// * `params` - Simulation parameters
    /// * `dt` - Delta time in seconds (spawn rates are per second)
    pub fn spawn(&mut self, params: &Params, dt: f32) {
        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        // Automatic asexual reproduction from graveyard
        // This complements organism-initiated reproduction
        let current_count = self.organisms.len();
        let max_allowed = params.max_organism.saturating_sub(current_count);

        let organisms_to_spawn_f = params.organism_spawn_rate * dt;
        let organisms_to_spawn = organisms_to_spawn_f.floor() as usize;
        let spawn_prob = organisms_to_spawn_f.fract();

        let total_organisms_to_spawn = if rand::rng().random::<f32>() < spawn_prob {
            organisms_to_spawn + 1
        } else {
            organisms_to_spawn
        };

        // Enforce hard cap
        let total_organisms_to_spawn = total_organisms_to_spawn.min(max_allowed);

        for _ in 0..total_organisms_to_spawn {
            // Select a random genetic pool for this organism
            let target_pool_id = rand::rng().random_range(0..params.num_genetic_pools);

            // Use evolution engine to spawn organism
            let new_organism = self.evolution_engine.spawn_organism(
                self.generation,
                target_pool_id,
                &center,
                params,
            );

            self.generation += 1;
            self.organisms.push(new_organism);
        }

        let current_food_count = self.food.len();
        let max_allowed_food = params.max_food.saturating_sub(current_food_count);
        if max_allowed_food > 0 {
            // Calculate spawn amount based on rate per second
            let food_to_spawn_f = params.food_spawn_rate * dt;
            let base_spawn = food_to_spawn_f.floor() as usize;

            // Fractional part determines probability of spawning one more
            let spawn_prob = food_to_spawn_f.fract();
            let extra = if spawn_prob > 0.0 && rand::rng().random::<f32>() < spawn_prob {
                1
            } else {
                0
            };

            // Enforce hard cap
            let total_food_to_spawn = (base_spawn + extra).min(max_allowed_food);

            for _ in 0..total_food_to_spawn {
                let food_item = food::Food::new_random(&center, params.food_energy);
                self.food.push(food_item);
            }
        }
    }

    /// Saves the ecosystem state to a JSON file.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Loads an ecosystem state from a JSON file.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let ecosystem = serde_json::from_str(&json)?;
        Ok(ecosystem)
    }

    /// Returns a reference to the graveyard.
    pub fn graveyard(&self) -> &[organism::Organism] {
        self.evolution_engine.graveyard()
    }
}
