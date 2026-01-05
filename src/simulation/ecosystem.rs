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
use super::locatable::Locatable;
use super::organism;
use super::projectile;
use super::spatial::SpatialIndex;
pub use super::spatial::SpatialTrees;

use super::event_log::EventLog;
use super::geometric_utils::wrap_around_mut;
use super::params::Params;
use super::reproduction::ReproductionStats;
use ndarray::{Array1, s};
use ndarray_rand::RandomExt;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Performance timing statistics for profiling the simulation.
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    /// Time to build spatial index (KD-trees)
    pub spatial_index_ms: f32,
    /// Time to clone ecosystem for parallel access
    pub ecosystem_clone_ms: f32,
    /// Time for parallel organism updates (sensing, thinking, actions)
    pub parallel_update_ms: f32,
    /// Time for projectile updates and collision detection
    pub projectile_update_ms: f32,
    /// Time to apply all events
    pub event_application_ms: f32,
    /// Time for cleanup (removing dead organisms, expired food)
    pub cleanup_ms: f32,
    /// Total step time
    pub total_ms: f32,
}

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
    /// Performance timing statistics
    #[serde(skip)]
    pub timing_stats: TimingStats,
    /// Spatial cluster centers for organism spawning
    #[serde(skip)]
    organism_cluster_centers: Vec<Array1<f32>>,
    /// Pool assignment for each organism cluster (same length as `organism_cluster_centers`)
    #[serde(skip)]
    cluster_pool_assignments: Vec<usize>,
    /// Spatial cluster centers for food spawning (disjoint from organism clusters)
    #[serde(skip)]
    food_cluster_centers: Vec<Array1<f32>>,
}

fn default_evolution_engine() -> EvolutionEngine {
    EvolutionEngine::new(1000, 5)
}

impl Ecosystem {
    /// Generates disjoint cluster centers for organisms and food.
    ///
    /// Creates two sets of cluster centers that are guaranteed not to overlap
    /// by ensuring minimum distance between organism and food clusters.
    /// Each organism cluster is assigned to a genetic pool.
    ///
    /// Returns: (`organism_clusters`, `pool_assignments`, `food_clusters`)
    fn generate_disjoint_clusters(
        params: &Params,
    ) -> (Vec<Array1<f32>>, Vec<usize>, Vec<Array1<f32>>) {
        let mut organism_clusters = Vec::with_capacity(params.num_spawn_clusters);
        let mut pool_assignments = Vec::with_capacity(params.num_spawn_clusters);
        let mut food_clusters = Vec::with_capacity(params.num_spawn_clusters);

        // Minimum distance between organism and food cluster centers
        // Should be at least 2x cluster radius to ensure no overlap
        let min_separation = params.cluster_radius * 2.5;
        let margin = params.cluster_radius;

        // Generate organism clusters first and assign them to pools
        // Distribute clusters evenly across pools
        for i in 0..params.num_spawn_clusters {
            let x = rand::rng().random_range(margin..params.box_width - margin);
            let y = rand::rng().random_range(margin..params.box_height - margin);
            organism_clusters.push(Array1::from_vec(vec![x, y]));

            // Assign cluster to pool (round-robin distribution)
            let pool_id = i % params.num_genetic_pools;
            pool_assignments.push(pool_id);
        }

        // Generate food clusters ensuring they're far from organism clusters
        for _ in 0..params.num_spawn_clusters {
            let mut attempts = 0;
            let max_attempts = 100;

            loop {
                let x = rand::rng().random_range(margin..params.box_width - margin);
                let y = rand::rng().random_range(margin..params.box_height - margin);
                let candidate = Array1::from_vec(vec![x, y]);

                // Check distance to all organism clusters
                let mut far_enough = true;
                for org_cluster in &organism_clusters {
                    let dist = (&candidate - org_cluster).mapv(|x| x.powi(2)).sum().sqrt();
                    if dist < min_separation {
                        far_enough = false;
                        break;
                    }
                }

                if far_enough || attempts >= max_attempts {
                    food_clusters.push(candidate);
                    break;
                }

                attempts += 1;
            }
        }

        (organism_clusters, pool_assignments, food_clusters)
    }

    /// Generates a random position within a random cluster
    fn random_cluster_position(clusters: &[Array1<f32>], cluster_radius: f32) -> Array1<f32> {
        // Pick a random cluster
        let cluster_idx = rand::rng().random_range(0..clusters.len());
        let cluster_center = &clusters[cluster_idx];

        // Generate random offset within cluster radius
        let angle = rand::rng().random::<f32>() * 2.0 * std::f32::consts::PI;
        let radius = rand::rng().random::<f32>().sqrt() * cluster_radius;

        let offset_x = angle.cos() * radius;
        let offset_y = angle.sin() * radius;

        Array1::from_vec(vec![
            cluster_center[0] + offset_x,
            cluster_center[1] + offset_y,
        ])
    }

    /// Generates a random position within a cluster belonging to a specific pool
    fn random_pool_cluster_position(
        clusters: &[Array1<f32>],
        pool_assignments: &[usize],
        pool_id: usize,
        cluster_radius: f32,
    ) -> Array1<f32> {
        // Find all clusters belonging to this pool
        let pool_cluster_indices: Vec<usize> = pool_assignments
            .iter()
            .enumerate()
            .filter(|(_, assigned_pool)| **assigned_pool == pool_id)
            .map(|(idx, _)| idx)
            .collect();

        if pool_cluster_indices.is_empty() {
            // Fallback: use any cluster if no clusters assigned to this pool
            return Self::random_cluster_position(clusters, cluster_radius);
        }

        // Pick a random cluster from those assigned to this pool
        let chosen_idx =
            pool_cluster_indices[rand::rng().random_range(0..pool_cluster_indices.len())];
        let cluster_center = &clusters[chosen_idx];

        // Generate random offset within cluster radius
        let angle = rand::rng().random::<f32>() * 2.0 * std::f32::consts::PI;
        let radius = rand::rng().random::<f32>().sqrt() * cluster_radius;

        let offset_x = angle.cos() * radius;
        let offset_y = angle.sin() * radius;

        Array1::from_vec(vec![
            cluster_center[0] + offset_x,
            cluster_center[1] + offset_y,
        ])
    }

    /// Creates a new ecosystem with random organisms and food.
    pub fn new(params: &Params) -> Self {
        let mut organisms = Vec::with_capacity(params.n_organism);
        let mut food = Vec::with_capacity(params.n_food);

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        // Generate disjoint cluster centers for organisms and food with pool assignments
        let (organism_cluster_centers, cluster_pool_assignments, food_cluster_centers) =
            Self::generate_disjoint_clusters(params);

        for i in 0..params.n_organism {
            // Distribute organisms evenly across genetic pools
            let pool_id = i % params.num_genetic_pools;

            // Spawn organism in a cluster belonging to its pool
            let pos = Self::random_pool_cluster_position(
                &organism_cluster_centers,
                &cluster_pool_assignments,
                pool_id,
                params.cluster_radius,
            );

            let mut entity = organism::Organism::new_random(
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

            // Override position with cluster position
            entity.pos = pos;

            organisms.push(entity);
        }

        for _i in 0..params.n_food {
            // Spawn food in a random food cluster (disjoint from organism clusters)
            let pos = Self::random_cluster_position(&food_cluster_centers, params.cluster_radius);
            let mut food_item = food::Food::new_random(&center, params.food_energy);
            food_item.pos = pos;
            food.push(food_item);
        }

        Self {
            organisms,
            food,
            projectiles: Vec::new(),
            time: 0.,
            generation: params.n_organism as u32,
            reproduction_stats: ReproductionStats::default(),
            evolution_engine: EvolutionEngine::new(params.graveyard_size, params.elite_pool_size),
            energy_shares: Vec::new(),
            reproduction_intents: Vec::new(),
            event_log: EventLog::default(),
            timing_stats: TimingStats::default(),
            organism_cluster_centers,
            cluster_pool_assignments,
            food_cluster_centers,
        }
    }

    /// Advances the simulation by one timestep with parallel organism updates.
    pub fn step(&mut self, params: &Params, dt: f32) {
        let step_start = Instant::now();

        // Build spatial index for efficient neighbor queries
        let spatial_start = Instant::now();
        let spatial_index = SpatialIndex::build(self).expect("Failed to build spatial index");
        self.timing_stats.spatial_index_ms = spatial_start.elapsed().as_secs_f32() * 1000.0;

        self.time += dt;

        // Create perception system for generating brain inputs
        let perception = organism::Perception::default();

        // Get reference to KD-trees for perception system
        let spatial_trees = spatial_index.as_trees();

        // Create Arc wrapper for shared read-only access
        // This does ONE clone before parallelization (unavoidable for now due to perception API)
        let clone_start = Instant::now();
        let ecosystem_snapshot = Arc::new(self.clone());
        self.timing_stats.ecosystem_clone_ms = clone_start.elapsed().as_secs_f32() * 1000.0;

        // Parallel phase: collect events from each organism without mutex contention
        // Use larger chunks to reduce cache line bouncing and task switching overhead
        let parallel_start = Instant::now();
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

                    // Update position based on velocity (via Locatable trait)
                    entity.update(dt);

                    // Apply velocity damping (simulates friction/drag)
                    // entity.vel *= 1.0 - params.velocity_damping;

                    // Consume idle energy
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
        self.timing_stats.parallel_update_ms = parallel_start.elapsed().as_secs_f32() * 1000.0;

        // Update projectiles and check for collisions using KD tree
        let projectile_start = Instant::now();
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
        self.timing_stats.projectile_update_ms = projectile_start.elapsed().as_secs_f32() * 1000.0;

        // Combine all events and apply them
        let event_start = Instant::now();
        let mut combined_events = events::EventQueue::new();
        for event in all_events {
            combined_events.push(event);
        }
        for event in projectile_events {
            combined_events.push(event);
        }

        events::apply_events(self, params, combined_events);
        self.timing_stats.event_application_ms = event_start.elapsed().as_secs_f32() * 1000.0;

        // Record deaths and add to graveyard before removing organisms
        let cleanup_start = Instant::now();
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

        // Clean up old energy share and reproduction visualizations (older than 0.5 seconds)
        let visualization_lifetime = 0.5;
        self.energy_shares
            .retain(|(_, _, timestamp)| self.time - timestamp < visualization_lifetime);
        self.reproduction_intents
            .retain(|(_, _, timestamp)| self.time - timestamp < visualization_lifetime);

        self.timing_stats.cleanup_ms = cleanup_start.elapsed().as_secs_f32() * 1000.0;

        // Update elite pool periodically (every 10 seconds of simulation time)
        if (self.time % 10.0) < dt {
            self.evolution_engine.update_elite_pool(&self.organisms);
        }

        self.timing_stats.total_ms = step_start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Spawns new organisms through evolution when population is below target.
    ///
    /// # Arguments
    /// * `params` - Simulation parameters
    /// * `dt` - Delta time in seconds (spawn rates are per second)
    pub fn spawn(&mut self, params: &Params, dt: f32) {
        // Regenerate cluster centers if empty (happens after loading from file)
        if self.organism_cluster_centers.is_empty() || self.food_cluster_centers.is_empty() {
            let (org_clusters, pool_assignments, food_clusters) =
                Self::generate_disjoint_clusters(params);
            self.organism_cluster_centers = org_clusters;
            self.cluster_pool_assignments = pool_assignments;
            self.food_cluster_centers = food_clusters;
        }

        // Automatic asexual reproduction from living organisms and elite pool
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
            // Select target pool based on sampling strategy
            let target_pool_id = if params.unbalanced_pool_sampling {
                // Sample from largest pool
                self.select_pool_weighted_by_size(params)
            } else {
                // Uniform random sampling
                rand::rng().random_range(0..params.num_genetic_pools)
            };

            // Generate position in a cluster belonging to the target pool
            let spawn_pos = Self::random_pool_cluster_position(
                &self.organism_cluster_centers,
                &self.cluster_pool_assignments,
                target_pool_id,
                params.cluster_radius,
            );

            // Decide which breeding pool to use
            let roll = rand::rng().random::<f32>();

            let mut new_organism = if roll < params.elite_spawn_probability
                && !self.evolution_engine.elite_pool().is_empty()
            {
                // Elite-based: spawn from elite pool (top scorers of all time)
                self.evolution_engine.spawn_organism(
                    self.generation,
                    target_pool_id,
                    &spawn_pos,
                    params,
                    self.evolution_engine.elite_pool(),
                )
            } else if !self.organisms.is_empty() {
                // Reproduction-based: spawn from living organisms using evolution engine
                self.evolution_engine.spawn_organism(
                    self.generation,
                    target_pool_id,
                    &spawn_pos,
                    params,
                    &self.organisms,
                )
            } else {
                // Fallback: create random organism if both pools are empty
                organism::Organism::new_random(
                    self.generation as usize,
                    &spawn_pos,
                    params.signal_size,
                    params.memory_size,
                    params.num_vision_directions,
                    params.vision_radius,
                    params.fov,
                    params.layer_sizes.clone(),
                    target_pool_id,
                    params,
                )
            };

            self.generation += 1;
            new_organism.pos = spawn_pos;
            self.organisms.push(new_organism);
        }

        // Seed empty pools with organisms from non-empty pools
        self.seed_empty_pools(params);

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
                let spawn_pos = Self::random_cluster_position(
                    &self.food_cluster_centers,
                    params.cluster_radius,
                );
                let food_item = food::Food::new_random(&spawn_pos, params.food_energy);
                // food_item.pos = spawn_pos;
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
        let mut ecosystem: Self = serde_json::from_str(&json)?;

        // Cluster centers are not saved (marked with #[serde(skip)])
        // They will be empty after deserialization and need to be regenerated
        // This will happen automatically on the first spawn() call
        ecosystem.organism_cluster_centers = Vec::new();
        ecosystem.cluster_pool_assignments = Vec::new();
        ecosystem.food_cluster_centers = Vec::new();

        Ok(ecosystem)
    }

    /// Generates a random position using the same method as `Organism::new_random`.
    fn random_spawn_position(center: &Array1<f32>, _params: &Params) -> Array1<f32> {
        use ndarray_rand::rand::distributions::Uniform;
        Array1::random(2, Uniform::new(0., 1.)) * center * 2.0
    }

    /// Returns a random position within a cluster belonging to a specific pool.
    /// Regenerates clusters if they're empty (e.g., after loading from file).
    pub fn random_organism_cluster_position_for_pool(
        &mut self,
        params: &Params,
        pool_id: usize,
    ) -> Array1<f32> {
        // Regenerate clusters if empty
        if self.organism_cluster_centers.is_empty() || self.food_cluster_centers.is_empty() {
            let (org_clusters, pool_assignments, food_clusters) =
                Self::generate_disjoint_clusters(params);
            self.organism_cluster_centers = org_clusters;
            self.cluster_pool_assignments = pool_assignments;
            self.food_cluster_centers = food_clusters;
        }

        Self::random_pool_cluster_position(
            &self.organism_cluster_centers,
            &self.cluster_pool_assignments,
            pool_id,
            params.cluster_radius,
        )
    }

    /// Returns a random position within a random food cluster.
    /// Regenerates clusters if they're empty (e.g., after loading from file).
    pub fn random_food_cluster_position(&mut self, params: &Params) -> Array1<f32> {
        // Regenerate clusters if empty
        if self.organism_cluster_centers.is_empty() || self.food_cluster_centers.is_empty() {
            let (org_clusters, pool_assignments, food_clusters) =
                Self::generate_disjoint_clusters(params);
            self.organism_cluster_centers = org_clusters;
            self.cluster_pool_assignments = pool_assignments;
            self.food_cluster_centers = food_clusters;
        }

        Self::random_cluster_position(&self.food_cluster_centers, params.cluster_radius)
    }

    /// Selects a pool ID weighted by pool size (larger pools more likely).
    fn select_pool_weighted_by_size(&self, params: &Params) -> usize {
        // Count organisms per pool
        let mut pool_counts = vec![0; params.num_genetic_pools];
        for organism in &self.organisms {
            if organism.pool_id < params.num_genetic_pools {
                pool_counts[organism.pool_id] += 1;
            }
        }

        let total_organisms: usize = pool_counts.iter().sum();
        if total_organisms == 0 {
            // No organisms, return random pool
            return rand::rng().random_range(0..params.num_genetic_pools);
        }

        // Sample weighted by pool size
        let mut cumulative = 0;
        let target = rand::rng().random_range(0..total_organisms);
        for (pool_id, &count) in pool_counts.iter().enumerate() {
            cumulative += count;
            if target < cumulative {
                return pool_id;
            }
        }

        // Fallback (shouldn't happen)
        params.num_genetic_pools - 1
    }

    /// Seeds empty pools with organisms from non-empty pools.
    fn seed_empty_pools(&mut self, params: &Params) {
        // Count organisms per pool
        let mut pool_counts = vec![0; params.num_genetic_pools];
        for organism in &self.organisms {
            if organism.pool_id < params.num_genetic_pools {
                pool_counts[organism.pool_id] += 1;
            }
        }

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        // Find empty and non-empty pools
        for pool_id in 0..params.num_genetic_pools {
            if pool_counts[pool_id] == 0 {
                // Pool is empty, find a non-empty pool to seed from
                let non_empty_pools: Vec<usize> = pool_counts
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(id, _)| id)
                    .collect();

                if non_empty_pools.is_empty() {
                    // All pools empty, create random organisms
                    for _ in 0..params.empty_pool_seed_count {
                        let new_organism = organism::Organism::new_random(
                            self.generation as usize,
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
                        self.generation += 1;
                        self.organisms.push(new_organism);
                    }
                } else {
                    // Select random non-empty pool
                    let source_pool_id =
                        non_empty_pools[rand::rng().random_range(0..non_empty_pools.len())];

                    // Clone organisms in source pool (collect ownership to avoid borrow issues)
                    let source_organisms: Vec<organism::Organism> = self
                        .organisms
                        .iter()
                        .filter(|org| org.pool_id == source_pool_id)
                        .cloned()
                        .collect();

                    // Create new organisms from cloned source pool to empty pool
                    for _ in 0..params.empty_pool_seed_count {
                        if let Some(parent) = source_organisms
                            .get(rand::rng().random_range(0..source_organisms.len()))
                        {
                            let mut child = parent.clone();
                            child.id = self.generation as usize;
                            child.pool_id = pool_id; // Change to empty pool
                            child.age = 0.0;
                            child.score = 0;
                            child.pos = Self::random_spawn_position(&center, params);
                            child.brain.mutate(0.1); // Mutate to create diversity

                            self.generation += 1;
                            self.organisms.push(child);
                        }
                    }
                }

                // Update pool counts after seeding
                pool_counts[pool_id] = params.empty_pool_seed_count;
            }
        }
    }

    /// Returns a reference to the graveyard.
    pub fn graveyard(&self) -> &[organism::Organism] {
        self.evolution_engine.graveyard()
    }
}
