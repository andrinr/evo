//! Main ecosystem simulation with parallel organism updates.
//!
//! The ecosystem manages all organisms, food, and projectiles. It handles:
//! - Parallel organism brain updates using rayon
//! - Spatial queries using k-d trees for efficient neighbor finding
//! - Event-driven state updates for thread safety
//! - Organism spawning, reproduction, and evolution

use super::brain;
use super::dna;
use super::events;
use super::food;
use super::organism;
use super::projectile;

use super::geometric_utils::wrap_around_mut;
use super::params::Params;
use super::reproduction::ReproductionStats;
use kdtree::distance::squared_euclidean;
use kdtree::{ErrorKind as KdTreeError, KdTree};
use ndarray::{Array1, s};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

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
    /// Graveyard of deceased organisms for breeding selection.
    /// Maintains the fittest organisms that have died, sorted by score (highest first).
    pub graveyard: Vec<organism::Organism>,
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
            graveyard: Vec::with_capacity(params.graveyard_size),
        }
    }

    /// Advances the simulation by one timestep with parallel organism updates.
    pub fn step(&mut self, params: &Params, dt: f32) {
        let (kd_tree_orgs, kd_tree_food, kd_tree_projectiles) =
            build_trees(self).expect("Failed to build kd-trees");

        // Clone the organisms vector
        let new_organisms = self.organisms.clone();

        let event_queue = Mutex::new(events::EventQueue::new());

        self.time += dt;

        // Create perception system for generating brain inputs
        let perception = organism::Perception::default();

        // Wrap trees in SpatialTrees struct for passing to perception system
        let spatial_trees = SpatialTrees {
            organisms: &kd_tree_orgs,
            food: &kd_tree_food,
            projectiles: &kd_tree_projectiles,
        };

        // Create a snapshot Ecosystem for read-only access in the parallel loop
        let ecosystem_snapshot = self.clone();

        // parallel phase, only apply updates to entity itself
        // for events involing other objects, use the event queue for thread safety
        self.organisms.par_iter_mut().for_each(|entity| {
            let mut local_events = Vec::new();

            // wrap around the screen
            wrap_around_mut(&mut entity.pos, params.box_width, params.box_height);

            // Get nearest neighbors for collision detection
            let neighbors_orgs = kd_tree_orgs
                .within(
                    &entity.pos.to_vec(),
                    params.vision_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_else(|e| {
                    panic!("Error finding neighbors: {:?}", e);
                });

            let neighbor_foods = kd_tree_food
                .within(
                    &entity.pos.to_vec(),
                    params.vision_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_else(|e| {
                    panic!("Error finding food neighbors: {:?}", e);
                });

            // Check for collisions with other organisms
            for (_, neighbor_id) in &neighbors_orgs {
                let neighbor_org = &new_organisms[**neighbor_id];
                if neighbor_org.id == entity.id {
                    continue; // skip self
                }
                let org_org_distance = (&entity.pos - &neighbor_org.pos).mapv(f32::abs).sum();
                if org_org_distance < params.body_radius * 2.0 {
                    entity.kill(); // collision with another organism
                }
            }

            // Generate brain inputs using perception system
            let brain_inputs =
                perception.perceive(entity, &ecosystem_snapshot, params, Some(&spatial_trees));

            // Store brain inputs for visualization
            entity.last_brain_inputs.clone_from(&brain_inputs);

            let brain_outputs = entity.brain.think(&brain_inputs);

            entity.signal = brain_outputs.slice(s![..params.signal_size]).to_owned();
            entity.memory = brain_outputs
                .slice(s![
                    params.signal_size..params.signal_size + params.memory_size
                ])
                .to_owned();

            let offset = params.signal_size + params.memory_size;
            let rot = brain_outputs[offset];
            entity.rot += rot * dt * 10.0; // rotation adjustment

            // update age and cooldown
            entity.age_by(dt);
            entity.update_cooldown(dt);

            let vel = brain_outputs[offset + 1]; // acceleration
            let attack_strength = brain_outputs[offset + 2]; // attack action
            let share_amount = brain_outputs[offset + 3]; // energy sharing

            let vel_vector = Array1::from_vec(vec![vel * entity.rot.cos(), vel * entity.rot.sin()])
                * params.move_multiplier; // scale acceleration

            entity.pos += &(&vel_vector * dt); // update velocity
            entity.consume_energy(vel.abs() * dt * params.move_energy_rate); // energy consumption for acceleration
            entity.consume_energy(rot.abs() * dt * params.rot_energy_rate); // energy consumption for rotation
            entity.consume_energy(params.idle_energy_rate * dt); // additional energy consumption

            // Handle attack/projectile shooting (with cooldown check)
            if attack_strength > 0.1
                && entity.energy > attack_strength * params.attack_cost_rate
                && entity.can_attack()
            {
                entity.consume_energy(attack_strength * params.attack_cost_rate);
                entity.reset_attack_cooldown(params.attack_cooldown);

                local_events.push(events::SimulationEvent::ProjectileCreated {
                    pos: entity.pos.clone(),
                    rotation: entity.rot,
                    damage: attack_strength * params.attack_damage_rate,
                    owner_id: entity.id,
                });
            }

            // Handle energy sharing with nearest organism
            if share_amount > 0.1 && entity.energy > 0.2 {
                // Find nearest organism within share_radius
                let mut nearest_dist = f32::MAX;
                let mut nearest_id = None;

                for (_, neighbor_id) in &neighbors_orgs {
                    let other = &new_organisms[**neighbor_id];
                    if other.id != entity.id {
                        let dist = (&entity.pos - &other.pos).mapv(f32::abs).sum();
                        if dist < params.share_radius && dist < nearest_dist {
                            nearest_dist = dist;
                            nearest_id = Some(other.id);
                        }
                    }
                }

                if let Some(receiver_id) = nearest_id {
                    local_events.push(events::SimulationEvent::EnergyShared {
                        giver_id: entity.id,
                        receiver_id,
                        amount: share_amount,
                    });
                }
            }

            // consume all food within BODY_RADIUS
            for (_, food_id) in &neighbor_foods {
                let food_item = &ecosystem_snapshot.food[**food_id];
                let org_food_dist = (&entity.pos - &food_item.pos).mapv(f32::abs).sum();
                if org_food_dist < params.body_radius * 2.0 && !food_item.is_consumed() {
                    entity.gain_energy(food_item.energy, params.max_energy);
                    entity.score += 1; // increase score for reproduction

                    local_events.push(events::SimulationEvent::FoodConsumed {
                        organism_id: entity.id,
                        food_id: **food_id,
                    });
                }
            }

            let mut queue = event_queue.lock().unwrap();
            for event in local_events {
                queue.push(event);
            }
        });

        // Update projectiles and check for collisions using KD tree
        let mut projectile_events = Vec::new();
        for (proj_idx, projectile) in self.projectiles.iter_mut().enumerate() {
            projectile.update(dt);

            // Use KD tree to find nearby organisms within collision range
            let collision_radius = params.body_radius + params.projectile_radius;
            let nearby_organisms = kd_tree_orgs
                .within(
                    &projectile.pos.to_vec(),
                    collision_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_else(|e| {
                    panic!("Error finding nearby organisms for projectile: {e:?}");
                });

            // Check collision with nearby organisms
            for (_, org_id) in &nearby_organisms {
                let organism = &self.organisms[**org_id];

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

        // Add projectile events to queue
        {
            let mut queue = event_queue.lock().unwrap();
            for event in projectile_events {
                queue.push(event);
            }
        }

        events::apply_events(self, params, event_queue.into_inner().unwrap());

        // Record deaths and add to graveyard before removing organisms
        for organism in &self.organisms {
            if !organism.is_alive() {
                self.reproduction_stats.record_death(organism);

                // Add to graveyard (skip organisms that died too quickly)
                if organism.age >= 0.5 {
                    self.graveyard.push(organism.clone());
                }
            }
        }

        // Maintain graveyard size by keeping only the fittest
        if self.graveyard.len() > params.graveyard_size {
            // Sort by fitness (age + score, highest first)
            self.graveyard
                .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
            // Keep only the top graveyard_size organisms
            self.graveyard.truncate(params.graveyard_size);
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
        // Sort graveyard by fitness in descending order (keep fittest at front)
        // Note: graveyard is already sorted when organisms are added, but we sort here for safety
        self.graveyard
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        // spawn new organisms if there are less than n_organism (respecting max_organism cap)
        // Use probabilistic spawning: spawn rate is per second, multiply by dt
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

            // Get organisms in this pool FROM GRAVEYARD (not from living organisms)
            let pool_organisms: Vec<usize> = self
                .graveyard
                .iter()
                .enumerate()
                .filter(|(_, org)| org.pool_id == target_pool_id)
                .map(|(idx, _)| idx)
                .collect();

            let mut new_organism = organism::Organism::new_random(
                self.generation as usize,
                &center,
                params.signal_size,
                params.memory_size,
                params.num_vision_directions,
                params.vision_radius,
                params.fov,
                params.layer_sizes.clone(),
                target_pool_id,
                params,
            );

            new_organism.birth_generation = self.generation;
            self.generation += 1;

            // Logarithmic random sampling for mutation scale
            let min = 0.002f32;
            let max = 0.2f32;
            let log_min = min.ln();
            let log_max = max.ln();
            let log_mutation_scale = rand::rng().random_range(log_min..log_max);
            let mutation_scale = log_mutation_scale.exp();

            // If pool is empty in graveyard, seed from other pools in graveyard
            if pool_organisms.is_empty() && !self.graveyard.is_empty() {
                // Pick a random organism from any other pool in graveyard as a seed
                let seed_idx = rand::rng().random_range(0..self.graveyard.len());
                let seed = &self.graveyard[seed_idx];

                // Clone and mutate the seed organism into the new pool
                let mut cloned_brain = seed.brain.clone();
                cloned_brain.mutate(mutation_scale * 2.0); // Extra mutation for diversity
                new_organism.brain = cloned_brain;
                new_organism.dna.clone_from(&seed.dna);
                dna::mutate(&mut new_organism.dna, params.dna_mutation_rate * 2.0);
            } else if pool_organisms.len() >= 2 {
                // choose reproduction strategy randomly
                let reproduction_strategy = rand::rng().random_range(0..2);

                if reproduction_strategy == 0 {
                    // Crossover: pick two organisms from top 15% (possibly from different pools)

                    // Decide if we allow inter-pool breeding for this organism
                    let allow_interbreeding =
                        rand::rng().random::<f32>() < params.pool_interbreed_prob;

                    let (candidates, is_same_pool) =
                        if allow_interbreeding && self.graveyard.len() >= 2 {
                            // Inter-pool breeding: select from ALL graveyard organisms
                            let all_indices: Vec<usize> = (0..self.graveyard.len()).collect();
                            (all_indices, false)
                        } else {
                            // Same-pool breeding: select from THIS pool only
                            (pool_organisms.clone(), true)
                        };

                    if candidates.len() >= 2 {
                        let top_count = (candidates.len() as f32 * 0.15).max(2.0) as usize;
                        let top_count = top_count.min(candidates.len());

                        // Pick two different parents from top 15%
                        let parent_1_idx = rand::rng().random_range(0..top_count);
                        let mut parent_2_idx = rand::rng().random_range(0..top_count);

                        // Ensure parents are different
                        while parent_2_idx == parent_1_idx && top_count > 1 {
                            parent_2_idx = rand::rng().random_range(0..top_count);
                        }

                        let parent_1 = &self.graveyard[candidates[parent_1_idx]];
                        let parent_2 = &self.graveyard[candidates[parent_2_idx]];

                        // Track parent scores for later comparison
                        let avg_parent_score = (parent_1.score + parent_2.score) as f64 / 2.0;
                        new_organism.parent_avg_score = avg_parent_score;

                        // Mark reproduction method
                        if !is_same_pool && parent_1.pool_id != parent_2.pool_id {
                            new_organism.reproduction_method = 3; // inter-pool sexual
                        } else {
                            new_organism.reproduction_method = 2; // same-pool sexual
                        }

                        // Perform crossover
                        let crossover_brain =
                            brain::Brain::crossover(&parent_1.brain, &parent_2.brain);
                        new_organism.brain = crossover_brain;

                        // Inherit DNA from parents with crossover and mutation
                        let alpha = rand::rng().random::<f32>();
                        new_organism.dna = dna::crossover(&parent_1.dna, &parent_2.dna, alpha);
                        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate);

                        // If parents from different pools, apply extra mutation for diversity
                        if !is_same_pool && parent_1.pool_id != parent_2.pool_id {
                            new_organism.brain.mutate(mutation_scale * 0.5);
                        }
                    }
                } else if pool_organisms.len() >= 10 {
                    // Asexual: clone with mutation from THIS POOL (from graveyard)
                    let parent_pool_idx = rand::rng().random_range(0..pool_organisms.len() / 10);
                    let parent = &self.graveyard[pool_organisms[parent_pool_idx]];

                    // Track parent score for later comparison
                    new_organism.parent_avg_score = parent.score as f64;
                    new_organism.reproduction_method = 1; // asexual

                    let mut cloned_brain = parent.brain.clone();
                    cloned_brain.mutate(mutation_scale);
                    new_organism.brain = cloned_brain;

                    // Inherit DNA with mutation
                    new_organism.dna.clone_from(&parent.dna);
                    for i in 0..2 {
                        let mutation =
                            rand::rng().random_range(-1.0..1.0) * params.dna_mutation_rate;
                        new_organism.dna[i] = (new_organism.dna[i] + mutation).clamp(0.0, 1.0);
                    }
                }
            } else if pool_organisms.len() == 1 {
                // Only one organism in pool - clone and mutate it (from graveyard)
                let parent = &self.graveyard[pool_organisms[0]];

                // Track parent score for later comparison
                new_organism.parent_avg_score = parent.score as f64;
                new_organism.reproduction_method = 1; // asexual

                let mut cloned_brain = parent.brain.clone();
                cloned_brain.mutate(mutation_scale);
                new_organism.brain = cloned_brain;
                new_organism.dna.clone_from(&parent.dna);
                dna::mutate(&mut new_organism.dna, params.dna_mutation_rate);
            }

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
}

/// Type alias for 2D spatial KD-tree used for efficient neighbor queries.
pub type Tree2D = KdTree<f32, usize, Vec<f32>>;

/// Container for pre-built KD-trees for spatial queries.
pub struct SpatialTrees<'a> {
    /// KD-tree for organism positions.
    pub organisms: &'a Tree2D,
    /// KD-tree for food positions.
    pub food: &'a Tree2D,
    /// KD-tree for projectile positions.
    pub projectiles: &'a Tree2D,
}

fn build_tree<T>(items: &[T], get_pos: impl Fn(&T) -> Vec<f32>) -> Result<Tree2D, KdTreeError> {
    let mut tree = KdTree::with_capacity(2, items.len());
    for (i, item) in items.iter().enumerate() {
        tree.add(get_pos(item), i)?;
    }
    Ok(tree)
}

fn build_trees(ecosystem: &Ecosystem) -> Result<(Tree2D, Tree2D, Tree2D), KdTreeError> {
    let kd_tree_orgs = build_tree(&ecosystem.organisms, |org| org.pos.to_vec())?;
    let kd_tree_food = build_tree(&ecosystem.food, |food| food.pos.to_vec())?;
    let kd_tree_projectiles = build_tree(&ecosystem.projectiles, |proj| proj.pos.to_vec())?;
    Ok((kd_tree_orgs, kd_tree_food, kd_tree_projectiles))
}
