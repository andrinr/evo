//! Main ecosystem simulation with parallel organism updates.
//!
//! The ecosystem manages all organisms, food, and projectiles. It handles:
//! - Parallel organism brain updates using rayon
//! - Spatial queries using k-d trees for efficient neighbor finding
//! - Event-driven state updates for thread safety
//! - Organism spawning, reproduction, and evolution

use super::brain;
use super::events;
use super::food;
use super::organism;
use super::projectile;

use geo::algorithm::Distance;
use geo::{Euclidean, Line, Point};
use kdtree::distance::squared_euclidean;
use kdtree::{ErrorKind as KdTreeError, KdTree};
use ndarray::{Array1, s};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// Simulation parameters that control ecosystem behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    /// Organism collision radius.
    pub body_radius: f32,
    /// Maximum distance organisms can see.
    pub vision_radius: f32,
    /// Maximum distance organisms can smell food.
    pub scent_radius: f32,
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
}

impl Ecosystem {
    /// Creates a new ecosystem with random organisms and food.
    pub fn new(params: &Params) -> Self {
        let mut organisms = Vec::with_capacity(params.n_organism);
        let mut food = Vec::with_capacity(params.n_food);

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        for i in 0..params.n_organism {
            let entity = organism::Organism::new_random(
                i,
                &center,
                params.signal_size,
                params.memory_size,
                params.num_vision_directions,
                params.vision_radius,
                params.fov,
                params.layer_sizes.clone(),
            );

            organisms.push(entity);
        }

        for _i in 0..params.n_food {
            let food_item = food::Food::new_random(&center);
            food.push(food_item);
        }

        Self {
            organisms,
            food,
            projectiles: Vec::new(),
            time: 0.,
            generation: params.n_organism as u32,
        }
    }

    /// Advances the simulation by one timestep with parallel organism updates.
    pub fn step(&mut self, params: &Params, dt: f32) {
        let (kd_tree_orgs, kd_tree_food) = build_trees(self).expect("Failed to build kd-trees");

        // Clone the organisms vector
        let new_organisms = self.organisms.clone();

        let event_queue = Mutex::new(events::EventQueue::new());

        self.time += dt;

        // parallel phase, only apply updates to entity itself
        // for events involing other objects, use the event queue for thread safety
        self.organisms.par_iter_mut().for_each(|entity| {
            let mut local_events = Vec::new();

            let vision_vectors = entity.get_vision_vectors();

            // wrap around the screen
            wrap_around_mut(&mut entity.pos, params.box_width, params.box_height);

            // get nearest neighbors
            let neighbors_orgs = kd_tree_orgs.within(
                &entity.pos.to_vec(),
                params.vision_radius.powi(2),
                &squared_euclidean,
            );

            let neighbor_foods = kd_tree_food.within(
                &entity.pos.to_vec(),
                params.vision_radius.powi(2),
                &squared_euclidean,
            );

            // the above returns a Result, so we need to handle the error
            let neighbors_orgs = match neighbors_orgs {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding neighbors: {:?}", e);
                }
            };

            let neighbor_foods = match neighbor_foods {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding food neighbors: {:?}", e);
                }
            };

            // collect all the signals from neighbor organisms and food
            // Input structure: [vision rays] + [scent] + [memory] + [energy]
            let mut brain_inputs = Array1::zeros(
                (params.signal_size + 1) * params.num_vision_directions
                    + params.signal_size // scent inputs (RGB)
                    + params.memory_size
                    + 1, // energy
            );

            for (i, vision_vector) in vision_vectors.iter().enumerate() {
                let end_point = &entity.pos + vision_vector;
                let mut min_distance = f32::MAX;

                // detect neighbor organisms within the vision vector
                for (_, neighbor_id) in &neighbors_orgs {
                    let neighbor_org = &new_organisms[**neighbor_id];

                    if neighbor_org.id == entity.id {
                        continue; // skip self
                    }
                    let distance = line_circle_distance(&entity.pos, &end_point, &neighbor_org.pos);
                    if distance < params.body_radius && distance < min_distance {
                        min_distance = distance;
                        brain_inputs[i * 2] = neighbor_org.signal[0];
                        brain_inputs[(i * 2) + 1] = neighbor_org.signal[1];
                        brain_inputs[(i * 2) + 2] = neighbor_org.signal[2];
                        brain_inputs[(i * 2) + 3] = distance;
                    }

                    let org_org_distance = (&entity.pos - &neighbor_org.pos).mapv(f32::abs).sum();

                    if org_org_distance < params.body_radius * 2.0 {
                        entity.kill(); // collision with another organism
                    }
                }

                // detect neighbor food within the vision vector
                for (_, food_id) in &neighbor_foods {
                    let food_item = &self.food[**food_id];
                    let distance = line_circle_distance(&entity.pos, &end_point, &food_item.pos);
                    if distance < params.body_radius && distance < min_distance {
                        min_distance = distance;
                        brain_inputs[(params.signal_size + 1) * i] = 0.0;
                        brain_inputs[(params.signal_size + 1) * i + 1] = 0.2; // food signal color (green)
                        brain_inputs[(params.signal_size + 1) * i + 2] = 1.0; // food y position
                        brain_inputs[(params.signal_size + 1) * i + 3] = distance; // distance to food
                    }
                }
            }

            // Calculate scent: average signal of nearby organisms and food within scent_radius
            let scent_orgs = kd_tree_orgs
                .within(
                    &entity.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_else(|e| {
                    panic!("Error finding scent neighbors (orgs): {e:?}");
                });

            let scent_foods = kd_tree_food
                .within(
                    &entity.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_else(|e| {
                    panic!("Error finding scent neighbors (food): {e:?}");
                });

            let mut scent_signal = Array1::zeros(params.signal_size);
            let mut scent_count = 0;

            // Add organism signals
            for (_, org_id) in &scent_orgs {
                let neighbor_org = &new_organisms[**org_id];
                if neighbor_org.id == entity.id {
                    continue; // Skip self
                }
                // Add all signal components (works for any signal_size)
                for i in 0..params.signal_size {
                    scent_signal[i] += neighbor_org.signal[i];
                }
                scent_count += 1;
            }

            // Add food signals
            for (_, _food_id) in &scent_foods {
                scent_count += 1;
                scent_signal[2] += 1.0; // B-channel
            }

            if scent_count > 0 {
                scent_signal /= scent_count as f32; // Average
            }

            let mut offset = (params.signal_size + 1) * params.num_vision_directions;
            // Add scent to inputs
            brain_inputs
                .slice_mut(s![offset..offset + params.signal_size])
                .assign(&scent_signal);

            offset += params.signal_size;
            // Add the organism's own memory to the inputs
            brain_inputs
                .slice_mut(s![offset..offset + params.memory_size])
                .assign(&entity.memory);
            brain_inputs[offset + params.memory_size] = entity.energy; // energy

            // Store brain inputs for visualization
            entity.last_brain_inputs = brain_inputs.clone();

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

            let vel_vector = Array1::from_vec(vec![vel * entity.rot.cos(), vel * entity.rot.sin()])
                * params.move_multiplier; // scale acceleration

            entity.pos += &(&vel_vector * dt); // update velocity
            entity.consume_energy(vel.abs() * dt * params.move_energy_rate); // energy consumption for acceleration
            // entity.consume_energy(rot.abs() * dt * params.rot_energy_rate); // energy consumption for rotation
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

            // consume all food within BODY_RADIUS
            for (_, food_id) in &neighbor_foods {
                let food_item = &self.food[**food_id];
                let org_food_dist = (&entity.pos - &food_item.pos).mapv(f32::abs).sum();
                if org_food_dist < params.body_radius * 2.0 && !food_item.is_consumed() {
                    entity.gain_energy(food_item.energy, 1.0);
                    entity.score += 1; // increase score for reproduction

                    local_events.push(events::SimulationEvent::FoodConsumed {
                        organism_id: entity.id,
                        food_id: **food_id,
                    });

                    println!("org {} consumed {}", entity.id, food_id);
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
        // sort organisms by score in descending order
        self.organisms.sort_by(|a, b| b.score.cmp(&a.score));

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
            let mut new_organism = organism::Organism::new_random(
                self.generation as usize,
                &center,
                params.signal_size,
                params.memory_size,
                params.num_vision_directions,
                params.vision_radius,
                params.fov,
                params.layer_sizes.clone(),
            );

            self.generation += 1;

            // Logarithmic random sampling for mutation scale
            let min = 0.0002f32;
            let max = 0.2f32;
            let log_min = min.ln();
            let log_max = max.ln();
            let log_mutation_scale = rand::rng().random_range(log_min..log_max);
            let mutation_scale = log_mutation_scale.exp();

            println!("mutation scale: {}", mutation_scale);

            // choose reproduction strategy randomly
            let reproduction_strategy = rand::rng().random_range(0..2);

            if reproduction_strategy == 0 && self.organisms.len() >= 10 {
                // pick two random organisms to reproduce
                let id_a = rand::rng().random_range(0..self.organisms.len() / 5); // pick from the top 10% of organisms
                let id_b = rand::rng().random_range(0..self.organisms.len() / 10); // pick from the top 10% of organisms

                let parent_1 = &self.organisms[id_a];
                let parent_2 = &self.organisms[id_b];

                let crossover_brain = brain::Brain::crossover(&parent_1.brain, &parent_2.brain);
                // crossover_brain.mutate(mutation_scale);

                new_organism.brain = crossover_brain;
            } else if reproduction_strategy == 1 && self.organisms.len() >= 10 {
                let id = rand::rng().random_range(0..self.organisms.len() / 15); // pick from the top 10% of organisms

                let parent = &self.organisms[id];

                let mut cloned_brain = parent.brain.clone();
                cloned_brain.mutate(mutation_scale);

                new_organism.brain = cloned_brain;
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
                let food_item = food::Food::new_random(&center);
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

type Tree2D = KdTree<f32, usize, Vec<f32>>;

fn build_tree<T>(items: &[T], get_pos: impl Fn(&T) -> Vec<f32>) -> Result<Tree2D, KdTreeError> {
    let mut tree = KdTree::with_capacity(2, items.len());
    for (i, item) in items.iter().enumerate() {
        tree.add(get_pos(item), i)?;
    }
    Ok(tree)
}

fn build_trees(ecosystem: &Ecosystem) -> Result<(Tree2D, Tree2D), KdTreeError> {
    let kd_tree_orgs = build_tree(&ecosystem.organisms, |org| org.pos.to_vec())?;
    let kd_tree_food = build_tree(&ecosystem.food, |food| food.pos.to_vec())?;
    Ok((kd_tree_orgs, kd_tree_food))
}

fn line_circle_distance(
    line_start: &Array1<f32>,
    line_end: &Array1<f32>,
    circle_center: &Array1<f32>,
) -> f32 {
    let p = Point::new(circle_center[0], circle_center[1]);
    let line = Line::new(
        Point::new(line_start[0], line_start[1]),
        Point::new(line_end[0], line_end[1]),
    );
    Euclidean.distance(&p, &line)
}

fn wrap_around_mut(v: &mut Array1<f32>, box_width: f32, box_height: f32) {
    v[0] = v[0].rem_euclid(box_width);
    v[1] = v[1].rem_euclid(box_height);
}
