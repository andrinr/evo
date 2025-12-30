//! Event system for thread-safe simulation state updates.
//!
//! Uses an event queue to collect state changes from parallel organism updates,
//! then applies them serially to avoid race conditions.

use super::ecosystem::Ecosystem;
use super::event_log::EventColor;
use super::params::Params;
use super::projectile;
use ndarray::Array1;
use std::collections::HashMap;

/// Events that modify simulation state.
///
/// Collected during parallel updates and applied serially to maintain consistency.
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    /// An organism consumed a food item.
    FoodConsumed {
        /// ID of the organism that consumed the food.
        organism_id: usize,
        /// Index of the food item in the food vector.
        food_id: usize,
    },
    /// An organism created a projectile attack.
    ProjectileCreated {
        /// Starting position of the projectile.
        pos: Array1<f32>,
        /// Direction the projectile is fired (in radians).
        rotation: f32,
        /// Damage dealt by the projectile on impact.
        damage: f32,
        /// ID of the organism that fired the projectile.
        owner_id: usize,
    },
    /// An organism died and should be removed.
    OrganismDied {
        /// ID of the organism that died.
        organism_id: usize,
        /// Position where the organism died (for corpse placement).
        pos: Array1<f32>,
    },
    /// A projectile hit an organism.
    ProjectileHit {
        /// Index of the projectile in the projectiles vector.
        projectile_idx: usize,
        /// ID of the organism that was hit.
        target_id: usize,
        /// Amount of damage to deal.
        damage: f32,
        /// ID of the organism that fired the projectile.
        owner_id: usize,
    },
    /// An organism shared energy with another organism.
    EnergyShared {
        /// ID of the organism giving energy.
        giver_id: usize,
        /// ID of the organism receiving energy.
        receiver_id: usize,
        /// Amount of energy to transfer.
        amount: f32,
    },
    /// An organism wants to reproduce asexually.
    AsexualReproduction {
        /// ID of the parent organism.
        parent_id: usize,
        /// Position of the parent.
        parent_pos: Array1<f32>,
        /// Energy contribution from parent to offspring.
        energy_contribution: f32,
    },
    /// An organism wants to reproduce sexually.
    SexualReproductionIntent {
        /// ID of the organism wanting to reproduce.
        organism_id: usize,
        /// ID of the potential partner.
        partner_id: usize,
        /// Energy this organism wants to contribute.
        energy_contribution: f32,
        /// Position of the organism.
        pos: Array1<f32>,
    },
}

/// Queue for collecting simulation events from parallel updates.
pub struct EventQueue {
    events: Vec<SimulationEvent>,
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl EventQueue {
    /// Creates an empty event queue.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Adds an event to the queue.
    pub fn push(&mut self, event: SimulationEvent) {
        self.events.push(event);
    }

    /// Drains all events from the queue.
    pub fn drain(&mut self) -> std::vec::Drain<'_, SimulationEvent> {
        self.events.drain(..)
    }
}

/// Applies all queued events to the ecosystem state.
pub fn apply_events(state: &mut Ecosystem, params: &Params, mut queue: EventQueue) {
    // Remove old interaction visualizations (older than 0.5 seconds)
    const VISUALIZATION_DURATION: f32 = 0.5;
    state
        .energy_shares
        .retain(|(_, _, timestamp)| state.time - timestamp < VISUALIZATION_DURATION);
    state
        .reproduction_intents
        .retain(|(_, _, timestamp)| state.time - timestamp < VISUALIZATION_DURATION);

    // Track which food items are contested
    let mut food_claims: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut dead_organisms_natural: Vec<(usize, Array1<f32>)> = Vec::new(); // Natural deaths (no corpse)
    let mut dead_organisms_combat: Vec<(usize, Array1<f32>)> = Vec::new(); // Combat deaths (spawn corpse)
    let mut projectiles_to_remove: Vec<usize> = Vec::new();
    let mut energy_transfers: Vec<(usize, usize, f32)> = Vec::new();
    let mut asexual_reproductions: Vec<(usize, Array1<f32>, f32)> = Vec::new();
    let mut sexual_reproduction_intents: HashMap<usize, Vec<(usize, f32, Array1<f32>)>> =
        HashMap::new();

    for event in queue.drain() {
        match event {
            SimulationEvent::FoodConsumed {
                organism_id,
                food_id,
            } => {
                food_claims.entry(food_id).or_default().push(organism_id);
            }
            SimulationEvent::ProjectileCreated {
                pos,
                rotation,
                damage,
                owner_id,
            } => {
                let projectile = projectile::Projectile::new(
                    pos,
                    rotation,
                    params.projectile_speed,
                    damage,
                    owner_id,
                    params.projectile_range,
                );
                state.projectiles.push(projectile);
            }
            SimulationEvent::OrganismDied { organism_id, pos } => {
                // Natural death - no corpse spawned
                dead_organisms_natural.push((organism_id, pos));
            }
            SimulationEvent::ProjectileHit {
                projectile_idx,
                target_id,
                damage,
                owner_id,
            } => {
                // Apply damage to target
                let mut target_killed = false;
                let mut target_pos = None;
                if let Some(org) = state.organisms.iter_mut().find(|o| o.id == target_id) {
                    org.consume_energy(damage);
                    if !org.is_alive() {
                        target_killed = true;
                        target_pos = Some(org.pos.clone());
                    }
                }
                // Award point to attacker if target was killed
                if target_killed
                    && let Some(_attacker) = state.organisms.iter_mut().find(|o| o.id == owner_id)
                {
                    // _attacker.score += 1;
                }
                // Create corpse if organism was killed by projectile
                if let Some(pos) = target_pos {
                    dead_organisms_combat.push((target_id, pos));
                }
                // Mark projectile for removal
                projectiles_to_remove.push(projectile_idx);
            }
            SimulationEvent::EnergyShared {
                giver_id,
                receiver_id,
                amount,
            } => {
                energy_transfers.push((giver_id, receiver_id, amount));
            }
            SimulationEvent::AsexualReproduction {
                parent_id,
                parent_pos,
                energy_contribution,
            } => {
                asexual_reproductions.push((parent_id, parent_pos, energy_contribution));
            }
            SimulationEvent::SexualReproductionIntent {
                organism_id,
                partner_id,
                energy_contribution,
                pos,
            } => {
                // Store intent - we'll match pairs later
                sexual_reproduction_intents
                    .entry(partner_id)
                    .or_default()
                    .push((organism_id, energy_contribution, pos));
            }
        }
    }

    // Resolve food consumption - first come first served
    for (food_id, claimants) in food_claims {
        if state.food[food_id].is_consumed() {
            continue;
        }

        if let Some(&winner_id) = claimants.first() {
            if let Some(org) = state.organisms.iter_mut().find(|o| o.id == winner_id) {
                org.gain_energy(state.food[food_id].energy, params.max_energy);
                org.score += 1;
            }
            state.food[food_id].consume();
        }
    }

    // Remove projectiles that hit targets (in reverse order to maintain indices)
    projectiles_to_remove.sort_unstable();
    projectiles_to_remove.reverse();
    for idx in projectiles_to_remove {
        if idx < state.projectiles.len() {
            state.projectiles.remove(idx);
        }
    }

    // Create corpses only from combat deaths (organisms killed by projectiles)
    // Natural deaths do not spawn corpses
    for (organism_id, pos) in dead_organisms_combat {
        let corpse = super::food::Food {
            pos,
            energy: params.corpse_energy_ratio,
            age: 0.0,
        };
        state.food.push(corpse);

        // Log combat death
        state.event_log.log(
            state.time,
            format!("Organism {} killed in combat", organism_id),
            EventColor::Combat,
        );
    }

    // Process energy transfers
    for (giver_id, receiver_id, amount) in energy_transfers {
        // Find giver and deduct energy
        let mut energy_to_give = 0.0;
        if let Some(giver) = state.organisms.iter_mut().find(|o| o.id == giver_id) {
            // Only share if giver has enough energy
            energy_to_give = amount.min(giver.energy * 0.5); // Max 50% of current energy
            giver.consume_energy(energy_to_give);
        }

        // Find receiver and add energy
        if energy_to_give > 0.0
            && let Some(receiver) = state.organisms.iter_mut().find(|o| o.id == receiver_id)
        {
            receiver.gain_energy(energy_to_give, params.max_energy);
            // Add to visualization with timestamp
            state
                .energy_shares
                .push((giver_id, receiver_id, state.time));

            // Log energy sharing
            state.event_log.log(
                state.time,
                format!(
                    "Organism {} shared {:.1} energy with organism {}",
                    giver_id, energy_to_give, receiver_id
                ),
                EventColor::Sharing,
            );
        }
    }

    // Process asexual reproductions
    for (parent_id, parent_pos, energy_contribution) in asexual_reproductions {
        if let Some(parent) = state.organisms.iter_mut().find(|o| o.id == parent_id)
            && parent.energy >= energy_contribution + 0.5
        {
            // Deduct energy from parent
            parent.consume_energy(energy_contribution);

            // Create offspring using parent's brain with mutation
            let mut offspring = super::organism::Organism::new_random(
                state.generation as usize,
                &parent_pos,
                params.signal_size,
                params.memory_size,
                params.num_vision_directions,
                params.vision_radius,
                params.fov,
                params.layer_sizes.clone(),
                parent.pool_id,
                params,
            );

            // Clone and mutate parent brain
            offspring.brain = parent.brain.clone();
            offspring.brain.mutate(0.05); // Small mutation for asexual reproduction

            // Set offspring properties - offspring gets multiplied energy
            offspring.energy = energy_contribution * params.reproduction_energy_multiplier;
            offspring.birth_generation = state.generation;
            offspring.reproduction_method = 1; // asexual
            offspring.parent_avg_score = parent.score as f64;
            offspring.dna.clone_from(&parent.dna);
            super::dna::mutate(&mut offspring.dna, params.dna_mutation_rate);

            state.generation += 1;
            let offspring_id = offspring.id;
            state.organisms.push(offspring);

            // Log asexual reproduction
            state.event_log.log(
                state.time,
                format!(
                    "Organism {} reproduced asexually (offspring: {})",
                    parent_id, offspring_id
                ),
                EventColor::Reproduction,
            );
        }
    }

    // Process sexual reproductions - match organisms that both want to reproduce with each other
    let mut sexual_reproductions: Vec<(usize, usize, f32, f32, Array1<f32>)> = Vec::new();
    for (partner_id, intents) in &sexual_reproduction_intents {
        // Check if the partner also wants to reproduce with any of these organisms
        if let Some(partner_intents) = sexual_reproduction_intents.get(partner_id) {
            for (organism_id, org_energy, org_pos) in intents {
                // Check if partner wants to reproduce with this organism
                if let Some((_, partner_energy, _)) =
                    partner_intents.iter().find(|(id, _, _)| id == organism_id)
                {
                    // Both organisms want to reproduce with each other
                    // Use the position of the first organism
                    sexual_reproductions.push((
                        *organism_id,
                        *partner_id,
                        *org_energy,
                        *partner_energy,
                        org_pos.clone(),
                    ));
                    // Add to visualization with timestamp
                    state
                        .reproduction_intents
                        .push((*organism_id, *partner_id, state.time));
                    break; // Only one match per organism
                }
            }
        }
    }

    // Execute sexual reproductions
    for (parent1_id, parent2_id, energy1, energy2, pos) in sexual_reproductions {
        // Find both parents
        let parent1_idx = state.organisms.iter().position(|o| o.id == parent1_id);
        let parent2_idx = state.organisms.iter().position(|o| o.id == parent2_id);

        if let (Some(p1_idx), Some(p2_idx)) = (parent1_idx, parent2_idx) {
            // Check both parents have enough energy
            if state.organisms[p1_idx].energy >= energy1 + 0.5
                && state.organisms[p2_idx].energy >= energy2 + 0.5
            {
                // Clone parents for genetic material
                let parent1 = state.organisms[p1_idx].clone();
                let parent2 = state.organisms[p2_idx].clone();

                // Deduct energy from both parents
                state.organisms[p1_idx].consume_energy(energy1);
                state.organisms[p2_idx].consume_energy(energy2);

                // Calculate weighted average ratio based on energy contribution
                let total_energy = energy1 + energy2;
                let weight1 = energy1 / total_energy;

                // Create offspring
                let mut offspring = super::organism::Organism::new_random(
                    state.generation as usize,
                    &pos,
                    params.signal_size,
                    params.memory_size,
                    params.num_vision_directions,
                    params.vision_radius,
                    params.fov,
                    params.layer_sizes.clone(),
                    parent1.pool_id, // Inherit pool from first parent
                    params,
                );

                // Perform weighted crossover based on energy contributions
                offspring.brain = super::brain::Brain::crossover_weighted(
                    &parent1.brain,
                    &parent2.brain,
                    weight1,
                );

                // Set offspring properties - offspring gets multiplied energy
                offspring.energy = total_energy * params.reproduction_energy_multiplier;
                offspring.birth_generation = state.generation;
                offspring.reproduction_method = if parent1.pool_id == parent2.pool_id {
                    2 // same-pool sexual
                } else {
                    3 // inter-pool sexual
                };
                offspring.parent_avg_score = (parent1.score + parent2.score) as f64 / 2.0;

                // DNA crossover
                offspring.dna = super::dna::crossover(&parent1.dna, &parent2.dna, weight1);
                super::dna::mutate(&mut offspring.dna, params.dna_mutation_rate);

                state.generation += 1;
                let offspring_id = offspring.id;
                state.organisms.push(offspring);

                // Log sexual reproduction
                state.event_log.log(
                    state.time,
                    format!(
                        "Organisms {} and {} reproduced sexually (offspring: {})",
                        parent1_id, parent2_id, offspring_id
                    ),
                    EventColor::Reproduction,
                );
            }
        }
    }
}
