//! Event system for thread-safe simulation state updates.
//!
//! Uses an event queue to collect state changes from parallel organism updates,
//! then applies them serially to avoid race conditions.

use super::ecosystem::{Ecosystem, Params};
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
    // Track which food items are contested
    let mut food_claims: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut dead_organisms: Vec<(usize, Array1<f32>)> = Vec::new();
    let mut projectiles_to_remove: Vec<usize> = Vec::new();
    let mut energy_transfers: Vec<(usize, usize, f32)> = Vec::new();

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
                dead_organisms.push((organism_id, pos));
            }
            SimulationEvent::ProjectileHit {
                projectile_idx,
                target_id,
                damage,
                owner_id,
            } => {
                // Apply damage to target
                let mut target_killed = false;
                if let Some(org) = state.organisms.iter_mut().find(|o| o.id == target_id) {
                    org.consume_energy(damage);
                    if !org.is_alive() {
                        target_killed = true;
                    }
                }
                // Award point to attacker if target was killed
                if target_killed
                    && let Some(attacker) = state.organisms.iter_mut().find(|o| o.id == owner_id)
                {
                    attacker.score += 1;
                    println!("Projectile kill!")
                }
                // Mark projectile for removal
                projectiles_to_remove.push(projectile_idx);
            }
            SimulationEvent::EnergyShared {
                giver_id,
                receiver_id,
                amount,
            } => {
                println!("Shared energy");
                energy_transfers.push((giver_id, receiver_id, amount));
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

    // Create corpses from dead organisms
    for (_organism_id, pos) in dead_organisms {
        let corpse = super::food::Food {
            pos,
            energy: params.corpse_energy_ratio,
            age: 0.0,
        };
        state.food.push(corpse);
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
        }
    }
}
