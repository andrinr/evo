use super::logic::{Params, State};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum SimulationEvent {
    FoodConsumed { organism_id: usize, food_id: usize },
}

pub struct EventQueue {
    events: Vec<SimulationEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn push(&mut self, event: SimulationEvent) {
        self.events.push(event);
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, SimulationEvent> {
        self.events.drain(..)
    }
}

pub fn apply_events(state: &mut State, _params: &Params, mut queue: EventQueue) {
    // Track which food items are contested
    let mut food_claims: HashMap<usize, Vec<usize>> = HashMap::new();

    for event in queue.drain() {
        match event {
            SimulationEvent::FoodConsumed {
                organism_id,
                food_id,
            } => {
                food_claims.entry(food_id).or_default().push(organism_id);
            }
        }
    }

    // Resolve food consumption - first come first served
    for (food_id, claimants) in food_claims {
        if state.food[food_id].energy <= 0.0 {
            continue;
        }

        if let Some(&winner_id) = claimants.first() {
            if let Some(org) = state.organisms.iter_mut().find(|o| o.id == winner_id) {
                org.energy = (org.energy + state.food[food_id].energy).min(1.0);
                org.score += 1;
            }
            state.food[food_id].energy = 0.0;
        }
    }
}
