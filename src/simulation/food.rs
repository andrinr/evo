//! Food items that organisms can consume for energy.
//!
//! Food can be either randomly spawned or created from organism corpses.

use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// A food item that organisms can consume for energy.
///
/// Food items have a position and energy value. When an organism consumes food,
/// it gains the food's energy. Corpses from dead organisms also become food items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Food {
    /// Position in 2D space.
    pub pos: Array1<f32>,
    /// Energy value remaining. Regular food has 1.0, corpses have less based on `corpse_energy_ratio`.
    pub energy: f32,
    /// Age of the food item in seconds.
    pub age: f32,
}

impl Food {
    /// Creates a new food item at a random position.
    ///
    /// # Arguments
    ///
    /// * `screen_center` - The center point of the simulation area, used to calculate bounds.
    /// * `energy` - The energy value for this food item.
    ///
    /// # Returns
    ///
    /// A new `Food` instance with random position and specified energy.
    pub fn new_random(screen_center: &Array1<f32>, energy: f32) -> Self {
        Self {
            pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
            energy,
            age: 0.0,
        }
    }

    /// Checks if this food item has been fully consumed.
    ///
    /// # Returns
    ///
    /// `true` if energy is <= 0, `false` otherwise.
    pub fn is_consumed(&self) -> bool {
        self.energy <= 0.0
    }

    /// Marks this food as consumed by setting energy to 0.
    pub fn consume(&mut self) {
        self.energy = 0.0;
    }
}
