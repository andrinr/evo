//! Projectile system for organism attacks.
//!
//! Organisms can shoot projectiles in the direction they're facing to attack other organisms.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A projectile fired by an organism during an attack.
///
/// Projectiles travel in a straight line at constant velocity and expire after
/// traveling a maximum distance. They deal damage on collision with organisms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projectile {
    /// Current position in 2D space.
    pub pos: Array1<f32>,
    /// Velocity vector (direction and speed).
    pub velocity: Array1<f32>,
    /// Damage dealt on impact.
    pub damage: f32,
    /// ID of the organism that fired this projectile (to prevent self-damage).
    pub owner_id: usize,
    /// Total distance traveled so far.
    pub distance_traveled: f32,
    /// Maximum distance before expiring.
    pub max_range: f32,
}

impl Projectile {
    /// Creates a new projectile.
    ///
    /// # Arguments
    ///
    /// * `pos` - Starting position
    /// * `rotation` - Firing direction in radians
    /// * `speed` - Projectile speed
    /// * `damage` - Damage dealt on impact
    /// * `owner_id` - ID of the firing organism
    /// * `max_range` - Maximum travel distance before expiring
    ///
    /// # Returns
    ///
    /// A new `Projectile` instance with velocity calculated from rotation and speed.
    pub fn new(
        pos: Array1<f32>,
        rotation: f32,
        speed: f32,
        damage: f32,
        owner_id: usize,
        max_range: f32,
    ) -> Self {
        let velocity = Array1::from_vec(vec![rotation.cos() * speed, rotation.sin() * speed]);

        Self {
            pos,
            velocity,
            damage,
            owner_id,
            distance_traveled: 0.0,
            max_range,
        }
    }

    /// Updates projectile position based on velocity and time delta.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time delta since last update
    pub fn update(&mut self, dt: f32) {
        let displacement = &self.velocity * dt;
        let distance = displacement.mapv(|x| x.powi(2)).sum().sqrt();

        self.pos += &displacement;
        self.distance_traveled += distance;
    }

    /// Checks if the projectile has exceeded its maximum range.
    ///
    /// # Returns
    ///
    /// `true` if the projectile should be removed, `false` otherwise.
    pub fn is_expired(&self) -> bool {
        self.distance_traveled >= self.max_range
    }
}
