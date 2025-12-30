//! Trait for entities that have a position and can be updated.
//!
//! This trait provides a common interface for all entities in the simulation
//! that have a position and need to be updated each frame.

use ndarray::Array1;

/// Trait for entities with a position that can be updated over time.
///
/// Any type that implements this trait:
/// - Has a position in 2D space
/// - Can be updated with a time delta
pub trait Locatable {
    /// Returns a reference to the entity's position.
    ///
    /// # Returns
    ///
    /// A reference to the 2D position vector.
    fn pos(&self) -> &Array1<f32>;

    /// Returns a mutable reference to the entity's position.
    ///
    /// # Returns
    ///
    /// A mutable reference to the 2D position vector.
    fn pos_mut(&mut self) -> &mut Array1<f32>;

    /// Updates the entity's state based on the time delta.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time delta since the last update in seconds.
    fn update(&mut self, dt: f32);
}
