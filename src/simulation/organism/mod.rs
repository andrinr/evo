//! Organism module containing organism behavior and perception systems.

mod organism;
mod perception;
mod proprioception;
mod scent;
mod sense;
mod vision;

// Re-export everything from the organism module
pub use organism::*;

// Re-export perception system components
pub use perception::Perception;
pub use proprioception::Proprioception;
pub use scent::Scent;
pub use sense::Sense;
pub use vision::Vision;
