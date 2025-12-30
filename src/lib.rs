//! # Evo - Evolutionary Organism Simulation
//!
//! A simulation of organisms with neural network brains that evolve through natural selection.
//! Organisms can move, eat food, reproduce, and now attack each other using projectiles.
//!
//! ## Features
//!
//! - Neural network brains (MLP with tanh activation)
//! - Genetic algorithm evolution (mutation and crossover)
//! - Vision-based perception system
//! - Food consumption and energy management
//! - Projectile-based combat system
//! - Corpse generation from dead organisms
//! - Real-time visualization with egui/macroquad
//! - Save/load simulation state
//!
//! ## Core Modules
//!
//! - [`simulation::organism`] - Organism behavior and state
//! - [`simulation::brain`] - Neural network implementation
//! - [`simulation::ecosystem`] - Main simulation logic
//! - [`simulation::food`] - Food items for organisms
//! - [`simulation::projectile`] - Attack projectiles
//! - [`simulation::events`] - Event system for thread-safe updates

/// Core simulation logic and data structures.
pub mod simulation {
    /// Neural network implementation for organism brains.
    pub mod brain;
    /// DNA utilities for genetic similarity and breeding.
    pub mod dna;
    /// Main ecosystem simulation with parallel updates.
    pub mod ecosystem;
    /// Event system for thread-safe state updates.
    pub mod events;
    /// Food items that organisms can consume.
    pub mod food;
    /// Geometric utility functions for distance calculations.
    pub mod geometric_utils;
    /// Trait for locatable entities that can be updated.
    ///
    /// The [`locatable::Locatable`] trait is implemented by all entities that have
    /// a position in 2D space and can be updated over time (Food, Organism, Projectile).
    pub mod locatable;
    /// Organism behavior, state, and lifecycle.
    pub mod organism;
    /// Simulation parameters.
    pub mod params;
    /// Attack projectiles fired by organisms.
    pub mod projectile;
    /// Reproduction statistics tracking.
    pub mod reproduction;
}
