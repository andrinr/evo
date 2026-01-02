//! Scent sense - allows organisms to smell nearby entities.
//!
//! Scent detects the chemical signals and DNA distance of nearby organisms.

use ndarray::Array1;

use super::super::ecosystem::Ecosystem;
use super::super::params::Params;
use super::Organism;
use super::sense::Sense;

/// Scent sense that detects chemical signals.
///
/// Outputs:
/// - Signal channels from nearby organisms (RGB color), weighted by distance (closer = stronger)
///
/// The scent strength falls off linearly with distance:
/// - 1.0 at distance 0
/// - 0.0 at `scent_radius`
pub struct Scent;

impl Scent {
    /// Creates a new scent sense.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Scent {
    fn default() -> Self {
        Self::new()
    }
}

impl Sense for Scent {
    fn sense(
        &self,
        organism: &Organism,
        ecosystem: &Ecosystem,
        params: &Params,
        trees: Option<&super::super::ecosystem::SpatialTrees>,
    ) -> Array1<f32> {
        use kdtree::KdTree;
        use kdtree::distance::squared_euclidean;

        let mut scent_outputs = Array1::zeros(params.signal_size);

        // Use provided trees or build them
        let (scent_orgs, scent_foods) = if let Some(spatial_trees) = trees {
            // Use pre-built trees (efficient path) and collect to owned
            let temp_orgs = spatial_trees
                .organisms
                .within(
                    &organism.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_default();
            let scent_orgs: Vec<(f32, usize)> =
                temp_orgs.iter().map(|(d, idx)| (*d, **idx)).collect();

            let temp_foods = spatial_trees
                .food
                .within(
                    &organism.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_default();
            let scent_foods: Vec<(f32, usize)> =
                temp_foods.iter().map(|(d, idx)| (*d, **idx)).collect();

            (scent_orgs, scent_foods)
        } else {
            // Build trees on demand (for testing) and collect results immediately
            let mut kd_tree_orgs = KdTree::with_capacity(2, ecosystem.organisms.len());
            for (i, org) in ecosystem.organisms.iter().enumerate() {
                let _ = kd_tree_orgs.add(org.pos.to_vec(), i);
            }

            let mut kd_tree_food = KdTree::with_capacity(2, ecosystem.food.len());
            for (i, food) in ecosystem.food.iter().enumerate() {
                let _ = kd_tree_food.add(food.pos.to_vec(), i);
            }

            let temp_orgs = kd_tree_orgs
                .within(
                    &organism.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_default();
            let scent_orgs: Vec<(f32, usize)> =
                temp_orgs.iter().map(|(d, idx)| (*d, **idx)).collect();

            let temp_foods = kd_tree_food
                .within(
                    &organism.pos.to_vec(),
                    params.scent_radius.powi(2),
                    &squared_euclidean,
                )
                .unwrap_or_default();
            let scent_foods: Vec<(f32, usize)> =
                temp_foods.iter().map(|(d, idx)| (*d, **idx)).collect();

            (scent_orgs, scent_foods)
        };

        // Scent: signal channels from nearby entities
        let mut scent_signal = Array1::zeros(params.signal_size);

        // Add organism signals weighted by distance (closer = stronger)
        for (_, org_id) in &scent_orgs {
            let neighbor_org = &ecosystem.organisms[*org_id];
            if neighbor_org.id == organism.id {
                continue; // Skip self
            }

            // Calculate distance
            let dist = (&organism.pos - &neighbor_org.pos)
                .mapv(|x| x * x)
                .sum()
                .sqrt();

            // Distance falloff: 1.0 at distance 0, 0.0 at scent_radius
            let distance_factor = (1.0 - (dist / params.scent_radius)).max(0.0);

            // Add signal components multiplied by distance factor
            for i in 0..params.signal_size {
                scent_signal[i] += neighbor_org.signal[i] * distance_factor;
            }
        }

        // Add food signals weighted by distance
        for (_, food_id) in &scent_foods {
            let food_item = &ecosystem.food[*food_id];

            // Calculate distance
            let dist = (&organism.pos - &food_item.pos)
                .mapv(|x| x * x)
                .sum()
                .sqrt();

            // Distance falloff: 1.0 at distance 0, 0.0 at scent_radius
            let distance_factor = (1.0 - (dist / params.scent_radius)).max(0.0);

            // Food adds to B-channel (blue) weighted by distance
            scent_signal[2] += 1.0 * distance_factor;
        }

        // Copy signal to outputs
        scent_outputs.assign(&scent_signal);

        scent_outputs
    }

    fn input_size(&self, params: &Params) -> usize {
        // signal_size channels
        params.signal_size
    }

    fn name(&self) -> &'static str {
        "Scent"
    }
}
