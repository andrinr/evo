//! Scent sense - allows organisms to smell nearby entities.
//!
//! Scent detects the chemical signals and DNA distance of nearby organisms.

use ndarray::Array1;

use super::super::ecosystem::Ecosystem;
use super::super::params::Params;
use super::Organism;
use super::sense::Sense;

/// Scent sense that detects chemical signals and DNA similarity.
///
/// Outputs:
/// - Signal channels from the nearest organism (RGB color)
/// - DNA distance to the nearest organism
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
        use super::super::dna;
        use kdtree::KdTree;
        use kdtree::distance::squared_euclidean;

        let mut scent_outputs = Array1::zeros(params.signal_size + 1);

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

        // Scent: signal (RGB) + DNA distance to nearest organism
        let mut scent_signal = Array1::zeros(params.signal_size);
        let mut scent_count = 0;
        let mut closest_dna_distance = 0.0f32;
        let mut min_org_distance = f32::MAX;

        // Add organism signals and find closest organism for DNA distance
        for (_, org_id) in &scent_orgs {
            let neighbor_org = &ecosystem.organisms[*org_id];
            if neighbor_org.id == organism.id {
                continue; // Skip self
            }
            // Add all signal components (RGB)
            for i in 0..params.signal_size {
                scent_signal[i] += neighbor_org.signal[i];
            }
            scent_count += 1;

            // Track closest organism for DNA distance
            let dist = (&organism.pos - &neighbor_org.pos)
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            if dist < min_org_distance {
                min_org_distance = dist;
                // Calculate DNA distance with periodic boundary conditions
                closest_dna_distance = dna::periodic_distance(&organism.dna, &neighbor_org.dna);
            }
        }

        // Add food signals (no DNA for food)
        for (_, _food_id) in &scent_foods {
            scent_count += 1;
            scent_signal[2] += 1.0; // B-channel
        }

        if scent_count > 0 {
            scent_signal /= scent_count as f32; // Average
        }

        // Copy signal to outputs
        for i in 0..params.signal_size {
            scent_outputs[i] = scent_signal[i];
        }
        // Add DNA distance to closest organism
        scent_outputs[params.signal_size] = closest_dna_distance;

        scent_outputs
    }

    fn input_size(&self, params: &Params) -> usize {
        // signal_size channels + 1 for DNA distance
        params.signal_size + 1
    }

    fn name(&self) -> &'static str {
        "Scent"
    }
}
