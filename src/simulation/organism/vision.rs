//! Vision sense - allows organisms to see nearby entities.
//!
//! Vision uses raycasting to detect organisms and food within the field of view.

use ndarray::Array1;

use super::super::ecosystem::Ecosystem;
use super::super::params::Params;
use super::Organism;
use super::sense::Sense;

/// Vision sense that detects organisms and food using raycasting.
///
/// For each vision direction, the sense outputs:
/// - Proximity to nearest entity (inverted distance: 1.0 = very close, 0.0 = far)
/// - Whether the entity is in the same genetic pool (1.0) or not (0.0)
/// - Whether the entity is an organism (1.0) or food (0.0)
pub struct Vision;

impl Vision {
    /// Creates a new vision sense.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Vision {
    fn default() -> Self {
        Self::new()
    }
}

impl Sense for Vision {
    fn sense(
        &self,
        organism: &Organism,
        ecosystem: &Ecosystem,
        params: &Params,
        trees: Option<&super::super::ecosystem::SpatialTrees>,
    ) -> Array1<f32> {
        use super::super::geometric_utils::line_circle_distance;
        use kdtree::KdTree;
        use kdtree::distance::squared_euclidean;

        let num_directions = params.num_vision_directions;
        let mut vision_outputs = Array1::zeros(num_directions * 3);

        // Get vision vectors
        let vision_vectors = organism.get_vision_vectors();

        // Use provided trees or build them
        let (neighbors_orgs, neighbor_foods, neighbor_projectiles) =
            if let Some(spatial_trees) = trees {
                // Use pre-built trees (efficient path) and collect to owned
                let temp_orgs = spatial_trees
                    .organisms
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbors_orgs: Vec<(f32, usize)> =
                    temp_orgs.iter().map(|(d, idx)| (*d, **idx)).collect();

                let temp_foods = spatial_trees
                    .food
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbor_foods: Vec<(f32, usize)> =
                    temp_foods.iter().map(|(d, idx)| (*d, **idx)).collect();

                let temp_projectiles = spatial_trees
                    .projectiles
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbor_projectiles: Vec<(f32, usize)> = temp_projectiles
                    .iter()
                    .map(|(d, idx)| (*d, **idx))
                    .collect();

                (neighbors_orgs, neighbor_foods, neighbor_projectiles)
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

                let mut kd_tree_projectiles = KdTree::with_capacity(2, ecosystem.projectiles.len());
                for (i, proj) in ecosystem.projectiles.iter().enumerate() {
                    let _ = kd_tree_projectiles.add(proj.pos.to_vec(), i);
                }

                // Collect owned indices to match the type from pre-built trees
                let temp_orgs = kd_tree_orgs
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbors_orgs: Vec<(f32, usize)> =
                    temp_orgs.iter().map(|(d, idx)| (*d, **idx)).collect();

                let temp_foods = kd_tree_food
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbor_foods: Vec<(f32, usize)> =
                    temp_foods.iter().map(|(d, idx)| (*d, **idx)).collect();

                let temp_projectiles = kd_tree_projectiles
                    .within(
                        &organism.pos.to_vec(),
                        params.vision_radius.powi(2),
                        &squared_euclidean,
                    )
                    .unwrap_or_default();
                let neighbor_projectiles: Vec<(f32, usize)> = temp_projectiles
                    .iter()
                    .map(|(d, idx)| (*d, **idx))
                    .collect();

                (neighbors_orgs, neighbor_foods, neighbor_projectiles)
            };

        // Raycast for each vision direction
        for (i, vision_vector) in vision_vectors.iter().enumerate() {
            let end_point = &organism.pos + vision_vector;
            let mut min_distance = f32::MAX;

            // Check organisms
            for (_, neighbor_id) in &neighbors_orgs {
                let neighbor_org = &ecosystem.organisms[*neighbor_id];

                if neighbor_org.id == organism.id {
                    continue; // skip self
                }

                let distance = line_circle_distance(&organism.pos, &end_point, &neighbor_org.pos);
                if distance < params.body_radius && distance < min_distance {
                    min_distance = distance;
                    let base_idx = 3 * i;
                    // Invert distance: closer = higher value
                    // Use vision_radius as max distance for normalization
                    let proximity = 1.0 - (distance / params.vision_radius).min(1.0);
                    vision_outputs[base_idx] = proximity;
                    // Pool match: 1.0 if same pool, 0.0 if different pool
                    vision_outputs[base_idx + 1] = if neighbor_org.pool_id == organism.pool_id {
                        1.0
                    } else {
                        0.0
                    };
                    // Is organism: 1.0 for organisms
                    vision_outputs[base_idx + 2] = 1.0;
                }
            }

            // Check food
            for (_, food_id) in &neighbor_foods {
                let food_item = &ecosystem.food[*food_id];
                let distance = line_circle_distance(&organism.pos, &end_point, &food_item.pos);
                if distance < params.body_radius && distance < min_distance {
                    min_distance = distance;
                    let base_idx = 3 * i;
                    // Invert distance: closer = higher value
                    let proximity = 1.0 - (distance / params.vision_radius).min(1.0);
                    vision_outputs[base_idx] = proximity;
                    vision_outputs[base_idx + 1] = 0.0; // no pool match for food
                    vision_outputs[base_idx + 2] = 0.0; // is_organism = 0 for food
                }
            }

            // Check projectiles
            for (_, projectile_id) in &neighbor_projectiles {
                let projectile_item = &ecosystem.projectiles[*projectile_id];

                // Skip projectiles owned by this organism
                if projectile_item.owner_id == organism.id {
                    continue;
                }

                let distance =
                    line_circle_distance(&organism.pos, &end_point, &projectile_item.pos);
                if distance < params.projectile_radius && distance < min_distance {
                    min_distance = distance;
                    let base_idx = 3 * i;
                    // Invert distance: closer = higher value
                    let proximity = 1.0 - (distance / params.vision_radius).min(1.0);
                    vision_outputs[base_idx] = proximity;
                    vision_outputs[base_idx + 1] = 0.0; // no pool match for projectiles
                    vision_outputs[base_idx + 2] = -1.0; // special marker for projectiles
                }
            }
        }

        vision_outputs
    }

    fn input_size(&self, params: &Params) -> usize {
        // 3 outputs per direction: proximity (inverted distance), pool_match, is_organism
        params.num_vision_directions * 3
    }

    fn name(&self) -> &'static str {
        "Vision"
    }
}
