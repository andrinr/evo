//! Spatial indexing for efficient neighbor queries.
//!
//! Provides a unified interface for building and querying KD-trees for spatial queries.

use kdtree::distance::squared_euclidean;
use kdtree::{ErrorKind as KdTreeError, KdTree};
use ndarray::Array1;

use super::ecosystem::Ecosystem;

/// Type alias for 2D spatial KD-tree used for efficient neighbor queries.
pub type Tree2D = KdTree<f32, usize, Vec<f32>>;

/// Container for pre-built KD-trees for spatial queries.
pub struct SpatialTrees<'a> {
    /// KD-tree for organism positions.
    pub organisms: &'a Tree2D,
    /// KD-tree for food positions.
    pub food: &'a Tree2D,
    /// KD-tree for projectile positions.
    pub projectiles: &'a Tree2D,
}

/// Spatial index containing KD-trees for all entity types.
pub struct SpatialIndex {
    /// KD-tree for organism positions.
    organisms: Tree2D,
    /// KD-tree for food positions.
    food: Tree2D,
    /// KD-tree for projectile positions.
    projectiles: Tree2D,
}

/// Result of a spatial radius query.
pub type SpatialQueryResult = Vec<(f32, usize)>;

impl SpatialIndex {
    /// Builds a spatial index from the current ecosystem state.
    ///
    /// # Arguments
    ///
    /// * `ecosystem` - The ecosystem to build indices for
    ///
    /// # Returns
    ///
    /// A spatial index or an error if tree building fails.
    pub fn build(ecosystem: &Ecosystem) -> Result<Self, KdTreeError> {
        let organisms = build_tree(&ecosystem.organisms, |org| org.pos.to_vec())?;
        let food = build_tree(&ecosystem.food, |f| f.pos.to_vec())?;
        let projectiles = build_tree(&ecosystem.projectiles, |proj| proj.pos.to_vec())?;

        Ok(Self {
            organisms,
            food,
            projectiles,
        })
    }

    /// Returns references to the trees as a `SpatialTrees` struct.
    pub fn as_trees(&self) -> SpatialTrees<'_> {
        SpatialTrees {
            organisms: &self.organisms,
            food: &self.food,
            projectiles: &self.projectiles,
        }
    }

    /// Query organisms within a radius.
    ///
    /// # Arguments
    ///
    /// * `pos` - Center position for the query
    /// * `radius` - Search radius (will be squared internally)
    ///
    /// # Returns
    ///
    /// Vector of (`distance_squared`, index) pairs for organisms within radius.
    pub fn query_organisms(&self, pos: &Array1<f32>, radius: f32) -> SpatialQueryResult {
        self.organisms
            .within(&pos.to_vec(), radius.powi(2), &squared_euclidean)
            .unwrap_or_default()
            .into_iter()
            .map(|(dist, &idx)| (dist, idx))
            .collect()
    }

    /// Query food within a radius.
    pub fn query_food(&self, pos: &Array1<f32>, radius: f32) -> SpatialQueryResult {
        self.food
            .within(&pos.to_vec(), radius.powi(2), &squared_euclidean)
            .unwrap_or_default()
            .into_iter()
            .map(|(dist, &idx)| (dist, idx))
            .collect()
    }

    /// Query projectiles within a radius.
    pub fn query_projectiles(&self, pos: &Array1<f32>, radius: f32) -> SpatialQueryResult {
        self.projectiles
            .within(&pos.to_vec(), radius.powi(2), &squared_euclidean)
            .unwrap_or_default()
            .into_iter()
            .map(|(dist, &idx)| (dist, idx))
            .collect()
    }

    /// Get direct references to the trees (for backwards compatibility).
    pub fn organisms_tree(&self) -> &Tree2D {
        &self.organisms
    }

    /// Get direct reference to the food KD-tree.
    pub fn food_tree(&self) -> &Tree2D {
        &self.food
    }

    /// Get direct reference to the projectiles KD-tree.
    pub fn projectiles_tree(&self) -> &Tree2D {
        &self.projectiles
    }
}

/// Helper function to build a KD-tree from a collection of items.
fn build_tree<T>(items: &[T], get_pos: impl Fn(&T) -> Vec<f32>) -> Result<Tree2D, KdTreeError> {
    let mut tree = KdTree::with_capacity(2, items.len());
    for (i, item) in items.iter().enumerate() {
        tree.add(get_pos(item), i)?;
    }
    Ok(tree)
}
