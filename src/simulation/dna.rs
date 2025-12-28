//! DNA utilities for genetic similarity and breeding.
//!
//! DNA is represented as a 2D vector in [0, 1] x [0, 1] space with periodic
//! boundary conditions (toroidal topology).

use ndarray::Array1;

/// Calculates the periodic distance between two DNA vectors.
///
/// DNA space is [0, 1] x [0, 1] with periodic boundary conditions (toroidal topology).
/// For each dimension, the distance is min(|a - b|, 1 - |a - b|).
///
/// # Arguments
///
/// * `dna1` - First DNA vector
/// * `dna2` - Second DNA vector
///
/// # Returns
///
/// Euclidean distance in periodic space (range: 0.0 to ~0.707)
pub fn periodic_distance(dna1: &Array1<f32>, dna2: &Array1<f32>) -> f32 {
    let mut sum_sq = 0.0;
    for i in 0..dna1.len() {
        let diff = (dna1[i] - dna2[i]).abs();
        let periodic_diff = diff.min(1.0 - diff);
        sum_sq += periodic_diff * periodic_diff;
    }
    sum_sq.sqrt()
}

/// Applies periodic wrapping to a DNA value.
///
/// # Arguments
///
/// * `value` - DNA value (may be outside [0, 1])
///
/// # Returns
///
/// Wrapped value in [0, 1]
pub fn wrap(value: f32) -> f32 {
    value.rem_euclid(1.0)
}

/// Mutates a DNA vector with periodic wrapping.
///
/// # Arguments
///
/// * `dna` - DNA vector to mutate (modified in place)
/// * `mutation_rate` - Standard deviation of Gaussian mutation
pub fn mutate(dna: &mut Array1<f32>, mutation_rate: f32) {
    for i in 0..dna.len() {
        let mutation = rand::random::<f32>() * 2.0 - 1.0; // Range: [-1, 1]
        dna[i] = wrap(dna[i] + mutation * mutation_rate);
    }
}

/// Performs DNA crossover between two parents.
///
/// # Arguments
///
/// * `parent1` - First parent DNA
/// * `parent2` - Second parent DNA
/// * `alpha` - Crossover ratio (0 = all parent2, 1 = all parent1)
///
/// # Returns
///
/// New DNA vector as weighted average of parents
pub fn crossover(parent1: &Array1<f32>, parent2: &Array1<f32>, alpha: f32) -> Array1<f32> {
    parent1 * alpha + parent2 * (1.0 - alpha)
}
