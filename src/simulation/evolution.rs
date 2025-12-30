//! Evolution and organism spawning system.
//!
//! Manages organism reproduction, mutation, crossover, and the graveyard
//! of deceased organisms used for breeding.

use ndarray::Array1;
use rand::Rng;

use super::brain::Brain;
use super::dna;
use super::organism::Organism;
use super::params::Params;
use super::reproduction::ReproductionStats;

/// Manages the graveyard and organism spawning/evolution.
#[derive(Debug, Clone)]
pub struct EvolutionEngine {
    /// Graveyard of deceased organisms for breeding selection.
    /// Maintained sorted by fitness (highest first).
    graveyard: Vec<Organism>,
    /// Maximum size of the graveyard.
    max_graveyard_size: usize,
}

impl EvolutionEngine {
    /// Creates a new evolution engine.
    pub fn new(max_graveyard_size: usize) -> Self {
        Self {
            graveyard: Vec::with_capacity(max_graveyard_size),
            max_graveyard_size,
        }
    }

    /// Records an organism's death and adds it to the graveyard.
    ///
    /// Only organisms that lived long enough (age >= 0.5) are added.
    /// Maintains graveyard sorted by fitness.
    pub fn record_death(&mut self, organism: &Organism, stats: &mut ReproductionStats) {
        stats.record_death(organism);

        // Only add organisms that lived long enough
        if organism.age >= 0.5 {
            self.graveyard.push(organism.clone());

            // Maintain graveyard size by keeping only the fittest
            if self.graveyard.len() > self.max_graveyard_size {
                self.graveyard
                    .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
                self.graveyard.truncate(self.max_graveyard_size);
            }
        }
    }

    /// Spawns a new organism through evolution.
    ///
    /// # Arguments
    ///
    /// * `generation` - Current generation number
    /// * `target_pool_id` - Genetic pool for the new organism
    /// * `center` - Spawn position
    /// * `params` - Simulation parameters
    ///
    /// # Returns
    ///
    /// A new organism with genetics from the graveyard.
    pub fn spawn_organism(
        &self,
        generation: u32,
        target_pool_id: usize,
        center: &Array1<f32>,
        params: &Params,
    ) -> Organism {
        // Sort graveyard by fitness
        let mut sorted_graveyard = self.graveyard.clone();
        sorted_graveyard.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        // Create base organism
        let mut new_organism = Organism::new_random(
            generation as usize,
            center,
            params.signal_size,
            params.memory_size,
            params.num_vision_directions,
            params.vision_radius,
            params.fov,
            params.layer_sizes.clone(),
            target_pool_id,
            params,
        );

        new_organism.birth_generation = generation;

        // Logarithmic random sampling for mutation scale
        let mutation_scale = sample_mutation_scale();

        // Get organisms in the target pool
        let pool_organisms: Vec<usize> = sorted_graveyard
            .iter()
            .enumerate()
            .filter(|(_, org)| org.pool_id == target_pool_id)
            .map(|(idx, _)| idx)
            .collect();

        // Apply evolution strategy based on pool population
        if pool_organisms.is_empty() && !sorted_graveyard.is_empty() {
            // Seed from other pools
            self.seed_from_other_pool(&mut new_organism, &sorted_graveyard, mutation_scale, params);
        } else if pool_organisms.len() >= 2 {
            // Choose reproduction strategy
            let reproduction_strategy = rand::rng().random_range(0..2);

            if reproduction_strategy == 0 {
                // Sexual reproduction (crossover)
                self.sexual_reproduction(
                    &mut new_organism,
                    &pool_organisms,
                    &sorted_graveyard,
                    mutation_scale,
                    params,
                );
            } else if pool_organisms.len() >= 10 {
                // Asexual reproduction (cloning with mutation)
                self.asexual_reproduction(
                    &mut new_organism,
                    &pool_organisms,
                    &sorted_graveyard,
                    mutation_scale,
                    params,
                );
            }
        } else if pool_organisms.len() == 1 {
            // Only one organism - clone and mutate
            self.clone_single_parent(
                &mut new_organism,
                &sorted_graveyard[pool_organisms[0]],
                mutation_scale,
                params,
            );
        }

        new_organism
    }

    /// Seeds a new organism from a different genetic pool.
    fn seed_from_other_pool(
        &self,
        new_organism: &mut Organism,
        graveyard: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        let seed_idx = rand::rng().random_range(0..graveyard.len());
        let seed = &graveyard[seed_idx];

        let mut cloned_brain = seed.brain.clone();
        cloned_brain.mutate(mutation_scale * 2.0); // Extra mutation for diversity
        new_organism.brain = cloned_brain;
        new_organism.dna.clone_from(&seed.dna);
        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate * 2.0);
    }

    /// Performs sexual reproduction (crossover between two parents).
    fn sexual_reproduction(
        &self,
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        graveyard: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Decide if we allow inter-pool breeding
        let allow_interbreeding = rand::rng().random::<f32>() < params.pool_interbreed_prob;

        let (candidates, is_same_pool) = if allow_interbreeding && graveyard.len() >= 2 {
            // Inter-pool breeding: select from ALL graveyard organisms
            let all_indices: Vec<usize> = (0..graveyard.len()).collect();
            (all_indices, false)
        } else {
            // Same-pool breeding: select from THIS pool only
            (pool_organisms.to_vec(), true)
        };

        if candidates.len() >= 2 {
            let top_count = (candidates.len() as f32 * 0.15).max(2.0) as usize;
            let top_count = top_count.min(candidates.len());

            // Pick two different parents from top 15%
            let parent_1_idx = rand::rng().random_range(0..top_count);
            let mut parent_2_idx = rand::rng().random_range(0..top_count);

            // Ensure parents are different
            while parent_2_idx == parent_1_idx && top_count > 1 {
                parent_2_idx = rand::rng().random_range(0..top_count);
            }

            let parent_1 = &graveyard[candidates[parent_1_idx]];
            let parent_2 = &graveyard[candidates[parent_2_idx]];

            // Track parent scores
            let avg_parent_score = (parent_1.score + parent_2.score) as f64 / 2.0;
            new_organism.parent_avg_score = avg_parent_score;

            // Mark reproduction method
            if !is_same_pool && parent_1.pool_id != parent_2.pool_id {
                new_organism.reproduction_method = 3; // inter-pool sexual
            } else {
                new_organism.reproduction_method = 2; // same-pool sexual
            }

            // Perform crossover
            let crossover_brain = Brain::crossover(&parent_1.brain, &parent_2.brain);
            new_organism.brain = crossover_brain;

            // Inherit DNA from parents with crossover and mutation
            let alpha = rand::rng().random::<f32>();
            new_organism.dna = dna::crossover(&parent_1.dna, &parent_2.dna, alpha);
            dna::mutate(&mut new_organism.dna, params.dna_mutation_rate);

            // Extra mutation for inter-pool breeding
            if !is_same_pool && parent_1.pool_id != parent_2.pool_id {
                new_organism.brain.mutate(mutation_scale * 0.5);
            }
        }
    }

    /// Performs asexual reproduction (cloning with mutation).
    fn asexual_reproduction(
        &self,
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        graveyard: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        let parent_pool_idx = rand::rng().random_range(0..pool_organisms.len() / 10);
        let parent = &graveyard[pool_organisms[parent_pool_idx]];

        new_organism.parent_avg_score = parent.score as f64;
        new_organism.reproduction_method = 1; // asexual

        let mut cloned_brain = parent.brain.clone();
        cloned_brain.mutate(mutation_scale);
        new_organism.brain = cloned_brain;

        // Inherit DNA with mutation
        new_organism.dna.clone_from(&parent.dna);
        for i in 0..2 {
            let mutation = rand::rng().random_range(-1.0..1.0) * params.dna_mutation_rate;
            new_organism.dna[i] = (new_organism.dna[i] + mutation).clamp(0.0, 1.0);
        }
    }

    /// Clones a single parent organism.
    fn clone_single_parent(
        &self,
        new_organism: &mut Organism,
        parent: &Organism,
        mutation_scale: f32,
        params: &Params,
    ) {
        new_organism.parent_avg_score = parent.score as f64;
        new_organism.reproduction_method = 1; // asexual

        let mut cloned_brain = parent.brain.clone();
        cloned_brain.mutate(mutation_scale);
        new_organism.brain = cloned_brain;
        new_organism.dna.clone_from(&parent.dna);
        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate);
    }

    /// Returns a reference to the graveyard.
    pub fn graveyard(&self) -> &[Organism] {
        &self.graveyard
    }
}

/// Samples a mutation scale using logarithmic random distribution.
fn sample_mutation_scale() -> f32 {
    let min = 0.002f32;
    let max = 0.2f32;
    let log_min = min.ln();
    let log_max = max.ln();
    let log_mutation_scale = rand::rng().random_range(log_min..log_max);
    log_mutation_scale.exp()
}
