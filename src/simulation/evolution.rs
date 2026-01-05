//! Evolution and organism spawning system.
//!
//! Manages organism reproduction, mutation, crossover, and breeding pool
//! of organisms (either deceased from graveyard or living organisms).

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
    /// Elite pool - top scoring organisms preserved across all time.
    /// These are the best organisms ever seen and are kept for breeding.
    elite_pool: Vec<Organism>,
    /// Maximum size of the elite pool.
    max_elite_pool_size: usize,
}

/// Source of organisms for breeding.
#[derive(Debug, Clone, Copy)]
pub enum BreedingSource {
    /// Use deceased organisms from the graveyard
    Graveyard,
    /// Use currently living organisms
    Living,
}

impl EvolutionEngine {
    /// Creates a new evolution engine.
    pub fn new(max_graveyard_size: usize, max_elite_pool_size: usize) -> Self {
        Self {
            graveyard: Vec::with_capacity(max_graveyard_size),
            max_graveyard_size,
            elite_pool: Vec::with_capacity(max_elite_pool_size),
            max_elite_pool_size,
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
    /// * `breeding_pool` - Pool of organisms to breed from (can be graveyard or living organisms)
    ///
    /// # Returns
    ///
    /// A new organism with genetics from the breeding pool.
    pub fn spawn_organism(
        &self,
        generation: u32,
        target_pool_id: usize,
        center: &Array1<f32>,
        params: &Params,
        breeding_pool: &[Organism],
    ) -> Organism {
        // Create indices sorted by fitness (avoid cloning all organisms)
        let mut sorted_indices: Vec<usize> = (0..breeding_pool.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            breeding_pool[b]
                .fitness()
                .partial_cmp(&breeding_pool[a].fitness())
                .unwrap()
        });

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

        // Get organisms in the target pool (indices into sorted_indices)
        let pool_organisms: Vec<usize> = sorted_indices
            .iter()
            .enumerate()
            .filter(|(_, idx)| breeding_pool[**idx].pool_id == target_pool_id)
            .map(|(pos, _)| pos)
            .collect();

        // Apply evolution strategy based on pool population
        if pool_organisms.is_empty() && !sorted_indices.is_empty() {
            // Seed from other pools
            Self::seed_from_other_pool_indexed(
                &mut new_organism,
                breeding_pool,
                &sorted_indices,
                mutation_scale,
                params,
            );
        } else if pool_organisms.len() >= 2 {
            // Choose reproduction strategy
            let reproduction_strategy = rand::rng().random_range(0..2);

            if reproduction_strategy == 0 {
                // Sexual reproduction (crossover)
                Self::sexual_reproduction_indexed(
                    &mut new_organism,
                    &pool_organisms,
                    breeding_pool,
                    &sorted_indices,
                    mutation_scale,
                    params,
                );
            } else if pool_organisms.len() >= 10 {
                // Asexual reproduction (cloning with mutation)
                Self::asexual_reproduction_indexed(
                    &mut new_organism,
                    &pool_organisms,
                    breeding_pool,
                    &sorted_indices,
                    mutation_scale,
                    params,
                );
            }
        } else if pool_organisms.len() == 1 {
            // Only one organism - clone and mutate
            let parent_idx = sorted_indices[pool_organisms[0]];
            Self::clone_single_parent(
                &mut new_organism,
                &breeding_pool[parent_idx],
                mutation_scale,
                params,
            );
        }

        new_organism
    }

    /// Helper method to spawn from graveyard using the old cloning approach.
    pub fn spawn_from_graveyard(
        &self,
        generation: u32,
        target_pool_id: usize,
        center: &Array1<f32>,
        params: &Params,
    ) -> Organism {
        // Use the old clone-based approach for graveyard (small size, OK to clone)
        Self::spawn_from_pool_cloned(generation, target_pool_id, center, params, &self.graveyard)
    }

    /// Spawns from a pool using the cloning approach (for small pools like graveyard).
    fn spawn_from_pool_cloned(
        generation: u32,
        target_pool_id: usize,
        center: &Array1<f32>,
        params: &Params,
        breeding_pool: &[Organism],
    ) -> Organism {
        // Sort breeding pool by fitness
        let mut sorted_pool = breeding_pool.to_vec();
        sorted_pool.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

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
        let pool_organisms: Vec<usize> = sorted_pool
            .iter()
            .enumerate()
            .filter(|(_, org)| org.pool_id == target_pool_id)
            .map(|(idx, _)| idx)
            .collect();

        // Apply evolution strategy based on pool population
        if pool_organisms.is_empty() && !sorted_pool.is_empty() {
            // Seed from other pools
            Self::seed_from_other_pool(&mut new_organism, &sorted_pool, mutation_scale, params);
        } else if pool_organisms.len() >= 2 {
            // Choose reproduction strategy
            let reproduction_strategy = rand::rng().random_range(0..2);

            if reproduction_strategy == 0 {
                // Sexual reproduction (crossover)
                Self::sexual_reproduction(
                    &mut new_organism,
                    &pool_organisms,
                    &sorted_pool,
                    mutation_scale,
                    params,
                );
            } else if pool_organisms.len() >= 10 {
                // Asexual reproduction (cloning with mutation)
                Self::asexual_reproduction(
                    &mut new_organism,
                    &pool_organisms,
                    &sorted_pool,
                    mutation_scale,
                    params,
                );
            }
        } else if pool_organisms.len() == 1 {
            // Only one organism - clone and mutate
            Self::clone_single_parent(
                &mut new_organism,
                &sorted_pool[pool_organisms[0]],
                mutation_scale,
                params,
            );
        }

        new_organism
    }

    /// Seeds a new organism from a different genetic pool (indexed version).
    fn seed_from_other_pool_indexed(
        new_organism: &mut Organism,
        breeding_pool: &[Organism],
        sorted_indices: &[usize],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Safety check: ensure breeding pool is not empty
        if sorted_indices.is_empty() {
            return;
        }

        let pos = rand::rng().random_range(0..sorted_indices.len());
        let seed = &breeding_pool[sorted_indices[pos]];

        let mut cloned_brain = seed.brain.clone();
        cloned_brain.mutate_with_params(mutation_scale * 2.0, params.use_targeted_mutation); // Extra mutation for diversity
        new_organism.brain = cloned_brain;
        new_organism.dna.clone_from(&seed.dna);
        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate * 2.0);
    }

    /// Seeds a new organism from a different genetic pool.
    fn seed_from_other_pool(
        new_organism: &mut Organism,
        breeding_pool: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Safety check: ensure breeding pool is not empty
        if breeding_pool.is_empty() {
            return;
        }

        let seed_idx = rand::rng().random_range(0..breeding_pool.len());
        let seed = &breeding_pool[seed_idx];

        let mut cloned_brain = seed.brain.clone();
        cloned_brain.mutate_with_params(mutation_scale * 2.0, params.use_targeted_mutation); // Extra mutation for diversity
        new_organism.brain = cloned_brain;
        new_organism.dna.clone_from(&seed.dna);
        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate * 2.0);
    }

    /// Performs sexual reproduction (crossover between two parents).
    fn sexual_reproduction(
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        breeding_pool: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Decide if we allow inter-pool breeding
        let allow_interbreeding = rand::rng().random::<f32>() < params.pool_interbreed_prob;

        let (candidates, is_same_pool) = if allow_interbreeding && breeding_pool.len() >= 2 {
            // Inter-pool breeding: select from ALL breeding pool organisms
            let all_indices: Vec<usize> = (0..breeding_pool.len()).collect();
            (all_indices, false)
        } else {
            // Same-pool breeding: select from THIS pool only
            (pool_organisms.to_vec(), true)
        };

        if candidates.len() >= 2 {
            // Sample from top 50% instead of top 15% for more diversity
            let top_count = (candidates.len() as f32 * 0.2).max(2.0) as usize;
            let top_count = top_count.min(candidates.len());

            // Pick two different parents from top 50%
            let parent_1_idx = rand::rng().random_range(0..top_count);
            let mut parent_2_idx = rand::rng().random_range(0..top_count);

            // Ensure parents are different
            while parent_2_idx == parent_1_idx && top_count > 1 {
                parent_2_idx = rand::rng().random_range(0..top_count);
            }

            let parent_1 = &breeding_pool[candidates[parent_1_idx]];
            let parent_2 = &breeding_pool[candidates[parent_2_idx]];

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
                new_organism
                    .brain
                    .mutate_with_params(mutation_scale * 0.5, params.use_targeted_mutation);
            }
        }
    }

    /// Performs asexual reproduction (cloning with mutation).
    fn asexual_reproduction(
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        breeding_pool: &[Organism],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Sample from top 50% instead of top 10% for more diversity
        let top_count = (pool_organisms.len() as f32 * 0.2).max(1.0) as usize;
        let parent_pool_idx = rand::rng().random_range(0..top_count.min(pool_organisms.len()));
        let parent = &breeding_pool[pool_organisms[parent_pool_idx]];

        new_organism.parent_avg_score = parent.score as f64;
        new_organism.reproduction_method = 1; // asexual

        let mut cloned_brain = parent.brain.clone();
        cloned_brain.mutate_with_params(mutation_scale, params.use_targeted_mutation);
        new_organism.brain = cloned_brain;

        // Inherit DNA with mutation
        new_organism.dna.clone_from(&parent.dna);
        for i in 0..2 {
            let mutation = rand::rng().random_range(-1.0..1.0) * params.dna_mutation_rate;
            new_organism.dna[i] = (new_organism.dna[i] + mutation).clamp(0.0, 1.0);
        }
    }

    /// Performs sexual reproduction (indexed version for living organisms).
    fn sexual_reproduction_indexed(
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        breeding_pool: &[Organism],
        sorted_indices: &[usize],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Decide if we allow inter-pool breeding
        let allow_interbreeding = rand::rng().random::<f32>() < params.pool_interbreed_prob;

        let (candidates, is_same_pool) = if allow_interbreeding && sorted_indices.len() >= 2 {
            // Inter-pool breeding: select from ALL organisms
            (sorted_indices, false)
        } else {
            // Same-pool breeding: select from THIS pool only
            (pool_organisms, true)
        };

        if candidates.len() >= 2 {
            // Sample from top 50% instead of top 15% for more diversity
            let top_count = (candidates.len() as f32 * 0.15).max(8.0) as usize;
            let top_count = top_count.min(candidates.len());

            // Pick two different parents from top 50%
            let parent_1_pos = rand::rng().random_range(0..top_count);
            let mut parent_2_pos = rand::rng().random_range(0..top_count);

            // Ensure parents are different
            while parent_2_pos == parent_1_pos && top_count > 1 {
                parent_2_pos = rand::rng().random_range(0..top_count);
            }

            let parent_1_idx = if is_same_pool {
                sorted_indices[candidates[parent_1_pos]]
            } else {
                candidates[parent_1_pos]
            };
            let parent_2_idx = if is_same_pool {
                sorted_indices[candidates[parent_2_pos]]
            } else {
                candidates[parent_2_pos]
            };

            let parent_1 = &breeding_pool[parent_1_idx];
            let parent_2 = &breeding_pool[parent_2_idx];

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
                new_organism
                    .brain
                    .mutate_with_params(mutation_scale * 0.5, params.use_targeted_mutation);
            }
        }
    }

    /// Performs asexual reproduction (indexed version for living organisms).
    fn asexual_reproduction_indexed(
        new_organism: &mut Organism,
        pool_organisms: &[usize],
        breeding_pool: &[Organism],
        sorted_indices: &[usize],
        mutation_scale: f32,
        params: &Params,
    ) {
        // Sample from top 50% instead of top 10% for more diversity
        let top_count = (pool_organisms.len() as f32 * 0.15).max(8.0) as usize;
        let parent_pool_pos = rand::rng().random_range(0..top_count.min(pool_organisms.len()));
        let parent_idx = sorted_indices[pool_organisms[parent_pool_pos]];
        let parent = &breeding_pool[parent_idx];

        new_organism.parent_avg_score = parent.score as f64;
        new_organism.reproduction_method = 1; // asexual

        let mut cloned_brain = parent.brain.clone();
        cloned_brain.mutate_with_params(mutation_scale, params.use_targeted_mutation);
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
        new_organism: &mut Organism,
        parent: &Organism,
        mutation_scale: f32,
        params: &Params,
    ) {
        new_organism.parent_avg_score = parent.score as f64;
        new_organism.reproduction_method = 1; // asexual

        let mut cloned_brain = parent.brain.clone();
        cloned_brain.mutate_with_params(mutation_scale, params.use_targeted_mutation);
        new_organism.brain = cloned_brain;
        new_organism.dna.clone_from(&parent.dna);
        dna::mutate(&mut new_organism.dna, params.dna_mutation_rate);
    }

    /// Returns a reference to the graveyard.
    pub fn graveyard(&self) -> &[Organism] {
        &self.graveyard
    }

    /// Returns a reference to the elite pool.
    pub fn elite_pool(&self) -> &[Organism] {
        &self.elite_pool
    }

    /// Updates the elite pool with living organisms.
    /// Keeps only the top N scorers ever seen across all time.
    pub fn update_elite_pool(&mut self, living_organisms: &[Organism]) {
        if self.max_elite_pool_size == 0 {
            return;
        }

        // Add all living organisms to elite pool temporarily
        for organism in living_organisms {
            self.elite_pool.push(organism.clone());
        }

        // Sort by fitness and keep only the best
        self.elite_pool
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        self.elite_pool.truncate(self.max_elite_pool_size);
    }
}

/// Samples a mutation scale using logarithmic random distribution.
fn sample_mutation_scale() -> f32 {
    let min = 0.001f32;
    let max = 0.2f32;
    let log_min = min.ln();
    let log_max = max.ln();
    let log_mutation_scale = rand::rng().random_range(log_min..log_max);
    log_mutation_scale.exp()
}
