#![allow(missing_docs)]

use evo::simulation::ecosystem::Ecosystem;
use evo::simulation::organism::{Perception, Proprioception, Scent, Sense, Vision};
use evo::simulation::params::Params;

fn create_test_params() -> Params {
    let signal_size: usize = 3;
    let num_vision_directions: usize = 5;
    let memory_size: usize = 8;

    let layer_sizes = vec![
        3 * num_vision_directions + (signal_size + 1) + memory_size + 1,
        16,
        signal_size + memory_size + 4,
    ];

    Params {
        body_radius: 3.0,
        vision_radius: 50.0,
        scent_radius: 20.0,
        share_radius: 15.0,
        dna_breeding_distance: 0.2,
        dna_mutation_rate: 0.1,
        idle_energy_rate: 0.01,
        move_energy_rate: 0.0001,
        move_multiplier: 50.0,
        rot_energy_rate: 0.00001,
        num_vision_directions,
        fov: std::f32::consts::PI / 2.0,
        signal_size,
        memory_size,
        n_organism: 10,
        max_organism: 20,
        n_food: 10,
        max_food: 20,
        box_width: 500.0,
        box_height: 500.0,
        layer_sizes,
        attack_cost_rate: 0.1,
        attack_damage_rate: 0.5,
        attack_cooldown: 1.0,
        corpse_energy_ratio: 0.5,
        max_energy: 2.0,
        food_energy: 1.0,
        projectile_speed: 100.0,
        projectile_range: 50.0,
        projectile_radius: 2.0,
        organism_spawn_rate: 1.0,
        food_spawn_rate: 1.0,
        food_lifetime: 0.0,
        num_genetic_pools: 1,
        pool_interbreed_prob: 0.0,
        brain_type: evo::simulation::brain::BrainType::MLP,
        transformer_model_dim: 64,
        transformer_num_blocks: 2,
        transformer_num_heads: 4,
        transformer_head_dim: 16,
        transformer_ff_dim: 128,
        graveyard_size: 100,
    }
}

#[test]
fn test_vision_sense_size() {
    let params = create_test_params();
    let vision = Vision::new();

    let expected_size = params.num_vision_directions * 3; // distance, pool_match, is_organism
    assert_eq!(vision.input_size(&params), expected_size);
    assert_eq!(vision.name(), "Vision");
}

#[test]
fn test_scent_sense_size() {
    let params = create_test_params();
    let scent = Scent::new();

    let expected_size = params.signal_size + 1; // signal channels + DNA distance
    assert_eq!(scent.input_size(&params), expected_size);
    assert_eq!(scent.name(), "Scent");
}

#[test]
fn test_proprioception_sense_size() {
    let params = create_test_params();
    let proprio = Proprioception::new();

    let expected_size = params.memory_size + 1; // memory + energy
    assert_eq!(proprio.input_size(&params), expected_size);
    assert_eq!(proprio.name(), "Proprioception");
}

#[test]
fn test_perception_combines_senses() {
    let params = create_test_params();
    let ecosystem = Ecosystem::new(&params);

    let perception = Perception::default();

    // Total size should be sum of all senses
    let expected_size = (params.num_vision_directions * 3) // vision
        + (params.signal_size + 1) // scent
        + (params.memory_size + 1); // proprioception

    assert_eq!(perception.total_input_size(&params), expected_size);

    // Test perception on first organism
    if let Some(organism) = ecosystem.organisms.first() {
        let inputs = perception.perceive(organism, &ecosystem, &params, None);
        assert_eq!(inputs.len(), expected_size);
    }
}

#[test]
fn test_custom_perception() {
    let params = create_test_params();

    // Create perception with only vision and proprioception
    let perception = Perception::new(vec![
        Box::new(Vision::new()),
        Box::new(Proprioception::new()),
    ]);

    let expected_size = (params.num_vision_directions * 3) + (params.memory_size + 1);
    assert_eq!(perception.total_input_size(&params), expected_size);
}

#[test]
fn test_proprioception_reads_organism_state() {
    let params = create_test_params();
    let ecosystem = Ecosystem::new(&params);
    let proprio = Proprioception::new();

    if let Some(organism) = ecosystem.organisms.first() {
        let outputs = proprio.sense(organism, &ecosystem, &params, None);

        // Should have memory + energy
        assert_eq!(outputs.len(), params.memory_size + 1);

        // Last value should be energy
        let energy_idx = params.memory_size;
        assert_eq!(outputs[energy_idx], organism.energy);
    }
}
