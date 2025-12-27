#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use evo::simulation::ecosystem::{Ecosystem, Params};
use std::fs;

fn create_test_params() -> Params {
    let signal_size: usize = 3;
    let num_vision_directions: usize = 3;
    let memory_size: usize = 3;

    let layer_sizes = vec![
        (signal_size + 1) * num_vision_directions + memory_size + 1,
        10,
        signal_size + memory_size + 3,
    ];

    let vision_radius = 30.0;

    Params {
        body_radius: 3.0,
        vision_radius,
        idle_energy_rate: 0.023,
        move_energy_rate: 0.0002,
        move_multiplier: 60.0,
        rot_energy_rate: 0.0003,
        num_vision_directions,
        fov: std::f32::consts::PI / 2.0,
        signal_size,
        memory_size,
        n_organism: 20,
        n_food: 15,
        box_width: 1000.0,
        box_height: 1000.0,
        layer_sizes,
        attack_cost_rate: 0.2,
        attack_damage_rate: 0.4,
        attack_cooldown: 1.0,
        corpse_energy_ratio: 0.8,
        projectile_speed: vision_radius * 2.0,
        projectile_range: vision_radius,
        projectile_radius: 1.0,
        organism_spawn_rate: 1.0,
        food_spawn_rate: 1.0,
    }
}

#[test]
fn test_save_and_load() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    // Run simulation for a bit to create some state
    for _ in 0..5 {
        ecosystem.step(&params, 0.05);
    }

    let save_path = "test_save.json";

    // Save the ecosystem
    ecosystem
        .save_to_file(save_path)
        .expect("Failed to save ecosystem");

    // Load it back
    let loaded_ecosystem = Ecosystem::load_from_file(save_path).expect("Failed to load ecosystem");

    // Verify the loaded state matches
    assert_eq!(loaded_ecosystem.organisms.len(), ecosystem.organisms.len());
    assert_eq!(loaded_ecosystem.food.len(), ecosystem.food.len());
    assert!((loaded_ecosystem.time - ecosystem.time).abs() < 0.001);
    assert_eq!(loaded_ecosystem.generation, ecosystem.generation);

    // Check organism properties
    for (original, loaded) in ecosystem
        .organisms
        .iter()
        .zip(loaded_ecosystem.organisms.iter())
    {
        assert_eq!(original.id, loaded.id);
        assert!((original.energy - loaded.energy).abs() < 0.001);
        assert!((original.age - loaded.age).abs() < 0.001);
        assert_eq!(original.score, loaded.score);
        assert_eq!(original.signal.len(), loaded.signal.len());
        assert_eq!(original.memory.len(), loaded.memory.len());
    }

    // Clean up
    fs::remove_file(save_path).ok();
}

#[test]
fn test_save_creates_valid_json() {
    let params = create_test_params();
    let ecosystem = Ecosystem::new(&params);

    let save_path = "test_json_valid.json";

    ecosystem.save_to_file(save_path).expect("Failed to save");

    // Read the file and verify it's valid JSON
    let json_content = fs::read_to_string(save_path).expect("Failed to read save file");
    let parsed: serde_json::Value = serde_json::from_str(&json_content).expect("Invalid JSON");

    // Verify key fields exist
    assert!(parsed.get("organisms").is_some());
    assert!(parsed.get("food").is_some());
    assert!(parsed.get("time").is_some());
    assert!(parsed.get("generation").is_some());

    // Clean up
    fs::remove_file(save_path).ok();
}

#[test]
fn test_load_nonexistent_file() {
    let result = Ecosystem::load_from_file("nonexistent_file.json");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_load_invalid_json() {
    let invalid_path = "test_invalid.json";
    fs::write(invalid_path, "{ this is not valid json }").expect("Failed to write test file");

    let result = Ecosystem::load_from_file(invalid_path);
    assert!(
        result.is_err(),
        "Loading invalid JSON should return an error"
    );

    // Clean up
    fs::remove_file(invalid_path).ok();
}

#[test]
fn test_save_and_load_preserves_brain_weights() {
    let params = create_test_params();
    let ecosystem = Ecosystem::new(&params);

    let save_path = "test_brain_weights.json";

    // Save
    ecosystem.save_to_file(save_path).expect("Failed to save");

    // Load
    let loaded_ecosystem = Ecosystem::load_from_file(save_path).expect("Failed to load");

    // Check that brain weights are preserved
    for (original, loaded) in ecosystem
        .organisms
        .iter()
        .zip(loaded_ecosystem.organisms.iter())
    {
        assert_eq!(original.brain.layers.len(), loaded.brain.layers.len());

        for (orig_layer, loaded_layer) in
            original.brain.layers.iter().zip(loaded.brain.layers.iter())
        {
            assert_eq!(orig_layer.weights.shape(), loaded_layer.weights.shape());
            assert_eq!(orig_layer.biases.len(), loaded_layer.biases.len());

            // Check that values match
            for (orig_val, loaded_val) in orig_layer.weights.iter().zip(loaded_layer.weights.iter())
            {
                assert!((orig_val - loaded_val).abs() < 0.0001);
            }

            for (orig_val, loaded_val) in orig_layer.biases.iter().zip(loaded_layer.biases.iter()) {
                assert!((orig_val - loaded_val).abs() < 0.0001);
            }
        }
    }

    // Clean up
    fs::remove_file(save_path).ok();
}

#[test]
fn test_load_and_continue_simulation() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    // Run simulation
    for _ in 0..3 {
        ecosystem.step(&params, 0.05);
    }

    let save_path = "test_continue.json";
    ecosystem.save_to_file(save_path).expect("Failed to save");

    // Load and continue
    let mut loaded_ecosystem = Ecosystem::load_from_file(save_path).expect("Failed to load");
    let loaded_time = loaded_ecosystem.time;

    // Continue simulation
    for _ in 0..3 {
        loaded_ecosystem.step(&params, 0.05);
    }

    // Time should have advanced from where it was loaded
    assert!(loaded_ecosystem.time > loaded_time);

    // Clean up
    fs::remove_file(save_path).ok();
}
