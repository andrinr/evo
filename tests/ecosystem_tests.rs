#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use evo::simulation::ecosystem::{Ecosystem, Params};
use ndarray::Array1;

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
        n_organism: 50,
        n_food: 40,
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
fn test_ecosystem_creation() {
    let params = create_test_params();
    let ecosystem = Ecosystem::new(&params);

    assert_eq!(ecosystem.organisms.len(), params.n_organism);
    assert_eq!(ecosystem.food.len(), params.n_food);
    assert_eq!(ecosystem.time, 0.0);
    assert_eq!(ecosystem.generation, params.n_organism as u32);

    // Check that organisms are initialized with valid properties
    for organism in &ecosystem.organisms {
        assert!(organism.energy > 0.0);
        assert!(organism.age >= 0.0);
        assert!(organism.is_alive());
        assert_eq!(organism.signal.len(), params.signal_size);
        assert_eq!(organism.memory.len(), params.memory_size);
    }
}

#[test]
fn test_simulation_step() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    let initial_time = ecosystem.time;
    let dt = 0.05;

    ecosystem.step(&params, dt);

    // Time should advance
    assert!((ecosystem.time - (initial_time + dt)).abs() < 0.001);

    // Organisms should have aged
    for organism in &ecosystem.organisms {
        assert!(organism.age >= dt);
    }
}

#[test]
fn test_organism_energy_consumption() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    // Get initial energies
    let initial_energies: Vec<f32> = ecosystem.organisms.iter().map(|o| o.energy).collect();

    // Run simulation for several steps
    for _ in 0..10 {
        ecosystem.step(&params, 0.05);
    }

    // At least some organisms should have consumed energy
    let mut energy_consumed = false;
    for (i, organism) in ecosystem.organisms.iter().enumerate() {
        if organism.energy < initial_energies[i] {
            energy_consumed = true;
            break;
        }
    }

    assert!(
        energy_consumed,
        "Organisms should consume energy during simulation"
    );
}

#[test]
fn test_organism_spawning() {
    let mut params = create_test_params();
    params.n_organism = 100;
    params.n_food = 80;

    let mut ecosystem = Ecosystem::new(&params);

    // Remove some organisms to trigger spawning
    ecosystem.organisms.truncate(50);

    let initial_count = ecosystem.organisms.len();

    ecosystem.spawn(&params);

    // Should spawn one new organism
    assert_eq!(ecosystem.organisms.len(), initial_count + 1);
    assert!(ecosystem.generation > params.n_organism as u32);
}

#[test]
fn test_food_spawning() {
    let mut params = create_test_params();
    params.n_food = 50;

    let mut ecosystem = Ecosystem::new(&params);

    // Remove some food
    ecosystem.food.truncate(25);

    let initial_food_count = ecosystem.food.len();

    ecosystem.spawn(&params);

    // Should spawn one new food item
    assert_eq!(ecosystem.food.len(), initial_food_count + 1);
}

#[test]
fn test_organism_death_from_no_energy() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    // Drain energy from all organisms and remove all food
    for organism in &mut ecosystem.organisms {
        organism.energy = 0.0;
    }
    ecosystem.food.clear();

    ecosystem.step(&params, 0.05);

    // All organisms should be dead and removed
    assert_eq!(ecosystem.organisms.len(), 0);
}

#[test]
fn test_wrap_around() {
    let params = create_test_params();
    let mut ecosystem = Ecosystem::new(&params);

    // Move an organism outside the bounds
    if let Some(organism) = ecosystem.organisms.first_mut() {
        organism.pos = Array1::from_vec(vec![params.box_width + 50.0, params.box_height + 50.0]);
    }

    ecosystem.step(&params, 0.05);

    // After step, organism should be wrapped around
    if let Some(organism) = ecosystem.organisms.first() {
        assert!(organism.pos[0] >= 0.0 && organism.pos[0] < params.box_width);
        assert!(organism.pos[1] >= 0.0 && organism.pos[1] < params.box_height);
    }
}
