use evo::simulation::locatable::Locatable;
use evo::simulation::{food::Food, organism::Organism, projectile::Projectile};
use ndarray::Array1;

#[test]
fn test_food_locatable() {
    let mut food = Food {
        pos: Array1::from_vec(vec![10.0, 20.0]),
        energy: 1.0,
        age: 0.0,
    };

    // Test pos accessor
    assert_eq!(food.pos()[0], 10.0);
    assert_eq!(food.pos()[1], 20.0);

    // Test update
    food.update(1.0);
    assert_eq!(food.age, 1.0);

    // Test pos_mut accessor
    food.pos_mut()[0] = 15.0;
    assert_eq!(food.pos()[0], 15.0);
}

#[test]
fn test_projectile_locatable() {
    let mut projectile = Projectile::new(
        Array1::from_vec(vec![0.0, 0.0]),
        0.0,   // rotation (facing right)
        100.0, // speed
        1.0,   // damage
        0,     // owner_id
        500.0, // max_range
    );

    // Test pos accessor
    assert_eq!(projectile.pos()[0], 0.0);
    assert_eq!(projectile.pos()[1], 0.0);

    // Test update - projectile should move in the direction of rotation
    projectile.update(0.1);

    // After 0.1 seconds at 100 speed, should move ~10 units in x direction
    assert!(projectile.pos()[0] > 9.0 && projectile.pos()[0] < 11.0);
    assert!(projectile.pos()[1].abs() < 1.0); // Should stay close to y=0

    // Test distance tracking
    assert!(projectile.distance_traveled > 0.0);
}

#[test]
fn test_locatable_trait_polymorphism() {
    // Demonstrate that we can use Locatable trait objects
    let food = Food {
        pos: Array1::from_vec(vec![5.0, 5.0]),
        energy: 1.0,
        age: 0.0,
    };

    let projectile = Projectile::new(
        Array1::from_vec(vec![10.0, 10.0]),
        std::f32::consts::PI / 4.0, // 45 degrees
        50.0,
        1.0,
        0,
        100.0,
    );

    // We can work with them through the trait
    fn get_distance(a: &dyn Locatable, b: &dyn Locatable) -> f32 {
        let dx = a.pos()[0] - b.pos()[0];
        let dy = a.pos()[1] - b.pos()[1];
        (dx * dx + dy * dy).sqrt()
    }

    let distance = get_distance(&food, &projectile);
    assert!((distance - 7.071).abs() < 0.1); // sqrt((10-5)^2 + (10-5)^2) ≈ 7.071
}

#[test]
fn test_organism_locatable_update() {
    use evo::simulation::brain::Brain;

    // Create a simple organism manually for testing
    let mut organism = Organism {
        id: 0,
        age: 0.0,
        score: 0,
        pos: Array1::from_vec(vec![10.0, 20.0]),
        rot: 0.0,
        energy: 100.0,
        signal: Array1::zeros(3),
        memory: Array1::zeros(8),
        brain: Brain::new(&[10, 8, 6], 0.1),
        attack_cooldown: 2.0,
        last_brain_inputs: Array1::zeros(10),
        vision_angles: Array1::zeros(5),
        vision_lengths: Array1::ones(5),
        dna: Array1::zeros(2),
        pool_id: 0,
        birth_generation: 0,
        reproduction_method: 0,
        parent_avg_score: 0.0,
    };

    // Test pos accessor
    assert_eq!(organism.pos()[0], 10.0);
    assert_eq!(organism.pos()[1], 20.0);

    let initial_cooldown = organism.attack_cooldown;

    // Test update through Locatable trait
    organism.update(1.0);

    // Age should increase
    assert_eq!(organism.age, 1.0);

    // Cooldown should decrease
    assert_eq!(organism.attack_cooldown, initial_cooldown - 1.0);

    // Update again
    organism.update(0.5);
    assert_eq!(organism.age, 1.5);
    assert_eq!(organism.attack_cooldown, initial_cooldown - 1.5);
}
