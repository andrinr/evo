use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::brain;

#[derive(Debug, Clone)]
pub struct Organism {
    pub id: usize,
    pub age: f32,
    pub score: i32, // score for reproduction
    pub pos: Array1<f32>,
    // pub vel: Array1<f32>,
    pub rot: f32,
    pub energy: f32,
    pub signal: Array1<f32>,
    pub memory: Array1<f32>,
    pub brain: brain::Brain,
}

pub fn init_random_organism(
    i: usize,
    screen_center: &Array1<f32>,
    // init_velocity : f32,
    signal_size: usize,
    memory_size: usize,
    layer_sizes: Vec<usize>,
) -> Organism {
    Organism {
        id: i,
        age: 0.0,
        score: 0, // initial score for reproduction
        pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
        // vel: Array1::random(2, Uniform::new(-init_velocity, init_velocity)),
        // random rotation in radians
        rot: rand::random::<f32>() * std::f32::consts::PI * 2.,
        energy: 1.,
        signal: Array1::random(signal_size, Uniform::new(0.0, 1.0)),
        memory: Array1::zeros(memory_size),
        brain: brain::Brain {
            layers: (0..layer_sizes.len() - 1)
                .map(|i| {
                    brain::init_mlp(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        0.1, // scale for weights and biases
                    )
                })
                .collect(),
        },
    }
}

pub fn get_vision_vectors(
    organism: &Organism,
    field_of_view: f32,
    num_vision_directions: usize,
    vision_length: f32,
) -> Vec<Array1<f32>> {
    let mut angles = Vec::new();
    let angle_step = field_of_view / (num_vision_directions as f32 - 1.0);
    for i in 0..num_vision_directions {
        let angle = -field_of_view / 2.0 + i as f32 * angle_step;
        angles.push(angle);
    }
    let mut vectors = Vec::new();

    for &angle in angles.iter() {
        let angle_rad = organism.rot + angle;
        let vision_vector = Array1::from_vec(vec![
            angle_rad.cos() * vision_length,
            angle_rad.sin() * vision_length,
        ]);
        vectors.push(vision_vector);
    }

    vectors
}
