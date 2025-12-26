use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::brain;

#[derive(Debug, Clone)]
pub struct Organism {
    pub id: usize,
    pub age: f32,
    pub score: i32,
    pub pos: Array1<f32>,
    pub rot: f32,
    pub energy: f32,
    pub signal: Array1<f32>,
    pub memory: Array1<f32>,
    pub brain: brain::Brain,
}

impl Organism {
    pub fn new_random(
        id: usize,
        screen_center: &Array1<f32>,
        signal_size: usize,
        memory_size: usize,
        layer_sizes: Vec<usize>,
    ) -> Self {
        Self {
            id,
            age: 0.0,
            score: 0,
            pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
            rot: rand::random::<f32>() * std::f32::consts::PI * 2.,
            energy: 1.0,
            signal: Array1::random(signal_size, Uniform::new(0.0, 1.0)),
            memory: Array1::zeros(memory_size),
            brain: brain::Brain::new(&layer_sizes, 0.1),
        }
    }

    pub fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    pub fn get_vision_vectors(
        &self,
        field_of_view: f32,
        num_vision_directions: usize,
        vision_length: f32,
    ) -> Vec<Array1<f32>> {
        let angle_step = field_of_view / (num_vision_directions as f32 - 1.0);

        (0..num_vision_directions)
            .map(|i| {
                let angle = -field_of_view / 2.0 + i as f32 * angle_step;
                let angle_rad = self.rot + angle;
                Array1::from_vec(vec![
                    angle_rad.cos() * vision_length,
                    angle_rad.sin() * vision_length,
                ])
            })
            .collect()
    }

    pub fn age_by(&mut self, dt: f32) {
        self.age += dt;
    }

    pub fn consume_energy(&mut self, amount: f32) {
        self.energy -= amount;
    }

    pub fn gain_energy(&mut self, amount: f32, max_energy: f32) {
        self.energy = (self.energy + amount).min(max_energy);
    }

    pub fn kill(&mut self) {
        self.energy = 0.0;
    }
}
