use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct Food {
    pub pos: Array1<f32>,
    pub energy: f32,
}

impl Food {
    pub fn new_random(screen_center: &Array1<f32>) -> Self {
        Self {
            pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
            energy: 1.0,
        }
    }

    pub fn is_consumed(&self) -> bool {
        self.energy <= 0.0
    }

    pub fn consume(&mut self) {
        self.energy = 0.0;
    }
}
