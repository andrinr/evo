use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct Food {
    pub pos: Array1<f32>,
    pub energy: f32,
}

pub fn init_random_food(screen_center: &Array1<f32>) -> Food {
    Food {
        pos: Array1::random(2, Uniform::new(0., 1.)) * screen_center * 2.0,
        energy: 1.0, // initial energy
    }
}
