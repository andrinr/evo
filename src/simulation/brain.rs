use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct Mlp {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

impl Mlp {
    pub fn new_random(input_size: usize, output_size: usize, scale: f32) -> Self {
        Self {
            weights: Array2::random((output_size, input_size), Uniform::new(-scale, scale)),
            biases: Array1::random(output_size, Uniform::new(-scale, scale)),
        }
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        // Dot product of weights and inputs, plus biases
        let output = self.weights.dot(inputs) + &self.biases;
        // Activation function (tanh)
        output.mapv(|x| x.tanh())
    }

    pub fn mutate(&mut self, mutation_scale: f32) {
        self.weights += &Array2::random(
            self.weights.dim(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
        self.biases += &Array1::random(
            self.biases.len(),
            Uniform::new(-mutation_scale, mutation_scale),
        );
    }

    pub fn crossover(parent1: &Mlp, parent2: &Mlp) -> Self {
        Self {
            weights: &parent1.weights * 0.5 + &parent2.weights * 0.5,
            biases: &parent1.biases * 0.5 + &parent2.biases * 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Brain {
    pub layers: Vec<Mlp>,
}

impl Brain {
    pub fn new(layer_sizes: &[usize], scale: f32) -> Self {
        let layers = (0..layer_sizes.len() - 1)
            .map(|i| Mlp::new_random(layer_sizes[i], layer_sizes[i + 1], scale))
            .collect();

        Self { layers }
    }

    pub fn think(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut output = inputs.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn crossover(parent1: &Brain, parent2: &Brain) -> Self {
        let new_layers = parent1
            .layers
            .iter()
            .zip(&parent2.layers)
            .map(|(layer1, layer2)| Mlp::crossover(layer1, layer2))
            .collect();

        Self { layers: new_layers }
    }

    pub fn mutate(&mut self, mutation_scale: f32) {
        for layer in &mut self.layers {
            layer.mutate(mutation_scale);
        }
    }
}
