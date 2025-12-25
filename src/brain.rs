use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct MLP {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

pub fn init_mlp(input_size: usize, output_size: usize, scale : f32) -> MLP {
    MLP {
        weights: Array2::random(
            (output_size, input_size),
             Uniform::new(-scale, scale)
        ),
        biases: Array1::random(
            output_size, 
            Uniform::new(-scale, scale)
        )
    }
}

#[derive(Debug, Clone)]
pub struct Brain {
    pub layers : Vec<MLP>,
}


pub fn think(brain : &Brain, inputs: &Array1<f32>) -> Array1<f32> {

    let mut output = inputs.clone();

    for layer in &brain.layers {
        // Dot product of weights and inputs, plus biases
        output = layer.weights.dot(&output) + &layer.biases;
        // Activation function (tanh)
        output = output.map(|x| x.tanh());
    }

    output
}

pub fn crossover(
    brain_1: &Brain,
    brain_2: &Brain,
) -> Brain {
    // Average crossover operartion
    let mut new_layers = Vec::new();

    for (layer_1, layer_2) in brain_1.layers.iter().zip(&brain_2.layers) {
        let mut new_weights = Array2::zeros(layer_1.weights.dim());
        let mut new_biases = Array1::zeros(layer_1.biases.len());

        new_weights = &layer_1.weights * 0.5 + &layer_2.weights * 0.5;
        new_biases = &layer_1.biases * 0.5 + &layer_2.biases * 0.5;

        new_layers.push(MLP {
            weights: new_weights,
            biases: new_biases,
        });
    }

    Brain {
        layers: new_layers,
    }

}

pub fn mutate_brain(
    brain: &mut Brain,
    mutation_scale: f32,
) {

    for layer in &mut brain.layers {
        layer.weights += &Array2::random(
            layer.weights.dim(),
            Uniform::new(-mutation_scale, mutation_scale)
        );
        layer.biases += &Array1::random(
            layer.biases.len(),
            Uniform::new(-mutation_scale, mutation_scale)
        );
    }
}
