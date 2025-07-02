use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct MLP {
    weights: Array2<f32>,
    biases: Array1<f32>,
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
    pub embedd: MLP,
    pub hidden: MLP,
    pub output: MLP,
}


pub fn think(brain : &Brain, inputs: Array1<f32>) -> Array1<f32> {

    let mut output = Array1::zeros(brain.output.weights.shape()[0]);

    // Embedding layer
    let embedded = brain.embedd.weights.dot(&inputs) + &brain.embedd.biases;
    let embedded = embedded.map(|x| x.tanh()); // Activation function for embedding
    // Hidden layer
    let hidden = brain.hidden.weights.dot(&embedded) + &brain.hidden.biases;
    let hidden = hidden.map(|x| x.tanh()); // Activation function for hidden layer

    // Output layer
    output = brain.output.weights.dot(&hidden) + &brain.output.biases;
    output = output.map(|x| x.tanh()); // Activation function for output layer

    output
}