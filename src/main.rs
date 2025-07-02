use geo::Distance;
use macroquad::prelude::*;
use ndarray::{Array2, Array1, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use geo::{Line, Euclidean, Point};

const BODY_RADIUS: f32 = 5.0;
const VISION_RADIUS: f32 = 60.0;
const ENERGY_CONSUMPTION: f32 = 0.01;
const INIT_VELOCITY: f32 = 200.0;
const DAMPING_FACTOR: f32 = 0.01;
const NUM_VISION_DIRECTIONS: usize = 3; // number of vision directions
const FIELD_OF_VIEW: f32 = std::f32::consts::PI / 2.0; // field of view in radians

const SIGNAL_SIZE: usize = 3; // size of the signal array
const MEMORY_SIZE: usize = 3; // size of the memory array
const HIDDEN_SIZE: usize = 4; // size of the hidden layer in the MLP

const N_ORGANISMS: usize = 100;

#[derive(Debug, Clone)]
struct MLP {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

fn init_mlp(input_size: usize, output_size: usize, scale : f32) -> MLP {
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
struct Brain {
    embedd : MLP,
    hidden: MLP,
    output: MLP,
}

#[derive(Debug, Clone)]
struct Organism {
    id : usize,
    pos: Array1<f32>,
    vel: Array1<f32>,
    rot: f32,
    energy: f32,
    signal: Array1<f32>,
    memory: Array1<f32>,
    brain: Brain,
}

fn get_vision_vectors(organism: &Organism) -> Vec<Array1<f32>> {
    let vision_length = 30.0;
    let mut angles = Vec::new();
    let angle_step = FIELD_OF_VIEW / (NUM_VISION_DIRECTIONS as f32 - 1.0);
    for i in 0..NUM_VISION_DIRECTIONS {
        let angle = -FIELD_OF_VIEW / 2.0 + i as f32 * angle_step;
        angles.push(angle);
    }
    let mut vectors = Vec::new();

    for &angle in angles.iter() {
        let angle_rad = organism.rot + angle;
        let vision_vector = Array1::from_vec(vec![
            angle_rad.cos() as f32 * vision_length,
            angle_rad.sin() as f32 * vision_length,
        ]);
        vectors.push(vision_vector);
    }

    vectors
}

fn line_circle_distance(
    line_start: &Array1<f32>,
    line_end: &Array1<f32>,
    circle_center: &Array1<f32>,
) -> f32 {
    let p = Point::new(circle_center[0], circle_center[1]);
    let line = Line::new(
        Point::new(line_start[0], line_start[1]),
        Point::new(line_end[0] , line_end[1]),
    );
    Euclidean.distance(&p, &line)
}
    

fn think(brain : &Brain, inputs: Array1<f32>) -> Array1<f32> {

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

fn wrap_around(v: &Array1<f32>) -> Array1<f32> {
    let mut wrapped = v.clone();
    if wrapped[0] < 0.0 {
        wrapped[0] += screen_width();
    } else if wrapped[0] > screen_width() {
        wrapped[0] -= screen_width();   
    }
    if wrapped[1] < 0.0 {
        wrapped[1] += screen_height();
    } else if wrapped[1] > screen_height() {
        wrapped[1] -= screen_height();
    }
    wrapped
}

#[macroquad::main("Evolutionary Organisms")]
async fn main() {

    let mut rng = rand::thread_rng();

    let mut organisms = Vec::new();

    let mut kdtree = KdTree::new(2);

    let mut genesis = true;

    let mut screen_center;

    println!("Starting evolutionary organisms simulation");

    loop {
        if genesis {
            
            clear_background(LIGHTGRAY);
            let text = "Start a new evolution by pressing Enter";
            let font_size = 30.;

            screen_center = Array1::from_vec(vec![
                screen_width() / 2.,
                screen_height() / 2.,
            ]);

            let text_size = measure_text(text, None, font_size as _, 1.0);
            draw_text(
                text,
                screen_width() / 2. - text_size.width / 2.,
                screen_height() / 2. - text_size.height / 2.,
                font_size,
                DARKGRAY,
            );

            if is_key_down(KeyCode::Enter) {

                genesis = false;

                for i in 0.. N_ORGANISMS {

                    let organism = Organism {
                        id: i,
                        pos: &screen_center + Array1::random(2, Uniform::new(-100., 100.)),
                        vel: Array1::random(2, Uniform::new(-INIT_VELOCITY, INIT_VELOCITY)),
                        // random rotation in radians
                        rot : rand::random::<f32>() * std::f32::consts::PI * 2.,
                        energy: 1.,
                        signal: Array1::random(
                            SIGNAL_SIZE, 
                            Uniform::new(0.0, 1.0)
                        ),
                        memory: Array1::zeros(
                            MEMORY_SIZE
                        ),
                        brain: Brain {
                            embedd: init_mlp(
                                (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1, // inputs: signal + vision vectors + memory + energy
                                HIDDEN_SIZE,
                                0.1
                            ),
                            hidden: init_mlp(
                                HIDDEN_SIZE,
                                HIDDEN_SIZE,
                                0.1
                            ),
                            output: init_mlp(
                                HIDDEN_SIZE,
                                SIGNAL_SIZE + MEMORY_SIZE + 1, // outputs: signal + memory + rotation + acceleration
                                0.1
                            ),
                        },
                    };

                    kdtree.add(
                        organism.pos.to_vec(),
                        i as usize,
                    ).unwrap();

                    organisms.push(organism);

                }
            }
            next_frame().await;
            continue;
        }

        clear_background(DARKGRAY);

        // Clone the organisms vector
        let new_organisms = organisms.clone();

        for organism in organisms.iter_mut() {

            let vision_vectors = get_vision_vectors(organism);
            
            // euler integration
            organism.pos += &(&organism.vel * get_frame_time());
            organism.vel *= 1.0 - DAMPING_FACTOR * get_frame_time(); // damping

            // wrap around the screen
            organism.pos = wrap_around(&organism.pos);

            organism.energy -= ENERGY_CONSUMPTION * get_frame_time(); // energy consumption

            // get nearest neighbors
            let neighbors = kdtree.within(
                &organism.pos.to_vec(),
                VISION_RADIUS,
                &squared_euclidean,
            );

            // the above returns a Result, so we need to handle the error
            let neighbors = match neighbors {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding neighbors: {:?}", e);
                }
            };

            let mut brain_inputs = Array1::zeros((SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1);

            for (i ,vision_vector) in vision_vectors.iter().enumerate() {
                let end_point = &organism.pos + vision_vector;
                let mut min_distance = f32::MAX;
            
                for (_, neighbor_id) in neighbors.iter() {
                    let neighbor_org = &new_organisms[**neighbor_id];
                    let distance = line_circle_distance(
                        &organism.pos,
                        &end_point,
                        &neighbor_org.pos
                    );
                    if distance < min_distance {
                        min_distance = distance;
                        brain_inputs[(i * 2) + 0] = neighbor_org.signal[0];
                        brain_inputs[(i * 2) + 1] = neighbor_org.signal[1];
                        brain_inputs[(i * 2) + 2] = neighbor_org.signal[2];
                        brain_inputs[(i * 2) + 3] = distance;
                    }
                }
            }

            brain_inputs[(NUM_VISION_DIRECTIONS * 2) + 0] = organism.energy; // energy
            brain_inputs[(NUM_VISION_DIRECTIONS * 2) + 1] = organism.memory[0]; // memory 1
            brain_inputs[(NUM_VISION_DIRECTIONS * 2) + 2] = organism.memory[1]; // memory 2
            brain_inputs[(NUM_VISION_DIRECTIONS * 2) + 3] = organism.memory[2]; // memory 3

            let brain_outputs = think(&organism.brain, brain_inputs);

            organism.signal = brain_outputs.slice(s![..SIGNAL_SIZE]).to_owned();
            organism.memory = brain_outputs.slice(s![SIGNAL_SIZE..SIGNAL_SIZE + MEMORY_SIZE]).to_owned();
            organism.rot += brain_outputs[brain_outputs.len() - 2]; // rotation adjustment

            let acc = brain_outputs[brain_outputs.len() - 1]; // acceleration
            let acc_vector = Array1::from_vec(vec![
                acc * organism.rot.cos(),
                acc * organism.rot.sin(),
            ]);
            organism.vel += &(&acc_vector * get_frame_time()); // update velocity

            // organism body, simple circle
            draw_circle(
                organism.pos[0], 
                organism.pos[1],
                BODY_RADIUS,
                Color::from_rgba(
                    (organism.signal[0] * 255.0) as u8, 
                    (organism.signal[1] * 255.0) as u8, 
                    (organism.signal[2] * 255.0) as u8, 
                    255)
            );

            for vision_vector in vision_vectors.iter() {
                let end_point = &organism.pos + vision_vector;
                // draw a line from the organism's position to the end point of the vision vector
                draw_line(
                    organism.pos[0],
                    organism.pos[1],
                    end_point[0],
                    end_point[1],
                    2.0,
                    RED
                );
              
            }
        }

        next_frame().await
    }
}
