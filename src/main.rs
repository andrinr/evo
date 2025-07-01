use geo::Distance;
use macroquad::prelude::*;
use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand;
use kdtree::KdTree;
use kdtree::ErrorKind;
use kdtree::distance::squared_euclidean;
use geo::{Line, Euclidean, Point};

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

struct Brain {
    embedd : MLP,
    hidden: MLP,
    output: MLP,
}

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

fn vision_vectors(organism: &Organism) -> Vec<Array1<f32>> {
    let vision_length = 30.0;
    let angles = [0.0, 45.0, -45.0]; // angles in degrees
    let mut vectors = Vec::new();

    for &angle in angles.iter() {
        let angle_rad = (organism.rot + angle).to_radians();
        let vision_vector = Array1::from_vec(vec![
            angle_rad.cos() as f32 * vision_length,
            angle_rad.sin() as f32 * vision_length,
        ]);
        vectors.push(vision_vector);
    }
    vectors
}

fn line_circle_intersect(
    line_start: &Array1<f32>,
    line_end: &Array1<f32>,
    circle_center: &Array1<f32>,
    circle_radius: f32,
) -> bool {
    let p = Point::new(circle_center[0], circle_center[1]);
    let line = Line::new(
        Point::new(line_start[0], line_start[1]),
        Point::new(line_end[0] , line_end[1]),
    );
    Euclidean.distance(&p, &line) <= circle_radius
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

fn wrap_around(v: &Vec2) -> Vec2 {
    let mut vr = Vec2::new(v.x, v.y);
    if vr.x > screen_width() {
        vr.x = 0.;
    }
    if vr.x < 0. {
        vr.x = screen_width()
    }
    if vr.y > screen_height() {
        vr.y = 0.;
    }
    if vr.y < 0. {
        vr.y = screen_height()
    }
    vr
}

#[macroquad::main("Evolutionary Organisms")]
async fn main() {

    let mut rng = rand::thread_rng();

    let mut organisms = Vec::new();

    let mut kdtree = KdTree::new(2);

    let organism_count = 10;
    let init_velocity = 200.;

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

                for i in 0.. organism_count {

                    let organism = Organism {
                        id: i,
                        pos: &screen_center + Array1::random(2, Uniform::new(-100., 100.)),
                        vel: Array1::random(2, Uniform::new(-init_velocity, init_velocity)),
                        // random rotation in radians
                        rot : rand::random::<f32>() * std::f32::consts::PI * 2.,
                        energy: 1.,
                        signal: Array1::random(3, Uniform::new(0., 1.)),
                        memory: Array1::zeros(3),
                        brain: Brain {
                            embedd: init_mlp(10, 10, 0.1),
                            hidden: init_mlp(10, 10, 0.1),
                            output: init_mlp(10, 3, 0.1),
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
        
        for organism in organisms.iter_mut() {

            
            // euler integration
            organism.pos += &(&organism.vel * get_frame_time());
            organism.vel *= 0.99; // friction
            organism.energy -= 0.01; // energy consumption

            // get nearest neighbors
            let neighbors = kdtree.within(
                &organism.pos.to_vec(),
                50.0,
                &squared_euclidean,
            );

            let vision = vision_vectors(organism);

            // // check if any of the neighbours intersects with the vision vectors
            // for neighbor in neighbors.iter() {
            //     let neighbour_org = &organisms[neighbor.1];

            //     for vision_vector in vision.iter() {
         
            //         if line_circle_intersect(
            //             &organism.pos,
            //             &(organism.pos + vision_vector),
            //             &neighbour_org.pos,
            //             10.0, // radius of the organism
            //         ) {
            //             // if there is an intersection, we can interact with the neighbor
            //             organism.signal[0] += 0.1; // increase signal
            //             organism.memory[0] += 0.1; // increase memory
            //             organism.energy += 0.1; // gain energy
            //         }
            //     }
            // }
        }

        clear_background(LIGHTGRAY);

        // draw the organisms
        for organism in organisms.iter() {
            // organism body, simple circle
            draw_circle(
                organism.pos[0], 
                organism.pos[1],
                10., 
                Color::from_rgba(
                    (organism.signal[0] * 255.0) as u8, 
                    (organism.signal[1] * 255.0) as u8, 
                    (organism.signal[2] * 255.0) as u8, 
                    255)
            );
                
            // visualize vision vectors by drawing dotted line for each direction
            let vision_length = 30.0;
            let angles = [0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0]; // angles in radians
            let colors = [BLACK, RED, BLUE];

            for (i, &angle) in angles.iter().enumerate() {
                let angle_rad = organism.rot + angle;
                // let vision_vector = Vec2::new(angle_rad.cos(), angle_rad.sin()) * vision_length;
                let vision_vector = Array1::from_vec(vec![
                    angle_rad.cos() as f32 * vision_length,
                    angle_rad.sin() as f32 * vision_length,
                ]);
                let end_point = &organism.pos + vision_vector;

                draw_line(
                    organism.pos[0],
                    organism.pos[1],
                    end_point[0],
                    end_point[1],
                    2.0,
                    colors[i],
                );
              
            }
        }

        next_frame().await
    }
}
