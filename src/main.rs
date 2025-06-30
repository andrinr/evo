use macroquad::prelude::*;
use ndarray::{Array2, Array1};

struct MLP {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

struct Brain {
    embedd : MLP,
    hidden: MLP,
    output: MLP,
}

struct Organism {
    pos: Array1<f32>,
    vel: Array1<f32>,
    rot: f32,
    energy: f32,
    signal: Vec3,
    memory: Vec<f32>,
    brain: Brain,
}

fn think(brain : &Brain, inputs: Vec<f32>) -> Vec<f32> {

    let mut output = vec![0.0; brain.output.weights.len()];
    
    for (i, weights) in brain.output.weights.iter().enumerate() {
        output[i] = weights.iter().zip(&inputs).map(|(w, i)| w * i).sum::<f32>() + brain.output.biases[i];
    }
    
    // Apply activation function (e.g., ReLU)
    output.iter_mut().for_each(|x| *x = x.max(0.0));

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

    let mut organisms = Vec::new();

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

            screen_center = Vec2::new(screen_width() / 2., screen_height() / 2.);

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

                print!("genesis done");
                
                for _ in 0.. organism_count {

                    organisms.push(Organism {
                        pos: screen_center
                            + Vec2::new(rand::gen_range(-1., 1.), rand::gen_range(-1., 1.))
                                .normalize()
                                * screen_width().min(screen_height())
                                / 2.,
                        vel: Vec2::new(rand::gen_range(init_velocity * -1., init_velocity),
                                       rand::gen_range(init_velocity * -1., init_velocity)),
                        rot: 0.,
                        energy: 1.,
                        signal: Vec3::new(0., 0., 0.),
                        memory: vec![0.; 10],
                        brain: Brain {
                            embedd: MLP {
                                weights: vec![vec![0.; 10]; 10],
                                biases: vec![0.; 10],
                            },
                            hidden: MLP {
                                weights: vec![vec![0.; 10]; 10],
                                biases: vec![0.; 10],
                            },
                            output: MLP {
                                weights: vec![vec![0.; 3]; 10],
                                biases: vec![0.; 3],
                            },
                        },
                    });
                }
            }
            next_frame().await;
            continue;
        }

        // euler integration of eac organism
        for organism in organisms.iter_mut() {
            println!("Organism velocity: {:?}", organism.vel);
            organism.pos += organism.vel * get_frame_time();
            organism.pos = wrap_around(&organism.pos);
            organism.vel *= 0.99; // friction
            organism.energy -= 0.01; // energy consumption
        }

        clear_background(LIGHTGRAY);

        // draw the organisms
        for organism in organisms.iter() {
            // organism body, simple circle
            draw_circle(organism.pos.x, organism.pos.y, 10., DARKBLUE);
            
            // visualize vision vectors by drawing dotted line for each direction
            let vision_length = 30.0;
            let angles = [0.0, 45.0, -45.0]; // angles in degrees
            let colors = [BLACK, RED, BLUE];

            for (i, &angle) in angles.iter().enumerate() {
                let angle_rad = (organism.rot + angle).to_radians();
                let vision_vector = Vec2::new(angle_rad.cos(), angle_rad.sin()) * vision_length;
                let end_point = organism.pos + vision_vector;

                // Draw the dotted line
                let mut current_point = organism.pos;
                let step = 10.0; // length of each segment
                while (current_point - end_point).length() > step {
                    draw_line(
                        current_point.x,
                        current_point.y,
                        (current_point + vision_vector.normalize() * step).x,
                        (current_point + vision_vector.normalize() * step).y,
                        2.0,
                        colors[i],
                    );
                    current_point += vision_vector.normalize() * step;
                }
            }
        }

        next_frame().await
    }
}
