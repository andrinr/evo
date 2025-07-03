use geo::Distance;
use macroquad::prelude::*;
use ndarray::{Array1, s};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use geo::{Line, Euclidean, Point};

mod brain;
mod organism;

const BODY_RADIUS: f32 = 4.0;
const VISION_RADIUS: f32 = 40.0;
const ENERGY_CONSUMPTION: f32 = 0.01;
const ACCELERATION_CONSUMPTION: f32 = 0.01;
const ROTATION_CONSUMPTION: f32 = 0.01;
const INIT_VELOCITY: f32 = 20.0;
const DAMPING_FACTOR: f32 = 0.1;
const NUM_VISION_DIRECTIONS: usize = 3; // number of vision directions
const FIELD_OF_VIEW: f32 = std::f32::consts::PI / 2.0; // field of view in radians

const SIGNAL_SIZE: usize = 3; // size of the signal array
const MEMORY_SIZE: usize = 3; // size of the memory array
const N_ORGANISMS: usize = 100;


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

    let mut organisms = Vec::new();

    let mut kdtree = KdTree::new(2);

    let mut genesis = true;

    let mut screen_center;

    println!("Starting evolutionary organisms simulation");

    let layer_sizes = vec![
        (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1, // input size
        10, // hidden layer size
        SIGNAL_SIZE + MEMORY_SIZE + 2, // output size (signal + memory + rotation + acceleration)
    ];

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

                    let entity = organism::init_random_organism(
                        i, 
                        &screen_center, 
                        INIT_VELOCITY,
                        SIGNAL_SIZE,
                        MEMORY_SIZE,
                        layer_sizes.clone()
                    );

                    kdtree.add(
                        entity.pos.to_vec(),
                        i as usize,
                    ).unwrap();

                    organisms.push(entity);

                }
            }
            next_frame().await;
            continue;
        }

        clear_background(LIGHTGRAY);

        // Clone the organisms vector
        let new_organisms = organisms.clone();

        let mut keep_organisms = Vec::new();

        for (organism_id, entity) in organisms.iter_mut().enumerate() {

            let vision_vectors = organism::get_vision_vectors(
                entity,
                FIELD_OF_VIEW,
                NUM_VISION_DIRECTIONS,
            );
            
            // euler integration
            entity.pos += &(&entity.vel * get_frame_time());
            entity.vel *= 1.0 - DAMPING_FACTOR * get_frame_time(); // damping

            // wrap around the screen
            entity.pos = wrap_around(&entity.pos);

            entity.energy -= ENERGY_CONSUMPTION * get_frame_time(); // energy consumption

            // get nearest neighbors
            let neighbors = kdtree.within(
                &entity.pos.to_vec(),
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
                let end_point = &entity.pos + vision_vector;
                let mut min_distance = f32::MAX;
            
                for (_, neighbor_id) in neighbors.iter() {
                    let neighbor_org = &new_organisms[**neighbor_id];
                    let distance = line_circle_distance(
                        &entity.pos,
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

            let offset = (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS;
            // Add the organism's own signal to the inputs
            brain_inputs.slice_mut(s![offset..offset + SIGNAL_SIZE]).assign(&entity.memory);
            brain_inputs[offset + SIGNAL_SIZE] = entity.energy; // energy

            let brain_outputs = brain::think(&entity.brain, brain_inputs);

            entity.signal = brain_outputs.slice(s![..SIGNAL_SIZE]).to_owned();
            // apply sigmoid activation to the signal
            entity.signal = entity.signal.mapv(|x| 1.0 / (1.0 + (-x).exp()));
            entity.memory = brain_outputs.slice(s![SIGNAL_SIZE..SIGNAL_SIZE + MEMORY_SIZE]).to_owned();
            entity.rot += brain_outputs[brain_outputs.len() - 2]; // rotation adjustment

            let acc = brain_outputs[brain_outputs.len() - 1]; // acceleration
            let acc_vector = Array1::from_vec(vec![
                acc * entity.rot.cos(),
                acc * entity.rot.sin(),
            ]);
            entity.vel += &(&acc_vector * get_frame_time()); // update velocity
            entity.energy -= acc.abs() * get_frame_time() * ACCELERATION_CONSUMPTION; // energy consumption for acceleration
            entity.energy -= entity.rot.abs() * get_frame_time() * ROTATION_CONSUMPTION; // energy consumption for rotation
            entity.energy -= ENERGY_CONSUMPTION * get_frame_time(); // additional energy consumption
            
            // println!("Organism {}: pos = {:?}, vel = {:?}, energy = {}, signal = {:?}", 
            //     entity.id, 
            //     entity.pos, 
            //     entity.vel, 
            //     entity.energy, 
            //     entity.signal
            // );
            if entity.energy > 0.0 {
                keep_organisms.push(organism_id);
            }   
            else {
                // remove the organism from the kdtree
                let res = kdtree.remove(&entity.pos.to_vec(), &organism_id);

                if res.is_err() {
                    println!("Error removing organism {}: {:?}", entity.id, res);
                }

                continue; // skip drawing this organism
                    
            }
    

            // organism body, simple circle
            draw_circle(
                entity.pos[0], 
                entity.pos[1],
                BODY_RADIUS,
                Color::from_rgba(
                    (entity.signal[0] * 255.0) as u8, 
                    (entity.signal[1] * 255.0) as u8, 
                    (entity.signal[2] * 255.0) as u8, 
                    255)
            );

            // organism health bar
            let health_bar_width = 20.0;
            let health_bar_height = 3.0;
            let health_bar_x = entity.pos[0] - health_bar_width / 2.0;
            let health_bar_y = entity.pos[1] - BODY_RADIUS - health_bar_height - 2.0;
            draw_rectangle(
                health_bar_x,
                health_bar_y,
                health_bar_width,
                health_bar_height,
                Color::from_rgba(100, 100, 100, 200)
            );
            draw_rectangle(
                health_bar_x,
                health_bar_y,
                health_bar_width * (entity.energy / 1.0).max(0.0).min(1.0),
                health_bar_height,
                Color::from_rgba(0, 255, 0, 200)
            );

            // organism memory, simple rectangles
            let memory_bar_width = 20.0;
            let memory_bar_height = 3.0;
            let memory_bar_x = entity.pos[0] - memory_bar_width / 2.0;
            let memory_bar_y = entity.pos[1] - BODY_RADIUS - health_bar_height - memory_bar_height - 2.0;
            for (i, &value) in entity.memory.iter().enumerate() {
                let color_value = (value * 255.0) as u8;
                draw_rectangle(
                    memory_bar_x + i as f32 * (memory_bar_width / MEMORY_SIZE as f32),
                    memory_bar_y,
                    memory_bar_width / MEMORY_SIZE as f32,
                    memory_bar_height,
                    Color::from_rgba(color_value, color_value, color_value, 200)
                );
            }

            for vision_vector in vision_vectors.iter() {
                let end_point = &entity.pos + vision_vector;
                // draw a line from the organism's position to the end point of the vision vector
                draw_line(
                    entity.pos[0],
                    entity.pos[1],
                    end_point[0],
                    end_point[1],
                    1.0,
                    WHITE
                );
              
            }
        }

        next_frame().await
    }
}
