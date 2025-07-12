use geo::Distance;
use macroquad::prelude::*;
use ndarray::{Array1, s};
use ndarray_rand::{RandomExt};
use ndarray_rand::rand_distr::Uniform;

use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use geo::{Line, Euclidean, Point};

mod brain;
mod organism;
mod food;

const BODY_RADIUS: f32 = 3.0;
const VISION_RADIUS: f32 = 30.0;
const ENERGY_CONSUMPTION: f32 = 0.009;
const ACCELERATION_CONSUMPTION: f32 = 0.0001;
const ROTATION_CONSUMPTION: f32 = 0.0001;
// const INIT_VELOCITY: f32 = 20.0;
const DAMPING_FACTOR: f32 = 0.6;
const NUM_VISION_DIRECTIONS: usize = 3; // number of vision directions
const FIELD_OF_VIEW: f32 = std::f32::consts::PI / 2.0; // field of view in radians


const SIGNAL_SIZE: usize = 3; // size of the signal array
const MEMORY_SIZE: usize = 3; // size of the memory array
const N_ORGANISMS: usize = 150;
const N_FOOD: usize = 800;


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
    let mut food = Vec::new();

    let mut kd_tree_orgs = KdTree::new(2);
    let mut kd_tree_food = KdTree::new(2);

    let mut genesis = true;

    let mut screen_center = Array1::zeros(2);

    println!("Starting evolutionary organisms simulation");

    let layer_sizes = vec![
        (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1, // input size
        10, // hidden layer size
        SIGNAL_SIZE + MEMORY_SIZE + 2, // output size (signal + memory + rotation + acceleration)
    ];

    let mut max_id = 0;

    loop {
        
        screen_center = Array1::from_vec(vec![
            screen_width() / 2.,
            screen_height() / 2.,
        ]);
        if genesis {
            
            clear_background(LIGHTGRAY);
            let text = "Start a new evolution by pressing Enter";
            let font_size = 30.0;

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
                        // INIT_VELOCITY,
                        SIGNAL_SIZE,
                        MEMORY_SIZE,
                        layer_sizes.clone()
                    );

                    kd_tree_orgs.add(
                        entity.pos.to_vec(),
                        i as usize,
                    ).unwrap();

                    organisms.push(entity);

                    max_id = i + 1; // update max_id to the last id used

                }

                for i in 0..N_FOOD {
                    let food_item = food::init_random_food(i, &screen_center);

                    kd_tree_food.add(
                        food_item.pos.to_vec(),
                        i as usize,
                    ).unwrap();

                    food.push(food_item);
                }
            }
            next_frame().await;
            continue;
        }

        clear_background(WHITE);

        // Clone the organisms vector
        let new_organisms = organisms.clone();

        for  entity in organisms.iter_mut() {

            let vision_vectors = organism::get_vision_vectors(
                entity,
                FIELD_OF_VIEW,
                NUM_VISION_DIRECTIONS,
                VISION_RADIUS,
            );
            
            // euler integration
            // entity.pos += &(&entity.vel * get_frame_time());
            // entity.vel *= 1.0 - DAMPING_FACTOR * get_frame_time(); // damping

            // wrap around the screen
            entity.pos = wrap_around(&entity.pos);

            entity.energy -= ENERGY_CONSUMPTION * get_frame_time(); // energy consumption

            // get nearest neighbors
            let neighbors_orgs = kd_tree_orgs.within(
                &entity.pos.to_vec(),
                VISION_RADIUS.powi(2),
                &squared_euclidean,
            );

            let neighbor_foods = kd_tree_food.within(
                &entity.pos.to_vec(),
                VISION_RADIUS.powi(2),
                &squared_euclidean,
            );

            // the above returns a Result, so we need to handle the error
            let neighbors_orgs = match neighbors_orgs {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding neighbors: {:?}", e);
                }
            };

            let neighbor_foods = match neighbor_foods {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding food neighbors: {:?}", e);
                }
            };

            // collect all the signals from neighbor organisms and food
            let mut brain_inputs = Array1::zeros((SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1);
            for (i ,vision_vector) in vision_vectors.iter().enumerate() {
                let end_point = &entity.pos + vision_vector;
                let mut min_distance = f32::MAX;
                // detect neighbor organisms within the vision vector
                for (_, neighbor_id) in neighbors_orgs.iter() {
       
                    let neighbor_org = &new_organisms[**neighbor_id];

                    if neighbor_org.id == entity.id {
                        continue; // skip self
                    }
                    let distance = line_circle_distance(
                        &entity.pos,
                        &end_point,
                        &neighbor_org.pos
                    );
                    if distance < BODY_RADIUS && distance < min_distance {
                        min_distance = distance;
                        brain_inputs[(i * 2) + 0] = neighbor_org.signal[0];
                        brain_inputs[(i * 2) + 1] = neighbor_org.signal[1];
                        brain_inputs[(i * 2) + 2] = neighbor_org.signal[2];
                        brain_inputs[(i * 2) + 3] = distance;
                    }

                    let org_org_distance = (&entity.pos - &neighbor_org.pos).mapv(|x| x.abs()).sum();

                    if org_org_distance < BODY_RADIUS * 2.0 {
                        entity.energy = 0.0; // collision with another organism, set energy to 0
                    }
                }

                // detect neighbor food within the vision vector
                for (_, food_id) in neighbor_foods.iter() {
                    let food_item = &food[**food_id];
                    let distance = line_circle_distance(
                        &entity.pos,
                        &end_point,
                        &food_item.pos
                    );
                    if distance < BODY_RADIUS && distance < min_distance {
                        min_distance = distance;
                        brain_inputs[(SIGNAL_SIZE + 1) * i + 0] = 0.0;
                        brain_inputs[(SIGNAL_SIZE + 1) * i + 1] = 0.2; // food signal color (green)
                        brain_inputs[(SIGNAL_SIZE + 1) * i + 2] = 1.0; // food y position
                        brain_inputs[(SIGNAL_SIZE + 1) * i + 3] = distance; // distance to food
                    }
                }
            }

            let offset = (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS;
            // Add the organism's own signal to the inputs
            brain_inputs.slice_mut(s![offset..offset + SIGNAL_SIZE]).assign(&entity.memory);
            brain_inputs[offset + SIGNAL_SIZE] = entity.energy; // energy

            let brain_outputs = brain::think(&entity.brain, &brain_inputs);

            entity.signal = brain_outputs.slice(s![..SIGNAL_SIZE]).to_owned();
            // apply sigmoid activation to the signal
            entity.signal = entity.signal.mapv(|x| 1.0 / (1.0 + (-x).exp()));
            entity.memory = brain_outputs.slice(s![SIGNAL_SIZE..SIGNAL_SIZE + MEMORY_SIZE]).to_owned();
            entity.rot += brain_outputs[brain_outputs.len() - 2]; // rotation adjustment

            // update age
            entity.age += get_frame_time();

            let vel = brain_outputs[brain_outputs.len() - 1]; // acceleration
            // println!("Organism {}: acc = {}, rot = {}, energy = {}", 
            //     entity.id, acc, entity.rot, entity.energy);
            let vel_vector = Array1::from_vec(vec![
                vel * entity.rot.cos(),
                vel * entity.rot.sin(),
            ]) * 40.0; // scale acceleration
   
            entity.pos += &(&vel_vector * get_frame_time()); // update velocity
            entity.energy -= vel.abs() * get_frame_time() * ACCELERATION_CONSUMPTION; // energy consumption for acceleration
            entity.energy -= entity.rot.abs() * get_frame_time() * ROTATION_CONSUMPTION; // energy consumption for rotation
            entity.energy -= ENERGY_CONSUMPTION * get_frame_time(); // additional energy consumption

            // handle food consumption
            let food_neighbors = kd_tree_food.within(
                &entity.pos.to_vec(),
                (BODY_RADIUS * 2.0).powi(2),
                &squared_euclidean,
            );

            let food_neighbors = match food_neighbors {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    panic!("Error finding food neighbors: {:?}", e);
                }
            };
            
            // consume all food within BODY_RADIUS
            for (_, food_id) in food_neighbors.iter() {
                let food_item = &mut food[**food_id];
                if food_item.energy > 0.0 {
                    entity.energy += food_item.energy; // consume the food
                    // cap energy to a maximum value
                    entity.energy = entity.energy.min(1.0);
                    entity.score += 1; // increase score for reproduction
                    food_item.energy = 0.0; // remove the food

                    // println!("Organism {} consumed food at {:?}", entity.id, food_item.pos);
                }
            }
            // println!("Organism {}: pos = {:?}, vel = {:?}, energy = {}, signal = {:?}", 
            //     entity.id, 
            //     entity.pos, 
            //     entity.vel, 
            //     entity.energy, 
            //     entity.signal
            // );

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
            let health_bar_height = 2.0;
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
                Color::from_rgba(255, 0, 0, 255)
            );

            // organism id
            let id_text = format!("ID:{}", entity.id);
            let id_text_size = measure_text(&id_text, None, 12, 1.0);
            draw_text(
                &id_text,
                entity.pos[0] - id_text_size.width / 2.0,
                entity.pos[1] - BODY_RADIUS - health_bar_height - 2.0 - 10.0,
                9.0,
                BLACK
            );

            // organism age
            let age_text = format!("Age: {:.1}", entity.age);
            let age_text_size = measure_text(&age_text, None, 12, 1.0);
            draw_text(
                &age_text,
                entity.pos[0] - age_text_size.width / 2.0,
                entity.pos[1] - BODY_RADIUS - health_bar_height - 2.0 - 20.0,
                9.0,
                BLACK
            );

            // organism score
            let score_text = format!("Score: {}", entity.score);
            let score_text_size = measure_text(&score_text, None, 12, 1.0);
            draw_text(
                &score_text,
                entity.pos[0] - score_text_size.width / 2.0,
                entity.pos[1] - BODY_RADIUS - health_bar_height - 2.0 - 30.0,
                9.0,
                BLACK
            );

            // // organism memory, simple rectangles
            // let memory_bar_width = 20.0;
            // let memory_bar_height = 3.0;
            // let memory_bar_x = entity.pos[0] - memory_bar_width / 2.0;
            // let memory_bar_y = entity.pos[1] - BODY_RADIUS - health_bar_height - memory_bar_height - 2.0;
            // for (i, &value) in entity.memory.iter().enumerate() {
            //     let color_value = (value * 255.0) as u8;
            //     draw_rectangle(
            //         memory_bar_x + i as f32 * (memory_bar_width / MEMORY_SIZE as f32),
            //         memory_bar_y,
            //         memory_bar_width / MEMORY_SIZE as f32,
            //         memory_bar_height,
            //         Color::from_rgba(color_value, color_value, color_value, 200)
            //     );
            // }

            for (i, vision_vector) in vision_vectors.iter().enumerate() {
                let end_point = &entity.pos + vision_vector;
                // draw a line from the organism's position to the end point of the vision vector
                draw_line(
                    entity.pos[0],
                    entity.pos[1],
                    end_point[0],
                    end_point[1],
                    1.0,
                    Color::from_rgba(
                        (brain_inputs[(SIGNAL_SIZE + 1) * i + 0]* 255.0) as u8,
                        (brain_inputs[(SIGNAL_SIZE + 1) * i + 1] * 255.0) as u8,
                        (brain_inputs[(SIGNAL_SIZE + 1) * i + 2] * 255.0) as u8,
                        255
                    )
                );
            }
        }

        // draw food
        for food_item in food.iter() {
            if food_item.energy > 0.0 {
                draw_circle(
                    food_item.pos[0], 
                    food_item.pos[1],
                    BODY_RADIUS,
                    Color::from_rgba(0, 100, 255, 255)
                );
            }
        }

        // Update the organisms vector to keep only the ones that are still alive
        organisms.retain(|entity| entity.energy > 0.0);
        food.retain(|food_item| food_item.energy > 0.0);

        // sort organisms by score in descending order
        organisms.sort_by(|a, b| b.score.cmp(&a.score));

        // spawn new organisms if there are less than N_ORGANISMS
        if organisms.len() < N_ORGANISMS {

            let mut new_organism = organism::init_random_organism(
                max_id,
                &screen_center, 
                // INIT_VELOCITY,
                SIGNAL_SIZE,
                MEMORY_SIZE,
                layer_sizes.clone(),
            );

            let mutation_scale = rand::gen_range(0.002, 0.2);

            println!("mutation scale: {}", mutation_scale);

            // choose reproduction strategy randomly

            // let reproduction_strategy = rand::gen_range(0, 2); 

            // if reproduction_strategy == 2 {
                
            //     // pick a random organism to reproduce
            //     let id_a = rand::gen_range(0, organisms.len() / 10); // pick from the top 10% of organisms
            //     let id_b = rand::gen_range(0, organisms.len() / 10); // pick from the top 10% of organisms

            //     let parent_1 = &organisms[id_a];
            //     let parent_2 = &organisms[id_b];

            //     let mut crossover_brain = brain::crossover(&parent_1.brain, &parent_2.brain);

            //     brain::mutate_brain(
            //         // pass as mutable
            //         &mut crossover_brain,
            //         0.1, // mutation rate
            //     );

            //     // set the new organism's brain to the crossover brain
            //     new_organism.brain = crossover_brain;

            // } else if reproduction_strategy == 1 {

                let id = rand::gen_range(0, organisms.len() / 10); // pick from the top 10% of organisms

                let parent = &organisms[id];

                // clone the parent's brain
                let mut cloned_brain = parent.brain.clone();

                brain::mutate_brain(
                    &mut cloned_brain,
                    mutation_scale, // mutation rate
                );

                // set the new organism's brain to the cloned brain
                new_organism.brain = cloned_brain;
            // } 

            max_id += 1; // increment max_id for the new organism

            kd_tree_orgs.add(
                new_organism.pos.to_vec(),
                organisms.len(),
            ).unwrap();

            organisms.push(new_organism);
        }

        // spawn new food if there are less than N_FOOD
        if food.len() < N_FOOD {
            let food_item = food::init_random_food(food.len(), &screen_center);
            kd_tree_food.add(
                food_item.pos.to_vec(),
                food.len(),
            ).unwrap();
            food.push(food_item);
        }


        // Update the kdtree with the new positions of the organisms
        kd_tree_orgs = KdTree::new(2);
        for (i, org_item) in organisms.iter().enumerate() {
            kd_tree_orgs.add(
                org_item.pos.to_vec(),
                i as usize,
            ).unwrap();
        };

        // Update the kdtree with the new positions of the food
        kd_tree_food = KdTree::new(2);
        for (i, food_item) in food.iter().enumerate() {
            kd_tree_food.add(
                food_item.pos.to_vec(),
                i as usize,
            ).unwrap();
        };   

        next_frame().await
    }
}
