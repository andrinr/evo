use geo::Distance;
use macroquad::prelude::*;
use ndarray::{Array1, s};

use kdtree::KdTree;
use geo::{Line, Euclidean, Point};

mod brain;
mod organism;
mod food;
mod evolution;
mod graphics;

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


#[macroquad::main("Evolutionary Organisms")]
async fn main() {

    let mut kd_tree_orgs = KdTree::new(2);
    let mut kd_tree_food = KdTree::new(2);

    let mut genesis = true;

    let mut state : evolution::State;

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

                evolution = evolution::init(
                    N_ORGANISMS,
                    N_FOOD,
                    &screen_center,
                    SIGNAL_SIZE,
                    MEMORY_SIZE,
                    layer_sizes
                )
            }
            next_frame().await;
            continue;
        }

        clear_background(WHITE);

        // Clone the organisms vector
        let new_organisms = organisms.clone();

        state = evolution::step(state);


        graphics::draw_food(state);
        graphics::draw_food(state);


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

            let reproduction_strategy = rand::gen_range(0, 2); 

            if reproduction_strategy == 2 {
                
                // pick a random organism to reproduce
                let id_a = rand::gen_range(0, organisms.len() / 10); // pick from the top 10% of organisms
                let id_b = rand::gen_range(0, organisms.len() / 10); // pick from the top 10% of organisms

                let parent_1 = &organisms[id_a];
                let parent_2 = &organisms[id_b];

                let mut crossover_brain = brain::crossover(&parent_1.brain, &parent_2.brain);

                brain::mutate_brain(
                    // pass as mutable
                    &mut crossover_brain,
                    0.1, // mutation rate
                );

                // set the new organism's brain to the crossover brain
                new_organism.brain = crossover_brain;

            } else if reproduction_strategy == 1 {

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
            } 

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
