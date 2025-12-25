use crate::brain;
use crate::food;
use crate::organism;

use geo::algorithm::Distance;
use geo::{Euclidean, Line, Point};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use ndarray::{Array1, s};
use rand::Rng;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct State {
    pub organisms: Vec<organism::Organism>,
    pub food: Vec<food::Food>,
    pub time: f32,
    pub generation: u32,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub body_radius: f32,
    pub vision_radius: f32,
    pub idle_energy_rate: f32,
    pub move_energy_rate: f32,
    pub rot_energy_rate: f32,
    pub num_vision_directions: usize,
    pub fov: f32,
    pub signal_size: usize,
    pub memory_size: usize,
    pub n_organism: usize,
    pub n_food: usize,
    pub box_width: f32,
    pub box_height: f32,
    pub layer_sizes: Vec<usize>,
}

pub fn init(params: &Params) -> State {
    let mut organisms = Vec::with_capacity(params.n_organism);
    let mut food = Vec::with_capacity(params.n_food);

    let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

    for i in 0..params.n_organism {
        let entity = organism::init_random_organism(
            i,
            &center,
            params.signal_size,
            params.memory_size,
            params.layer_sizes.clone(),
        );

        organisms.push(entity);
    }

    for _i in 0..params.n_food {
        let food_item = food::init_random_food(&center);

        food.push(food_item);
    }

    State {
        organisms,
        food,
        time: 0.,
        generation: params.n_organism as u32,
    }
}

type Tree2D = KdTree<f32, usize, Vec<f32>>;

fn build_trees(state: &State) -> (Tree2D, Tree2D) {
    let mut kd_tree_orgs = KdTree::with_capacity(2, state.organisms.len());
    for (i, org_item) in state.organisms.iter().enumerate() {
        kd_tree_orgs.add(org_item.pos.to_vec(), i).unwrap();
    }

    let mut kd_tree_food = KdTree::with_capacity(2, state.food.len());
    for (i, food_item) in state.food.iter().enumerate() {
        kd_tree_food.add(food_item.pos.to_vec(), i).unwrap();
    }

    (kd_tree_orgs, kd_tree_food)
}

fn line_circle_distance(
    line_start: &Array1<f32>,
    line_end: &Array1<f32>,
    circle_center: &Array1<f32>,
) -> f32 {
    let p = Point::new(circle_center[0], circle_center[1]);
    let line = Line::new(
        Point::new(line_start[0], line_start[1]),
        Point::new(line_end[0], line_end[1]),
    );
    Euclidean.distance(&p, &line)
}

fn wrap_around(v: &Array1<f32>, box_width: f32, box_height: f32) -> Array1<f32> {
    let mut wrapped = v.clone();
    if wrapped[0] < 0.0 {
        wrapped[0] += box_width;
    } else if wrapped[0] > box_width {
        wrapped[0] -= box_width;
    }
    if wrapped[1] < 0.0 {
        wrapped[1] += box_height;
    } else if wrapped[1] > box_height {
        wrapped[1] -= box_height;
    }
    wrapped
}

pub fn step(state: &mut State, params: &Params, dt: f32) {
    let (kd_tree_orgs, kd_tree_food) = build_trees(state);

    // Clone the organisms vector
    let new_organisms = state.organisms.clone();
    // let new_food = state.food.clone();

    // Track consumed food items across threads
    let consumed_food = Mutex::new(Vec::new());

    state.time += dt;

    state.organisms.par_iter_mut().for_each(|entity| {
        let vision_vectors = organism::get_vision_vectors(
            entity,
            params.fov,
            params.num_vision_directions,
            params.vision_radius,
        );

        // wrap around the screen
        entity.pos = wrap_around(&entity.pos, params.box_width, params.box_width);

        // get nearest neighbors
        let neighbors_orgs = kd_tree_orgs.within(
            &entity.pos.to_vec(),
            params.vision_radius.powi(2),
            &squared_euclidean,
        );

        let neighbor_foods = kd_tree_food.within(
            &entity.pos.to_vec(),
            params.vision_radius.powi(2),
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
        let mut brain_inputs = Array1::zeros(
            (params.signal_size + 1) * params.num_vision_directions + params.memory_size + 1,
        );

        for (i, vision_vector) in vision_vectors.iter().enumerate() {
            let end_point = &entity.pos + vision_vector;
            let mut min_distance = f32::MAX;
            // detect neighbor organisms within the vision vector
            for (_, neighbor_id) in neighbors_orgs.iter() {
                let neighbor_org = &new_organisms[**neighbor_id];

                if neighbor_org.id == entity.id {
                    continue; // skip self
                }
                let distance = line_circle_distance(&entity.pos, &end_point, &neighbor_org.pos);
                if distance < params.body_radius && distance < min_distance {
                    min_distance = distance;
                    brain_inputs[i * 2] = neighbor_org.signal[0];
                    brain_inputs[(i * 2) + 1] = neighbor_org.signal[1];
                    brain_inputs[(i * 2) + 2] = neighbor_org.signal[2];
                    brain_inputs[(i * 2) + 3] = distance;
                }

                let org_org_distance = (&entity.pos - &neighbor_org.pos).mapv(|x| x.abs()).sum();

                if org_org_distance < params.body_radius * 2.0 {
                    entity.energy = 0.0; // collision with another organism, set energy to 0
                }
            }

            // detect neighbor food within the vision vector
            for (_, food_id) in neighbor_foods.iter() {
                let food_item = &state.food[**food_id];
                let distance = line_circle_distance(&entity.pos, &end_point, &food_item.pos);
                if distance < params.body_radius && distance < min_distance {
                    min_distance = distance;
                    brain_inputs[(params.signal_size + 1) * i] = 0.0;
                    brain_inputs[(params.signal_size + 1) * i + 1] = 0.2; // food signal color (green)
                    brain_inputs[(params.signal_size + 1) * i + 2] = 1.0; // food y position
                    brain_inputs[(params.signal_size + 1) * i + 3] = distance; // distance to food
                }
            }
        }

        let offset = (params.signal_size + 1) * params.num_vision_directions;
        // Add the organism's own signal to the inputs
        brain_inputs
            .slice_mut(s![offset..offset + params.signal_size])
            .assign(&entity.memory);
        brain_inputs[offset + params.signal_size] = entity.energy; // energy

        let brain_outputs = brain::think(&entity.brain, &brain_inputs);

        entity.signal = brain_outputs.slice(s![..params.signal_size]).to_owned();
        // apply sigmoid activation to the signal
        entity.signal = entity.signal.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        entity.memory = brain_outputs
            .slice(s![
                params.signal_size..params.signal_size + params.memory_size
            ])
            .to_owned();
        entity.rot += brain_outputs[brain_outputs.len() - 2]; // rotation adjustment

        // update age
        entity.age += dt;

        let vel = brain_outputs[brain_outputs.len() - 1]; // acceleration
        // println!("Organism {}: acc = {}, rot = {}, energy = {}",
        //     entity.id, acc, entity.rot, entity.energy);
        let vel_vector =
            Array1::from_vec(vec![vel * entity.rot.cos(), vel * entity.rot.sin()]) * 40.0; // scale acceleration

        entity.pos += &(&vel_vector * dt); // update velocity
        entity.energy -= vel.abs() * dt * params.move_energy_rate; // energy consumption for acceleration
        entity.energy -= entity.rot.abs() * dt * params.rot_energy_rate; // energy consumption for rotation
        entity.energy -= params.idle_energy_rate * dt; // additional energy consumption

        // handle food consumption
        let food_neighbors = kd_tree_food.within(
            &entity.pos.to_vec(),
            (params.body_radius * 2.0).powi(2),
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
            let food_item = &state.food[**food_id];
            if food_item.energy > 0.0 {
                entity.energy += food_item.energy; // consume the food
                // cap energy to a maximum value
                entity.energy = entity.energy.min(1.0);
                entity.score += 1; // increase score for reproduction

                // Record this food as consumed
                consumed_food.lock().unwrap().push(**food_id);

                // println!("Organism {} consumed food at {:?}", entity.id, food_item.pos);
            }
        }
    });

    // Mark all consumed food as depleted
    for food_id in consumed_food.lock().unwrap().iter() {
        state.food[*food_id].energy = 0.0;
    }
}

pub fn spawn(state: &mut State, params: &Params) {
    // sort organisms by score in descending order
    state.organisms.sort_by(|a, b| b.score.cmp(&a.score));

    let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

    // spawn new organisms if there are less than N_ORGANISMS
    if state.organisms.len() < params.n_organism {
        let mut new_organism = organism::init_random_organism(
            state.generation as usize,
            &center,
            params.signal_size,
            params.memory_size,
            params.layer_sizes.clone(),
        );

        state.generation += 1;

        let mutation_scale = rand::rng().random_range(0.002..0.2);

        println!("mutation scale: {}", mutation_scale);

        // choose reproduction strategy randomly

        let reproduction_strategy = rand::rng().random_range(0..3);

        if reproduction_strategy == 2 {
            // pick a random organism to reproduce
            let id_a = rand::rng().random_range(0..state.organisms.len() / 10); // pick from the top 10% of organisms
            let id_b = rand::rng().random_range(0..state.organisms.len() / 10); // pick from the top 10% of organisms

            let parent_1 = &state.organisms[id_a];
            let parent_2 = &state.organisms[id_b];

            let mut crossover_brain = brain::crossover(&parent_1.brain, &parent_2.brain);

            brain::mutate_brain(
                // pass as mutable
                &mut crossover_brain,
                0.1, // mutation rate
            );

            // set the new organism's brain to the crossover brain
            new_organism.brain = crossover_brain;
        } else if reproduction_strategy == 1 {
            let id = rand::rng().random_range(0..state.organisms.len() / 10); // pick from the top 10% of organisms

            let parent = &state.organisms[id];

            // clone the parent's brain
            let mut cloned_brain = parent.brain.clone();

            brain::mutate_brain(
                &mut cloned_brain,
                mutation_scale, // mutation rate
            );

            new_organism.brain = cloned_brain;
        }
        state.organisms.push(new_organism);
    }

    if state.food.len() < params.n_food {
        let food_item = food::init_random_food(&center);
        state.food.push(food_item);
    }
}
