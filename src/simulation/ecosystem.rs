use super::brain;
use super::events;
use super::food;
use super::organism;

use geo::algorithm::Distance;
use geo::{Euclidean, Line, Point};
use kdtree::distance::squared_euclidean;
use kdtree::{ErrorKind as KdTreeError, KdTree};
use ndarray::{Array1, s};
use rand::Rng;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct Params {
    pub body_radius: f32,
    pub vision_radius: f32,
    pub idle_energy_rate: f32,
    pub move_energy_rate: f32,
    pub move_multiplier: f32,
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

#[derive(Debug, Clone)]
pub struct Ecosystem {
    pub organisms: Vec<organism::Organism>,
    pub food: Vec<food::Food>,
    pub time: f32,
    pub generation: u32,
}

impl Ecosystem {
    pub fn new(params: &Params) -> Self {
        let mut organisms = Vec::with_capacity(params.n_organism);
        let mut food = Vec::with_capacity(params.n_food);

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        for i in 0..params.n_organism {
            let entity = organism::Organism::new_random(
                i,
                &center,
                params.signal_size,
                params.memory_size,
                params.layer_sizes.clone(),
            );

            organisms.push(entity);
        }

        for _i in 0..params.n_food {
            let food_item = food::Food::new_random(&center);
            food.push(food_item);
        }

        Self {
            organisms,
            food,
            time: 0.,
            generation: params.n_organism as u32,
        }
    }

    pub fn step(&mut self, params: &Params, dt: f32) {
        let (kd_tree_orgs, kd_tree_food) = build_trees(self).expect("Failed to build kd-trees");

        // Clone the organisms vector
        let new_organisms = self.organisms.clone();

        let event_queue = Mutex::new(events::EventQueue::new());

        self.time += dt;

        // parallel phase, only apply updates to entity itself
        // for events involing other objects, use the event queue for thread safety
        self.organisms.par_iter_mut().for_each(|entity| {
            let mut local_events = Vec::new();

            let vision_vectors = entity.get_vision_vectors(
                params.fov,
                params.num_vision_directions,
                params.vision_radius,
            );

            // wrap around the screen
            wrap_around_mut(&mut entity.pos, params.box_width, params.box_height);

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

                    let org_org_distance =
                        (&entity.pos - &neighbor_org.pos).mapv(|x| x.abs()).sum();

                    if org_org_distance < params.body_radius * 2.0 {
                        entity.kill(); // collision with another organism
                    }
                }

                // detect neighbor food within the vision vector
                for (_, food_id) in neighbor_foods.iter() {
                    let food_item = &self.food[**food_id];
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

            let brain_outputs = entity.brain.think(&brain_inputs);

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
            entity.age_by(dt);

            let vel = brain_outputs[brain_outputs.len() - 1]; // acceleration

            let vel_vector = Array1::from_vec(vec![vel * entity.rot.cos(), vel * entity.rot.sin()])
                * params.move_multiplier; // scale acceleration

            entity.pos += &(&vel_vector * dt); // update velocity
            entity.consume_energy(vel.abs() * dt * params.move_energy_rate); // energy consumption for acceleration
            entity.consume_energy(entity.rot.abs() * dt * params.rot_energy_rate); // energy consumption for rotation
            entity.consume_energy(params.idle_energy_rate * dt); // additional energy consumption

            // consume all food within BODY_RADIUS
            for (_, food_id) in neighbor_foods.iter() {
                let food_item = &self.food[**food_id];
                let org_food_dist = (&entity.pos - &food_item.pos).mapv(|x| x.abs()).sum();
                if org_food_dist < params.body_radius * 2.0 && !food_item.is_consumed() {
                    entity.gain_energy(food_item.energy, 1.0);
                    entity.score += 1; // increase score for reproduction

                    local_events.push(events::SimulationEvent::FoodConsumed {
                        organism_id: entity.id,
                        food_id: **food_id,
                    });

                    println!("org {} consumed {}", entity.id, food_id);
                }
            }

            let mut queue = event_queue.lock().unwrap();
            for event in local_events {
                queue.push(event);
            }
        });

        events::apply_events(self, params, event_queue.into_inner().unwrap());

        self.organisms.retain(|entity| entity.is_alive());
        self.food.retain(|food_item| !food_item.is_consumed());
    }

    pub fn spawn(&mut self, params: &Params) {
        // sort organisms by score in descending order
        self.organisms.sort_by(|a, b| b.score.cmp(&a.score));

        let center = Array1::from_vec(vec![params.box_width / 2., params.box_height / 2.]);

        // spawn new organisms if there are less than N_ORGANISMS
        if self.organisms.len() < params.n_organism {
            let mut new_organism = organism::Organism::new_random(
                self.generation as usize,
                &center,
                params.signal_size,
                params.memory_size,
                params.layer_sizes.clone(),
            );

            self.generation += 1;

            let mutation_scale = rand::rng().random_range(0.002..0.2);

            println!("mutation scale: {}", mutation_scale);

            // choose reproduction strategy randomly

            let reproduction_strategy = rand::rng().random_range(0..3);

            if reproduction_strategy == 2 {
                // pick a random organism to reproduce
                let id_a = rand::rng().random_range(0..self.organisms.len() / 10); // pick from the top 10% of organisms
                let id_b = rand::rng().random_range(0..self.organisms.len() / 10); // pick from the top 10% of organisms

                let parent_1 = &self.organisms[id_a];
                let parent_2 = &self.organisms[id_b];

                let mut crossover_brain = brain::Brain::crossover(&parent_1.brain, &parent_2.brain);
                crossover_brain.mutate(0.1);

                new_organism.brain = crossover_brain;
            } else if reproduction_strategy == 1 {
                let id = rand::rng().random_range(0..self.organisms.len() / 10); // pick from the top 10% of organisms

                let parent = &self.organisms[id];

                let mut cloned_brain = parent.brain.clone();
                cloned_brain.mutate(mutation_scale);

                new_organism.brain = cloned_brain;
            }
            self.organisms.push(new_organism);
        }

        if self.food.len() < params.n_food {
            let food_item = food::Food::new_random(&center);
            self.food.push(food_item);
        }
    }
}

type Tree2D = KdTree<f32, usize, Vec<f32>>;

fn build_tree<T>(items: &[T], get_pos: impl Fn(&T) -> Vec<f32>) -> Result<Tree2D, KdTreeError> {
    let mut tree = KdTree::with_capacity(2, items.len());
    for (i, item) in items.iter().enumerate() {
        tree.add(get_pos(item), i)?;
    }
    Ok(tree)
}

fn build_trees(ecosystem: &Ecosystem) -> Result<(Tree2D, Tree2D), KdTreeError> {
    let kd_tree_orgs = build_tree(&ecosystem.organisms, |org| org.pos.to_vec())?;
    let kd_tree_food = build_tree(&ecosystem.food, |food| food.pos.to_vec())?;
    Ok((kd_tree_orgs, kd_tree_food))
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

fn wrap_around_mut(v: &mut Array1<f32>, box_width: f32, box_height: f32) {
    v[0] = v[0].rem_euclid(box_width);
    v[1] = v[1].rem_euclid(box_height);
}
