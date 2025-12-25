use macroquad::prelude::*;
use ndarray::{Array1};

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

    let mut genesis = true;

    let mut state : evolution::State;

    let layer_sizes = vec![
        (SIGNAL_SIZE + 1) * NUM_VISION_DIRECTIONS + MEMORY_SIZE + 1, // input size
        10, // hidden layer size
        SIGNAL_SIZE + MEMORY_SIZE + 2, // output size (signal + memory + rotation + acceleration)
    ];

    let params = evolution::Params {
        body_radius: 3.0,
        vision_radius: 30.0,
        idle_energy_rate: 0.009,
        move_energy_rate: 0.0001,
        rot_energy_rate: 0.0001,
        num_vision_directions: 3,
        fov: std::f32::consts::PI / 2.0,
        signal_size: 3,
        memory_size: 3,
        n_organism: 150,
        n_food: 800,
        box_width : 1.0,
        box_height : 1.0,
        layer_sizes
    };

    let mut screen_center = Array1::zeros(2);

    println!("Starting evolutionary organisms simulation");

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

                state = evolution::init(&params)
            }
            next_frame().await;
            continue;
        }

        clear_background(WHITE);

        if let Some(ref mut state) = state {
            
            evolution::step(&state, &params, 0.01);
            evolution::spawn(&state, &params);

            graphics::draw_food(&state, &params);
            graphics::draw_organisms(&state, &params);

        }

        next_frame().await
    }
}
