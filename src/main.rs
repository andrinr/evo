use macroquad::prelude::*;

mod graphics;
mod simulation;

#[macroquad::main("Evolutionary Organisms")]
async fn main() {
    let mut genesis = true;

    let mut ecosystem: Option<simulation::ecosystem::Ecosystem> = None;

    let signal_size: usize = 3;
    let num_vision_directions: usize = 3;
    let memory_size: usize = 3;

    let layer_sizes = vec![
        (signal_size + 1) * num_vision_directions + memory_size + 1, // input size
        10,                                                          // hidden layer size
        signal_size + memory_size + 2, // output size (signal + memory + rotation + acceleration)
    ];

    let params = simulation::ecosystem::Params {
        body_radius: 3.0,
        vision_radius: 30.0,
        idle_energy_rate: 0.023,
        move_energy_rate: 0.0002,
        move_multiplier: 60.0,
        rot_energy_rate: 0.0003,
        num_vision_directions,
        fov: std::f32::consts::PI / 2.0,
        signal_size,
        memory_size,
        n_organism: 500,
        n_food: 400,
        box_width: 1000.0,
        box_height: 1000.0,
        layer_sizes,
    };

    println!("Starting evolutionary organisms simulation");

    // Simulation timing
    let simulation_fps = 20.0; // How many simulation steps per second
    let simulation_dt = 1.0 / simulation_fps;
    let mut accumulator = 0.0;
    let mut last_time = get_time();

    loop {
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
                ecosystem = Some(simulation::ecosystem::Ecosystem::new(&params));
                last_time = get_time(); // Reset time when starting
            }
            next_frame().await;
            continue;
        }

        // Calculate delta time since last frame
        let current_time = get_time();
        let frame_time = (current_time - last_time) as f32;
        last_time = current_time;

        // Accumulate time for fixed timestep updates
        accumulator += frame_time;

        // Run simulation at fixed timestep
        if let Some(ref mut eco) = ecosystem {
            // Update simulation as many times as needed to catch up
            while accumulator >= simulation_dt {
                eco.step(&params, simulation_dt);
                eco.spawn(&params);
                accumulator -= simulation_dt;
            }
        }

        // Render at display refresh rate (uncapped)
        clear_background(WHITE);
        if let Some(ref eco) = ecosystem {
            graphics::draw_food(eco, &params);
            graphics::draw_organisms(eco, &params);
        }

        next_frame().await
    }
}
