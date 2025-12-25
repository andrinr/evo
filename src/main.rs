use macroquad::prelude::*;

mod brain;
mod evolution;
mod food;
mod graphics;
mod organism;

#[macroquad::main("Evolutionary Organisms")]
async fn main() {
    let mut genesis = true;

    let mut state: Option<evolution::State> = None;

    let signal_size: usize = 3;
    let num_vision_directions: usize = 3;
    let memory_size: usize = 3;

    let layer_sizes = vec![
        (signal_size + 1) * num_vision_directions + memory_size + 1, // input size
        10,                                                          // hidden layer size
        signal_size + memory_size + 2, // output size (signal + memory + rotation + acceleration)
    ];

    let params = evolution::Params {
        body_radius: 3.0,
        vision_radius: 30.0,
        idle_energy_rate: 0.009,
        move_energy_rate: 0.0001,
        rot_energy_rate: 0.0001,
        num_vision_directions,
        fov: std::f32::consts::PI / 2.0,
        signal_size,
        memory_size,
        n_organism: 150,
        n_food: 800,
        box_width: 1.0,
        box_height: 1.0,
        layer_sizes,
    };

    println!("Starting evolutionary organisms simulation");

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

                state = Some(evolution::init(&params));
            }
            next_frame().await;
            continue;
        }

        clear_background(WHITE);

        if let Some(ref mut state) = state {
            evolution::step(state, &params, 0.01);
            evolution::spawn(state, &params);

            graphics::draw_food(state, &params);
            graphics::draw_organisms(state, &params);
        }

        next_frame().await
    }
}
