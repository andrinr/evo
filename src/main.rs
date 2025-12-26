use macroquad::prelude::*;

mod graphics;
mod simulation;
mod ui;

#[macroquad::main("Evolutionary Organisms")]
async fn main() {
    let mut genesis = true;

    let mut ecosystem: Option<simulation::ecosystem::Ecosystem> = None;
    let mut ui_state = ui::UIState::new();

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
        if let Some(ref mut eco) = ecosystem {
            // Handle keyboard shortcuts
            if is_key_pressed(KeyCode::S)
                && (is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl))
            {
                ui_state.save_requested = true;
            }
            if is_key_pressed(KeyCode::L)
                && (is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl))
            {
                ui_state.load_requested = true;
            }

            // Handle save request
            if ui_state.save_requested {
                ui_state.save_requested = false;
                let save_path = format!(
                    "evolution_save_{}.json",
                    chrono::Local::now().format("%Y%m%d_%H%M%S")
                );
                match eco.save_to_file(&save_path) {
                    Ok(_) => {
                        ui_state.status_message = Some(format!("✓ Saved to {}", save_path));
                        println!("Saved evolution state to {}", save_path);
                    }
                    Err(e) => {
                        ui_state.status_message = Some(format!("✗ Save failed: {}", e));
                        eprintln!("Failed to save: {}", e);
                    }
                }
            }

            // Handle load request
            if ui_state.load_requested {
                ui_state.load_requested = false;
                // Find the most recent save file
                if let Ok(entries) = std::fs::read_dir(".") {
                    let mut save_files: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path()
                                .file_name()
                                .and_then(|n| n.to_str())
                                .map(|s| s.starts_with("evolution_save_") && s.ends_with(".json"))
                                .unwrap_or(false)
                        })
                        .collect();

                    save_files.sort_by_key(|e| std::cmp::Reverse(e.path().clone()));

                    if let Some(latest_file) = save_files.first() {
                        let load_path = latest_file.path();
                        match simulation::ecosystem::Ecosystem::load_from_file(
                            load_path.to_str().unwrap(),
                        ) {
                            Ok(loaded_eco) => {
                                *eco = loaded_eco;
                                ui_state.status_message =
                                    Some(format!("✓ Loaded from {}", load_path.display()));
                                println!("Loaded evolution state from {}", load_path.display());
                                // Clear history as it's from a different timeline
                                ui_state.avg_age_history.clear();
                                ui_state.avg_score_history.clear();
                                ui_state.set_last_update_time(eco.time);
                            }
                            Err(e) => {
                                ui_state.status_message = Some(format!("✗ Load failed: {}", e));
                                eprintln!("Failed to load: {}", e);
                            }
                        }
                    } else {
                        ui_state.status_message = Some("✗ No save files found".to_string());
                    }
                }
            }

            // Update history data
            ui_state.update_history(eco);

            // Handle organism click (select/deselect)
            if let Some(clicked_id) =
                graphics::handle_organism_click(eco, &params, ui_state.stats_panel_width)
            {
                // Toggle selection: if clicking the same organism, deselect it
                if ui_state.selected_organism_id == Some(clicked_id) {
                    ui_state.selected_organism_id = None;
                } else {
                    ui_state.selected_organism_id = Some(clicked_id);
                }
            }

            // Update hovered organism
            ui_state.hovered_organism_id =
                graphics::get_hovered_organism(eco, &params, ui_state.stats_panel_width);

            // Draw simulation
            graphics::draw_food(eco, &params, ui_state.stats_panel_width);
            graphics::draw_organisms(eco, &params, ui_state.stats_panel_width);

            // Draw UI
            ui::draw_ui(&mut ui_state, eco, &params);
        }

        // Process egui rendering
        ui::process_egui();

        next_frame().await
    }
}
