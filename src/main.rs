//! Evolutionary simulation with neural networks and projectile combat.
//!
//! This is the main binary that runs the simulation with a graphical interface.
//! Organisms evolve behaviors through genetic algorithms while competing for food
//! and attacking each other with projectiles.

use macroquad::prelude::*;

mod graphics;
mod simulation;
mod ui;

fn create_simulation_params() -> simulation::ecosystem::Params {
    let signal_size: usize = 3;
    let num_vision_directions: usize = 3;
    let memory_size: usize = 3;

    let layer_sizes = vec![
        (signal_size + 1) * num_vision_directions + memory_size + 1, // input size
        10,                                                          // hidden layer size
        signal_size + memory_size + 3, // output size (signal + memory + rotation + acceleration + attack)
    ];

    let vision_radius = 30.0;

    simulation::ecosystem::Params {
        body_radius: 3.0,
        vision_radius,
        idle_energy_rate: 0.023,
        move_energy_rate: 0.0002,
        move_multiplier: 60.0,
        rot_energy_rate: 0.0003,
        num_vision_directions,
        fov: std::f32::consts::PI / 2.0,
        signal_size,
        memory_size,
        n_organism: 200,
        n_food: 100,
        box_width: 1000.0,
        box_height: 1000.0,
        layer_sizes,
        attack_cost_rate: 0.2,
        attack_damage_rate: 0.4,
        attack_cooldown: 1.0,
        corpse_energy_ratio: 0.8,
        projectile_speed: vision_radius * 2.0,
        projectile_range: vision_radius,
        projectile_radius: 1.0,
        organism_spawn_rate: 0.5,
        food_spawn_rate: 0.01,
    }
}

fn draw_genesis_screen() {
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
}

fn handle_keyboard_shortcuts(ui_state: &mut ui::UIState) {
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
}

fn handle_save_request(eco: &simulation::ecosystem::Ecosystem, ui_state: &mut ui::UIState) {
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

fn find_latest_save_file() -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(".").ok()?;

    let mut save_files: Vec<_> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("evolution_save_") && s.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();

    save_files.sort_by_key(|e| std::cmp::Reverse(e.path().clone()));
    save_files.first().map(std::fs::DirEntry::path)
}

fn handle_load_request(eco: &mut simulation::ecosystem::Ecosystem, ui_state: &mut ui::UIState) {
    let Some(load_path) = find_latest_save_file() else {
        ui_state.status_message = Some("✗ No save files found".to_string());
        return;
    };

    match simulation::ecosystem::Ecosystem::load_from_file(load_path.to_str().unwrap()) {
        Ok(loaded_eco) => {
            *eco = loaded_eco;
            ui_state.status_message = Some(format!("✓ Loaded from {}", load_path.display()));
            println!("Loaded evolution state from {}", load_path.display());
            // Clear history as it's from a different timeline
            ui_state.organism_count_history.clear();
            ui_state.food_count_history.clear();
            ui_state.avg_age_history.clear();
            ui_state.avg_score_history.clear();
            ui_state.set_last_update_time(eco.time);
        }
        Err(e) => {
            ui_state.status_message = Some(format!("✗ Load failed: {}", e));
            eprintln!("Failed to load: {}", e);
        }
    }
}

fn handle_organism_selection(
    eco: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_state: &mut ui::UIState,
) {
    if let Some(clicked_id) =
        graphics::handle_organism_click(eco, params, ui_state.stats_panel_width)
    {
        // Toggle selection: if clicking the same organism, deselect it
        if ui_state.selected_organism_id == Some(clicked_id) {
            ui_state.selected_organism_id = None;
        } else {
            ui_state.selected_organism_id = Some(clicked_id);
        }
    }
}

fn update_and_render(
    eco: &mut simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_state: &mut ui::UIState,
) {
    handle_keyboard_shortcuts(ui_state);

    // Handle save request
    if ui_state.save_requested {
        ui_state.save_requested = false;
        handle_save_request(eco, ui_state);
    }

    // Handle load request
    if ui_state.load_requested {
        ui_state.load_requested = false;
        handle_load_request(eco, ui_state);
    }

    // Update history data
    ui_state.update_history(eco);

    // Handle organism selection
    handle_organism_selection(eco, params, ui_state);

    // Update hovered organism
    ui_state.hovered_organism_id =
        graphics::get_hovered_organism(eco, params, ui_state.stats_panel_width);

    // Draw simulation
    graphics::draw_food(eco, params, ui_state.stats_panel_width);
    graphics::draw_projectiles(eco, params, ui_state.stats_panel_width);
    graphics::draw_organisms(eco, params, ui_state.stats_panel_width);

    // Draw UI
    ui::draw_ui(ui_state, eco, params);
}

#[macroquad::main("Evolutionary Organisms")]
async fn main() {
    let params = create_simulation_params();
    let mut ui_state = ui::UIState::new();
    let mut ecosystem: Option<simulation::ecosystem::Ecosystem> = None;

    println!("Starting evolutionary organisms simulation");

    // Simulation timing
    let simulation_fps = 20.0;
    let simulation_dt = 1.0 / simulation_fps;
    let mut accumulator = 0.0;
    let mut last_time = get_time();

    loop {
        // Genesis screen
        if ecosystem.is_none() {
            draw_genesis_screen();
            if is_key_down(KeyCode::Enter) {
                ecosystem = Some(simulation::ecosystem::Ecosystem::new(&params));
                last_time = get_time();
            }
            next_frame().await;
            continue;
        }

        // Update timing
        let current_time = get_time();
        let frame_time = (current_time - last_time) as f32;
        last_time = current_time;
        accumulator += frame_time;

        let eco = ecosystem.as_mut().unwrap();

        // Fixed timestep simulation updates
        while accumulator >= simulation_dt {
            eco.step(&params, simulation_dt);
            eco.spawn(&params);
            accumulator -= simulation_dt;
        }

        // Render at display refresh rate
        clear_background(WHITE);
        update_and_render(eco, &params, &mut ui_state);
        ui::process_egui();

        next_frame().await
    }
}
