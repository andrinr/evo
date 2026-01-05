//! Evolutionary simulation with neural networks and projectile combat.
//!
//! This is the main binary that runs the simulation with a graphical interface.
//! Organisms evolve behaviors through genetic algorithms while competing for food
//! and attacking each other with projectiles.

use evo::simulation;
use evo::simulation::params::Params;
use macroquad::prelude::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

mod graphics;
mod ui;

/// Calculates the neural network layer sizes based on simulation parameters.
fn calculate_layer_sizes(
    num_vision_directions: usize,
    signal_size: usize,
    memory_size: usize,
) -> Vec<usize> {
    vec![
        3 * num_vision_directions + signal_size + memory_size + 9, // input: vision(dist+pool+type) + scent + memory + energy + position(sin_x,cos_x,sin_y,cos_y) + rotation_diff(sin,cos) + time(sin,cos)
        128,
        64,
        signal_size + memory_size + 5, // output: signal + memory + movement_rotation + vision_rotation + velocity + attack + share
    ]
}

fn create_simulation_params() -> Params {
    let signal_size: usize = 8;
    let num_vision_directions: usize = 21;
    let memory_size: usize = 32;

    let layer_sizes = calculate_layer_sizes(num_vision_directions, signal_size, memory_size);

    let vision_radius = 70.0;
    let scent_radius = 60.0;
    let share_radius = 10.0;
    let reproduction_radius = 10.0;
    let dna_breeding_distance = 0.2; // Max DNA distance for breeding (hard cutoff)
    let dna_mutation_rate = 0.1; // Standard deviation of DNA mutation

    Params {
        body_radius: 5.0,
        vision_radius,
        scent_radius,
        share_radius,
        reproduction_radius,
        dna_breeding_distance,
        dna_mutation_rate,
        idle_energy_rate: 0.15,
        move_energy_rate: 0.000001,
        move_multiplier: 140.0,
        rot_multiplier: 15.0,
        rot_energy_rate: 0.00001,
        num_vision_directions,
        fov: std::f32::consts::PI * 0.75,
        signal_size,
        memory_size,
        n_organism: 90,
        max_organism: 250,
        n_food: 150,
        max_food: 300,
        box_width: 1200.0,
        box_height: 1000.0,
        layer_sizes,
        attack_cost_rate: 0.3,
        attack_damage_rate: 4.0,
        attack_cooldown: 0.3,
        corpse_energy_ratio: 2.0,
        max_energy: 3.0,
        food_energy: 1.0,
        projectile_speed: vision_radius * 3.0,
        projectile_range: vision_radius,
        projectile_radius: 3.0,
        organism_spawn_rate: 17.0,
        food_spawn_rate: 10.0,
        food_lifetime: 30.0, // 0 = unlimited
        num_genetic_pools: 3,
        pool_interbreed_prob: 0.01, // 5% chance of inter-pool breeding
        brain_type: simulation::brain::BrainType::Transformer,
        transformer_model_dim: 32,
        transformer_num_blocks: 1,
        transformer_num_heads: 4,
        transformer_head_dim: 32,
        transformer_ff_dim: 64,
        graveyard_size: 60,
        reproduction_energy_multiplier: 0.9,
        unbalanced_pool_sampling: true,
        empty_pool_seed_count: 5,
        velocity_damping: 0.1,
        elite_pool_size: 20,
        elite_spawn_probability: 0.2,
        num_spawn_clusters: 6,
        cluster_radius: 180.0,
    }
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

fn handle_load_request(
    eco_option: &mut Option<simulation::ecosystem::Ecosystem>,
    ui_state: &mut ui::UIState,
) {
    let Some(load_path) = find_latest_save_file() else {
        ui_state.status_message = Some("✗ No save files found".to_string());
        return;
    };

    match simulation::ecosystem::Ecosystem::load_from_file(load_path.to_str().unwrap()) {
        Ok(loaded_eco) => {
            let time = loaded_eco.time;
            *eco_option = Some(loaded_eco);
            ui_state.status_message = Some(format!("✓ Loaded from {}", load_path.display()));
            println!("Loaded evolution state from {}", load_path.display());
            // Clear history as it's from a different timeline
            ui_state.organism_count_history.clear();
            ui_state.food_count_history.clear();
            ui_state.set_last_update_time(time);
            ui_state.reset_plot_time();
        }
        Err(e) => {
            ui_state.status_message = Some(format!("✗ Load failed: {}", e));
            eprintln!("Failed to load: {}", e);
        }
    }
}

fn handle_organism_selection(
    eco: &simulation::ecosystem::Ecosystem,
    params: &Params,
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
    eco: &simulation::ecosystem::Ecosystem,
    params: &mut Params,
    ui_state: &mut ui::UIState,
) {
    handle_keyboard_shortcuts(ui_state);

    // Update history data
    ui_state.update_history(eco);
    ui_state.update_pool_scores(eco, params);
    ui_state.update_pool_ages(eco, params);
    ui_state.update_pool_energy_consumption(eco, params);

    // Handle organism selection
    handle_organism_selection(eco, params, ui_state);

    // Auto-select best organism if none selected or if selected one died
    if ui_state.selected_organism_id.is_none()
        || !eco
            .organisms
            .iter()
            .any(|o| Some(o.id) == ui_state.selected_organism_id)
    {
        // Find organism with highest score
        if let Some(best_org) = eco.organisms.iter().max_by_key(|o| o.score) {
            ui_state.selected_organism_id = Some(best_org.id);
        }
    }

    // Update hovered organism (only if rendering enabled)
    if ui_state.rendering_enabled {
        ui_state.hovered_organism_id =
            graphics::get_hovered_organism(eco, params, ui_state.stats_panel_width);

        // Draw simulation
        graphics::draw_food(eco, params, ui_state.stats_panel_width);
        graphics::draw_projectiles(eco, params, ui_state.stats_panel_width);
        graphics::draw_interactions(
            eco,
            params,
            ui_state.stats_panel_width,
            &eco.energy_shares,
            &eco.reproduction_intents,
        );
        graphics::draw_organisms(
            eco,
            params,
            ui_state.stats_panel_width,
            ui_state.selected_organism_id,
        );
    }

    // Draw UI (always show UI even when rendering is disabled)
    ui::draw_ui(ui_state, eco, params);
}

fn window_conf() -> macroquad::window::Conf {
    macroquad::window::Conf {
        window_title: "Evolutionary Organisms".to_owned(),
        window_width: 1400,
        window_height: 900,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let params = Arc::new(Mutex::new(create_simulation_params()));
    let mut ui_state = ui::UIState::new();

    println!("Starting evolutionary organisms simulation");

    // Shared ecosystem state wrapped in Arc<RwLock> for concurrent reads
    let ecosystem: Arc<RwLock<Option<simulation::ecosystem::Ecosystem>>> =
        Arc::new(RwLock::new(None));
    let ecosystem_clone = ecosystem.clone();

    // Shared simulation speed
    let simulation_speed: Arc<Mutex<f32>> = Arc::new(Mutex::new(1.0));
    let speed_clone = simulation_speed.clone();

    // Shared performance metrics
    let perf_metrics = Arc::new(Mutex::new((0.0f32, 0.0f32))); // (step_time_ms, steps_per_sec)
    let perf_metrics_clone = perf_metrics.clone();

    // Simulation thread
    let params_clone = params.clone();
    thread::spawn(move || {
        let simulation_fps = 13.0; // Higher base FPS for smoother high-speed simulation
        let simulation_dt = 1.0 / simulation_fps;
        let base_frame_time = Duration::from_secs_f32(simulation_dt);

        loop {
            let loop_start = Instant::now();

            // Get current simulation speed and run appropriate number of steps
            let speed = *speed_clone.lock().unwrap();
            let steps_to_run = speed.max(0.1).round() as usize;

            let step_start = Instant::now();

            // OPTIMIZATION: Batch steps, but release lock periodically for UI responsiveness
            // At high speeds, release lock every ~5 steps to let UI read
            let batch_size = if steps_to_run > 5 { 5 } else { steps_to_run };
            let mut remaining_steps = steps_to_run;

            while remaining_steps > 0 {
                let current_batch = remaining_steps.min(batch_size);

                {
                    let mut eco_lock = ecosystem_clone.write().unwrap();
                    let params_lock = params_clone.lock().unwrap();

                    if let Some(ref mut eco) = *eco_lock {
                        for _ in 0..current_batch {
                            eco.step(&params_lock, simulation_dt);
                            eco.spawn(&params_lock, simulation_dt);
                        }
                    }
                    // Locks dropped at end of scope, giving UI a chance to read
                }

                remaining_steps -= current_batch;

                // Give UI thread a chance to acquire read lock between batches
                if remaining_steps > 0 && steps_to_run > 5 {
                    thread::sleep(Duration::from_micros(100));
                }
            }

            let step_duration = step_start.elapsed();

            // Calculate performance metrics
            let step_time_ms = step_duration.as_secs_f32() * 1000.0 / steps_to_run as f32;
            let total_elapsed = loop_start.elapsed().as_secs_f32();
            let steps_per_sec = if total_elapsed > 0.0 {
                steps_to_run as f32 / total_elapsed
            } else {
                0.0
            };

            // Update shared metrics
            {
                let mut metrics = perf_metrics_clone.lock().unwrap();
                *metrics = (step_time_ms, steps_per_sec);
            }

            // At high speeds, don't sleep - just run as fast as possible
            // At lower speeds, maintain consistent frame timing
            // if steps_to_run < 10 {
            let elapsed = loop_start.elapsed();
            if elapsed < base_frame_time {
                thread::sleep(base_frame_time.checked_sub(elapsed).unwrap());
            }
            // }
        }
    });

    // UI frame timing
    let ui_fps = 20.0;
    let ui_frame_time = Duration::from_secs_f32(1.0 / ui_fps);

    loop {
        let frame_start = Instant::now();

        // Update simulation speed from UI
        {
            let mut speed_lock = simulation_speed.lock().unwrap();
            *speed_lock = ui_state.simulation_speed;
        }

        // Update performance metrics from simulation thread
        {
            let metrics = perf_metrics.lock().unwrap();
            ui_state.last_step_time_ms = metrics.0;
            ui_state.actual_steps_per_sec = metrics.1;
        }

        // Genesis screen
        let is_genesis = {
            let eco_lock = ecosystem.read().unwrap();
            eco_lock.is_none()
        };

        if is_genesis {
            let should_start = {
                let mut params_lock = params.lock().unwrap();
                let should_start = ui::draw_genesis_screen(&mut params_lock);
                if should_start {
                    // Recalculate layer sizes based on current parameters
                    params_lock.layer_sizes = calculate_layer_sizes(
                        params_lock.num_vision_directions,
                        params_lock.signal_size,
                        params_lock.memory_size,
                    );
                }
                should_start
            }; // params_lock dropped here

            if should_start {
                let mut eco_lock = ecosystem.write().unwrap();
                let params_lock = params.lock().unwrap();
                *eco_lock = Some(simulation::ecosystem::Ecosystem::new(&params_lock));
            }

            next_frame().await;
            continue;
        }

        // Check if reset was requested
        if ui_state.reset_requested {
            {
                let mut eco_lock = ecosystem.write().unwrap();
                *eco_lock = None;
                ui_state.reset_plot_time();
            } // eco_lock dropped here
            next_frame().await;
            continue;
        }

        // Handle save/load requests (need write lock)
        if ui_state.save_requested {
            ui_state.save_requested = false;
            let eco_lock = ecosystem.read().unwrap();
            if let Some(ref eco) = *eco_lock {
                handle_save_request(eco, &mut ui_state);
            }
        }

        if ui_state.load_requested {
            ui_state.load_requested = false;
            let mut eco_lock = ecosystem.write().unwrap();
            handle_load_request(&mut eco_lock, &mut ui_state);
        }

        // Render at display refresh rate (using read lock for concurrent access)
        // Use try_read to avoid blocking UI thread - skip frame if simulation is writing
        clear_background(WHITE);
        {
            if let Ok(eco_lock) = ecosystem.try_read()
                && let Some(ref eco) = *eco_lock
            {
                let mut params_lock = params.lock().unwrap();
                update_and_render(eco, &mut params_lock, &mut ui_state);
            }
            // If we can't get lock or ecosystem is None, just skip rendering ecosystem
            // UI will still be drawn below
        } // Locks drop here
        ui::process_egui();

        // Cap UI frame rate to 20 FPS
        let elapsed = frame_start.elapsed();
        if elapsed < ui_frame_time {
            let sleep_time = ui_frame_time.checked_sub(elapsed).unwrap();
            std::thread::sleep(sleep_time);
        }

        next_frame().await
    }
}
