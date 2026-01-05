use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;

use super::ui::UIState;

/// Get a distinct color for each genetic pool matching the organism rendering colors
fn get_pool_color(pool_id: usize) -> egui::Color32 {
    match pool_id % 10 {
        0 => egui::Color32::from_rgb(255, 100, 100), // Red
        1 => egui::Color32::from_rgb(100, 150, 255), // Blue
        2 => egui::Color32::from_rgb(255, 255, 100), // Yellow
        3 => egui::Color32::from_rgb(255, 100, 255), // Magenta
        4 => egui::Color32::from_rgb(100, 255, 255), // Cyan
        _ => egui::Color32::from_rgb(200, 200, 200), // Gray (fallback)
    }
}

pub(super) fn draw_stats_panel(
    egui_ctx: &egui::Context,
    state: &mut UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &mut Params,
) {
    egui::SidePanel::right("stats_panel")
        .default_width(state.stats_panel_width)
        .resizable(true)
        .show(egui_ctx, |ui| {
            ui.heading("Simulation Stats");
            ui.separator();

            // Save/Load/Reset buttons
            ui.horizontal(|ui| {
                if ui.button("💾 Save").clicked() {
                    state.save_requested = true;
                }
                if ui.button("📂 Load").clicked() {
                    state.load_requested = true;
                }
                if ui.button("🔄 Reset").clicked() {
                    state.reset_requested = true;
                }
            });

            // Rendering toggle
            ui.horizontal(|ui| {
                let button_text = if state.rendering_enabled {
                    "🎨 Rendering: ON"
                } else {
                    "🎨 Rendering: OFF"
                };
                if ui.button(button_text).clicked() {
                    state.rendering_enabled = !state.rendering_enabled;
                }
            });

            // Show status message if any
            if let Some(ref msg) = state.status_message {
                ui.label(msg);
            }

            ui.separator();

            // Simulation speed slider
            ui.label("Simulation Speed");
            ui.add(
                egui::Slider::new(&mut state.simulation_speed, 0.1..=50.0)
                    .text("x")
                    .logarithmic(false),
            );
            ui.label(format!("Speed: {:.1}x", state.simulation_speed));
            ui.label(format!("Steps/sec: {:.1}", state.actual_steps_per_sec));
            ui.label(format!("Step time: {:.2}ms", state.last_step_time_ms));

            // Detailed timing breakdown
            ui.collapsing("Timing Breakdown", |ui| {
                let timing = &ecosystem.timing_stats;
                ui.label(format!("Spatial index: {:.2}ms", timing.spatial_index_ms));
                ui.label(format!(
                    "Ecosystem clone: {:.2}ms",
                    timing.ecosystem_clone_ms
                ));
                ui.label(format!(
                    "Parallel update: {:.2}ms",
                    timing.parallel_update_ms
                ));
                ui.label(format!("Projectiles: {:.2}ms", timing.projectile_update_ms));
                ui.label(format!("Event apply: {:.2}ms", timing.event_application_ms));
                ui.label(format!("Cleanup: {:.2}ms", timing.cleanup_ms));
                ui.label(format!("Total: {:.2}ms", timing.total_ms));
            });

            ui.separator();

            ui.label(format!("Time: {:.1}s", ecosystem.time));
            ui.label(format!("Generation: {}", ecosystem.generation));
            ui.separator();

            ui.label(format!(
                "Organisms: {}/{}",
                ecosystem.organisms.len(),
                params.n_organism
            ));
            ui.label(format!("Food: {}/{}", ecosystem.food.len(), params.n_food));

            // Show pool populations
            if params.num_genetic_pools > 1 {
                ui.separator();
                ui.label("Genetic Pool Populations:");
                for pool_id in 0..params.num_genetic_pools {
                    let pool_count = ecosystem
                        .organisms
                        .iter()
                        .filter(|org| org.pool_id == pool_id)
                        .count();
                    ui.label(format!("  Pool {}: {}", pool_id, pool_count));
                }
            }

            ui.separator();

            // Reproduction Strategy Effectiveness (based on recent deaths)
            ui.label("Reproduction Strategy Stats:");
            ui.label("(avg final score - parent score)");
            let stats = &ecosystem.reproduction_stats;

            if stats.asexual_count() > 0 {
                ui.label(format!(
                    "  Asexual (n={}): {:.2}",
                    stats.asexual_count(),
                    stats.avg_asexual_delta()
                ));
            }

            if stats.sexual_count() > 0 {
                ui.label(format!(
                    "  Sexual (n={}): {:.2}",
                    stats.sexual_count(),
                    stats.avg_sexual_delta()
                ));
            }

            if stats.interpool_count() > 0 {
                ui.label(format!(
                    "  Inter-Pool (n={}): {:.2}",
                    stats.interpool_count(),
                    stats.avg_interpool_delta()
                ));
            }

            ui.separator();

            // Runtime Parameters
            ui.collapsing("⚙ Simulation Parameters", |ui| {
                ui.label("Energy Rates");
                ui.add(
                    egui::Slider::new(&mut params.idle_energy_rate, 0.001..=0.1)
                        .text("Idle")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut params.move_energy_rate, 0.00001..=0.01)
                        .text("Move")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut params.rot_energy_rate, 0.00001..=0.01)
                        .text("Rotation")
                        .logarithmic(true),
                );

                ui.separator();
                ui.label("Movement");
                ui.add(egui::Slider::new(&mut params.move_multiplier, 10.0..=200.0).text("Speed"));

                ui.separator();
                ui.label("Spawn Rates (per second)");
                ui.add(
                    egui::Slider::new(&mut params.organism_spawn_rate, 0.1..=30.0)
                        .text("Organisms")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut params.food_spawn_rate, 0.01..=30.0)
                        .text("Food")
                        .logarithmic(true),
                );

                ui.separator();
                ui.label("Attack Parameters");
                ui.add(egui::Slider::new(&mut params.attack_cost_rate, 0.01..=1.0).text("Cost"));
                ui.add(egui::Slider::new(&mut params.attack_damage_rate, 0.1..=2.0).text("Damage"));
                ui.add(
                    egui::Slider::new(&mut params.attack_cooldown, 0.1..=5.0).text("Cooldown (s)"),
                );

                ui.separator();
                ui.label("Food");
                ui.add(
                    egui::Slider::new(&mut params.food_lifetime, 0.0..=180.0).text("Lifetime (s)"),
                );
                ui.label("(0 = unlimited)");

                ui.separator();
                ui.label("Other");
                ui.add(
                    egui::Slider::new(&mut params.corpse_energy_ratio, 0.0..=4.0)
                        .text("Corpse Energy"),
                );
            });

            ui.separator();

            // Organism statistics
            if !ecosystem.organisms.is_empty() {
                ui.heading("Organism Stats");

                let total_age: f32 = ecosystem.organisms.iter().map(|o| o.age).sum();
                let avg_age = total_age / ecosystem.organisms.len() as f32;

                let total_energy: f32 = ecosystem.organisms.iter().map(|o| o.energy).sum();
                let avg_energy = total_energy / ecosystem.organisms.len() as f32;

                let max_age = ecosystem
                    .organisms
                    .iter()
                    .map(|o| o.age)
                    .fold(0.0f32, f32::max);
                let max_score = ecosystem
                    .organisms
                    .iter()
                    .map(|o| o.score)
                    .max()
                    .unwrap_or(0);

                ui.label(format!("Avg Age: {:.2}", avg_age));
                ui.label(format!("Max Age: {:.2}", max_age));
                ui.label(format!("Avg Energy: {:.3}", avg_energy));
                ui.label(format!("Max Score: {}", max_score));

                ui.separator();

                ui.separator();
            }

            // Per-pool population plot
            ui.heading("Population Per Pool Over Time");
            draw_pool_population_plot(ui, state, params);

            ui.separator();

            // Pool score plot
            if params.num_genetic_pools > 1 {
                ui.heading("Average Score Per Pool Over Time");
                draw_pool_scores_plot(ui, state, params);
                ui.separator();
            }

            // Pool kill heatmap
            if params.num_genetic_pools > 1 {
                ui.heading("Inter-Pool Kills");
                ui.label("(Attacker → Victim)");
                draw_kill_heatmap(ui, ecosystem);
                ui.separator();
            }

            // Pool energy sharing heatmap
            if params.num_genetic_pools > 1 {
                ui.heading("Inter-Pool Energy Sharing");
                ui.label("(Giver → Receiver)");
                draw_energy_sharing_heatmap(ui, ecosystem);
                ui.separator();
            }
        });
}

#[allow(dead_code)]
fn draw_time_series_plot(
    ui: &mut egui::Ui,
    id: &str,
    data: &VecDeque<(f64, f64)>,
    x_label: &str,
    y_label: &str,
) {
    if data.is_empty() {
        ui.label("Collecting data...");
        return;
    }

    let points: PlotPoints = data.iter().map(|&(x, y)| [x, y]).collect();
    let line = Line::new(points);

    Plot::new(id)
        .height(150.0)
        .show_axes([true, true])
        .label_formatter(|_name, value| {
            format!("{}: {:.1}\n{}: {:.2}", x_label, value.x, y_label, value.y)
        })
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

fn draw_pool_scores_plot(ui: &mut egui::Ui, state: &UIState, params: &Params) {
    if state.pool_score_histories.is_empty() {
        ui.label("Collecting data...");
        return;
    }

    Plot::new("pool_scores_plot")
        .height(200.0)
        .show_axes([true, true])
        .label_formatter(|name, value| {
            format!("{}: Time: {:.1}s, Score: {:.1}", name, value.x, value.y)
        })
        .show(ui, |plot_ui| {
            for pool_id in 0..params
                .num_genetic_pools
                .min(state.pool_score_histories.len())
            {
                if !state.pool_score_histories[pool_id].is_empty() {
                    let points: PlotPoints = state.pool_score_histories[pool_id]
                        .iter()
                        .map(|&(x, y)| [x, y])
                        .collect();

                    let color = get_pool_color(pool_id);
                    let line = Line::new(points)
                        .color(color)
                        .name(format!("Pool {}", pool_id));

                    plot_ui.line(line);
                }
            }
        });
}

fn draw_pool_ages_plot(ui: &mut egui::Ui, state: &UIState, params: &Params) {
    if state.pool_age_histories.is_empty() {
        ui.label("Collecting data...");
        return;
    }

    Plot::new("pool_ages_plot")
        .height(200.0)
        .show_axes([true, true])
        .label_formatter(|name, value| {
            format!("{}: Time: {:.1}s, Age: {:.1}", name, value.x, value.y)
        })
        .show(ui, |plot_ui| {
            for pool_id in 0..params.num_genetic_pools.min(state.pool_age_histories.len()) {
                if !state.pool_age_histories[pool_id].is_empty() {
                    let points: PlotPoints = state.pool_age_histories[pool_id]
                        .iter()
                        .map(|&(x, y)| [x, y])
                        .collect();

                    let color = get_pool_color(pool_id);
                    let line = Line::new(points)
                        .color(color)
                        .name(format!("Pool {}", pool_id));

                    plot_ui.line(line);
                }
            }
        });
}

fn draw_kill_heatmap(ui: &mut egui::Ui, ecosystem: &simulation::ecosystem::Ecosystem) {
    if ecosystem.kill_matrix.is_empty() {
        ui.label("No kill data yet...");
        return;
    }

    let num_pools = ecosystem.kill_matrix.len();
    if num_pools == 0 {
        return;
    }

    // Find max value for normalization
    let max_kills = ecosystem
        .kill_matrix
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0_f32, f32::max)
        .max(0.01); // Avoid division by zero

    // Calculate cell size based on available space and number of pools
    let cell_size = if num_pools <= 4 {
        40.0
    } else if num_pools <= 8 {
        30.0
    } else {
        25.0
    };

    ui.horizontal(|ui| {
        ui.add_space(cell_size); // Space for row labels
        ui.vertical(|ui| {
            // Column labels (Victim)
            ui.horizontal(|ui| {
                ui.label("V:");
                for col in 0..num_pools {
                    ui.label(
                        egui::RichText::new(format!("{}", col))
                            .color(get_pool_color(col))
                            .size(10.0),
                    )
                    .on_hover_text(format!("Victim: Pool {}", col));
                    if col < num_pools - 1 {
                        ui.add_space(cell_size - 15.0);
                    }
                }
            });

            // Heatmap grid with row labels
            ui.horizontal(|ui| {
                // Row labels (Attacker)
                ui.vertical(|ui| {
                    ui.label("A:");
                    for row in 0..num_pools {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{}", row))
                                    .color(get_pool_color(row))
                                    .size(10.0),
                            )
                            .on_hover_text(format!("Attacker: Pool {}", row));
                        });
                        if row < num_pools - 1 {
                            ui.add_space(cell_size - 15.0);
                        }
                    }
                });

                // Grid
                ui.vertical(|ui| {
                    ui.add_space(15.0);
                    for row in 0..num_pools {
                        ui.horizontal(|ui| {
                            for col in 0..num_pools {
                                let kill_rate = ecosystem.kill_matrix[row][col];
                                let intensity = (kill_rate / max_kills).min(1.0);

                                // Color based on intensity (red gradient for all cells)
                                let base_color = egui::Color32::from_rgb(
                                    (255.0 * (0.3 + intensity * 0.7)) as u8,
                                    (50.0 * (1.0 - intensity)) as u8,
                                    (50.0 * (1.0 - intensity)) as u8,
                                );

                                // Format display: show value if > 0.01, otherwise show empty
                                let display_text = if kill_rate > 0.01 {
                                    format!("{:.1}", kill_rate * 10.0) // Scale by 10 for readability
                                } else {
                                    String::new()
                                };

                                ui.add(
                                    egui::Button::new(egui::RichText::new(display_text).size(9.0))
                                        .fill(base_color)
                                        .min_size(egui::vec2(cell_size - 2.0, cell_size - 2.0)),
                                )
                                .on_hover_text(format!(
                                    "Pool {} → Pool {}: {:.2}",
                                    row, col, kill_rate
                                ));
                            }
                        });
                    }
                });
            });
        });
    });
}

fn draw_pool_population_plot(ui: &mut egui::Ui, state: &UIState, params: &Params) {
    if state.pool_organism_count_histories.is_empty() {
        ui.label("Collecting data...");
        return;
    }

    Plot::new("pool_population_plot")
        .height(200.0)
        .show_axes([true, true])
        .legend(egui_plot::Legend::default())
        .label_formatter(|name, value| {
            format!("{}\nTime: {:.1}s\nCount: {:.0}", name, value.x, value.y)
        })
        .show(ui, |plot_ui| {
            // Draw per-pool organism counts
            for pool_id in 0..params
                .num_genetic_pools
                .min(state.pool_organism_count_histories.len())
            {
                if !state.pool_organism_count_histories[pool_id].is_empty() {
                    let points: PlotPoints = state.pool_organism_count_histories[pool_id]
                        .iter()
                        .map(|&(x, y)| [x, y])
                        .collect();

                    let color = get_pool_color(pool_id);
                    let line = Line::new(points)
                        .color(color)
                        .name(format!("Pool {}", pool_id));

                    plot_ui.line(line);
                }
            }

            // Draw food count
            if !state.food_count_history.is_empty() {
                let food_points: PlotPoints = state
                    .food_count_history
                    .iter()
                    .map(|&(x, y)| [x, y])
                    .collect();
                let food_line = Line::new(food_points)
                    .color(egui::Color32::from_rgb(100, 200, 100))
                    .name("Food");
                plot_ui.line(food_line);
            }
        });
}

fn draw_energy_sharing_heatmap(ui: &mut egui::Ui, ecosystem: &simulation::ecosystem::Ecosystem) {
    if ecosystem.energy_sharing_matrix.is_empty() {
        ui.label("No energy sharing data yet...");
        return;
    }

    let num_pools = ecosystem.energy_sharing_matrix.len();
    if num_pools == 0 {
        return;
    }

    // Find max value for normalization
    let max_sharing = ecosystem
        .energy_sharing_matrix
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0_f32, f32::max)
        .max(0.01); // Avoid division by zero

    // Calculate cell size based on available space and number of pools
    let cell_size = if num_pools <= 4 {
        40.0
    } else if num_pools <= 8 {
        30.0
    } else {
        25.0
    };

    ui.horizontal(|ui| {
        ui.add_space(cell_size); // Space for row labels
        ui.vertical(|ui| {
            // Column labels (Receiver)
            ui.horizontal(|ui| {
                ui.label("R:");
                for col in 0..num_pools {
                    ui.label(
                        egui::RichText::new(format!("{}", col))
                            .color(get_pool_color(col))
                            .size(10.0),
                    )
                    .on_hover_text(format!("Receiver: Pool {}", col));
                    if col < num_pools - 1 {
                        ui.add_space(cell_size - 15.0);
                    }
                }
            });

            // Heatmap grid with row labels
            ui.horizontal(|ui| {
                // Row labels (Giver)
                ui.vertical(|ui| {
                    ui.label("G:");
                    for row in 0..num_pools {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{}", row))
                                    .color(get_pool_color(row))
                                    .size(10.0),
                            )
                            .on_hover_text(format!("Giver: Pool {}", row));
                        });
                        if row < num_pools - 1 {
                            ui.add_space(cell_size - 15.0);
                        }
                    }
                });

                // Grid
                ui.vertical(|ui| {
                    ui.add_space(15.0);
                    for row in 0..num_pools {
                        ui.horizontal(|ui| {
                            for col in 0..num_pools {
                                let share_rate = ecosystem.energy_sharing_matrix[row][col];
                                let intensity = (share_rate / max_sharing).min(1.0);

                                // Color based on intensity (green gradient for all cells)
                                let base_color = egui::Color32::from_rgb(
                                    (100.0 * (1.0 - intensity)) as u8,
                                    (255.0 * (0.4 + intensity * 0.6)) as u8,
                                    (100.0 * (1.0 - intensity)) as u8,
                                );

                                // Format display: show value if > 0.01, otherwise show empty
                                let display_text = if share_rate > 0.01 {
                                    format!("{:.1}", share_rate * 10.0) // Scale by 10 for readability
                                } else {
                                    String::new()
                                };

                                ui.add(
                                    egui::Button::new(egui::RichText::new(display_text).size(9.0))
                                        .fill(base_color)
                                        .min_size(egui::vec2(cell_size - 2.0, cell_size - 2.0)),
                                )
                                .on_hover_text(format!(
                                    "Pool {} → Pool {}: {:.2}",
                                    row, col, share_rate
                                ));
                            }
                        });
                    }
                });
            });
        });
    });
}
