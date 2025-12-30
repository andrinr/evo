use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;

use super::ui::UIState;

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
                egui::Slider::new(&mut state.simulation_speed, 0.1..=500.0)
                    .text("x")
                    .logarithmic(false),
            );
            ui.label(format!("Speed: {:.1}x", state.simulation_speed));

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
            ui.label(format!(
                "Graveyard: {}/{}",
                ecosystem.graveyard.len(),
                params.graveyard_size
            ));

            // Show top graveyard fitness (what's being selected for breeding)
            if !ecosystem.graveyard.is_empty() {
                let top_fitness = ecosystem.graveyard[0].fitness();
                let top_age = ecosystem.graveyard[0].age;
                let top_score = ecosystem.graveyard[0].score;
                ui.label(format!(
                    "Top Graveyard: fitness={:.1} (age={:.1}, score={})",
                    top_fitness, top_age, top_score
                ));
            }

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
                    egui::Slider::new(&mut params.organism_spawn_rate, 0.1..=40.0)
                        .text("Organisms")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut params.food_spawn_rate, 0.01..=10.0)
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
                    egui::Slider::new(&mut params.corpse_energy_ratio, 0.0..=1.0)
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

                // Time series plots
                ui.heading("Average Age Over Time");
                draw_time_series_plot_compact(
                    ui,
                    "avg_age_plot",
                    &state.avg_age_history,
                    "Time (s)",
                    "Avg Age",
                );

                ui.separator();
            }

            // Combined population plot (shown even when no organisms exist)
            ui.heading("Population Over Time");
            draw_population_plot(ui, &state.organism_count_history, &state.food_count_history);

            ui.separator();

            // Pool score plot
            if params.num_genetic_pools > 1 {
                ui.heading("Average Score Per Pool Over Time");
                draw_pool_scores_plot(ui, state, params);
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

fn draw_time_series_plot_compact(
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
        .height(200.0)  // Same as pool score plot
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

    // Define colors for each pool (matching the graphics colors)
    let pool_colors = [
        egui::Color32::from_rgb(255, 100, 100), // Red
        egui::Color32::from_rgb(100, 150, 255), // Blue
        egui::Color32::from_rgb(100, 255, 100), // Green
        egui::Color32::from_rgb(255, 255, 100), // Yellow
        egui::Color32::from_rgb(255, 100, 255), // Magenta
        egui::Color32::from_rgb(100, 255, 255), // Cyan
        egui::Color32::from_rgb(255, 150, 100), // Orange
        egui::Color32::from_rgb(200, 100, 255), // Purple
        egui::Color32::from_rgb(255, 200, 150), // Peach
        egui::Color32::from_rgb(150, 255, 200), // Mint
    ];

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

                    let color = pool_colors[pool_id % pool_colors.len()];
                    let line = Line::new(points)
                        .color(color)
                        .name(format!("Pool {}", pool_id));

                    plot_ui.line(line);
                }
            }
        });
}

fn draw_population_plot(
    ui: &mut egui::Ui,
    organism_data: &VecDeque<(f64, f64)>,
    food_data: &VecDeque<(f64, f64)>,
) {
    if organism_data.is_empty() && food_data.is_empty() {
        ui.label("Collecting data...");
        return;
    }

    Plot::new("population_plot")
        .height(150.0)
        .show_axes([true, true])
        .legend(egui_plot::Legend::default())
        .label_formatter(|name, value| {
            format!("{}\nTime: {:.1}s\nCount: {:.0}", name, value.x, value.y)
        })
        .show(ui, |plot_ui| {
            if !organism_data.is_empty() {
                let org_points: PlotPoints = organism_data.iter().map(|&(x, y)| [x, y]).collect();
                let org_line = Line::new(org_points)
                    .color(egui::Color32::from_rgb(100, 150, 255))
                    .name("Organisms");
                plot_ui.line(org_line);
            }

            if !food_data.is_empty() {
                let food_points: PlotPoints = food_data.iter().map(|&(x, y)| [x, y]).collect();
                let food_line = Line::new(food_points)
                    .color(egui::Color32::from_rgb(100, 200, 100))
                    .name("Food");
                plot_ui.line(food_line);
            }
        });
}
