use crate::simulation;
use egui_macroquad::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;

const MAX_HISTORY_POINTS: usize = 500;

#[allow(clippy::struct_excessive_bools)]
pub struct UIState {
    pub hovered_organism_id: Option<usize>,
    pub selected_organism_id: Option<usize>,
    pub stats_panel_width: f32,
    pub avg_age_history: VecDeque<(f64, f64)>,
    pub avg_score_history: VecDeque<(f64, f64)>,
    pub organism_count_history: VecDeque<(f64, f64)>,
    pub food_count_history: VecDeque<(f64, f64)>,
    pub projectile_count_history: VecDeque<(f64, f64)>,
    pub step_time_history: VecDeque<(f64, f64)>,
    pub steps_per_sec_history: VecDeque<(f64, f64)>,
    last_update_time: f32,
    update_interval: f32,
    pub save_requested: bool,
    pub load_requested: bool,
    pub reset_requested: bool,
    pub status_message: Option<String>,
    pub simulation_speed: f32,
    pub rendering_enabled: bool,
    plot_time_counter: f64,
    pub last_step_time_ms: f32,
    pub actual_steps_per_sec: f32,
}

impl UIState {
    pub fn new() -> Self {
        Self {
            hovered_organism_id: None,
            selected_organism_id: None,
            stats_panel_width: 300.0,
            avg_age_history: VecDeque::new(),
            avg_score_history: VecDeque::new(),
            organism_count_history: VecDeque::new(),
            food_count_history: VecDeque::new(),
            projectile_count_history: VecDeque::new(),
            step_time_history: VecDeque::new(),
            steps_per_sec_history: VecDeque::new(),
            last_update_time: 0.0,
            update_interval: 0.5, // Update every 0.5 seconds
            save_requested: false,
            load_requested: false,
            reset_requested: false,
            status_message: None,
            simulation_speed: 1.0, // Default 1x speed
            rendering_enabled: true,
            plot_time_counter: 0.0,
            last_step_time_ms: 0.0,
            actual_steps_per_sec: 0.0,
        }
    }

    pub fn set_last_update_time(&mut self, time: f32) {
        self.last_update_time = time;
    }

    pub fn reset_plot_time(&mut self) {
        self.plot_time_counter = 0.0;
    }

    pub fn update_history(&mut self, ecosystem: &simulation::ecosystem::Ecosystem) {
        if ecosystem.time - self.last_update_time >= self.update_interval {
            self.last_update_time = ecosystem.time;

            // Track population counts
            self.organism_count_history
                .push_back((ecosystem.time as f64, ecosystem.organisms.len() as f64));
            self.food_count_history
                .push_back((ecosystem.time as f64, ecosystem.food.len() as f64));
            self.projectile_count_history
                .push_back((ecosystem.time as f64, ecosystem.projectiles.len() as f64));

            if !ecosystem.organisms.is_empty() {
                let avg_age: f32 = ecosystem.organisms.iter().map(|o| o.age).sum::<f32>()
                    / ecosystem.organisms.len() as f32;
                let avg_score: f32 = ecosystem
                    .organisms
                    .iter()
                    .map(|o| o.score as f32)
                    .sum::<f32>()
                    / ecosystem.organisms.len() as f32;

                self.avg_age_history
                    .push_back((ecosystem.time as f64, avg_age as f64));
                self.avg_score_history
                    .push_back((ecosystem.time as f64, avg_score as f64));

                if self.avg_age_history.len() > MAX_HISTORY_POINTS {
                    self.avg_age_history.pop_front();
                }
                if self.avg_score_history.len() > MAX_HISTORY_POINTS {
                    self.avg_score_history.pop_front();
                }
            }

            if self.organism_count_history.len() > MAX_HISTORY_POINTS {
                self.organism_count_history.pop_front();
            }
            if self.food_count_history.len() > MAX_HISTORY_POINTS {
                self.food_count_history.pop_front();
            }
            if self.projectile_count_history.len() > MAX_HISTORY_POINTS {
                self.projectile_count_history.pop_front();
            }

            // Track performance metrics
            self.step_time_history
                .push_back((ecosystem.time as f64, self.last_step_time_ms as f64));
            self.steps_per_sec_history
                .push_back((ecosystem.time as f64, self.actual_steps_per_sec as f64));

            if self.step_time_history.len() > MAX_HISTORY_POINTS {
                self.step_time_history.pop_front();
            }
            if self.steps_per_sec_history.len() > MAX_HISTORY_POINTS {
                self.steps_per_sec_history.pop_front();
            }
        }
    }
}

pub fn draw_ui(
    state: &mut UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &mut simulation::ecosystem::Params,
) {
    egui_macroquad::ui(|egui_ctx| {
        // Configure brighter text and UI
        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(egui::Color32::from_rgb(240, 240, 240));
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(220, 220, 220);
        visuals.widgets.inactive.fg_stroke.color = egui::Color32::from_rgb(200, 200, 200);
        visuals.widgets.hovered.fg_stroke.color = egui::Color32::WHITE;
        visuals.widgets.active.fg_stroke.color = egui::Color32::WHITE;
        egui_ctx.set_visuals(visuals);

        // Right-side stats panel
        draw_stats_panel(egui_ctx, state, ecosystem, params);

        // Detail panel - show selected organism, or hovered if nothing selected
        let display_id = state.selected_organism_id.or(state.hovered_organism_id);
        if let Some(org_id) = display_id {
            if let Some(organism) = ecosystem.organisms.iter().find(|o| o.id == org_id) {
                draw_organism_detail_panel(
                    egui_ctx,
                    organism,
                    params,
                    state.selected_organism_id.is_some(),
                );
            } else if state.selected_organism_id == Some(org_id) {
                // Selected organism died, clear selection
                state.selected_organism_id = None;
            }
        }
    });
}

fn draw_stats_panel(
    egui_ctx: &egui::Context,
    state: &mut UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &mut simulation::ecosystem::Params,
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
                    egui::Slider::new(&mut params.organism_spawn_rate, 0.1..=10.0)
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
                draw_time_series_plot(
                    ui,
                    "avg_age_plot",
                    &state.avg_age_history,
                    "Time (s)",
                    "Avg Age",
                );

                ui.separator();

                ui.heading("Average Score Over Time");
                draw_time_series_plot(
                    ui,
                    "avg_score_plot",
                    &state.avg_score_history,
                    "Time (s)",
                    "Avg Score",
                );

                ui.separator();
            }

            // Combined population plot (shown even when no organisms exist)
            ui.heading("Population Over Time");
            draw_population_plot(ui, &state.organism_count_history, &state.food_count_history);

            ui.separator();

            // Projectile count plot
            ui.heading("Projectiles Over Time");
            draw_time_series_plot(
                ui,
                "projectile_count_plot",
                &state.projectile_count_history,
                "Time (s)",
                "Projectiles",
            );

            ui.separator();

            // Performance metrics
            ui.heading("Performance Metrics");
            ui.label(format!("Step Time: {:.2} ms", state.last_step_time_ms));
            ui.label(format!("Steps/Sec: {:.1}", state.actual_steps_per_sec));

            ui.separator();

            // Step time plot
            ui.heading("Step Time Over Time");
            draw_time_series_plot(
                ui,
                "step_time_plot",
                &state.step_time_history,
                "Time (s)",
                "Step Time (ms)",
            );

            ui.separator();

            // Steps per second plot
            ui.heading("Steps/Sec Over Time");
            draw_time_series_plot(
                ui,
                "steps_per_sec_plot",
                &state.steps_per_sec_history,
                "Time (s)",
                "Steps/Sec",
            );

            if !ecosystem.organisms.is_empty() {
                ui.separator();

                // Brain scatter plot
                ui.heading("Brain Weights (Mean vs Std Dev)");
                draw_dna_scatter_plot(ui, &ecosystem.organisms);
            }
        });
}

fn draw_organism_detail_panel(
    egui_ctx: &egui::Context,
    organism: &simulation::organism::Organism,
    params: &simulation::ecosystem::Params,
    is_selected: bool,
) {
    let title = if is_selected {
        format!("Organism #{} [SELECTED]", organism.id)
    } else {
        format!("Organism #{} (hover)", organism.id)
    };

    egui::Window::new(title)
        .default_pos([20.0, 20.0])
        .resizable(true)
        .show(egui_ctx, |ui| {
            if is_selected {
                ui.label("Click elsewhere to deselect");
                ui.separator();
            }
            ui.label(format!("Age: {:.2}", organism.age));
            ui.label(format!("Energy: {:.3}", organism.energy));
            ui.label(format!("Score: {}", organism.score));
            ui.label(format!(
                "Position: ({:.1}, {:.1})",
                organism.pos[0], organism.pos[1]
            ));
            ui.label(format!("Rotation: {:.2}", organism.rot));
            ui.label(format!("Genetic Pool: {}", organism.pool_id));

            ui.separator();

            // Signal visualization
            ui.heading("Signal");
            draw_signal_bars(ui, &organism.signal);

            ui.separator();

            // Memory visualization
            ui.heading("Memory");
            draw_memory_bars(ui, &organism.memory);

            ui.separator();

            // Brain structure info
            ui.heading("Brain Structure");
            ui.label(format!("Total Layers: {}", organism.brain.layers.len()));
            ui.label(format!(
                "Input: {} neurons",
                organism.last_brain_inputs.len()
            ));

            // Calculate total parameters
            let mut total_params = 0;
            for layer in &organism.brain.layers {
                // weights + biases
                total_params += layer.weights.len() + layer.biases.len();
            }
            ui.label(format!("Total Parameters: {}", total_params));

            ui.separator();

            for (i, layer) in organism.brain.layers.iter().enumerate() {
                let layer_name = if i == organism.brain.layers.len() - 1 {
                    "Output"
                } else {
                    "Hidden"
                };
                let layer_params = layer.weights.len() + layer.biases.len();
                ui.label(format!(
                    "Layer {}: {} neurons ({}) - {} params",
                    i + 1,
                    layer.weights.nrows(),
                    layer_name,
                    layer_params
                ));
            }

            ui.separator();

            // Neural network visualization
            ui.heading("Neural Network");
            draw_neural_network(ui, organism, params);
        });
}

/// Computes simple 2D representation of brain weights for visualization.
///
/// Uses mean and standard deviation of all weights as (x, y) coordinates.
/// This is much faster than PCA and still shows genetic diversity.
fn compute_brain_scatter_2d(organisms: &[simulation::organism::Organism]) -> Vec<[f64; 2]> {
    organisms
        .iter()
        .map(|org| {
            let weights = org.brain.to_flat_vector();

            // Compute mean of all weights
            let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;

            // Compute standard deviation
            let variance: f32 = weights
                .iter()
                .map(|&w| {
                    let diff = w - mean;
                    diff * diff
                })
                .sum::<f32>()
                / weights.len() as f32;
            let std_dev = variance.sqrt();

            [mean as f64, std_dev as f64]
        })
        .collect()
}

fn draw_dna_scatter_plot(ui: &mut egui::Ui, organisms: &[simulation::organism::Organism]) {
    use egui_plot::Points;

    // Use mean/std_dev to visualize brain weights in 2D
    let points = compute_brain_scatter_2d(organisms);

    let scatter_points = Points::new(points)
        .radius(3.0)
        .color(egui::Color32::from_rgb(100, 150, 255));

    Plot::new("brain_scatter")
        .height(200.0)
        .width(200.0)
        .show_axes([true, true])
        .show_grid([true, true])
        .allow_boxed_zoom(false)
        .allow_zoom(false)
        .allow_drag(false)
        .label_formatter(|_name, value| format!("Mean: {:.3}, StdDev: {:.3}", value.x, value.y))
        .show(ui, |plot_ui| {
            plot_ui.points(scatter_points);
        });
}

fn draw_memory_bars(ui: &mut egui::Ui, memory: &ndarray::Array1<f32>) {
    const ITEMS_PER_ROW: usize = 8;
    const BAR_WIDTH: f32 = 40.0;
    const BAR_HEIGHT: f32 = 20.0;
    const BAR_SPACING: f32 = 5.0;

    // Split memory into chunks for multiple rows
    let memory_vec: Vec<f32> = memory.iter().copied().collect();

    for chunk in memory_vec.chunks(ITEMS_PER_ROW) {
        // Draw colored bars
        ui.horizontal(|ui| {
            for &value in chunk {
                // Map from tanh range [-1, 1] to [0, 1]
                let normalized = f32::midpoint(value.clamp(-1.0, 1.0), 1.0);

                // Purple to orange gradient: purple (low) -> gray (mid) -> orange (high)
                let color = if normalized < 0.5 {
                    // Purple to gray (for values -1 to 0)
                    let t = normalized * 2.0;
                    let r = (120.0 + t * 80.0) as u8;
                    let g = (80.0 + t * 70.0) as u8;
                    let b = (200.0 - t * 50.0) as u8;
                    egui::Color32::from_rgb(r, g, b)
                } else {
                    // Gray to orange (for values 0 to 1)
                    let t = (normalized - 0.5) * 2.0;
                    let r = (200.0 + t * 55.0) as u8;
                    let g = (150.0 - t * 30.0) as u8;
                    let b = (150.0 - t * 150.0) as u8;
                    egui::Color32::from_rgb(r, g, b)
                };

                ui.painter().rect_filled(
                    egui::Rect::from_min_size(ui.cursor().min, egui::vec2(BAR_WIDTH, BAR_HEIGHT)),
                    2.0,
                    color,
                );
                ui.add_space(BAR_WIDTH + BAR_SPACING);
            }
        });

        // Draw values aligned with bars
        ui.horizontal(|ui| {
            for &value in chunk {
                // Use fixed width label to ensure alignment
                ui.allocate_ui_with_layout(
                    egui::vec2(BAR_WIDTH, 15.0),
                    egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                    |ui| {
                        ui.colored_label(egui::Color32::WHITE, format!("{:.2}", value));
                    },
                );
                ui.add_space(BAR_SPACING);
            }
        });

        ui.add_space(5.0); // Space between rows
    }
}

fn draw_signal_bars(ui: &mut egui::Ui, signal: &ndarray::Array1<f32>) {
    ui.horizontal(|ui| {
        for (i, &value) in signal.iter().enumerate() {
            // Map from tanh range [-1, 1] to [0, 1]
            let normalized = f32::midpoint(value.clamp(-1.0, 1.0), 1.0);
            let rect_height = 20.0;
            let rect_width = 30.0;

            // Use actual RGB color mapping for the 3 signal channels
            // This represents the organism's visual "color"
            let color = if i == 0 {
                // Red channel: low = dark, high = bright red
                let intensity = (normalized * 255.0) as u8;
                egui::Color32::from_rgb(intensity, 50, 50)
            } else if i == 1 {
                // Green channel: low = dark, high = bright green
                let intensity = (normalized * 255.0) as u8;
                egui::Color32::from_rgb(50, intensity, 50)
            } else {
                // Blue channel: low = dark, high = bright blue
                let intensity = (normalized * 255.0) as u8;
                egui::Color32::from_rgb(50, 50, intensity)
            };

            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(rect_width, rect_height)),
                2.0,
                color,
            );
            ui.add_space(35.0);
        }
    });

    ui.horizontal(|ui| {
        for &value in signal {
            ui.colored_label(egui::Color32::WHITE, format!("{:.2}", value));
            ui.add_space(5.0);
        }
    });
}

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

fn draw_neural_network(
    ui: &mut egui::Ui,
    organism: &simulation::organism::Organism,
    params: &simulation::ecosystem::Params,
) {
    // Compute forward pass to get all layer activations using real brain inputs
    let mut layer_activations: Vec<ndarray::Array1<f32>> = Vec::new();

    // Use the actual brain inputs from the last simulation step
    let mut current_activation = organism.last_brain_inputs.clone();

    layer_activations.push(current_activation.clone());

    // Forward pass through all layers
    for layer in &organism.brain.layers {
        current_activation = layer.forward(&current_activation);
        layer_activations.push(current_activation.clone());
    }

    // Draw the network - wider to accommodate all layers
    let width = 700.0; // Increased width for more layers
    let height = 600.0;
    let (response, painter) = ui.allocate_painter(egui::vec2(width, height), egui::Sense::hover());

    let rect = response.rect;
    let layer_count = layer_activations.len();

    if layer_count == 0 {
        ui.label("No layers to display");
        return;
    }

    // Debug: show layer count
    ui.label(format!("Displaying {} layers", layer_count));

    // Calculate spacing
    let layer_spacing = rect.width() / (layer_count as f32 + 1.0);

    // Draw connections first (so they appear behind neurons)
    for (layer_idx, layer) in organism.brain.layers.iter().enumerate() {
        let input_activations = &layer_activations[layer_idx];
        let output_activations = &layer_activations[layer_idx + 1];

        let x1 = rect.left() + layer_spacing * (layer_idx + 1) as f32;
        let x2 = rect.left() + layer_spacing * (layer_idx + 2) as f32;

        let input_count = input_activations.len();
        let output_count = output_activations.len();

        // Draw all connections (but thin them out if there are too many)
        let max_connections = 200;
        let draw_all = input_count * output_count <= max_connections;

        for out_idx in 0..output_count {
            let y2 =
                rect.top() + (rect.height() * (out_idx as f32 + 1.0) / (output_count as f32 + 1.0));

            for in_idx in 0..input_count {
                if !draw_all && (in_idx + out_idx) % 5 != 0 {
                    continue; // Skip some connections for clarity when there are many
                }

                let y1 = rect.top()
                    + (rect.height() * (in_idx as f32 + 1.0) / (input_count as f32 + 1.0));

                let weight = layer.weights[[out_idx, in_idx]];
                let input_activation = input_activations[in_idx];

                // Calculate the signal flowing through this connection
                // Signal = input_activation * weight
                let signal = input_activation * weight;
                let signal_strength = signal.abs().min(1.0);

                // Brighter base opacity and intensity
                let base_alpha = (weight.abs().min(1.0) * 120.0) as u8;
                let flow_intensity = (signal_strength * 255.0) as u8;

                // Color based on signal direction and strength - much brighter
                let color = if signal > 0.0 {
                    // Positive signal flow (excitatory) - bright green
                    let alpha = base_alpha.max((signal_strength * 200.0) as u8);
                    egui::Color32::from_rgba_unmultiplied(0, flow_intensity, 50, alpha)
                } else if signal < 0.0 {
                    // Negative signal flow (inhibitory) - bright red
                    let alpha = base_alpha.max((signal_strength * 200.0) as u8);
                    egui::Color32::from_rgba_unmultiplied(flow_intensity, 0, 50, alpha)
                } else {
                    // No signal - lighter gray
                    egui::Color32::from_rgba_unmultiplied(150, 150, 150, base_alpha)
                };

                // Line thickness based on signal strength
                let line_width = 0.8 + (signal_strength * 2.5);

                painter.line_segment(
                    [egui::pos2(x1, y1), egui::pos2(x2, y2)],
                    egui::Stroke::new(line_width, color),
                );
            }
        }
    }

    // Draw neurons
    for (layer_idx, activations) in layer_activations.iter().enumerate() {
        let x = rect.left() + layer_spacing * (layer_idx + 1) as f32;
        let neuron_count = activations.len(); // Show all neurons

        let is_input_layer = layer_idx == 0;
        let is_output_layer = layer_idx == layer_activations.len() - 1;

        for (neuron_idx, &activation) in activations.iter().enumerate() {
            let y = rect.top()
                + (rect.height() * (neuron_idx as f32 + 1.0) / (neuron_count as f32 + 1.0));

            // Color based on activation value (tanh output is -1 to 1)
            // Use color-coded neurons: blue for negative, yellow/orange for positive
            let normalized = f32::midpoint(activation, 1.0).clamp(0.0, 1.0);

            let color = if activation > 0.1 {
                // Positive activation: yellow to orange
                let intensity = (normalized * 255.0) as u8;
                egui::Color32::from_rgb(255, intensity, 0)
            } else if activation < -0.1 {
                // Negative activation: cyan to blue
                let intensity = ((1.0 - normalized) * 255.0) as u8;
                egui::Color32::from_rgb(0, intensity, 255)
            } else {
                // Near zero: gray
                egui::Color32::from_rgb(150, 150, 150)
            };

            painter.circle_filled(egui::pos2(x, y), 5.0, color);
            painter.circle_stroke(
                egui::pos2(x, y),
                5.0,
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            );

            // Add labels for input and output neurons
            if is_input_layer || is_output_layer {
                let label = if is_input_layer {
                    get_input_label(neuron_idx, params)
                } else {
                    get_output_label(neuron_idx, params)
                };

                if let Some(label_text) = label {
                    painter.text(
                        egui::pos2(if is_input_layer { x - 30.0 } else { x + 30.0 }, y),
                        if is_input_layer {
                            egui::Align2::RIGHT_CENTER
                        } else {
                            egui::Align2::LEFT_CENTER
                        },
                        label_text,
                        egui::FontId::proportional(9.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        }

        // Draw layer label
        let layer_name = if layer_idx == 0 {
            "Input"
        } else if layer_idx == layer_activations.len() - 1 {
            "Output"
        } else {
            "Hidden"
        };

        painter.text(
            egui::pos2(x, rect.bottom() + 5.0),
            egui::Align2::CENTER_TOP,
            layer_name,
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
    }
}

fn get_input_label(neuron_idx: usize, params: &simulation::ecosystem::Params) -> Option<String> {
    // Input structure: vision rays (distance+pool_match+is_organism for each direction) + scent (signal+dna_dist) + memory + energy
    // vision: 3 * num_vision_directions
    // scent: signal_size + 1
    // memory: memory_size
    // energy: 1

    let vision_inputs = 3 * params.num_vision_directions;
    let scent_start = vision_inputs;
    let scent_end = scent_start + params.signal_size + 1;
    let memory_start = scent_end;
    let memory_end = memory_start + params.memory_size;

    if neuron_idx < vision_inputs {
        let direction = neuron_idx / 3;
        let offset = neuron_idx % 3;
        if offset == 0 {
            Some(format!("V{} D", direction)) // Distance
        } else if offset == 1 {
            Some(format!("V{} P", direction)) // Pool match
        } else {
            Some(format!("V{} T", direction)) // Type (organism vs food)
        }
    } else if neuron_idx < scent_end - 1 {
        let signal_idx = neuron_idx - scent_start;
        Some(format!("Scent {}", signal_idx))
    } else if neuron_idx == scent_end - 1 {
        Some("DNA Dist".to_string())
    } else if neuron_idx < memory_end {
        let mem_idx = neuron_idx - memory_start;
        Some(format!("Mem {}", mem_idx))
    } else if neuron_idx == memory_end {
        Some("Energy".to_string())
    } else {
        None
    }
}

fn get_output_label(neuron_idx: usize, params: &simulation::ecosystem::Params) -> Option<String> {
    // Output structure: signal + memory + rotation + acceleration + attack + share
    // signal: signal_size
    // memory: memory_size
    // rotation: 1
    // acceleration: 1
    // attack: 1
    // share: 1

    let signal_end = params.signal_size;
    let memory_end = signal_end + params.memory_size;
    let rotation_idx = memory_end;
    let accel_idx = rotation_idx + 1;
    let attack_idx = accel_idx + 1;
    let share_idx = attack_idx + 1;

    if neuron_idx < signal_end {
        Some(format!("Signal {}", neuron_idx))
    } else if neuron_idx < memory_end {
        let mem_idx = neuron_idx - signal_end;
        Some(format!("Mem {}", mem_idx))
    } else if neuron_idx == rotation_idx {
        Some("Rotation".to_string())
    } else if neuron_idx == accel_idx {
        Some("Accel".to_string())
    } else if neuron_idx == attack_idx {
        Some("Attack".to_string())
    } else if neuron_idx == share_idx {
        Some("Share".to_string())
    } else {
        None
    }
}

pub fn process_egui() {
    egui_macroquad::draw();
}
