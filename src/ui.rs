use crate::simulation;
use egui_macroquad::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};
use std::collections::VecDeque;

const MAX_HISTORY_POINTS: usize = 500;

pub struct UIState {
    pub hovered_organism_id: Option<usize>,
    pub selected_organism_id: Option<usize>,
    pub stats_panel_width: f32,
    pub avg_age_history: VecDeque<(f64, f64)>,
    pub avg_score_history: VecDeque<(f64, f64)>,
    last_update_time: f32,
    update_interval: f32,
    pub save_requested: bool,
    pub load_requested: bool,
    pub status_message: Option<String>,
}

impl UIState {
    pub fn new() -> Self {
        Self {
            hovered_organism_id: None,
            selected_organism_id: None,
            stats_panel_width: 300.0,
            avg_age_history: VecDeque::new(),
            avg_score_history: VecDeque::new(),
            last_update_time: 0.0,
            update_interval: 0.5, // Update every 0.5 seconds
            save_requested: false,
            load_requested: false,
            status_message: None,
        }
    }

    pub fn set_last_update_time(&mut self, time: f32) {
        self.last_update_time = time;
    }

    pub fn update_history(&mut self, ecosystem: &simulation::ecosystem::Ecosystem) {
        if ecosystem.time - self.last_update_time >= self.update_interval {
            self.last_update_time = ecosystem.time;

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
        }
    }
}

pub fn draw_ui(
    state: &mut UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
) {
    egui_macroquad::ui(|egui_ctx| {
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
    params: &simulation::ecosystem::Params,
) {
    egui::SidePanel::right("stats_panel")
        .default_width(state.stats_panel_width)
        .resizable(true)
        .show(egui_ctx, |ui| {
            ui.heading("Simulation Stats");
            ui.separator();

            // Save/Load buttons
            ui.horizontal(|ui| {
                if ui.button("💾 Save").clicked() {
                    state.save_requested = true;
                }
                if ui.button("📂 Load").clicked() {
                    state.load_requested = true;
                }
            });

            // Show status message if any
            if let Some(ref msg) = state.status_message {
                ui.label(msg);
            }

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

                // Age distribution plot
                ui.heading("Age Distribution");
                draw_age_histogram(ui, &ecosystem.organisms);

                ui.separator();

                // Energy distribution plot
                ui.heading("Energy Distribution");
                draw_energy_histogram(ui, &ecosystem.organisms);
            }
        });
}

fn draw_organism_detail_panel(
    egui_ctx: &egui::Context,
    organism: &simulation::organism::Organism,
    _params: &simulation::ecosystem::Params,
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

            ui.separator();

            // Signal visualization
            ui.heading("Signal (Color)");
            ui.horizontal(|ui| {
                ui.label(format!("R: {:.2}", organism.signal[0]));
                ui.label(format!("G: {:.2}", organism.signal[1]));
                ui.label(format!("B: {:.2}", organism.signal[2]));
            });

            // Color preview
            let color = egui::Color32::from_rgb(
                (organism.signal[0] * 255.0) as u8,
                (organism.signal[1] * 255.0) as u8,
                (organism.signal[2] * 255.0) as u8,
            );
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(50.0, 20.0)),
                0.0,
                color,
            );
            ui.add_space(25.0);

            ui.separator();

            // Memory visualization
            ui.heading("Memory");
            draw_memory_bars(ui, &organism.memory);

            ui.separator();

            // Brain structure info
            ui.heading("Brain Structure");
            ui.label(format!("Layers: {}", organism.brain.layers.len()));
            for (i, layer) in organism.brain.layers.iter().enumerate() {
                ui.label(format!(
                    "Layer {}: {}x{}",
                    i,
                    layer.weights.nrows(),
                    layer.weights.ncols()
                ));
            }
        });
}

fn draw_age_histogram(ui: &mut egui::Ui, organisms: &[simulation::organism::Organism]) {
    // Create age bins
    let max_age = organisms.iter().map(|o| o.age).fold(0.0f32, f32::max);
    let num_bins = 10;
    let bin_size = (max_age / num_bins as f32).max(1.0);

    let mut bins = vec![0; num_bins];
    for org in organisms {
        let bin_index = ((org.age / bin_size) as usize).min(num_bins - 1);
        bins[bin_index] += 1;
    }

    let bars: Vec<Bar> = bins
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            Bar::new(i as f64 * bin_size as f64, count as f64).width(bin_size as f64 * 0.8)
        })
        .collect();

    let chart = BarChart::new(bars);

    Plot::new("age_histogram")
        .height(150.0)
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(chart);
        });
}

fn draw_energy_histogram(ui: &mut egui::Ui, organisms: &[simulation::organism::Organism]) {
    // Create energy bins (0.0 to 1.0)
    let num_bins = 10;
    let bin_size = 0.1f32;

    let mut bins = vec![0; num_bins];
    for org in organisms {
        let bin_index = ((org.energy / bin_size) as usize).min(num_bins - 1);
        bins[bin_index] += 1;
    }

    let bars: Vec<Bar> = bins
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            Bar::new(i as f64 * bin_size as f64, count as f64).width(bin_size as f64 * 0.8)
        })
        .collect();

    let chart = BarChart::new(bars);

    Plot::new("energy_histogram")
        .height(150.0)
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(chart);
        });
}

fn draw_memory_bars(ui: &mut egui::Ui, memory: &ndarray::Array1<f32>) {
    ui.horizontal(|ui| {
        for &value in memory.iter() {
            let color_value = (value * 255.0).clamp(0.0, 255.0) as u8;
            let color = egui::Color32::from_rgb(color_value, color_value, color_value);

            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(30.0, 20.0)),
                2.0,
                color,
            );
            ui.add_space(35.0);
        }
    });

    ui.horizontal(|ui| {
        for &value in memory.iter() {
            ui.label(format!("{:.2}", value));
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

pub fn process_egui() {
    egui_macroquad::draw();
}
