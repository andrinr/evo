use crate::simulation;
use egui_macroquad::egui;
use egui_plot::{Bar, BarChart, Plot};

pub struct UIState {
    pub hovered_organism_id: Option<usize>,
    pub stats_panel_width: f32,
}

impl UIState {
    pub fn new() -> Self {
        Self {
            hovered_organism_id: None,
            stats_panel_width: 300.0,
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

        // Hover detail panel (if organism is hovered)
        if let Some(org_id) = state.hovered_organism_id
            && let Some(organism) = ecosystem.organisms.iter().find(|o| o.id == org_id)
        {
            draw_organism_detail_panel(egui_ctx, organism, params);
        }
    });
}

fn draw_stats_panel(
    egui_ctx: &egui::Context,
    state: &UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
) {
    egui::SidePanel::right("stats_panel")
        .default_width(state.stats_panel_width)
        .resizable(true)
        .show(egui_ctx, |ui| {
            ui.heading("Simulation Stats");
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
) {
    egui::Window::new(format!("Organism #{}", organism.id))
        .default_pos([20.0, 20.0])
        .resizable(true)
        .show(egui_ctx, |ui| {
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

pub fn process_egui() {
    egui_macroquad::draw();
}
