use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;

pub(super) fn draw_organism_detail_panel(
    egui_ctx: &egui::Context,
    organism: &simulation::organism::Organism,
    params: &Params,
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
            let brain_type_str = match organism.brain.brain_type() {
                simulation::brain::BrainType::MLP => "MLP",
                simulation::brain::BrainType::Transformer => "Transformer",
            };
            ui.label(format!("Brain Type: {}", brain_type_str));

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

            // Calculate total parameters using flat vector
            let total_params = organism.brain.to_flat_vector().len();
            ui.label(format!("Total Parameters: {}", total_params));
            ui.label(format!(
                "Input: {} neurons",
                organism.last_brain_inputs.len()
            ));

            match &organism.brain {
                simulation::brain::Brain::MLP { layers } => {
                    ui.label(format!("Architecture: MLP ({} layers)", layers.len()));
                    ui.separator();

                    for (i, layer) in layers.iter().enumerate() {
                        let layer_name = if i == layers.len() - 1 {
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
                }
                simulation::brain::Brain::Transformer {
                    input_embed,
                    blocks,
                    output_proj,
                } => {
                    ui.label(format!(
                        "Architecture: Transformer ({} blocks)",
                        blocks.len()
                    ));
                    ui.separator();

                    ui.label(format!(
                        "Input Embed: {} → {} neurons",
                        input_embed.weights.ncols(),
                        input_embed.weights.nrows()
                    ));

                    for (i, block) in blocks.iter().enumerate() {
                        ui.label(format!(
                            "Block {}: {} heads, {} dims",
                            i + 1,
                            block.heads.len(),
                            block.heads.first().map(|h| h.w_q.nrows()).unwrap_or(0)
                        ));
                    }

                    ui.label(format!(
                        "Output Proj: {} → {} neurons",
                        output_proj.weights.ncols(),
                        output_proj.weights.nrows()
                    ));
                }
            }

            ui.separator();

            // Neural network visualization
            ui.heading("Neural Network");
            super::nn::draw_neural_network(ui, organism, params);
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

pub(super) fn get_input_label(neuron_idx: usize, params: &Params) -> Option<String> {
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

pub(super) fn get_output_label(neuron_idx: usize, params: &Params) -> Option<String> {
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
