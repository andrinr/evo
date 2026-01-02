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
            ui.add_space(8.0);
            draw_signal_bars(ui, &organism.signal);
            ui.add_space(8.0);

            ui.separator();

            // Memory visualization
            ui.heading("Memory");
            ui.add_space(8.0);
            draw_memory_bars(ui, &organism.memory);
            ui.add_space(8.0);

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
            match &organism.brain {
                simulation::brain::Brain::Transformer {
                    input_embed,
                    blocks,
                    output_proj,
                } => {
                    ui.heading("Transformer Architecture");
                    draw_transformer_visualization(
                        ui,
                        input_embed,
                        blocks,
                        output_proj,
                        &organism.last_brain_inputs,
                    );
                }
                simulation::brain::Brain::MLP { .. } => {
                    ui.heading("Neural Network");
                    super::nn::draw_neural_network(ui, organism, params);
                }
            }
        });
}

/// Inferno colormap similar to matplotlib's inferno
/// Maps a value from 0.0 to 1.0 to a color from dark purple/black to yellow/white
fn inferno_colormap(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);

    // Inferno colormap approximation (dark purple -> red -> orange -> yellow)
    let r = if t < 0.5 {
        (t * 2.0 * 100.0) as u8
    } else {
        (100.0 + (t - 0.5) * 2.0 * 155.0) as u8
    };

    let g = if t < 0.25 {
        0
    } else if t < 0.75 {
        ((t - 0.25) * 2.0 * 200.0) as u8
    } else {
        (200.0 + (t - 0.75) * 4.0 * 55.0) as u8
    };

    let b = if t < 0.33 {
        (50.0 + t * 3.0 * 100.0) as u8
    } else if t < 0.66 {
        (150.0 - (t - 0.33) * 3.0 * 100.0) as u8
    } else {
        (50.0 - (t - 0.66) * 3.0 * 50.0) as u8
    };

    egui::Color32::from_rgb(r, g, b)
}

fn draw_memory_bars(ui: &mut egui::Ui, memory: &ndarray::Array1<f32>) {
    const ITEMS_PER_ROW: usize = 8;
    const BAR_WIDTH: f32 = 40.0;
    const BAR_HEIGHT: f32 = 20.0;
    const BAR_SPACING: f32 = 8.0;

    // Split memory into chunks for multiple rows
    let memory_vec: Vec<f32> = memory.iter().copied().collect();

    for chunk in memory_vec.chunks(ITEMS_PER_ROW) {
        // Draw colored bars
        ui.horizontal(|ui| {
            for &value in chunk {
                // Map from tanh range [-1, 1] to [0, 1]
                let normalized = f32::midpoint(value.clamp(-1.0, 1.0), 1.0);

                // Use inferno colormap
                let color = inferno_colormap(normalized);

                ui.painter().rect_filled(
                    egui::Rect::from_min_size(ui.cursor().min, egui::vec2(BAR_WIDTH, BAR_HEIGHT)),
                    2.0,
                    color,
                );
                ui.add_space(BAR_WIDTH + BAR_SPACING);
            }
        });

        ui.add_space(8.0); // Space between rows
    }
}

fn draw_signal_bars(ui: &mut egui::Ui, signal: &ndarray::Array1<f32>) {
    ui.horizontal(|ui| {
        for &value in signal.iter() {
            // Map from tanh range [-1, 1] to [0, 1]
            let normalized = f32::midpoint(value.clamp(-1.0, 1.0), 1.0);
            let rect_height = 20.0;
            let rect_width = 30.0;

            // Use inferno colormap for all signal values
            let color = inferno_colormap(normalized);

            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(rect_width, rect_height)),
                2.0,
                color,
            );
            ui.add_space(38.0);
        }
    });
}

pub(super) fn get_input_label(neuron_idx: usize, params: &Params) -> Option<String> {
    // Input structure: vision rays (distance+pool_match+is_organism for each direction) + scent (signal) + memory + energy + rotation + position
    // vision: 3 * num_vision_directions
    // scent: signal_size
    // memory: memory_size
    // energy: 1
    // rotation: 2 (sin, cos)
    // position: 4 (sin_x, cos_x, sin_y, cos_y)

    let vision_inputs = 3 * params.num_vision_directions;
    let scent_start = vision_inputs;
    let scent_end = scent_start + params.signal_size;
    let memory_start = scent_end;
    let memory_end = memory_start + params.memory_size;
    let energy_idx = memory_end;
    let rotation_start = energy_idx + 1;
    let rotation_end = rotation_start + 2;
    let position_start = rotation_end;
    let position_end = position_start + 4;

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
    } else if neuron_idx < scent_end {
        let signal_idx = neuron_idx - scent_start;
        Some(format!("Scent {}", signal_idx))
    } else if neuron_idx < memory_end {
        let mem_idx = neuron_idx - memory_start;
        Some(format!("Mem {}", mem_idx))
    } else if neuron_idx == energy_idx {
        Some("Energy".to_string())
    } else if neuron_idx < rotation_end {
        let offset = neuron_idx - rotation_start;
        if offset == 0 {
            Some("Rot Sin".to_string())
        } else {
            Some("Rot Cos".to_string())
        }
    } else if neuron_idx < position_end {
        let offset = neuron_idx - position_start;
        match offset {
            0 => Some("Pos X Sin".to_string()),
            1 => Some("Pos X Cos".to_string()),
            2 => Some("Pos Y Sin".to_string()),
            3 => Some("Pos Y Cos".to_string()),
            _ => None,
        }
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

/// Visualizes transformer architecture with weight heatmaps
fn draw_transformer_visualization(
    ui: &mut egui::Ui,
    input_embed: &crate::simulation::brain::Mlp,
    blocks: &[crate::simulation::brain::TransformerBlock],
    output_proj: &crate::simulation::brain::Mlp,
    inputs: &ndarray::Array1<f32>,
) {
    const CELL_SIZE: f32 = 3.0;
    const MAX_DISPLAY_SIZE: usize = 64;

    ui.label(format!("Input: {} dims", inputs.len()));

    // Show input embedding weights as heatmap
    ui.collapsing("Input Embedding Weights", |ui| {
        let weights = &input_embed.weights;
        let (rows, cols) = (
            weights.nrows().min(MAX_DISPLAY_SIZE),
            weights.ncols().min(MAX_DISPLAY_SIZE),
        );

        ui.label(format!(
            "Shape: {} × {} (showing {}×{})",
            weights.nrows(),
            weights.ncols(),
            rows,
            cols
        ));

        let (response, painter) = ui.allocate_painter(
            egui::vec2((cols as f32) * CELL_SIZE, (rows as f32) * CELL_SIZE),
            egui::Sense::hover(),
        );

        for i in 0..rows {
            for j in 0..cols {
                let weight = weights[[i, j]];
                let normalized = f32::midpoint(weight.tanh(), 1.0);
                let color = inferno_colormap(normalized);

                let rect = egui::Rect::from_min_size(
                    response.rect.min + egui::vec2(j as f32 * CELL_SIZE, i as f32 * CELL_SIZE),
                    egui::vec2(CELL_SIZE, CELL_SIZE),
                );
                painter.rect_filled(rect, 0.0, color);
            }
        }
    });

    // Show each transformer block
    for (block_idx, block) in blocks.iter().enumerate() {
        ui.collapsing(
            format!("Block {} ({} heads)", block_idx + 1, block.heads.len()),
            |ui| {
                // Show attention head weights
                for (head_idx, head) in block.heads.iter().enumerate().take(4) {
                    ui.collapsing(format!("Head {}", head_idx + 1), |ui| {
                        ui.horizontal(|ui| {
                            // Query weights
                            ui.vertical(|ui| {
                                ui.label("Q");
                                draw_weight_grid(ui, &head.w_q, CELL_SIZE, MAX_DISPLAY_SIZE);
                            });

                            // Key weights
                            ui.vertical(|ui| {
                                ui.label("K");
                                draw_weight_grid(ui, &head.w_k, CELL_SIZE, MAX_DISPLAY_SIZE);
                            });

                            // Value weights
                            ui.vertical(|ui| {
                                ui.label("V");
                                draw_weight_grid(ui, &head.w_v, CELL_SIZE, MAX_DISPLAY_SIZE);
                            });
                        });
                    });
                }

                if block.heads.len() > 4 {
                    ui.label(format!("... and {} more heads", block.heads.len() - 4));
                }
            },
        );
    }

    // Show output projection
    ui.collapsing("Output Projection Weights", |ui| {
        let weights = &output_proj.weights;
        let (rows, cols) = (
            weights.nrows().min(MAX_DISPLAY_SIZE),
            weights.ncols().min(MAX_DISPLAY_SIZE),
        );

        ui.label(format!(
            "Shape: {} × {} (showing {}×{})",
            weights.nrows(),
            weights.ncols(),
            rows,
            cols
        ));

        let (response, painter) = ui.allocate_painter(
            egui::vec2((cols as f32) * CELL_SIZE, (rows as f32) * CELL_SIZE),
            egui::Sense::hover(),
        );

        for i in 0..rows {
            for j in 0..cols {
                let weight = weights[[i, j]];
                let normalized = f32::midpoint(weight.tanh(), 1.0);
                let color = inferno_colormap(normalized);

                let rect = egui::Rect::from_min_size(
                    response.rect.min + egui::vec2(j as f32 * CELL_SIZE, i as f32 * CELL_SIZE),
                    egui::vec2(CELL_SIZE, CELL_SIZE),
                );
                painter.rect_filled(rect, 0.0, color);
            }
        }
    });
}

/// Helper function to draw a weight matrix as a colored grid
fn draw_weight_grid(
    ui: &mut egui::Ui,
    weights: &ndarray::Array2<f32>,
    cell_size: f32,
    max_size: usize,
) {
    let (rows, cols) = (weights.nrows().min(max_size), weights.ncols().min(max_size));

    let (response, painter) = ui.allocate_painter(
        egui::vec2((cols as f32) * cell_size, (rows as f32) * cell_size),
        egui::Sense::hover(),
    );

    for i in 0..rows {
        for j in 0..cols {
            let weight = weights[[i, j]];
            let normalized = f32::midpoint(weight.tanh(), 1.0);
            let color = inferno_colormap(normalized);

            let rect = egui::Rect::from_min_size(
                response.rect.min + egui::vec2(j as f32 * cell_size, i as f32 * cell_size),
                egui::vec2(cell_size, cell_size),
            );
            painter.rect_filled(rect, 0.0, color);
        }
    }
}
