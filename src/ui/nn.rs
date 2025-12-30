use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;

pub(super) fn draw_neural_network(
    ui: &mut egui::Ui,
    organism: &simulation::organism::Organism,
    params: &Params,
) {
    // Compute forward pass to get all layer activations using real brain inputs
    let mut layer_activations: Vec<ndarray::Array1<f32>> = Vec::new();

    // Use the actual brain inputs from the last simulation step
    let mut current_activation = organism.last_brain_inputs.clone();

    layer_activations.push(current_activation.clone());

    // Forward pass through all layers based on brain type
    match &organism.brain {
        simulation::brain::Brain::MLP { layers } => {
            for layer in layers {
                current_activation = layer.forward(&current_activation);
                layer_activations.push(current_activation.clone());
            }
        }
        simulation::brain::Brain::Transformer {
            input_embed,
            blocks,
            output_proj,
        } => {
            // Input embedding
            current_activation = input_embed.forward(&current_activation);
            layer_activations.push(current_activation.clone());

            // Transformer blocks
            for block in blocks {
                current_activation = block.forward(&current_activation);
                layer_activations.push(current_activation.clone());
            }

            // Output projection
            current_activation = output_proj.forward(&current_activation);
            layer_activations.push(current_activation.clone());
        }
    }

    // Only draw detailed visualization for MLP (transformers are too complex)
    if let simulation::brain::Brain::Transformer { .. } = &organism.brain {
        ui.label("Transformer architecture visualization not yet implemented.");
        ui.label("Use the Brain Structure section above for details.");
        return;
    }

    let simulation::brain::Brain::MLP { layers } = &organism.brain else {
        return;
    };

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
    for (layer_idx, layer) in layers.iter().enumerate() {
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
                    super::organisms::get_input_label(neuron_idx, params)
                } else {
                    super::organisms::get_output_label(neuron_idx, params)
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
