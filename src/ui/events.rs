use crate::simulation;
use egui_macroquad::egui;

/// Draws a transparent panel showing recent events
pub fn draw_events_panel(egui_ctx: &egui::Context, ecosystem: &simulation::ecosystem::Ecosystem) {
    // Get screen height to position at bottom
    let screen_height = egui_ctx.screen_rect().height();
    let panel_height = 400.0;

    egui::Window::new("Recent Events")
        .fixed_pos(egui::pos2(10.0, screen_height - panel_height - 10.0))
        .fixed_size(egui::vec2(350.0, panel_height))
        .frame(
            egui::Frame::window(&egui_ctx.style())
                .fill(egui::Color32::from_rgba_premultiplied(20, 20, 30, 200))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(100, 100, 120),
                )),
        )
        .show(egui_ctx, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 4.0;

                let events = ecosystem.event_log.events();

                if events.is_empty() {
                    ui.label(
                        egui::RichText::new("No events yet...")
                            .color(egui::Color32::from_rgb(150, 150, 150))
                            .size(12.0),
                    );
                } else {
                    for event in events {
                        // Choose color based on event type
                        let color = match event.color {
                            simulation::event_log::EventColor::Reproduction => {
                                egui::Color32::from_rgb(100, 255, 100) // Green
                            }
                            simulation::event_log::EventColor::Combat => {
                                egui::Color32::from_rgb(255, 100, 100) // Red
                            }
                            simulation::event_log::EventColor::Sharing => {
                                egui::Color32::from_rgb(100, 200, 255) // Blue
                            }
                            simulation::event_log::EventColor::Death => {
                                egui::Color32::from_rgb(150, 150, 150) // Gray
                            }
                            simulation::event_log::EventColor::Food => {
                                egui::Color32::from_rgb(255, 200, 100) // Yellow
                            }
                        };

                        // Display time and event description
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("[{:.1}s]", event.time))
                                    .color(egui::Color32::from_rgb(180, 180, 200))
                                    .size(11.0)
                                    .monospace(),
                            );
                            ui.label(
                                egui::RichText::new(&event.description)
                                    .color(color)
                                    .size(11.0),
                            );
                        });
                    }
                }
            });
        });
}
