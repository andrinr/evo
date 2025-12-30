use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;
use macroquad::prelude::*;

pub fn draw_genesis_screen(params: &mut Params) -> bool {
    clear_background(LIGHTGRAY);

    let mut start_simulation = false;

    egui_macroquad::ui(|egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Evolution Simulation - Configuration");
                ui.add_space(10.0);

                ui.collapsing("Organism Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.body_radius, 1.0..=10.0).text("Body Radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.vision_radius, 10.0..=200.0)
                            .text("Vision Radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.scent_radius, 10.0..=200.0)
                            .text("Scent Radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.share_radius, 5.0..=50.0)
                            .text("Share Radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.num_vision_directions, 1..=16)
                            .text("Vision Directions"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.fov, 0.1..=std::f32::consts::PI)
                            .text("Field of View"),
                    );
                    ui.add(egui::Slider::new(&mut params.signal_size, 1..=10).text("Signal Size"));
                    ui.add(egui::Slider::new(&mut params.memory_size, 1..=20).text("Memory Size"));
                    ui.add(
                        egui::Slider::new(&mut params.num_genetic_pools, 1..=10)
                            .text("Genetic Pools"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.pool_interbreed_prob, 0.0..=1.0)
                            .text("Inter-Pool Breeding Prob"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.graveyard_size, 50..=1000)
                            .text("Graveyard Size (breeding pool)"),
                    );
                });

                ui.collapsing("Brain Architecture", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Brain Type:");
                        ui.radio_value(
                            &mut params.brain_type,
                            simulation::brain::BrainType::MLP,
                            "MLP (Fast)",
                        );
                        ui.radio_value(
                            &mut params.brain_type,
                            simulation::brain::BrainType::Transformer,
                            "Transformer (Advanced)",
                        );
                    });

                    ui.add_space(5.0);

                    // Show transformer-specific params only when transformer is selected
                    if params.brain_type == simulation::brain::BrainType::Transformer {
                        ui.label("Transformer Configuration:");
                        ui.add(
                            egui::Slider::new(&mut params.transformer_model_dim, 32..=256)
                                .text("Model Dimension"),
                        );
                        ui.add(
                            egui::Slider::new(&mut params.transformer_num_blocks, 1..=6)
                                .text("Number of Blocks"),
                        );
                        ui.add(
                            egui::Slider::new(&mut params.transformer_num_heads, 1..=16)
                                .text("Number of Heads"),
                        );
                        ui.add(
                            egui::Slider::new(&mut params.transformer_head_dim, 8..=64)
                                .text("Head Dimension"),
                        );
                        ui.add(
                            egui::Slider::new(&mut params.transformer_ff_dim, 64..=512)
                                .text("Feed-Forward Dimension"),
                        );
                    }
                });

                ui.collapsing("Energy Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.max_energy, 0.5..=10.0).text("Max Energy"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.food_energy, 0.1..=5.0).text("Food Energy"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.idle_energy_rate, 0.001..=0.5)
                            .text("Idle Energy Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.move_energy_rate, 0.0001..=0.01)
                            .text("Move Energy Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.rot_energy_rate, 0.000_001..=0.001)
                            .text("Rotation Energy Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.move_multiplier, 10.0..=200.0)
                            .text("Move Multiplier"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.corpse_energy_ratio, 0.1..=2.0)
                            .text("Corpse Energy Ratio"),
                    );
                });

                ui.collapsing("DNA & Breeding", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.dna_breeding_distance, 0.01..=1.0)
                            .text("DNA Breeding Distance"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.dna_mutation_rate, 0.001..=0.5)
                            .text("DNA Mutation Rate"),
                    );
                });

                ui.collapsing("Population Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.n_organism, 1..=200)
                            .text("Initial Organisms"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.max_organism, 10..=500).text("Max Organisms"),
                    );
                    ui.add(egui::Slider::new(&mut params.n_food, 1..=200).text("Initial Food"));
                    ui.add(egui::Slider::new(&mut params.max_food, 10..=500).text("Max Food"));
                    ui.add(
                        egui::Slider::new(&mut params.organism_spawn_rate, 0.1..=10.0)
                            .text("Organism Spawn Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.food_spawn_rate, 0.1..=10.0)
                            .text("Food Spawn Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.food_lifetime, 0.0..=100.0)
                            .text("Food Lifetime"),
                    );
                });

                ui.collapsing("World Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.box_width, 100.0..=5000.0)
                            .text("World Width"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.box_height, 100.0..=5000.0)
                            .text("World Height"),
                    );
                });

                ui.collapsing("Combat Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut params.attack_cost_rate, 0.0..=1.0)
                            .text("Attack Cost Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.attack_damage_rate, 0.0..=2.0)
                            .text("Attack Damage Rate"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.attack_cooldown, 0.1..=10.0)
                            .text("Attack Cooldown"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.projectile_speed, 10.0..=500.0)
                            .text("Projectile Speed"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.projectile_range, 10.0..=500.0)
                            .text("Projectile Range"),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.projectile_radius, 0.5..=10.0)
                            .text("Projectile Radius"),
                    );
                });

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    if ui.button("Start Simulation").clicked() {
                        start_simulation = true;
                    }
                    ui.label("Configure parameters above, then click to start");
                });
            });
        });
    });

    egui_macroquad::draw();

    start_simulation
}
