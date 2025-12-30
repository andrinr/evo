use crate::simulation;
use crate::simulation::params::Params;
use egui_macroquad::egui;
use std::collections::VecDeque;

const MAX_HISTORY_POINTS: usize = 500;

#[allow(clippy::struct_excessive_bools)]
pub struct UIState {
    pub hovered_organism_id: Option<usize>,
    pub selected_organism_id: Option<usize>,
    pub stats_panel_width: f32,
    pub avg_age_history: VecDeque<(f64, f64)>,
    pub organism_count_history: VecDeque<(f64, f64)>,
    pub food_count_history: VecDeque<(f64, f64)>,
    pub pool_score_histories: Vec<VecDeque<(f64, f64)>>, // One history per pool
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
            organism_count_history: VecDeque::new(),
            food_count_history: VecDeque::new(),
            pool_score_histories: Vec::new(),
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

            if !ecosystem.organisms.is_empty() {
                let avg_age: f32 = ecosystem.organisms.iter().map(|o| o.age).sum::<f32>()
                    / ecosystem.organisms.len() as f32;

                self.avg_age_history
                    .push_back((ecosystem.time as f64, avg_age as f64));

                if self.avg_age_history.len() > MAX_HISTORY_POINTS {
                    self.avg_age_history.pop_front();
                }
            }

            if self.organism_count_history.len() > MAX_HISTORY_POINTS {
                self.organism_count_history.pop_front();
            }
            if self.food_count_history.len() > MAX_HISTORY_POINTS {
                self.food_count_history.pop_front();
            }
        }
    }

    pub fn update_pool_scores(
        &mut self,
        ecosystem: &simulation::ecosystem::Ecosystem,
        params: &Params,
    ) {
        // Ensure we have enough histories for all pools
        while self.pool_score_histories.len() < params.num_genetic_pools {
            self.pool_score_histories.push(VecDeque::new());
        }

        // Calculate average score for each pool
        for pool_id in 0..params.num_genetic_pools {
            let pool_organisms: Vec<&simulation::organism::Organism> = ecosystem
                .organisms
                .iter()
                .filter(|org| org.pool_id == pool_id)
                .collect();

            let avg_score = if pool_organisms.is_empty() {
                0.0
            } else {
                pool_organisms.iter().map(|o| o.score as f64).sum::<f64>()
                    / pool_organisms.len() as f64
            };

            self.pool_score_histories[pool_id].push_back((ecosystem.time as f64, avg_score));

            if self.pool_score_histories[pool_id].len() > MAX_HISTORY_POINTS {
                self.pool_score_histories[pool_id].pop_front();
            }
        }
    }
}

pub fn draw_ui(
    state: &mut UIState,
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &mut Params,
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
        super::stats::draw_stats_panel(egui_ctx, state, ecosystem, params);

        // Detail panel - show selected organism, or hovered if nothing selected
        let display_id = state.selected_organism_id.or(state.hovered_organism_id);
        if let Some(org_id) = display_id {
            if let Some(organism) = ecosystem.organisms.iter().find(|o| o.id == org_id) {
                super::organisms::draw_organism_detail_panel(
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

pub fn process_egui() {
    egui_macroquad::draw();
}
