use crate::simulation;
use crate::simulation::params::Params;
use macroquad::prelude::*;
use ndarray::Array1;

/// Get a distinct color for each genetic pool
fn get_pool_color(pool_id: usize) -> Color {
    match pool_id % 10 {
        0 => Color::from_rgba(255, 100, 100, 255), // Red
        1 => Color::from_rgba(100, 150, 255, 255), // Blue
        2 => Color::from_rgba(255, 255, 100, 255), // Yellow
        3 => Color::from_rgba(255, 100, 255, 255), // Magenta
        4 => Color::from_rgba(100, 255, 255, 255), // Cyan
        _ => Color::from_rgba(200, 200, 200, 255), // Gray (fallback)
    }
}

fn get_organism_at_mouse(
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
) -> Option<usize> {
    let (mouse_x, mouse_y) = mouse_position();

    // Don't detect if mouse is over UI panel
    if mouse_x > screen_width() - ui_panel_width {
        return None;
    }

    // Convert mouse position to simulation coordinates accounting for camera zoom and offset
    let screen_w = screen_width() - ui_panel_width;
    let screen_h = screen_height();

    // Reverse the camera transformation to get simulation coordinates
    let sim_x = (mouse_x / screen_w) * params.box_width / camera_zoom + camera_offset_x;
    let sim_y = (mouse_y / screen_h) * params.box_height / camera_zoom + camera_offset_y;

    // Find the closest organism within a larger click radius for easier selection
    let click_radius = params.body_radius * 3.0; // 3x larger for easier clicking
    let mouse_pos = Array1::from_vec(vec![sim_x, sim_y]);

    for organism in &ecosystem.organisms {
        let distance = (&organism.pos - &mouse_pos)
            .mapv(|x| x.powi(2))
            .sum()
            .sqrt();
        if distance < click_radius {
            return Some(organism.id);
        }
    }

    None
}

pub fn get_hovered_organism(
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
) -> Option<usize> {
    get_organism_at_mouse(
        ecosystem,
        params,
        ui_panel_width,
        camera_zoom,
        camera_offset_x,
        camera_offset_y,
    )
}

pub fn handle_organism_click(
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
) -> Option<usize> {
    if is_mouse_button_pressed(MouseButton::Left) {
        get_organism_at_mouse(
            ecosystem,
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        )
    } else {
        None
    }
}

/// Handles mouse wheel zoom centered on cursor position
pub fn handle_camera_zoom(
    camera_zoom: &mut f32,
    camera_offset_x: &mut f32,
    camera_offset_y: &mut f32,
    params: &Params,
    ui_panel_width: f32,
) {
    let (mouse_x, mouse_y) = mouse_position();

    // Don't zoom if mouse is over UI panel
    if mouse_x > screen_width() - ui_panel_width {
        return;
    }

    let (_scroll_x, scroll_y) = mouse_wheel();

    if scroll_y.abs() < 0.01 {
        return; // No scroll detected
    }

    // Get mouse position in simulation coordinates BEFORE zoom
    let screen_w = screen_width() - ui_panel_width;
    let screen_h = screen_height();

    // Convert screen mouse position to simulation coordinates
    let mouse_sim_x = (mouse_x / screen_w) * params.box_width / *camera_zoom + *camera_offset_x;
    let mouse_sim_y = (mouse_y / screen_h) * params.box_height / *camera_zoom + *camera_offset_y;

    // Update zoom level
    let zoom_speed = 0.1;
    if scroll_y > 0.0 {
        // Zoom in
        *camera_zoom = (*camera_zoom * (1.0 + zoom_speed)).min(10.0);
    } else {
        // Zoom out - don't allow zooming out beyond the initial view (1.0)
        *camera_zoom = (*camera_zoom * (1.0 - zoom_speed)).max(1.0);
    }

    // Adjust camera offset to keep mouse position fixed in sim coordinates
    let new_mouse_sim_x = (mouse_x / screen_w) * params.box_width / *camera_zoom + *camera_offset_x;
    let new_mouse_sim_y =
        (mouse_y / screen_h) * params.box_height / *camera_zoom + *camera_offset_y;

    *camera_offset_x += mouse_sim_x - new_mouse_sim_x;
    *camera_offset_y += mouse_sim_y - new_mouse_sim_y;

    // Clamp camera offset to keep view within bounds
    let max_offset_x = params.box_width * (1.0 - 1.0 / *camera_zoom).max(0.0);
    let max_offset_y = params.box_height * (1.0 - 1.0 / *camera_zoom).max(0.0);

    *camera_offset_x = camera_offset_x.clamp(0.0, max_offset_x);
    *camera_offset_y = camera_offset_y.clamp(0.0, max_offset_y);
}

trait ToScreen {
    type Output;
    fn to_screen(
        &self,
        params: &Params,
        ui_panel_width: f32,
        camera_zoom: f32,
        camera_offset_x: f32,
        camera_offset_y: f32,
    ) -> Self::Output;
}

impl ToScreen for Array1<f32> {
    type Output = Array1<f32>;
    fn to_screen(
        &self,
        params: &Params,
        ui_panel_width: f32,
        camera_zoom: f32,
        camera_offset_x: f32,
        camera_offset_y: f32,
    ) -> Array1<f32> {
        let screen_w = screen_width() - ui_panel_width;
        let screen_h = screen_height();

        // Apply camera offset and zoom
        let view_x = (self[0] - camera_offset_x) * camera_zoom;
        let view_y = (self[1] - camera_offset_y) * camera_zoom;

        // Convert to screen coordinates
        Array1::from_vec(vec![
            view_x * screen_w / params.box_width,
            view_y * screen_h / params.box_height,
        ])
    }
}

impl ToScreen for f32 {
    type Output = f32;
    fn to_screen(
        &self,
        params: &Params,
        ui_panel_width: f32,
        camera_zoom: f32,
        _camera_offset_x: f32,
        _camera_offset_y: f32,
    ) -> f32 {
        let screen_w = screen_width() - ui_panel_width;
        let screen_h = screen_height();
        let scale_x = screen_w / params.box_width;
        let scale_y = screen_h / params.box_height;
        let scale = scale_x.min(scale_y);
        self * scale * camera_zoom
    }
}

/// Trait for converting relative vectors (like vision directions) to screen space
/// These should only be scaled by zoom, not offset
trait ToScreenRelative {
    type Output;
    fn to_screen_relative(
        &self,
        params: &Params,
        ui_panel_width: f32,
        camera_zoom: f32,
    ) -> Self::Output;
}

impl ToScreenRelative for Array1<f32> {
    type Output = Array1<f32>;
    fn to_screen_relative(
        &self,
        params: &Params,
        ui_panel_width: f32,
        camera_zoom: f32,
    ) -> Array1<f32> {
        let screen_w = screen_width() - ui_panel_width;
        let screen_h = screen_height();

        // Only scale by zoom, don't apply camera offset (this is a relative vector)
        Array1::from_vec(vec![
            self[0] * screen_w / params.box_width * camera_zoom,
            self[1] * screen_h / params.box_height * camera_zoom,
        ])
    }
}

/// Draws lines between organisms that are interacting (energy sharing or reproduction).
pub fn draw_interactions(
    state: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
    energy_shares: &[(usize, usize, f32)],
    reproduction_intents: &[(usize, usize, f32)],
) {
    // Draw energy sharing lines (green)
    for (giver_id, receiver_id, _timestamp) in energy_shares {
        if let (Some(giver), Some(receiver)) = (
            state.organisms.iter().find(|o| o.id == *giver_id),
            state.organisms.iter().find(|o| o.id == *receiver_id),
        ) {
            let start = giver.pos.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            let end = receiver.pos.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            draw_line(
                start[0],
                start[1],
                end[0],
                end[1],
                2.0,
                Color::from_rgba(0, 155, 155, 255),
            );
        }
    }

    // Draw reproduction intent lines (pink/magenta)
    for (org1_id, org2_id, _timestamp) in reproduction_intents {
        if let (Some(org1), Some(org2)) = (
            state.organisms.iter().find(|o| o.id == *org1_id),
            state.organisms.iter().find(|o| o.id == *org2_id),
        ) {
            let start = org1.pos.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            let end = org2.pos.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            draw_line(
                start[0],
                start[1],
                end[0],
                end[1],
                3.0,
                Color::from_rgba(255, 100, 200, 200),
            );
        }
    }
}

pub fn draw_food(
    state: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
) {
    // draw food
    state.food.iter().for_each(|entity| {
        if entity.energy > 0.0 {
            let screen_pos = entity.pos.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            let scaled_radius = params.body_radius.to_screen(
                params,
                ui_panel_width,
                camera_zoom,
                camera_offset_x,
                camera_offset_y,
            );
            draw_circle(
                screen_pos[0],
                screen_pos[1],
                scaled_radius,
                Color::from_rgba(0, 200, 100, 255),
            );
        }
    });
}

pub fn draw_projectiles(
    state: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
) {
    state.projectiles.iter().for_each(|projectile| {
        let screen_pos = projectile.pos.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );
        let scaled_radius = params.projectile_radius.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );

        // Map damage to alpha (transparency) to visualize projectile strength
        // Damage ranges from 0 to attack_damage_rate (typically ~4.0)
        // Map to alpha range 100-255 to keep projectiles visible
        let max_damage = params.attack_damage_rate;
        let normalized_damage = (projectile.damage / max_damage).clamp(0.0, 1.0);
        let alpha = (100.0 + normalized_damage * 155.0) as u8;

        draw_circle(
            screen_pos[0],
            screen_pos[1],
            scaled_radius,
            Color::from_rgba(0, 0, 0, alpha),
        );
    });
}

pub fn draw_organisms(
    state: &simulation::ecosystem::Ecosystem,
    params: &Params,
    ui_panel_width: f32,
    camera_zoom: f32,
    camera_offset_x: f32,
    camera_offset_y: f32,
    selected_id: Option<usize>,
) {
    state.organisms.iter().for_each(|entity| {
        let screen_pos = entity.pos.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );
        let screen_radius = params.body_radius.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );
        let is_selected = selected_id == Some(entity.id);

        // Draw scent radius (faint circle)
        let scent_radius_screen = params.scent_radius.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );
        draw_circle_lines(
            screen_pos[0],
            screen_pos[1],
            scent_radius_screen,
            2.0,
            Color::from_rgba(100, 100, 100, 20),
        );

        // Draw scent radius (faint circle)
        let share_radius_screen = params.share_radius.to_screen(
            params,
            ui_panel_width,
            camera_zoom,
            camera_offset_x,
            camera_offset_y,
        );
        draw_circle_lines(
            screen_pos[0],
            screen_pos[1],
            share_radius_screen,
            2.0,
            Color::from_rgba(100, 100, 100, 20),
        );

        // Highlight selected organism with a bright outline
        if is_selected {
            draw_circle_lines(
                screen_pos[0],
                screen_pos[1],
                screen_radius + 5.0,
                3.0,
                Color::from_rgba(255, 0, 0, 255),
            );
        }

        // Draw organism body with pool color
        let pool_color = get_pool_color(entity.pool_id);
        draw_circle(screen_pos[0], screen_pos[1], screen_radius, pool_color);

        // Draw movement axis indicator (direction of movement/orientation)
        let movement_line_length = screen_radius * 2.0;
        let movement_end_x = screen_pos[0] + entity.rot.cos() * movement_line_length;
        let movement_end_y = screen_pos[1] + entity.rot.sin() * movement_line_length;
        draw_line(
            screen_pos[0],
            screen_pos[1],
            movement_end_x,
            movement_end_y,
            2.0,
            Color::from_rgba(0, 0, 0, 180), // Dark line for movement direction
        );

        // organism health bar (scaled)
        let health_bar_width = 20.0;
        let health_bar_height = 2.0;
        let health_bar_offset = 2.0;
        let health_bar_x = screen_pos[0] - health_bar_width / 2.0;
        let health_bar_y = screen_pos[1] - screen_radius - health_bar_height - health_bar_offset;
        draw_rectangle(
            health_bar_x,
            health_bar_y,
            health_bar_width,
            health_bar_height,
            Color::from_rgba(100, 100, 100, 200),
        );
        draw_rectangle(
            health_bar_x,
            health_bar_y,
            health_bar_width * (entity.energy / 1.0).clamp(0.0, 1.0),
            health_bar_height,
            Color::from_rgba(255, 0, 0, 255),
        );

        // text scaling
        // let _font_size = 9.0; // minimum font size of 8
        // let _text_spacing = 10.0;

        // // organism id
        // let id_text = format!("ID:{}", entity.id);
        // let id_text_size = measure_text(&id_text, None, font_size as u16, 1.0);
        // draw_text(
        //     &id_text,
        //     screen_pos[0] - id_text_size.width / 2.0,
        //     health_bar_y - text_spacing,
        //     font_size,
        //     BLACK,
        // );

        // // organism age
        // let age_text = format!("Age: {:.1}", entity.age);
        // let age_text_size = measure_text(&age_text, None, font_size as u16, 1.0);
        // draw_text(
        //     &age_text,
        //     screen_pos[0] - age_text_size.width / 2.0,
        //     health_bar_y - text_spacing * 2.0,
        //     font_size,
        //     BLACK,
        // );

        // // organism score
        // let score_text = format!("Score: {}", entity.score);
        // let score_text_size = measure_text(&score_text, None, font_size as u16, 1.0);
        // draw_text(
        //     &score_text,
        //     screen_pos[0] - score_text_size.width / 2.0,
        //     health_bar_y - text_spacing * 3.0,
        //     font_size,
        //     BLACK,
        // );

        let vision_vectors = entity.get_vision_vectors();
        // // organism memory, simple rectangles

        // let memory_bar_width = 20.0;
        // let memory_bar_height = 3.0;
        // let memory_bar_x = screen_pos[0] - memory_bar_width / 2.0;
        // let memory_bar_y =
        //     screen_pos[1] - screen_radius - health_bar_height - memory_bar_height - 2.0;
        // for (i, &value) in entity.memory.iter().enumerate() {
        //     let color_value = (value * 255.0) as u8;
        //     draw_rectangle(
        //         memory_bar_x + i as f32 * (memory_bar_width / params.memory_size as f32),
        //         memory_bar_y,
        //         memory_bar_width / params.memory_size as f32,
        //         memory_bar_height,
        //         Color::from_rgba(color_value, color_value, color_value, 200),
        //     );
        // }

        for vision_vector in vision_vectors.iter() {
            // Decrease accuracy: only show 30% of the actual vision length
            let shortened_vector = vision_vector;
            let end_point = &screen_pos
                + shortened_vector.to_screen_relative(params, ui_panel_width, camera_zoom);

            // Draw a line from the organism's position to the shortened end point
            draw_line(
                screen_pos[0],
                screen_pos[1],
                end_point[0],
                end_point[1],
                1.0,
                Color::from_rgba(0, 0, 0, 50), // Semi-transparent black for less visual clutter
            );
        }
    });
}
