use crate::simulation;
use macroquad::prelude::*;
use ndarray::Array1;

fn get_organism_at_mouse(
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
) -> Option<usize> {
    let (mouse_x, mouse_y) = mouse_position();

    // Don't detect if mouse is over UI panel
    if mouse_x > screen_width() - ui_panel_width {
        return None;
    }

    // Convert mouse position to simulation coordinates
    let screen_w = screen_width() - ui_panel_width;
    let screen_h = screen_height();
    let scale_x = params.box_width / screen_w;
    let scale_y = params.box_height / screen_h;

    let sim_x = mouse_x * scale_x;
    let sim_y = mouse_y * scale_y;

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
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
) -> Option<usize> {
    get_organism_at_mouse(ecosystem, params, ui_panel_width)
}

pub fn handle_organism_click(
    ecosystem: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
) -> Option<usize> {
    if is_mouse_button_pressed(MouseButton::Left) {
        get_organism_at_mouse(ecosystem, params, ui_panel_width)
    } else {
        None
    }
}

trait ToScreen {
    type Output;
    fn to_screen(
        &self,
        params: &simulation::ecosystem::Params,
        ui_panel_width: f32,
    ) -> Self::Output;
}

impl ToScreen for Array1<f32> {
    type Output = Array1<f32>;
    fn to_screen(
        &self,
        params: &simulation::ecosystem::Params,
        ui_panel_width: f32,
    ) -> Array1<f32> {
        let screen_w = screen_width() - ui_panel_width;
        let screen_h = screen_height();
        let scale_x = screen_w / params.box_width;
        let scale_y = screen_h / params.box_height;
        Array1::from_vec(vec![self[0] * scale_x, self[1] * scale_y])
    }
}

impl ToScreen for f32 {
    type Output = f32;
    fn to_screen(&self, params: &simulation::ecosystem::Params, ui_panel_width: f32) -> f32 {
        let screen_w = screen_width() - ui_panel_width;
        let screen_h = screen_height();
        let scale_x = screen_w / params.box_width;
        let scale_y = screen_h / params.box_height;
        let scale = scale_x.min(scale_y);
        self * scale
    }
}

pub fn draw_food(
    state: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
) {
    // draw food
    state.food.iter().for_each(|entity| {
        if entity.energy > 0.0 {
            let screen_pos = entity.pos.to_screen(params, ui_panel_width);
            let scaled_radius = params.body_radius.to_screen(params, ui_panel_width);
            draw_circle(
                screen_pos[0],
                screen_pos[1],
                scaled_radius,
                Color::from_rgba(0, 100, 255, 255),
            );
        }
    });
}

pub fn draw_projectiles(
    state: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
) {
    state.projectiles.iter().for_each(|projectile| {
        let screen_pos = projectile.pos.to_screen(params, ui_panel_width);
        let scaled_radius = params.projectile_radius.to_screen(params, ui_panel_width);
        draw_circle(
            screen_pos[0],
            screen_pos[1],
            scaled_radius,
            Color::from_rgba(255, 0, 0, 255),
        );
    });
}

pub fn draw_organisms(
    state: &simulation::ecosystem::Ecosystem,
    params: &simulation::ecosystem::Params,
    ui_panel_width: f32,
    selected_id: Option<usize>,
) {
    state.organisms.iter().for_each(|entity| {
        let screen_pos = entity.pos.to_screen(params, ui_panel_width);
        let screen_radius = params.body_radius.to_screen(params, ui_panel_width);
        let is_selected = selected_id == Some(entity.id);

        // Draw scent radius (faint circle)
        let scent_radius_screen = params.scent_radius.to_screen(params, ui_panel_width);
        draw_circle_lines(
            screen_pos[0],
            screen_pos[1],
            scent_radius_screen,
            2.0,
            Color::from_rgba(100, 100, 100, 30),
        );

        // Highlight selected organism with a bright outline
        if is_selected {
            draw_circle_lines(
                screen_pos[0],
                screen_pos[1],
                screen_radius + 3.0,
                3.0,
                Color::from_rgba(255, 255, 0, 255),
            );
        }

        // Draw organism body
        draw_circle(
            screen_pos[0],
            screen_pos[1],
            screen_radius,
            Color::from_rgba(
                (entity.signal[0] * 255.0) as u8,
                (entity.signal[1] * 255.0) as u8,
                (entity.signal[2] * 255.0) as u8,
                255,
            ),
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
        let _font_size = 9.0; // minimum font size of 8
        let _text_spacing = 10.0;

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
            let end_point = &screen_pos + vision_vector.to_screen(params, ui_panel_width);
            // draw a line from the organism's position to the end point of the vision vector
            draw_line(
                screen_pos[0],
                screen_pos[1],
                end_point[0],
                end_point[1],
                1.0,
                BLACK, // Color::from_rgba(
                       //     (brain_inputs[(params.signal_size + 1) * i + 0]* 255.0) as u8,
                       //     (brain_inputs[(params.signal_size + 1) * i + 1] * 255.0) as u8,
                       //     (brain_inputs[(params.signal_size + 1) * i + 2] * 255.0) as u8,
                       //     255
                       // )
            );
        }
    });
}
