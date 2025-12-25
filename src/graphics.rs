use crate::evolution;

use macroquad::prelude::*;

pub fn draw_food(state: &evolution::State, params: &evolution::Params) {
    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale_x = screen_w / params.box_width;
    let scale_y = screen_h / params.box_height;
    let scale = scale_x.min(scale_y); // uniform scaling factor

    // draw food
    state.food.iter().for_each(|entity| {
        if entity.energy > 0.0 {
            draw_circle(
                entity.pos[0] * scale_x,
                entity.pos[1] * scale_y,
                params.body_radius * scale,
                Color::from_rgba(0, 100, 255, 255),
            );
        }
    });
}
pub fn draw_organisms(state: &evolution::State, params: &evolution::Params) {
    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale_x = screen_w / params.box_width;
    let scale_y = screen_h / params.box_height;
    let scale = scale_x.min(scale_y); // uniform scaling factor

    state.organisms.iter().for_each(|entity| {
        let screen_x = entity.pos[0] * scale_x;
        let screen_y = entity.pos[1] * scale_y;
        let scaled_radius = params.body_radius * scale;

        draw_circle(
            screen_x,
            screen_y,
            scaled_radius,
            Color::from_rgba(
                (entity.signal[0] * 255.0) as u8,
                (entity.signal[1] * 255.0) as u8,
                (entity.signal[2] * 255.0) as u8,
                255,
            ),
        );

        // organism health bar (scaled)
        let health_bar_width = 20.0 * scale;
        let health_bar_height = 2.0 * scale;
        let health_bar_offset = 2.0 * scale;
        let health_bar_x = screen_x - health_bar_width / 2.0;
        let health_bar_y = screen_y - scaled_radius - health_bar_height - health_bar_offset;
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
        let font_size = (9.0 * scale).max(8.0); // minimum font size of 8
        let text_spacing = 10.0 * scale;

        // organism id
        let id_text = format!("ID:{}", entity.id);
        let id_text_size = measure_text(&id_text, None, font_size as u16, 1.0);
        draw_text(
            &id_text,
            screen_x - id_text_size.width / 2.0,
            health_bar_y - text_spacing,
            font_size,
            BLACK,
        );

        // organism age
        let age_text = format!("Age: {:.1}", entity.age);
        let age_text_size = measure_text(&age_text, None, font_size as u16, 1.0);
        draw_text(
            &age_text,
            screen_x - age_text_size.width / 2.0,
            health_bar_y - text_spacing * 2.0,
            font_size,
            BLACK,
        );

        // organism score
        let score_text = format!("Score: {}", entity.score);
        let score_text_size = measure_text(&score_text, None, font_size as u16, 1.0);
        draw_text(
            &score_text,
            screen_x - score_text_size.width / 2.0,
            health_bar_y - text_spacing * 3.0,
            font_size,
            BLACK,
        );

        // let vision_vectors = organism::get_vision_vectors(
        //     entity,
        //     params.fov,
        //     params.num_vision_directions,
        //     params.vision_radius,
        // );
        // // organism memory, simple rectangles
        // let memory_bar_width = 20.0;
        // let memory_bar_height = 3.0;
        // let memory_bar_x = entity.pos[0] - memory_bar_width / 2.0;
        // let memory_bar_y = entity.pos[1] - BODY_RADIUS - health_bar_height - memory_bar_height - 2.0;
        // for (i, &value) in entity.memory.iter().enumerate() {
        //     let color_value = (value * 255.0) as u8;
        //     draw_rectangle(
        //         memory_bar_x + i as f32 * (memory_bar_width / MEMORY_SIZE as f32),
        //         memory_bar_y,
        //         memory_bar_width / MEMORY_SIZE as f32,
        //         memory_bar_height,
        //         Color::from_rgba(color_value, color_value, color_value, 200)
        //     );
        // }

        // for (i, vision_vector) in vision_vectors.iter().enumerate() {
        //     let end_point = &entity.pos + vision_vector;
        //     // draw a line from the organism's position to the end point of the vision vector
        //     draw_line(
        //         entity.pos[0],
        //         entity.pos[1],
        //         end_point[0],
        //         end_point[1],
        //         1.0,
        //         Color::from_rgba(
        //             (brain_inputs[(params.signal_size + 1) * i + 0]* 255.0) as u8,
        //             (brain_inputs[(params.signal_size + 1) * i + 1] * 255.0) as u8,
        //             (brain_inputs[(params.signal_size + 1) * i + 2] * 255.0) as u8,
        //             255
        //         )
        //     );
        // }
    });
}
