use crate::evolution;

use macroquad::prelude::*;
use rayon::prelude::*;

pub fn draw_food(state: &evolution::State, params: &evolution::Params) {
    // draw food
    state.food.par_iter().for_each(|entity| {
        if entity.energy > 0.0 {
            draw_circle(
                entity.pos[0],
                entity.pos[1],
                params.body_radius,
                Color::from_rgba(0, 100, 255, 255),
            );
        }
    });
}
pub fn draw_organisms(state: &evolution::State, params: &evolution::Params) {
    state.organisms.par_iter_mut().for_each(|entity| {
        draw_circle(
            entity.pos[0],
            entity.pos[1],
            params.body_radius,
            Color::from_rgba(
                (entity.signal[0] * 255.0) as u8,
                (entity.signal[1] * 255.0) as u8,
                (entity.signal[2] * 255.0) as u8,
                255,
            ),
        );

        // organism health bar
        let health_bar_width = 20.0;
        let health_bar_height = 2.0;
        let health_bar_x = entity.pos[0] - health_bar_width / 2.0;
        let health_bar_y = entity.pos[1] - params.body_radius - health_bar_height - 2.0;
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
            health_bar_width * (entity.energy / 1.0).max(0.0).min(1.0),
            health_bar_height,
            Color::from_rgba(255, 0, 0, 255),
        );

        // organism id
        let id_text = format!("ID:{}", entity.id);
        let id_text_size = measure_text(&id_text, None, 12, 1.0);
        draw_text(
            &id_text,
            entity.pos[0] - id_text_size.width / 2.0,
            entity.pos[1] - params.body_radius - health_bar_height - 2.0 - 10.0,
            9.0,
            BLACK,
        );

        // organism age
        let age_text = format!("Age: {:.1}", entity.age);
        let age_text_size = measure_text(&age_text, None, 12, 1.0);
        draw_text(
            &age_text,
            entity.pos[0] - age_text_size.width / 2.0,
            entity.pos[1] - params.body_radius - health_bar_height - 2.0 - 20.0,
            9.0,
            BLACK,
        );

        // organism score
        let score_text = format!("Score: {}", entity.score);
        let score_text_size = measure_text(&score_text, None, 12, 1.0);
        draw_text(
            &score_text,
            entity.pos[0] - score_text_size.width / 2.0,
            entity.pos[1] - params.body_radius - health_bar_height - 2.0 - 30.0,
            9.0,
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
