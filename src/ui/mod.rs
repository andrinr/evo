// UI module - handles all user interface rendering

mod genesis;
mod nn;
mod organisms;
mod stats;
mod ui;

// Re-export the public interface
pub use genesis::draw_genesis_screen;
pub use ui::{UIState, draw_ui, process_egui};
