pub mod graphics_api;

use graphics_api::GraphicsAPI;

pub struct Engine<T: GraphicsAPI> {
    graphics_api: T
}

impl<T: GraphicsAPI> Engine<T> {
    pub fn new(graphics_api: T) -> Self {
        Engine {
            graphics_api
        }
    }

    pub fn run(&self) {
    }
}