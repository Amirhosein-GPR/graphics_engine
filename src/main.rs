mod lib;

use lib::graphics_api;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graphics_api = match graphics_api::Vulkan::new() {
        Ok(api) => api,
        Err(error) => panic!("Error creating a Vulkan instance: {}", error)
    };

    let engine = lib::Engine::new(graphics_api);
    engine.run();
    
    Ok(())
}