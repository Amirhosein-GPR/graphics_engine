mod lib;
use lib::graphics_api;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;
    
    let vulkan_module = match graphics_api::vulkan::VulkanModule::new(window) {
        Ok(api) => api,
        Err(error) => panic!("Error creating a Vulkan instance: {}", error)
    };

    let engine = lib::Engine::new(vulkan_module, event_loop);
    engine.run();

    Ok(())
}