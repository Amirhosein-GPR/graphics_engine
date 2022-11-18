mod lib;
use lib::graphics_api;
use lib::graphics_api::vulkan::{ModelModule, InstanceData};
use nalgebra::{Matrix4, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;
    
    let mut vulkan_module = match graphics_api::vulkan::VulkanModule::new(window) {
        Ok(api) => api,
        Err(error) => panic!("Error creating a Vulkan instance: {}", error)
    };

    let mut cube = ModelModule::cube();
    let mut model_handles = vec![];
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::new_translation(&Vector3::new(0.0, 0.0, 0.1))
            * Matrix4::new_scaling(0.1)),
        color: [0.2, 0.4, 1.0]
    }));
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::new_translation(&Vector3::new(0.05, 0.05, 0.0))
            * Matrix4::new_scaling(0.1)),
        color: [1.0, 1.0, 0.2]
    }));
    for i in 0..10 {
        for j in 0..10 {
            model_handles.push(cube.insert_visibly(InstanceData {
                model_matrix: (Matrix4::new_translation(&Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    j as f32 * 0.2 - 1.0,
                    0.5
                )) * Matrix4::new_scaling(0.03)),
                color: [1.0, i as f32 * 0.07, j as f32 * 0.07]
            }));
            model_handles.push(cube.insert_visibly(InstanceData {
                model_matrix: (Matrix4::new_translation(&Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    0.0,
                    j as f32 * 0.2 - 1.0,
                )) * Matrix4::new_scaling(0.02)),
                color: [i as f32 * 0.07, j as f32 * 0.07, 1.0]
            }));
        }
    }
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::from_scaled_axis(Vector3::new(0.0, 0.0, 1.4))
            * Matrix4::new_translation(&Vector3::new(0.0, 0.5, 0.0))
            * Matrix4::new_scaling(0.1)),
        color: [0.0, 0.5, 0.0]
    }));
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::from_scaled_axis(Vector3::new(0.5, 0.0, 0.0))
            * Matrix4::new_translation(&Vector3::new(0.5, 0.01, 0.01))),
        color: [1.0, 0.5, 0.5]
    }));
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::from_scaled_axis(Vector3::new(0.0, 0.5, 0.0))
            * Matrix4::new_translation(&Vector3::new(0.01, 0.5, 0.01))),
        color: [0.5, 1.0, 0.5]
    }));
    model_handles.push(cube.insert_visibly(InstanceData {
        model_matrix: (Matrix4::from_scaled_axis(Vector3::new(0.0, 0.0, 0.0))
            * Matrix4::new_translation(&Vector3::new(0.01, 0.01, 0.5))),
        color: [0.5, 0.5, 1.0]
    }));

    cube.update_vertex_buffer(&vulkan_module.device, &mut vulkan_module.allocator.as_mut().unwrap())?;
    cube.update_index_buffer(&vulkan_module.device, &mut vulkan_module.allocator.as_mut().unwrap())?;
    cube.update_instance_buffer(&vulkan_module.device, &mut vulkan_module.allocator.as_mut().unwrap())?;
    vulkan_module.model_modules = vec![cube];

    let engine = lib::Engine::new(vulkan_module, event_loop, model_handles);
    engine.run();

    Ok(())
}