use graphics_api::vulkan::Camera;
use ash::vk;
use winit::{event, event_loop};
use nalgebra::Matrix4;

pub mod graphics_api;
use graphics_api::vulkan;

pub struct Engine {
    vulkan_module: vulkan::VulkanModule,
    event_loop: event_loop::EventLoop<()>,
    model_handles: Vec<usize>
}

impl Engine {
    pub fn new(vulkan_module: vulkan::VulkanModule, event_loop: event_loop::EventLoop<()>, model_handles: Vec<usize>) -> Self {
        Engine {
            vulkan_module,
            event_loop,
            model_handles
        }
    }

    pub fn run(mut self) {
        let mut camera = Camera::default();
        self.event_loop.run(move |event, _event_loop_window_target, control_flow| {
            match event {
                event::Event::WindowEvent {
                    event: event::WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = event_loop::ControlFlow::Exit;
                },
                event::Event::WindowEvent {
                    event: event::WindowEvent::KeyboardInput { input, .. },
                    ..
                } => match input {
                    event::KeyboardInput {
                        state: event::ElementState::Pressed,
                        virtual_keycode: Some(key_code),
                        ..
                    } => match key_code {
                        event::VirtualKeyCode::Right => {
                            camera.turn_right(0.1);
                        },
                        event::VirtualKeyCode::Left => {
                            camera.turn_left(0.1);
                        },
                        event::VirtualKeyCode::Up => {
                            camera.move_forward(0.05);
                        },
                        event::VirtualKeyCode::Down => {
                            camera.move_backward(0.05);
                        },
                        event::VirtualKeyCode::PageUp => {
                            camera.turn_up(0.02);
                        },
                        event::VirtualKeyCode::PageDown => {
                            camera.turn_down(0.02);
                        }
                        _ => {}
                    },
                    _ => {}
                },
                event::Event::MainEventsCleared => {
                    self.vulkan_module.window.request_redraw();
                },
                event::Event::RedrawRequested(_window_id) => {
                    let (image_index, _is_suboptimal) = unsafe {
                        self.vulkan_module
                            .swapchain_module
                            .loader
                            .acquire_next_image(
                                self.vulkan_module.swapchain_module.swapchain,
                                std::u64::MAX,
                                self.vulkan_module.swapchain_module.image_available_semaphores[self.vulkan_module.swapchain_module.current_image],
                                vk::Fence::null()
                            )
                            .expect("Error: Can't aquire image from swapchain!")
                    };

                    unsafe {
                        self.vulkan_module
                        .device.wait_for_fences(
                            &[self.vulkan_module.swapchain_module.begin_drawing_fences[self.vulkan_module.swapchain_module.current_image]],
                            true,
                            std::u64::MAX
                        )
                        .expect("Error: Can't wait on fences!");

                        self.vulkan_module
                            .device
                            .reset_fences(&[self.vulkan_module.swapchain_module.begin_drawing_fences[self.vulkan_module.swapchain_module.current_image]])
                            .expect("Error: Can't reset fences!")
                    }

                    camera.update_buffer(&self.vulkan_module.device, self.vulkan_module.allocator.as_mut().unwrap(), &mut self.vulkan_module.uniform_buffer);
                    for model_module in &mut self.vulkan_module.model_modules {
                        model_module.update_instance_buffer(&self.vulkan_module.device, self.vulkan_module.allocator.as_mut().unwrap())
                            .expect("Error: Can't update instance buffer!");
                    }
                    // self.vulkan_module.update
                    self.vulkan_module.update_command_buffers(image_index as usize).expect("Error: Updating the command buffer failed!");

                    let available_image_semaphores = [self.vulkan_module.swapchain_module.image_available_semaphores[self.vulkan_module.swapchain_module.current_image]];
                    let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let rendering_finished_semaphores = [self.vulkan_module.swapchain_module.rendering_finished_semaphores[self.vulkan_module.swapchain_module.current_image]];
                    let command_buffers = [self.vulkan_module.command_buffers[image_index as usize]];
                    let submit_infos = [vk::SubmitInfo::builder()
                        .wait_semaphores(&available_image_semaphores)
                        .wait_dst_stage_mask(&waiting_stages)
                        .command_buffers(&command_buffers)
                        .signal_semaphores(&rendering_finished_semaphores)
                        .build()];

                    unsafe {
                        self.vulkan_module
                            .device
                            .queue_submit(
                                self.vulkan_module.queue_module.graphics_queue,
                                &submit_infos,
                                self.vulkan_module.swapchain_module.begin_drawing_fences[self.vulkan_module.swapchain_module.current_image]
                            )
                            .expect("Error: Can't submit queues");
                    }

                    let swapchains = [self.vulkan_module.swapchain_module.swapchain];
                    let image_indices = [image_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&rendering_finished_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices);

                    unsafe {
                        self.vulkan_module
                            .swapchain_module
                            .loader
                            .queue_present(self.vulkan_module.queue_module.graphics_queue, &present_info)
                            .expect("Error: Can't present queue!");
                    }

                    self.vulkan_module.swapchain_module.current_image = (self.vulkan_module.swapchain_module.current_image + 1) % self.vulkan_module.swapchain_module.amount_of_images as usize;
                }
                _ => {}
            }
        });
    }

    // pub fn cleanup(&self) {
    //     std::mem::drop(self);
    // }
}