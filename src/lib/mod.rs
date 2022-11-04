use ash::vk;
use winit::{event, event_loop};
pub mod graphics_api;
use graphics_api::vulkan;

pub struct Engine {
    vulkan_module: vulkan::VulkanModule,
    event_loop: event_loop::EventLoop<()>
}

impl Engine {
    pub fn new(vulkan_module: vulkan::VulkanModule, event_loop: event_loop::EventLoop<()>) -> Self {
        Engine {
            vulkan_module,
            event_loop
        }
    }

    pub fn run(mut self) {
        self.event_loop.run(move |event, _event_loop_window_target, control_flow| {
            match event {
                event::Event::WindowEvent {
                    event: event::WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = event_loop::ControlFlow::Exit;
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