use std::error;
use ash::Entry;
use ash::vk;
use ash::prelude::VkResult;
use winit::platform::unix::WindowExtUnix;
use vk_shader_macros;
use nalgebra::{Matrix4, Vector3};

pub struct VulkanModule {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_utils_module: Option<DebugUtilsModule>,
    pub surface_module: Option<SurfaceModule>,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub window: winit::window::Window,
    pub device: ash::Device,
    pub queue_module: QueueModule,
    pub swapchain_module: SwapchainModule,
    pub render_pass: vk::RenderPass,
    pub pipeline_module: PipelineModule,
    pub command_pool_module: CommandPoolModule,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub allocator: Option<gpu_allocator::vulkan::Allocator>,
    pub model_modules: Vec<ModelModule<[f32; 3], InstanceData>>,
    pub uniform_buffer: BufferModule,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>
}

pub struct DebugUtilsModule {
    pub loader: ash::extensions::ext::DebugUtils,
    pub messenger: vk::DebugUtilsMessengerEXT
}

pub struct SurfaceModule {
    pub xlib_loader: ash::extensions::khr::XlibSurface,
    pub surface: vk::SurfaceKHR,
    pub loader: ash::extensions::khr::Surface
}

pub struct QueueModule {
    pub graphics_queue: vk::Queue,
    pub graphics_queue_index: u32,
    pub transfer_queue: vk::Queue,
    pub transfer_queue_index: u32
}

pub struct SwapchainModule {
    pub swapchain: vk::SwapchainKHR,
    pub loader: ash::extensions::khr::Swapchain,
    pub image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_buffer: vk::Buffer,
    pub depth_image_allocator: Option<gpu_allocator::vulkan::Allocator>,
    pub depth_image_allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub depth_image_view: vk::ImageView,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub rendering_finished_semaphores: Vec<vk::Semaphore>,
    pub begin_drawing_fences: Vec<vk::Fence>,
    pub amount_of_images: u32,
    pub current_image: usize
}

pub struct PipelineModule {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>
}

pub struct CommandPoolModule {
    pub graphics_command_pool: vk::CommandPool,
    pub transfer_command_pool: vk::CommandPool
}

pub struct BufferModule {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    size_in_bytes: u64,
    buffer_usage: vk::BufferUsageFlags,
    memory_location: gpu_allocator::MemoryLocation,
    bind_buffer: bool
}

#[derive(Debug, Clone)]
pub struct InvalidHandle;

pub struct ModelModule<V, I> {
    vertex_data: Vec<V>,
    index_data: Vec<u32>,
    handle_to_index: std::collections::HashMap<usize, usize>,
    handles: Vec<usize>,
    instances: Vec<I>,
    first_invisible_instance: usize,
    next_handle: usize,
    vertex_buffer_module: Option<BufferModule>,
    index_buffer_module: Option<BufferModule>,
    instance_buffer_module: Option<BufferModule>,
}

#[repr(C)]
pub struct InstanceData {
    pub model_matrix: Matrix4<f32>,
    pub color: [f32; 3]
}

pub struct Camera {
    pub view_matrix: Matrix4<f32>,
    pub position: Vector3<f32>,
    pub view_direction: nalgebra::Unit<Vector3<f32>>,
    pub down_direction: nalgebra::Unit<Vector3<f32>>,
    pub fovy: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub projection_matrix: Matrix4<f32>
}

pub struct CameraBuilder {
    pub position: Vector3<f32>,
    pub view_direction: nalgebra::Unit<Vector3<f32>>,
    pub down_direction: nalgebra::Unit<Vector3<f32>>,
    pub fovy: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32
}

impl VulkanModule {
    pub fn new(window: winit::window::Window) -> Result<Self, Box<dyn error::Error>> {
        let entry = unsafe { Entry::load()? };

        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(VulkanModule::vulkan_debug_utils_callback));

        let layer_names = [std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let extension_names = [
            std::ffi::CString::from(ash::extensions::ext::DebugUtils::name()),
            std::ffi::CString::from(ash::extensions::khr::Surface::name()),
            std::ffi::CString::from(ash::extensions::khr::XlibSurface::name())
        ];

        let instance = Self::initialize_instance(&entry, &mut debug_utils_messenger_create_info, &layer_names, &extension_names)?;
        let debug_utils_module = DebugUtilsModule::initialize(&entry, &instance, &debug_utils_messenger_create_info)?;
        let surface_module = SurfaceModule::initialize(&entry, &instance, &window)?;

        let (physical_device, physical_device_properties, physical_device_features) = Self::initialize_physical_device_and_properties(&instance)?;

        let (device, queue_module) = Self::initialize_device_and_queues(&instance, &surface_module, physical_device, &physical_device_features, &layer_names)?;

        let mut swapchain_module = SwapchainModule::initialize(&instance, physical_device, &device, &surface_module, &queue_module)?;

        let render_pass = Self::initialize_render_pass(&device, physical_device, &swapchain_module.surface_format.format)?;

        swapchain_module.create_framebuffers(&device, render_pass)?;

        let pipeline_module = PipelineModule::initialize(&device, &swapchain_module, &render_pass)?;

        let command_pool_module = CommandPoolModule::initialize(&device, &queue_module)?;

        let allocator_create_description = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            device: device.clone(),
            instance: instance.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false
        };
        let mut allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_description)?;

        let command_buffers = Self::create_command_buffers(&device, &command_pool_module, swapchain_module.framebuffers.len() as u32)?;

        let mut uniform_buffer = BufferModule::new(
            &device,
            &mut allocator,
            128,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu
        )?;
        let camera_transform: [[[f32; 4]; 4]; 2] = [Matrix4::identity().into(), Matrix4::identity().into()];
        uniform_buffer.fill(&device, &mut allocator, &camera_transform)?;
        let descriptor_pool_size = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain_module.amount_of_images
        }];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain_module.amount_of_images)
            .pool_sizes(&descriptor_pool_size);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? };
        let descriptor_layouts = vec![pipeline_module.descriptor_set_layouts[0]; swapchain_module.amount_of_images as usize];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info)? };

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 128
            }];
            let write_descriptor_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(*descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .build()];
            unsafe {device.update_descriptor_sets(&write_descriptor_sets, &[])};
        }

        Ok(VulkanModule {
            entry,
            instance,
            debug_utils_module: Some(debug_utils_module),
            surface_module: Some(surface_module),
            physical_device,
            physical_device_properties,
            window,
            device,
            queue_module,
            swapchain_module: swapchain_module,
            render_pass,
            pipeline_module,
            command_pool_module,
            command_buffers,
            allocator: Some(allocator),
            model_modules: vec![],
            uniform_buffer,
            descriptor_pool,
            descriptor_sets
        })
    }

    fn initialize_instance(entry: &ash::Entry, debug_utils_messenger_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT, layer_names: &[std::ffi::CString], extension_names: &[std::ffi::CString]) -> VkResult<ash::Instance> {
        let engine_name = std::ffi::CString::new("SparkEngine").unwrap();
        let application_name = std::ffi::CString::new("Game").unwrap();
        
        let application_info = vk::ApplicationInfo::builder()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let layer_name_pointers: Vec<*const i8> = layer_names.iter().map(|layer_name| { layer_name.as_ptr() }).collect();
        let extension_name_pointers: Vec<*const i8> = extension_names.iter().map(|extension_name| { extension_name.as_ptr() }).collect();

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .push_next(debug_utils_messenger_create_info)
            .application_info(&application_info)
            .enabled_layer_names(&layer_name_pointers)
            .enabled_extension_names(&extension_name_pointers);

        unsafe { entry.create_instance(&instance_create_info, None)}
    }

    fn initialize_physical_device_and_properties(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, vk::PhysicalDeviceFeatures), vk::Result> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen = None;
        for physical_device in physical_devices {
            let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let physical_device_features = unsafe { instance.get_physical_device_features(physical_device) };
            if physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU && physical_device_features.fill_mode_non_solid == vk::TRUE {
                chosen = Some((physical_device, physical_device_properties, physical_device_features));
                break;
            }
        }
        Ok(chosen.expect("Error: No proper graphics card found! (DISCRETE_GPU needed!)"))
    }

    fn initialize_device_and_queues(instance: &ash::Instance, surface_module: &SurfaceModule, physical_device: vk::PhysicalDevice, physical_device_features: &vk::PhysicalDeviceFeatures, layer_names: &[std::ffi::CString]) -> Result<(ash::Device, QueueModule), vk::Result> {
        let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics_queue_index = None;
        let mut transfer_queue_index = None;

        for (index, queue_family) in queue_family_properties.iter().enumerate() {
            if queue_family.queue_count > 0 {
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && unsafe { surface_module.loader.get_physical_device_surface_support(physical_device, index as u32, surface_module.surface)? } {
                    graphics_queue_index = Some(index as u32);
                }

                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    if transfer_queue_index.is_none() || !queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        transfer_queue_index = Some(index as u32);
                    }
                }
            }
        }

        let graphics_queue_index = graphics_queue_index.expect("Error: No graphics queue family found!");
        let transfer_queue_index = transfer_queue_index.expect("Error: No transfer queue family found!");


        let mut priorities = vec![1.0f32];
        let mut queue_create_infos = Vec::new();
        if graphics_queue_index == transfer_queue_index {
            priorities.push(1.0);
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_index)
                    .queue_priorities(&priorities)
                    .build()
            );
        } else {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_index)
                    .queue_priorities(&priorities)
                    .build()
            );
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(transfer_queue_index)
                .queue_priorities(&priorities)
                .build()
            );
        }

        let device_extension_name_pointers = vec![ash::extensions::khr::Swapchain::name().as_ptr()];
        let layer_name_pointers: Vec<*const i8> = layer_names.iter().map(|layer_name| { layer_name.as_ptr() }).collect();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_name_pointers)
            .enabled_layer_names(&layer_name_pointers)
            .enabled_features(&physical_device_features);
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let queue_module = QueueModule::initialize(&device, graphics_queue_index, transfer_queue_index, &queue_family_properties);

        Ok((device, queue_module))
    }

    fn initialize_render_pass(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        format: &vk::Format
    ) -> Result<vk::RenderPass, vk::Result> {
        let attachment_descriptions = [
            vk::AttachmentDescription::builder()
                .format(*format)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
        ];

        let color_attachment_refrences = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        }];
        let depth_attachment_refrence = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let subpass_descriptions = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refrences)
            .depth_stencil_attachment(&depth_attachment_refrence)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            )
            .build()];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies);
        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };
        Ok(render_pass)
    }

    fn create_command_buffers(device: &ash::Device, command_pool_module: &CommandPoolModule, amount: u32) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool_module.graphics_command_pool)
            .command_buffer_count(amount);
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
    }

    fn fill_command_buffers(command_buffers: &[vk::CommandBuffer], device: &ash::Device, render_pass: &vk::RenderPass, swapchain_module: &SwapchainModule, pipeline_module: &PipelineModule, model_modules: &Vec<ModelModule<[f32; 3], InstanceData>>) -> Result<(), vk::Result> {
        for (index, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
            unsafe {
                device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0]
                    }
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0
                    }
                }
            ];
            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(*render_pass)
                .framebuffer(swapchain_module.framebuffers[index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: 0
                    },
                    extent: swapchain_module.extent
                })
                .clear_values(&clear_values);

            unsafe {
                device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_module.pipeline);
                for model_module in model_modules {
                    model_module.draw(device, command_buffer);
                }
                device.cmd_end_render_pass(command_buffer);
                device.end_command_buffer(command_buffer)?;
            }
        }

        Ok(())
    }

    pub fn update_command_buffers(&mut self, index: usize) -> Result<(), vk::Result> {
        let command_buffer = self.command_buffers[index];
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.8, 1.0]
                }
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0
                }
            }
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_module.framebuffers[index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_module.extent
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_module.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_module.layout,
                0,
                &[self.descriptor_sets[index]],
                &[]
            );
            for model_module in &self.model_modules {
                model_module.draw(&self.device, command_buffer);
            }
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    extern "system" fn vulkan_debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void
    ) -> vk::Bool32 {
        let message = unsafe { std::ffi::CStr::from_ptr((*p_callback_data).p_message) };
        let severity = format!("{:?}", message_severity).to_lowercase();
        let ty = format!("{:?}", message_type).to_lowercase();
        println!("[Debug][{}][{}] {:?}", severity, ty, message);
        vk::FALSE
    }
}

impl Drop for VulkanModule {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Error: Device can't wait on vulkan module when being dropped!");
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_buffer(self.uniform_buffer.buffer, None);
            let mut allocator = self.allocator.take().unwrap();
            for model_module in &mut self.model_modules {
                if let Some(vertex_buffer_module) = &mut model_module.vertex_buffer_module {
                    self.device.destroy_buffer(vertex_buffer_module.buffer, None);
                    allocator.free(vertex_buffer_module.allocation.take().unwrap()).unwrap();
                }
                if let Some(instance_buffer_module) = &mut model_module.instance_buffer_module {
                    self.device.destroy_buffer(instance_buffer_module.buffer, None);
                    allocator.free(instance_buffer_module.allocation.take().unwrap()).unwrap();
                }
                if let Some(index_buffer_module) = &mut model_module.index_buffer_module {
                    self.device.destroy_buffer(index_buffer_module.buffer, None);
                    allocator.free(index_buffer_module.allocation.take().unwrap()).unwrap();
                }
            }
            std::mem::drop(allocator);
            self.command_pool_module.cleanup(&self.device);
            self.pipeline_module.cleanup(&self.device);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_module.cleanup(&self.device);
            std::mem::drop(self.surface_module.take().unwrap());
            self.device.destroy_device(None);
            std::mem::drop(self.debug_utils_module.take().unwrap());
            self.instance.destroy_instance(None);
        }
    }
}

impl DebugUtilsModule {
    fn initialize(entry: &ash::Entry, instance: &ash::Instance, debug_utils_messenger_create_info: &vk::DebugUtilsMessengerCreateInfoEXT) -> Result<Self, vk::Result> {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_utils_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_utils_messenger_create_info, None)? };
        Ok(Self {
            loader: debug_utils_loader,
            messenger: debug_utils_messenger
        })
    }
}

impl Drop for DebugUtilsModule {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

impl SurfaceModule {
    fn initialize(entry: &ash::Entry, instance: &ash::Instance, window: &winit::window::Window) -> Result<Self, vk::Result> {
        let xlib_display = window.xlib_display().unwrap();
        let xlib_window = window.xlib_window().unwrap();
        let xlib_surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(xlib_window)
            .dpy(xlib_display as *mut vk::Display);
        let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(&entry, &instance);
        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&xlib_surface_create_info, None)? };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        
        Ok(Self {
            xlib_loader: xlib_surface_loader,
            surface,
            loader: surface_loader
        })
    }
}

impl Drop for SurfaceModule {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

impl QueueModule {
    fn initialize(device: &ash::Device, graphics_queue_index: u32, transfer_queue_index: u32, queue_family_properties: &[vk::QueueFamilyProperties]) -> Self {
        let graphics_queue;
        let transfer_queue;
        if graphics_queue_index == transfer_queue_index {
            if queue_family_properties[graphics_queue_index as usize].queue_count > 1 {
                graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
                transfer_queue = unsafe { device.get_device_queue(graphics_queue_index, 1) };
            } else {
                graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
                transfer_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
            }
        } else {
            graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
            transfer_queue = unsafe { device.get_device_queue(transfer_queue_index, 0) };
        }

        Self {
            graphics_queue,
            graphics_queue_index,
            transfer_queue,
            transfer_queue_index
        }
    }
}

impl SwapchainModule {
    fn initialize(instance: &ash::Instance, physical_device: vk::PhysicalDevice, device: &ash::Device, surface_module: &SurfaceModule, queue_module: &QueueModule) -> Result<Self, vk::Result> {
        let surface_capabilities = unsafe {
            surface_module.loader.get_physical_device_surface_capabilities(physical_device, surface_module.surface)?
        };
        let surface_pressent_modes = unsafe {
            surface_module.loader.get_physical_device_surface_present_modes(physical_device, surface_module.surface)?
        };
        let surface_format = *unsafe {
            surface_module.loader.get_physical_device_surface_formats(physical_device, surface_module.surface)?
        }.first().unwrap();

        let extent = surface_capabilities.current_extent;

        let graphics_queue_family_indices = [queue_module.graphics_queue_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface_module.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count)
            )
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&graphics_queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let amount_of_images = swapchain_images.len() as u32;
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());
        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(*subresource_range);
            let image_views = unsafe { device.create_image_view(&image_view_create_info, None)? };
            swapchain_image_views.push(image_views);
        }

        let extent_3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1
        };
        let depth_image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(extent_3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&graphics_queue_family_indices);
        let depth_image = unsafe { device.create_image(&depth_image_create_info, None)? };
        let depth_image_size = unsafe { device.get_image_memory_requirements(depth_image).size };
        let depth_image_allocator_create_description = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            device: device.clone(),
            instance: instance.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false
        };
        let mut depth_image_allocator = gpu_allocator::vulkan::Allocator::new(&depth_image_allocator_create_description).expect("Error: Can't create memory allocator for depth buffer!");
        let depth_image_buffer_create_info = vk::BufferCreateInfo::builder()
            .size(depth_image_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER);
        let depth_image_buffer = unsafe { device.create_buffer(&depth_image_buffer_create_info, None)? };
        let depth_image_buffer_memory_requirements = unsafe { device.get_buffer_memory_requirements(depth_image_buffer) };
        let depth_image_allocation = depth_image_allocator.allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Depth buffer allocation.",
                requirements: depth_image_buffer_memory_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true
            }
        ).expect("Error: Could'nt allocate memory");
        unsafe { device.bind_image_memory(depth_image, depth_image_allocation.memory(), depth_image_allocation.offset())? };
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let depth_image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(*subresource_range);
        let depth_image_view = unsafe { device.create_image_view(&depth_image_view_create_info, None)? };

        let mut image_available_semaphores = vec![];
        let mut rendering_finished_semaphores = vec![];
        let mut begin_drawing_fences = vec![];
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..amount_of_images {
            image_available_semaphores.push(unsafe { device.create_semaphore(&semaphore_create_info, None)? });
            rendering_finished_semaphores.push(unsafe { device.create_semaphore(&semaphore_create_info, None)? });
            begin_drawing_fences.push(unsafe { device.create_fence(&fence_create_info, None)? });
        }


        Ok (Self {
            swapchain,
            loader: swapchain_loader,
            image_views: swapchain_image_views,
            depth_image,
            depth_image_buffer,
            depth_image_allocator: Some(depth_image_allocator),
            depth_image_allocation: Some(depth_image_allocation),
            depth_image_view,
            framebuffers: vec![],
            surface_format,
            extent,
            image_available_semaphores,
            rendering_finished_semaphores,
            begin_drawing_fences,
            amount_of_images,
            current_image: 0
        })
    }

    fn create_framebuffers(&mut self, device: &ash::Device, render_pass: vk::RenderPass) -> Result<(), vk::Result> {
        for image_view in &self.image_views {
            let image_view_array = [*image_view, self.depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&image_view_array)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None)? };
            self.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        let mut depth_image_allocator = self.depth_image_allocator.take().unwrap();
        device.destroy_buffer(self.depth_image_buffer, None);
        depth_image_allocator.free(self.depth_image_allocation.take().unwrap()).unwrap();
        std::mem::drop(depth_image_allocator);
        device.destroy_image_view(self.depth_image_view, None);
        device.destroy_image(self.depth_image, None);
        for begin_drawing_fence in &self.begin_drawing_fences {
            device.destroy_fence(*begin_drawing_fence, None);
        }

        for image_available_semaphore in &self.image_available_semaphores {
            device.destroy_semaphore(*image_available_semaphore, None);
        }

        for rendering_finished_semaphore in &self.rendering_finished_semaphores {
            device.destroy_semaphore(*rendering_finished_semaphore, None);
        }

        for frame_buffer in &self.framebuffers {
            device.destroy_framebuffer(*frame_buffer, None);
        }

        for image_view in &self.image_views {
            device.destroy_image_view(*image_view, None);
        }

        self.loader.destroy_swapchain(self.swapchain, None);
    }
}


impl PipelineModule {
    fn initialize(device: &ash::Device, swapchain_module: &SwapchainModule, render_pass: &vk::RenderPass) -> Result<Self, vk::Result> {
        let vertex_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(vk_shader_macros::include_glsl!("./shaders/shader.vert"));
        let vertex_shader_module = unsafe { device.create_shader_module(&vertex_shader_module_create_info, None)? };
        let fragment_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(vk_shader_macros::include_glsl!("./shaders/shader.frag"));
        let fragment_shader_module = unsafe { device.create_shader_module(&fragment_shader_module_create_info, None)? };

        let main_function_name = std::ffi::CString::new("main").unwrap();

        let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&main_function_name);
        let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&main_function_name);
        
        let shader_stages = [vertex_shader_stage_create_info.build(), fragment_shader_stage_create_info.build()];

        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: 0,
                format: vk::Format::R32G32B32_SFLOAT
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                offset: 0,
                format: vk::Format::R32G32B32A32_SFLOAT
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                offset: 16,
                format: vk::Format::R32G32B32A32_SFLOAT
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                offset: 32,
                format: vk::Format::R32G32B32A32_SFLOAT
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                offset: 48,
                format: vk::Format::R32G32B32A32_SFLOAT
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                offset: 64,
                format: vk::Format::R32G32B32_SFLOAT
            }
        ];
        let vertex_input_binding_descriptions = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: 12,
                input_rate: vk::VertexInputRate::VERTEX
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: 76,
                input_rate: vk::VertexInputRate::INSTANCE
            },
        ];
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_input_binding_descriptions);

        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_module.extent.width as f32,
            height: swapchain_module.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_module.extent
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A
            )
            .build()];
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachment_states);

        let descriptor_set_layout_binding = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()
        ];
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&descriptor_set_layout_binding);
        let descriptor_set_layouts = vec![unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)? }];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let graphics_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&input_assembly_state_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_create_info)
            .multisample_state(&multisample_state_create_info)
            .depth_stencil_state(&depth_stencil_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .build()];

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &graphics_pipeline_create_infos, None).expect("Error: Can not create pipeline!")
        }[0];

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        Ok(Self {
            pipeline,
            layout: pipeline_layout,
            descriptor_set_layouts
        })
    }

    unsafe fn cleanup(&self, device: &ash::Device) {
        for descriptor_set_layout in &self.descriptor_set_layouts {
            device.destroy_descriptor_set_layout(*descriptor_set_layout, None);
        }
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}

impl CommandPoolModule {
    fn initialize(device: &ash::Device, queue_module: &QueueModule) -> Result<Self, vk::Result> {
        let graphics_command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_module.graphics_queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let graphics_command_pool = unsafe { device.create_command_pool(&graphics_command_pool_create_info, None)? };

        let transfer_command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_module.transfer_queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let transfer_command_pool = unsafe { device.create_command_pool(&transfer_command_pool_create_info, None)? };

        Ok(Self {
            graphics_command_pool,
            transfer_command_pool
        })
    }

    unsafe fn cleanup(&self, device: &ash::Device) {
        device.destroy_command_pool(self.graphics_command_pool, None);
        device.destroy_command_pool(self.transfer_command_pool, None);
    }
}

impl BufferModule {
    fn new(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size_in_bytes: u64,
        buffer_usage: vk::BufferUsageFlags,
        memory_location: gpu_allocator::MemoryLocation
    ) -> Result<Self, vk::Result> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size_in_bytes)
            .usage(buffer_usage);
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };
        let buffer_memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator.allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Vertex buffer allocation.",
                requirements: buffer_memory_requirements,
                location: memory_location,
                linear: true
            }
        ).expect("Error: Could'nt allocate memory");
        
        Ok(Self {
            allocation: Some(allocation),
            buffer,
            size_in_bytes,
            buffer_usage,
            memory_location,
            bind_buffer: true
        })
    }

    fn fill<T: Sized>(
        &mut self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        data: &[T]
    ) -> Result<(), vk::Result> {
        let bytes_to_write_size = (data.len() * std::mem::size_of::<T>()) as u64;

        if bytes_to_write_size > self.size_in_bytes {
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            allocator.free(self.allocation.take().unwrap()).unwrap();

            let new_buffer = BufferModule::new(device, allocator, bytes_to_write_size, self.buffer_usage, self.memory_location)?;
            *self = new_buffer;
        }
        
        let allocation = self.allocation.take().unwrap();

        if self.bind_buffer {
            unsafe { device.bind_buffer_memory(self.buffer, allocation.memory(), allocation.offset())? };
            self.bind_buffer = false;
        }
        let data_pointer = allocation.mapped_ptr().expect("Error: Can't map to vertex buffer's memory for position!").as_ptr() as *mut T;
        unsafe { data_pointer.copy_from_nonoverlapping(data.as_ptr(), data.len()) };

        self.allocation = Some(allocation);

        Ok(())
    }
}

impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid handle!")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl<V, I> ModelModule<V, I> {
    pub fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }

    pub fn swap_by_handle(&mut self, handle1: usize, handle2: usize) -> Result<(), InvalidHandle> {
        if handle1 == handle2 {
            return Ok(());
        }

        if let (Some(&index1), Some(&index2)) = (
            self.handle_to_index.get(&handle1),
            self.handle_to_index.get(&handle2)
        ) {
            self.handles.swap(index1, index2);
            self.instances.swap(index1, index2);
            self.handle_to_index.insert(handle1, index2);
            self.handle_to_index.insert(handle2, index1);
            // self.handle_to_index.insert(index1, handle2);
            // self.handle_to_index.insert(index2, handle1);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn swap_by_index(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }

        let handle1 = self.handles[index1];
        let handle2 = self.handles[index2];
        self.handles.swap(index1, index2);
        self.instances.swap(index1, index2);
        self.handle_to_index.insert(handle1, index2);
        self.handle_to_index.insert(handle2, index1);
        // self.handle_to_index.insert(index1, handle2);
        // self.handle_to_index.insert(index2, handle1);
    }

    pub fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible_instance)
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible_instance {
                return Ok(());
            }

            self.swap_by_index(index, self.first_invisible_instance);
            self.first_invisible_instance += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn make_invisible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index >= self.first_invisible_instance {
                return Ok(());
            }

            self.first_invisible_instance -= 1;
            self.swap_by_index(index, self.first_invisible_instance);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn insert(&mut self, instance: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        let index = self.instances.len();
        self.instances.push(instance);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }

    pub fn insert_visibly(&mut self, instance: I) -> usize {
        let handle = self.insert(instance);
        self.make_visible(handle).ok();
        handle
    }

    pub fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible_instance {
                self.first_invisible_instance -= 1;
                self.swap_by_index(index, self.first_invisible_instance);
            }
            self.swap_by_index(index, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);
            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn update_vertex_buffer(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) -> Result<(), vk::Result> {
        if let Some(vertex_buffer_module) = &mut self.vertex_buffer_module {
            vertex_buffer_module.fill(
                device,
                allocator,
                &self.vertex_data
            )?;
            Ok(())
        } else {
            let size_in_bytes = (self.vertex_data.len() * std::mem::size_of::<V>()) as u64;
            let mut vertex_buffer_module = BufferModule::new(&device, allocator, size_in_bytes, vk::BufferUsageFlags::VERTEX_BUFFER, gpu_allocator::MemoryLocation::CpuToGpu)?;
            vertex_buffer_module.fill(
                device,
                allocator,
                &self.vertex_data
            )?;
            self.vertex_buffer_module = Some(vertex_buffer_module);
            Ok(())
        }
    }

    pub fn update_index_buffer(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) -> Result<(), vk::Result> {
        if let Some(index_buffer_module) = &mut self.index_buffer_module {
            index_buffer_module.fill(
                device,
                allocator,
                &self.index_data
            )?;
            Ok(())
        } else {
            let size_in_bytes = (self.index_data.len() * std::mem::size_of::<f32>()) as u64;
            let mut index_buffer_module = BufferModule::new(&device, allocator, size_in_bytes, vk::BufferUsageFlags::INDEX_BUFFER, gpu_allocator::MemoryLocation::CpuToGpu)?;
            index_buffer_module.fill(
                device,
                allocator,
                &self.index_data
            )?;
            self.index_buffer_module = Some(index_buffer_module);
            Ok(())
        }
    }

    pub fn update_instance_buffer(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) -> Result<(), vk::Result> {
        if let Some(instance_buffer_module) = &mut self.instance_buffer_module {
            instance_buffer_module.fill(
                device,
                allocator,
                &self.instances[0..self.first_invisible_instance]
            )?;
            Ok(())
        } else {
            let size_in_bytes = (self.vertex_data.len() * std::mem::size_of::<V>()) as u64;
            let mut instance_buffer_module = BufferModule::new(&device, allocator, size_in_bytes, vk::BufferUsageFlags::VERTEX_BUFFER, gpu_allocator::MemoryLocation::CpuToGpu)?;
            instance_buffer_module.fill(
                device,
                allocator,
                &self.instances[0..self.first_invisible_instance]
            )?;
            self.instance_buffer_module = Some(instance_buffer_module);
            Ok(())
        }
    }

    fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if let Some(vertex_buffer_module) = &self.vertex_buffer_module {
            if let Some(index_buffer_module) = &self.index_buffer_module {
                if let Some(instance_buffer_module) = &self.instance_buffer_module {
                    if self.first_invisible_instance > 0 {
                        unsafe {
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[vertex_buffer_module.buffer],
                                &[0]
                            );
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                1,
                                &[instance_buffer_module.buffer],
                                &[0]
                            );
                            device.cmd_bind_index_buffer(
                                command_buffer,
                                index_buffer_module.buffer,
                                0,
                                vk::IndexType::UINT32
                            );
                            device.cmd_draw_indexed(
                                command_buffer,
                                self.index_data.len() as u32,
                                self.first_invisible_instance as u32,
                                0,
                                0,
                                0
                            );
                        }
                    }
                }
            }
        }
    }
}

impl ModelModule<[f32; 3], InstanceData> {
    pub fn cube() -> Self {
        let lbf = [-1.0, 1.0, -1.0]; //lbf: left-bottom-front
        let lbb = [-1.0, 1.0, 1.0];
        let ltf = [-1.0, -1.0, -1.0];
        let ltb = [-1.0, -1.0, 1.0];
        let rbf = [1.0, 1.0, -1.0];
        let rbb = [1.0, 1.0, 1.0];
        let rtf = [1.0, -1.0, -1.0];
        let rtb = [1.0, -1.0, 1.0];
        Self {
            vertex_data: vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            index_data: vec![
                0, 1, 5, 0, 5, 4, //bottom
                2, 7, 3, 2, 6, 7, //top
                0, 6, 2, 0, 4, 6, //front
                1, 3, 7, 1, 7, 5, //back
                0, 2, 1, 1, 2, 3, //left
                4, 5, 6, 5, 7, 6 //right
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible_instance: 0,
            next_handle: 0,
            vertex_buffer_module: None,
            index_buffer_module: None,
            instance_buffer_module: None
        }
    }
}

impl Camera {
    pub fn builder() -> CameraBuilder {
        CameraBuilder {
            position: Vector3::new(0.0, -3.0, -3.0),
            view_direction: nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 1.0)),
            down_direction: nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, -1.0)),
            fovy: std::f32::consts::FRAC_PI_3,
            aspect: 800.0 / 600.0,
            near: 0.1,
            far: 100.0
        }
    }

    pub fn update_buffer(&self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator, buffer: &mut BufferModule) {
        let data: [[[f32; 4]; 4]; 2] = [self.view_matrix.into(), self.projection_matrix.into()];
        buffer.fill(device, allocator, &data).expect("Error: Can't fill uniform buffer.");
    }

    fn update_view_matrix(&mut self) {
        let right = nalgebra::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        self.view_matrix = nalgebra::Matrix4::new(
            right.x,
            right.y,
            right.z,
            -right.dot(&self.position),
            self.down_direction.x,
            self.down_direction.y,
            self.down_direction.z,
            -self.down_direction.dot(&self.position),
            self.view_direction.x,
            self.view_direction.y,
            self.view_direction.z,
            -self.view_direction.dot(&self.position),
            0.0,
            0.0,
            0.0,
            1.0
        );
    }

    fn update_projection_matrix(&mut self) {
        let d = 1.0 / (0.5 * self.fovy).tan();
        self.projection_matrix = Matrix4::new(
            d / self.aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            d,
            0.0,
            0.0,
            0.0,
            0.0,
            self.far / (self.far - self.near),
            -self.near * self.far / (self.far - self.near),
            0.0,
            0.0,
            1.0,
            0.0
        );
    }

    pub fn move_forward(&mut self, distance: f32) {
        self.position += distance * self.view_direction.as_ref();
        self.update_view_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn turn_right(&mut self, angle: f32) {
        let rotation = nalgebra::Rotation3::from_axis_angle(&self.down_direction, angle);
        self.view_direction = rotation * self.view_direction;
        self.update_view_matrix();
    }

    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let right = nalgebra::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let rotation = nalgebra::Rotation3::from_axis_angle(&right, angle);
        self.view_direction = rotation * self.view_direction;
        self.down_direction = rotation * self.down_direction;
        self.update_view_matrix();
    }

    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }
}

impl CameraBuilder {
    pub fn position(mut self, pos: Vector3<f32>) -> CameraBuilder {
        self.position = pos;
        self
    }

    pub fn fovy(mut self, fovy: f32) -> CameraBuilder {
        self.fovy = fovy.max(0.01).min(std::f32::consts::PI - 0.01);
        self
    }

    pub fn aspect(mut self, aspect: f32) -> CameraBuilder {
        self.aspect = aspect;
        self
    }

    pub fn near(mut self, near: f32) -> CameraBuilder {
        if near <= 0.0 {
            println!("Warning: Near plane is set to negative value: {}. This can be a bug!", near);
        }
        self.near = near;
        self
    }

    pub fn far(mut self, far: f32) -> CameraBuilder {
        if far <= 0.0 {
            println!("Warning: Far plane is set to negative value: {}. This can be a bug!", far);
        }
        self.far = far;
        self
    }

    pub fn view_direction(mut self, direction: Vector3<f32>) -> CameraBuilder {
        self.view_direction = nalgebra::Unit::new_normalize(direction);
        self
    }

    pub fn down_direction(mut self, direction: Vector3<f32>) -> CameraBuilder {
        self.down_direction = nalgebra::Unit::new_normalize(direction);
        self
    }

    pub fn build(self) -> Camera {
        if self.far < self.near {
            println!(
                "Warning: Far plane (at {}) is closer than near plane (at{}). This can be a bug!",
                self.far,
                self.near
            );
        }
        let mut camera = Camera {
            position: self.position,
            view_direction: self.view_direction,
            down_direction: nalgebra::Unit::new_normalize(
                self.down_direction.as_ref() -
                    self.down_direction.as_ref().dot(self.view_direction.as_ref()) * self.view_direction.as_ref()
            ),
            fovy: self.fovy,
            aspect: self.aspect,
            near: self.near,
            far: self.far,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity()
        };
        camera.update_projection_matrix();
        camera.update_view_matrix();
        camera
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        
    }
}