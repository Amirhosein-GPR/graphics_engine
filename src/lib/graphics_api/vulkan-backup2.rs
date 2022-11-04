use std::error;
use ash::Entry;
use ash::vk;
use ash::prelude::VkResult;
use winit::platform::unix::WindowExtUnix;

use super::GraphicsAPI;

pub struct Vulkan {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_module: Option<DebugUtilsModule>,
    surface_module: Option<SurfaceModule>,
    device_module: Option<DeviceModule>
}

struct DebugUtilsModule {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT
}

struct SurfaceModule {
    xlib_loader: ash::extensions::khr::XlibSurface,
    surface: vk::SurfaceKHR,
    loader: ash::extensions::khr::Surface,
    window: winit::window::Window
}

struct QueueModule {
    graphics_queue: vk::Queue,
    graphics_queue_index: u32,
    transfer_queue: vk::Queue,
    transfer_queue_index: u32
}

struct SwapchainModule {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    image_views: Vec<vk::ImageView>
}

struct DeviceModule {
    device: ash::Device,
    queue_module: QueueModule,
    swapchain_module: SwapchainModule
}

impl Vulkan {
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
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
        .pfn_user_callback(Some(Vulkan::vulkan_debug_utils_callback));

        let layer_names = [std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let extension_names = [
            std::ffi::CString::from(ash::extensions::ext::DebugUtils::name()),
            std::ffi::CString::from(ash::extensions::khr::Surface::name()),
            std::ffi::CString::from(ash::extensions::khr::XlibSurface::name())
        ];

        let instance = Self::initialize_instance(&entry, &mut debug_utils_messenger_create_info, &layer_names, &extension_names)?;

        let debug_utils_module = DebugUtilsModule::initialize(&entry, &instance, &debug_utils_messenger_create_info)?;

        let (physical_device, physical_device_properties) = Self::initialize_physical_device_and_properties(&instance)?;

        let event_loop = winit::event_loop::EventLoop::new();
        let window = winit::window::Window::new(&event_loop)?;
        let surface_module = SurfaceModule::initialize(&entry, &instance, window)?;

        let device_module = DeviceModule::initialize(&instance, &surface_module, physical_device, &layer_names)?;

        Ok(Vulkan {
            entry,
            instance,
            debug_utils_module: Some(debug_utils_module),
            surface_module: Some(surface_module),
            device_module: Some(device_module)
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

    fn initialize_physical_device_and_properties(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties), vk::Result> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen = None;
        for physical_device in physical_devices {
            let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
            if physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                chosen = Some((physical_device, physical_device_properties));
                break;
            }
        }
        Ok(chosen.expect("Error: No proper graphics card found! (DISCRETE_GPU needed!)"))
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

impl GraphicsAPI for Vulkan {}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            std::mem::drop(self.device_module.take());
            std::mem::drop(self.surface_module.take());
            std::mem::drop(self.debug_utils_module.take());
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
    fn initialize(entry: &ash::Entry, instance: &ash::Instance, window: winit::window::Window) -> Result<Self, vk::Result> {
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
            loader: surface_loader,
            window
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
        let surface_formats = unsafe {
            surface_module.loader.get_physical_device_surface_formats(physical_device, surface_module.surface)?
        };

        let graphics_queue_family_indices = [queue_module.graphics_queue_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface_module.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count)
            )
            .image_format(surface_formats.first().unwrap().format)
            .image_color_space(surface_formats.first().unwrap().color_space)
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
                .format(surface_formats.first().unwrap().format)
                .subresource_range(*subresource_range);
            let image_views = unsafe { device.create_image_view(&image_view_create_info, None)? };
            swapchain_image_views.push(image_views);
        }

        Ok (Self {
            swapchain,
            loader: swapchain_loader,
            image_views: swapchain_image_views
        })
    }
}

impl Drop for SwapchainModule {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl DeviceModule {
    fn initialize(instance: &ash::Instance, surface_module: &SurfaceModule, physical_device: vk::PhysicalDevice, layer_names: &[std::ffi::CString]) -> Result<Self, vk::Result> {
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
            .enabled_layer_names(&layer_name_pointers);
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let queue_module = QueueModule::initialize(&device, graphics_queue_index, transfer_queue_index, &queue_family_properties);
        let swapchain_module = SwapchainModule::initialize(instance, physical_device, &device, surface_module, &queue_module)?;

        Ok(Self {
            device,
            queue_module,
            swapchain_module
        })
    }
}

impl Drop for DeviceModule {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.swapchain_module.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain_module.loader.destroy_swapchain(self.swapchain_module.swapchain, None);
            self.device.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        
    }
}