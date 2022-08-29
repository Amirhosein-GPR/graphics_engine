use std::error;
use ash::Entry;

use super::GraphicsAPI;

pub struct Vulkan {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Vulkan {
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
        let entry = unsafe { Entry::load()? };
        let instance = unsafe { entry.create_instance(&Default::default(), None)? };
        Ok(Vulkan {
            entry,
            instance
        })
    }
}

impl GraphicsAPI for Vulkan {}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}