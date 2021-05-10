use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub supported_extensions: Vec<vk::ExtensionProperties>,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub features: vk::PhysicalDeviceFeatures,
}


impl DeviceInfo {
    pub unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        Self {
            supported_extensions: entry.enumerate_instance_extension_properties().unwrap(),
            physical_device_properties: instance.get_physical_device_properties(physical_device),
            memory_properties: instance.get_physical_device_memory_properties(physical_device),
            features: instance.get_physical_device_features(physical_device),
        }
    }
    pub fn api_version(&self) -> u32 {
        self.physical_device_properties.api_version
    }
    fn supports_extension(&self, extension: &CStr) -> bool {
        self.supported_extensions
            .iter()
            .any(|ep| unsafe { CStr::from_ptr(ep.extension_name.as_ptr()) } == extension)
    }
    pub fn required_device_extensions(&self) -> Vec<&CStr> {
        let mut requested_extensions: Vec<&CStr> = Vec::new();

        requested_extensions.push(ash::extensions::khr::Swapchain::name());

        requested_extensions
    }
}
