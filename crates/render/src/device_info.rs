use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;

pub struct DeviceInfo {
    pub supported_extensions: Vec<vk::ExtensionProperties>,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
}

#[derive(Default, Debug)]
pub struct Quirks {
    pub flip_y_requires_shift: bool,
}

impl DeviceInfo {
    pub unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let supported_extensions = entry.enumerate_instance_extension_properties().unwrap();
        let physical_device_properties = instance.get_physical_device_properties(physical_device);

        Self {
            supported_extensions,
            physical_device_properties,
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
    pub fn required_device_extensions_and_quirks(&self) -> (Vec<&CStr>, Quirks) {
        let mut requested_extensions: Vec<&CStr> = Vec::new();

        requested_extensions.push(ash::extensions::khr::Swapchain::name());

        let mut quirks = Quirks::default();
        if self.api_version() < vk::make_version(1, 1, 0) {
            if self.supports_extension(vk::KhrMaintenance1Fn::name()) {
                requested_extensions.push(vk::KhrMaintenance1Fn::name());
                quirks.flip_y_requires_shift = true;
            } else {
                requested_extensions.push(vk::AmdNegativeViewportHeightFn::name());
                quirks.flip_y_requires_shift = false;
            }
        } else {
            quirks.flip_y_requires_shift = true;
        }
        (requested_extensions, quirks)
    }
}
