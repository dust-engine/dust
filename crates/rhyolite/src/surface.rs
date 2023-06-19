use crate::{Instance, PhysicalDevice};
use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use std::ops::Deref;
use std::sync::Arc;

pub struct SurfaceLoader {
    instance: Arc<Instance>,
    loader: khr::Surface,
}

impl SurfaceLoader {
    pub fn new(instance: Arc<Instance>) -> Self {
        let loader = khr::Surface::new(instance.entry(), &instance);
        SurfaceLoader { instance, loader }
    }
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl Deref for SurfaceLoader {
    type Target = khr::Surface;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

pub struct Surface {
    instance: Arc<Instance>,
    pub(crate) surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.surface
    }
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
    pub fn create(
        instance: Arc<Instance>,
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
        display_handle: &impl raw_window_handle::HasRawDisplayHandle,
    ) -> VkResult<Surface> {
        let surface = unsafe {
            ash_window::create_surface(
                instance.entry(),
                &instance,
                display_handle.raw_display_handle(),
                window_handle.raw_window_handle(),
                None,
            )?
        };
        Ok(Surface { instance, surface })
    }

    /// Query the basic capabilities of a surface, needed in order to create a swapchain
    pub fn get_capabilities(
        &self,
        pdevice: &PhysicalDevice,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        assert_eq!(pdevice.instance().handle(), self.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.instance
                .surface_loader()
                .get_physical_device_surface_capabilities(pdevice.raw(), self.surface)
        }
    }

    /// Determine whether a queue family of a physical device supports presentation to a given surface
    pub fn supports_queue_family(
        &self,
        pdevice: &PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<bool> {
        assert_eq!(pdevice.instance().handle(), self.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.instance
                .surface_loader()
                .get_physical_device_surface_support(
                    pdevice.raw(),
                    queue_family_index,
                    self.surface,
                )
        }
    }

    /// Query color formats supported by surface
    pub fn get_formats(&self, pdevice: &PhysicalDevice) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        assert_eq!(pdevice.instance().handle(), self.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.instance
                .surface_loader()
                .get_physical_device_surface_formats(pdevice.raw(), self.surface)
        }
    }

    pub fn pick_format(
        &self,
        pdevice: &PhysicalDevice,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<Option<vk::SurfaceFormatKHR>> {
        assert!(!usage.is_empty());
        let formats = self.get_formats(pdevice)?;
        let format = formats
            .into_iter()
            .filter(|f| {
                let properties = pdevice
                    .image_format_properties(&vk::PhysicalDeviceImageFormatInfo2 {
                        format: f.format,
                        ty: vk::ImageType::TYPE_2D, // We're gonna use this for presentation on a surface (screen) so the image should be 2D
                        tiling: vk::ImageTiling::OPTIMAL, // Of course you want optimal tiling for presentation right
                        usage,
                        flags: vk::ImageCreateFlags::empty(), // Nothing fancy should be needed for presentation here
                        ..Default::default()
                    })
                    .unwrap();
                properties.is_some()
            })
            .next();
        Ok(format)
    }

    /// Query color formats supported by surface
    pub fn get_present_modes(&self, pdevice: &PhysicalDevice) -> VkResult<Vec<vk::PresentModeKHR>> {
        assert_eq!(pdevice.instance().handle(), self.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.instance
                .surface_loader()
                .get_physical_device_surface_present_modes(pdevice.raw(), self.surface)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        tracing::info!(surface = ?self.surface, "drop surface");
        unsafe {
            self.instance
                .surface_loader()
                .destroy_surface(self.surface, None);
        }
    }
}
