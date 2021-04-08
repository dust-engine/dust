use crate::device_info::{DeviceInfo, Quirks};
use crate::raytracer::RayTracer;

use crate::swapchain::Swapchain;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use vk_mem as vma;

use crate::material::Material;
use crate::material_repo::TextureRepo;
use dust_core::svo::alloc::{BlockAllocator, BLOCK_SIZE};
use std::sync::Arc;

pub struct RenderContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,

    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::extensions::khr::Surface,
}
pub struct Renderer {
    pub context: Arc<RenderContext>,
    pub physical_device: vk::PhysicalDevice,
    pub quirks: Quirks,

    pub graphics_queue: vk::Queue,
    pub transfer_binding_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub transfer_binding_queue_family: u32,
    pub info: DeviceInfo,
}

impl Renderer {
    /// Returns a Renderer and its associated BlockAllocator.
    /// Note that the two objects must be dropped at the same time.
    /// The application needs to ensure that when the Renderer was dropped,
    /// the BlockAllocator will not be used anymore.
    pub unsafe fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Renderer {
        unsafe {
            let entry = ash::Entry::new().unwrap();
            let mut extensions = ash_window::enumerate_required_extensions(window_handle).unwrap();
            extensions.push(ash::extensions::ext::DebugUtils::name());

            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::builder()
                        .application_info(
                            &vk::ApplicationInfo::builder()
                                .application_name(&CStr::from_bytes_with_nul_unchecked(
                                    b"Dust Application\0",
                                ))
                                .application_version(0)
                                .engine_name(&CStr::from_bytes_with_nul_unchecked(b"Dust Engine\0"))
                                .engine_version(0)
                                .api_version(vk::make_version(1, 2, 0)),
                        )
                        .enabled_extension_names(
                            &extensions
                                .iter()
                                .map(|&str| str.as_ptr())
                                .collect::<Vec<_>>(),
                        ),
                    None,
                )
                .unwrap();

            let surface =
                ash_window::create_surface(&entry, &instance, window_handle, None).unwrap();
            let available_physical_devices: Vec<_> = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .map(|physical_device| {
                    let device_info = DeviceInfo::new(&entry, &instance, physical_device);
                    (physical_device, device_info)
                })
                .filter(|(physical_device, device_info)| {
                    device_info.features.sparse_residency_buffer != 0
                        && device_info.features.sparse_binding != 0
                })
                .collect();
            let (physical_device, device_info) = available_physical_devices
                .iter()
                .find(|(physical_device, device_info)| {
                    device_info.physical_device_properties.device_type
                        == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .or_else(|| {
                    available_physical_devices
                        .iter()
                        .find(|(physical_device, device_info)| {
                            device_info.physical_device_properties.device_type
                                == vk::PhysicalDeviceType::INTEGRATED_GPU
                        })
                })
                .expect("Unable to find a supported graphics card");
            let physical_device = *physical_device;
            let device_info = device_info.clone();
            println!(
                "Selected graphics card: {}",
                CStr::from_ptr(&device_info.physical_device_properties.device_name as *const _)
                    .to_string_lossy()
            );
            let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

            let available_queue_family =
                instance.get_physical_device_queue_family_properties(physical_device);
            let graphics_queue_family = available_queue_family
                .iter()
                .enumerate()
                .find(|&(i, family)| {
                    family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap_or(false)
                })
                .unwrap()
                .0 as u32;
            let transfer_binding_queue_family = available_queue_family
                .iter()
                .enumerate()
                .find(|&(_, family)| {
                    !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && family
                            .queue_flags
                            .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING)
                })
                .or_else(|| {
                    available_queue_family
                        .iter()
                        .enumerate()
                        .find(|&(_, family)| {
                            !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && family.queue_flags.contains(
                                    vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING,
                                )
                        })
                })
                .or_else(|| {
                    available_queue_family
                        .iter()
                        .enumerate()
                        .find(|&(_, family)| {
                            family
                                .queue_flags
                                .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING)
                        })
                })
                .unwrap()
                .0 as u32;

            let (extension_names, quirks) = device_info.required_device_extensions_and_quirks();
            let device = instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::builder()
                        .queue_create_infos(&[
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(graphics_queue_family)
                                .queue_priorities(&[1.0])
                                .build(),
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(transfer_binding_queue_family)
                                .queue_priorities(&[0.5])
                                .build(),
                        ])
                        .enabled_extension_names(
                            &extension_names
                                .into_iter()
                                .map(|str| str.as_ptr())
                                .collect::<Vec<_>>(),
                        )
                        .enabled_features(&vk::PhysicalDeviceFeatures {
                            sparse_binding: 1,
                            sparse_residency_buffer: 1,
                            ..Default::default()
                        })
                        .push_next(
                            &mut vk::PhysicalDevice16BitStorageFeatures::builder()
                                .storage_buffer16_bit_access(true)
                                .build(),
                        )
                        .push_next(
                            &mut vk::PhysicalDevice8BitStorageFeatures::builder()
                                .uniform_and_storage_buffer8_bit_access(true)
                                .build(),
                        ),
                    None,
                )
                .unwrap();
            let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
            let transfer_binding_queue = device.get_device_queue(transfer_binding_queue_family, 0);
            let context = RenderContext {
                entry,
                device,
                surface,
                instance,
                surface_loader
            };
            let renderer = Self {
                context: Arc::new(context),
                physical_device,
                quirks,
                graphics_queue,
                transfer_binding_queue,
                graphics_queue_family,
                transfer_binding_queue_family,
                info: device_info,
            };

            renderer
        }
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
