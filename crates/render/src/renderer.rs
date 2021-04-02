use crate::device_info::{DeviceInfo, Quirks};
use crate::raytracer::RayTracer;

use crate::swapchain::Swapchain;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;
use std::mem::ManuallyDrop;

use dust_core::svo::alloc::BlockAllocator;

pub struct Renderer {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub raytracer: Option<RayTracer>,
    pub physical_device: vk::PhysicalDevice,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::extensions::khr::Surface,
    pub quirks: Quirks,

    pub graphics_queue: vk::Queue,
    pub transfer_binding_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub transfer_binding_queue_family: u32,
    pub info: DeviceInfo,
    pub swapchain: ManuallyDrop<Swapchain>,
}

impl Renderer {
    /// Returns a Renderer and its associated BlockAllocator.
    /// Note that the two objects must be dropped at the same time.
    /// The application needs to ensure that when the Renderer was dropped,
    /// the BlockAllocator will not be used anymore.
    pub fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Renderer {
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
            let swapchain_config =
                Swapchain::get_config(physical_device, surface, &surface_loader, &quirks);
            let swapchain = Swapchain::new(
                &instance,
                device.clone(),
                surface,
                swapchain_config,
                graphics_queue_family,
                graphics_queue,
            );
            let renderer = Self {
                entry,
                device,
                raytracer: None,
                physical_device,
                surface,
                surface_loader,
                quirks,
                graphics_queue,
                transfer_binding_queue,
                graphics_queue_family,
                instance,
                transfer_binding_queue_family,
                info: device_info,
                swapchain: ManuallyDrop::new(swapchain),
            };

            renderer
        }
    }
    pub fn resize(&mut self) {
        // Assuming the swapchain format is still the same.
        unsafe {
            self.device.device_wait_idle().unwrap();
            let config = Swapchain::get_config(
                self.physical_device,
                self.surface,
                &self.surface_loader,
                &self.quirks,
            );
            self.swapchain.recreate(config);
            // swapchain.recreate is going to clear its command buffers
            // so we have to rebind the render pass here.
            if let Some(raytracer) = self.raytracer.as_mut() {
                self.swapchain.bind_render_pass(raytracer);
            }
        }
    }
    pub fn update(&mut self, state: &crate::State) {
        unsafe {
            if let Some(raytracer) = self.raytracer.as_mut() {
                let extent = self.swapchain.config.extent;
                raytracer.update(state, extent.width as f32 / extent.height as f32);
            }
            self.swapchain.render_frame();
        }
    }
    pub fn create_raytracer(&mut self) {
        unsafe {
            let raytracer = RayTracer::new(
                self.device.clone(),
                self.swapchain.config.format,
                &self.info,
                self.graphics_queue,
                self.graphics_queue_family,
            );
            self.raytracer = Some(raytracer);
        }
    }
    pub fn create_block_allocator(&mut self, block_size: u64) -> Box<dyn BlockAllocator> {
        unsafe {
            let node_pool_buffer: vk::Buffer;
            let device_type = self.info.physical_device_properties.device_type;
            let allocator: Box<dyn BlockAllocator> = match device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => {
                    let allocator = crate::block_alloc::DiscreteBlockAllocator::new(
                        self.device.clone(),
                        self.transfer_binding_queue,
                        self.transfer_binding_queue_family,
                        self.graphics_queue_family,
                        block_size,
                        self.info
                            .physical_device_properties
                            .limits
                            .max_storage_buffer_range as u64,
                        &self.info,
                    );
                    node_pool_buffer = allocator.device_buffer;
                    Box::new(allocator)
                }
                vk::PhysicalDeviceType::INTEGRATED_GPU => {
                    let allocator = crate::block_alloc::IntegratedBlockAllocator::new(
                        self.device.clone(),
                        self.transfer_binding_queue,
                        self.transfer_binding_queue_family,
                        self.graphics_queue_family,
                        block_size,
                        self.info
                            .physical_device_properties
                            .limits
                            .max_storage_buffer_range as u64,
                        &self.info,
                    );
                    node_pool_buffer = allocator.buffer;
                    Box::new(allocator)
                }
                _ => panic!("Unsupported GPU"),
            };
            if let Some(raytracer) = self.raytracer.as_mut() {
                raytracer.bind_block_allocator_buffer(node_pool_buffer);
                self.swapchain.bind_render_pass(raytracer);
            }
            allocator
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            if let Some(raytracer) = self.raytracer.take() {
                drop(raytracer);
            }
            ManuallyDrop::drop(&mut self.swapchain);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
