use crate::device_info::{DeviceInfo, Quirks};
use crate::raytracer::RayTracer;
use crate::shared_buffer::SharedBuffer;
use crate::swapchain::Swapchain;
use crate::State;
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;
use svo::alloc::BlockAllocator;

pub struct Renderer {
    device: ash::Device,
    swapchain: Swapchain,
    raytracer: RayTracer,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    quirks: Quirks,
}

impl Renderer {
    pub fn new(
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
        block_size: u64,
    ) -> (Self, Box<dyn BlockAllocator>) {
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
            let mut physical_devices = instance.enumerate_physical_devices().unwrap();
            let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
            let physical_device = physical_devices.pop().unwrap();
            let device_info = DeviceInfo::new(&entry, &instance, physical_device);

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
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
                .unwrap();
            let graphics_queue_family = (graphics_queue_family.0 as u32, graphics_queue_family.1);
            let transfer_binding_queue_family = available_queue_family
                .iter()
                .enumerate()
                .find(|&(_, family)| {
                    !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
                })
                .unwrap();
            let transfer_binding_queue_family = (
                transfer_binding_queue_family.0 as u32,
                transfer_binding_queue_family.1,
            );

            let (extension_names, quirks) = device_info.required_device_extensions_and_quirks();
            let device = instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::builder()
                        .queue_create_infos(&[
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(graphics_queue_family.0)
                                .queue_priorities(&[1.0])
                                .build(),
                            vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(transfer_binding_queue_family.0)
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
            let graphics_queue = device.get_device_queue(graphics_queue_family.0, 0);
            let transfer_binding_queue =
                device.get_device_queue(transfer_binding_queue_family.0, 0);

            let node_pool_buffer: vk::Buffer;
            let device_type = device_info.physical_device_properties.device_type;
            let allocator: Box<dyn BlockAllocator> = match device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => {
                    let allocator = crate::block_alloc::DiscreteBlockAllocator::new(
                        device.clone(),
                        transfer_binding_queue,
                        transfer_binding_queue_family.0,
                        graphics_queue_family.0,
                        block_size,
                        device_info
                            .physical_device_properties
                            .limits
                            .max_storage_buffer_range as u64,
                        &device_info,
                    );
                    node_pool_buffer = allocator.device_buffer;
                    Box::new(allocator)
                }
                vk::PhysicalDeviceType::INTEGRATED_GPU => {
                    let allocator = crate::block_alloc::IntegratedBlockAllocator::new(
                        device.clone(),
                        transfer_binding_queue,
                        transfer_binding_queue_family.0,
                        graphics_queue_family.0,
                        block_size,
                        device_info
                            .physical_device_properties
                            .limits
                            .max_storage_buffer_range as u64,
                        &device_info,
                    );
                    node_pool_buffer = allocator.buffer;
                    Box::new(allocator)
                }
                _ => panic!("Unsupported GPU"),
            };

            let swapchain_config =
                Swapchain::get_config(physical_device, surface, &surface_loader, &quirks);
            let shared_buffer = SharedBuffer::new(
                device.clone(),
                &memory_properties,
                graphics_queue,
                graphics_queue_family.0,
            );
            let raytracer = RayTracer::new(
                device.clone(),
                shared_buffer,
                node_pool_buffer,
                swapchain_config.format,
            );

            let swapchain = Swapchain::new(
                &instance,
                device.clone(),
                surface,
                swapchain_config,
                graphics_queue_family.0,
                graphics_queue,
                &raytracer,
            );
            let renderer = Self {
                device,
                swapchain,
                raytracer,
                physical_device,
                surface,
                surface_loader,
                quirks,
            };
            (renderer, allocator)
        }
    }

    pub fn update(&mut self, state: &State) {
        unsafe {
            self.raytracer.shared_buffer.update_camera(
                state.camera_projection,
                state.camera_transform,
                self.swapchain.config.extent.width as f32
                    / self.swapchain.config.extent.height as f32,
            );
            self.swapchain.render_frame();
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
            self.swapchain.recreate(config, &self.raytracer);
        }
    }
}
