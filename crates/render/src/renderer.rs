use ash::version::{EntryV1_0, InstanceV1_0, InstanceV1_1, DeviceV1_0, DeviceV1_2};
use ash::vk;
use std::ffi::{CStr, CString};
use raw_window_handle::HasRawWindowHandle;
use std::sync::Arc;
use crate::swapchain::Swapchain;
use crate::State;


pub struct Renderer {
    device: ash::Device,
    swapchain: Swapchain
}

struct RayTracer {
}

impl RayTracer {
    pub unsafe fn new(
        device: ash::Device,
        swapchain: &Swapchain,
    ) -> Self {
        let desc_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(2)
                .pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 1
                    }
                ]),
            None
        ).unwrap();
        let uniform_desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                .bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
                        .descriptor_count(1)
                        .build()
                ]),
            None
        ).unwrap();
        let storage_desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                .bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .descriptor_count(1)
                        .build()
                ]),
            None
        ).unwrap();
        let mut desc_sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&[uniform_desc_layout, storage_desc_layout])
                .build()
        ).unwrap();
        assert_eq!(desc_sets.len(), 2);
        let storage_desc_set = desc_sets.pop().unwrap();
        let uniform_desc_set = desc_sets.pop().unwrap();



        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[uniform_desc_layout, storage_desc_layout]),
            None
        );
        Self {
        }
    }
}

impl Renderer {
    pub fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Self {
        unsafe {
            let entry = ash::Entry::new().unwrap();
            let mut extensions = ash_window::enumerate_required_extensions(window_handle).unwrap();
            extensions.push(ash::extensions::ext::DebugUtils::name());

            let available_instance_extensions = entry.enumerate_instance_extension_properties().unwrap();
            tracing::info!("Supported instance extensions: {:?}", available_instance_extensions);

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(&CString::new("Dust Application").unwrap())
                            .application_version(0)
                            .engine_name(&CString::new("Dust Engine").unwrap())
                            .engine_version(0)
                            .api_version(vk::make_version(1, 2, 0))
                    )
                    .enabled_extension_names(&extensions.iter().map(|&str| str.as_ptr()).collect::<Vec<_>>()),
                None
            ).unwrap();

            let surface = ash_window::create_surface(&entry, &instance, window_handle, None).unwrap();
            let mut physical_devices = instance
                .enumerate_physical_devices()
                .unwrap();
            let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
            let physical_device = physical_devices.pop().unwrap();


            let available_queue_family = instance.get_physical_device_queue_family_properties(physical_device);
            let graphics_queue_family = available_queue_family
                .iter()
                .enumerate()
                .find(|&(i, family)| {
                    family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && surface_loader.get_physical_device_surface_support(
                        physical_device,
                        i as u32,
                        surface,
                    ).unwrap_or(false)
                })
                .unwrap();
            let graphics_queue_family = (graphics_queue_family.0 as u32, graphics_queue_family.1);
            let transfer_binding_queue_family = available_queue_family
                .iter()
                .enumerate()
                .find(|&(i, family)| {
                    !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                        !family.queue_flags.contains(vk::QueueFlags::COMPUTE) &&
                        family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
                })
                .unwrap();
            let transfer_binding_queue_family = (transfer_binding_queue_family.0 as u32, transfer_binding_queue_family.1);


            let extension_names = [
                ash::extensions::khr::Swapchain::name()
            ];
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
                        .enabled_extension_names(&extension_names.map(|str| str.as_ptr()))
                        .enabled_features(&vk::PhysicalDeviceFeatures {
                            sparse_binding: 1,
                            sparse_residency_buffer: 1,
                            ..Default::default()
                        }),
                    None,
                ).unwrap();
            let graphics_queue = device.get_device_queue(graphics_queue_family.0, 0);
            let transfer_binding_queue = device.get_device_queue(transfer_binding_queue_family.0, 0);

            let swapchain = Swapchain::new(
                &instance,
                device.clone(),
                physical_device,
                surface,
                surface_loader,
                graphics_queue_family.0,
                graphics_queue,
            );
            Self {
                device,
                swapchain
            }
        }

    }

    pub fn update(&mut self, state: &State) {
        unsafe {
            //self.swapchain.render_frame();
            //self.device.cmd_bind_
        }
    }

    pub fn resize(&mut self) {

    }
}
