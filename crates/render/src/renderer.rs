use ash::version::{EntryV1_0, InstanceV1_0, InstanceV1_1, DeviceV1_0, DeviceV1_2};
use ash::vk;
use std::ffi::{CStr, CString};
use raw_window_handle::HasRawWindowHandle;
use std::sync::Arc;
use crate::swapchain::{Swapchain, SwapchainConfig};
use crate::State;
use crate::shared_buffer::SharedBuffer;
use std::io::Cursor;

const CUBE_INDICES: [u16; 14] = [
    3, 7, 1, 5, 4, 7, 6, 3, 2, 1, 0, 4, 2, 6,
];
const CUBE_POSITIONS: [(f32, f32, f32); 8] = [
    (0.0, 0.0, 0.0), // 0
    (0.0, 0.0, 1.0), // 1
    (0.0, 1.0, 0.0), // 2
    (0.0, 1.0, 1.0), // 3
    (1.0, 0.0, 0.0), // 4
    (1.0, 0.0, 1.0), // 5
    (1.0, 1.0, 0.0), // 6
    (1.0, 1.0, 1.0), // 7
];


pub struct Renderer {
    device: ash::Device,
    swapchain: Swapchain,
    shared_buffer: SharedBuffer,
    raytracer: RayTracer,
}

struct RayTracer {
    device: ash::Device,
    storage_desc_set: vk::DescriptorSet,
    uniform_desc_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

impl RayTracer {
    pub unsafe fn bind_shared_buffer(
        &self,
        shared_buffer: &SharedBuffer,
    ) {
        // TODO: temp code to test things out
        let storage_buffer = self.device
            .create_buffer(&vk::BufferCreateInfo::builder()
                .flags(vk::BufferCreateFlags::SPARSE_RESIDENCY | vk::BufferCreateFlags::SPARSE_BINDING)
                .size(4294967295)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .build(), None)
            .unwrap();

        self.device
            .update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.uniform_desc_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[
                            vk::DescriptorBufferInfo {
                                buffer: shared_buffer.buffer,
                                offset: 128,
                                range: 128
                            }
                        ])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.storage_desc_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[
                            vk::DescriptorBufferInfo {
                                buffer: storage_buffer,
                                offset: 0,
                                range: 4294967295
                            }
                        ])
                        .build(),
                ],
                &[]
            );
    }
    pub unsafe fn new(
        device: ash::Device,
        swapchain_config: &SwapchainConfig
    ) -> Self {
        let render_pass = device.create_render_pass(
            &vk::RenderPassCreateInfo::builder()
                .attachments(&[
                    vk::AttachmentDescription::builder()
                        .format(swapchain_config.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .build(),
                ])
                .subpasses(&[
                    vk::SubpassDescription::builder()
                        .color_attachments(&[
                            vk::AttachmentReference {
                                attachment: 0,
                                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                            }
                        ])
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .build()
                ])
                .build(),
            None
        ).unwrap();
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
        ).unwrap();
        let vertex_shader_module = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder()
                .code(&ash::util::read_spv(&mut Cursor::new(&include_bytes!("./ray.vert.spv")[..])).unwrap())
                .build(),
            None
        ).unwrap();
        let fragment_shader_module = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder()
                .code(&ash::util::read_spv(&mut Cursor::new(&include_bytes!("./ray.frag.spv")[..])).unwrap())
                .build(),
            None
        ).unwrap();
        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[
                    vk::GraphicsPipelineCreateInfo::builder()
                        .flags(vk::PipelineCreateFlags::empty())
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::builder()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                                .module(vertex_shader_module)
                                .build(),
                            vk::PipelineShaderStageCreateInfo::builder()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                                .module(fragment_shader_module)
                                .build(),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::builder()
                            .vertex_attribute_descriptions(&[
                                vk::VertexInputAttributeDescription {
                                    location: 0,
                                    binding: 0,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: 0
                                }
                            ])
                            .vertex_binding_descriptions(&[
                                vk::VertexInputBindingDescription {
                                    binding: 0,
                                    stride: std::mem::size_of::<[f32; 3]>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                }
                            ])
                            .build()
                        )
                        .input_assembly_state(&vk::PipelineInputAssemblyStateCreateInfo::builder()
                            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
                            .primitive_restart_enable(false)
                            .build()
                        )
                        .viewport_state(&vk::PipelineViewportStateCreateInfo::builder()
                            .viewports(&[
                                vk::Viewport {
                                    x: 0.0,
                                    y: 0.0,
                                    width: swapchain_config.extent.width as f32,
                                    height: swapchain_config.extent.height as f32,
                                    min_depth: 0.0,
                                    max_depth: 1.0
                                }
                            ])
                            .scissors(&[
                                vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: 0,
                                        y: 0
                                    },
                                    extent: swapchain_config.extent
                                }
                            ])
                            .build()
                        )
                        .rasterization_state(&vk::PipelineRasterizationStateCreateInfo::builder()
                            .depth_clamp_enable(false)
                            .rasterizer_discard_enable(false)
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(vk::CullModeFlags::NONE)
                            .front_face(vk::FrontFace::CLOCKWISE)
                            .depth_bias_enable(false)
                            .line_width(1.0)
                            .build()
                        )
                        .multisample_state(&vk::PipelineMultisampleStateCreateInfo::builder()
                            .sample_shading_enable(false)
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                            .build()
                        )
                        .color_blend_state(&vk::PipelineColorBlendStateCreateInfo::builder()
                            .logic_op_enable(false)
                            .attachments(&[
                                vk::PipelineColorBlendAttachmentState::builder()
                                    .blend_enable(false)
                                    .build()
                            ])
                            .build()
                        )
                        .dynamic_state(&vk::PipelineDynamicStateCreateInfo::builder()
                            .dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ])
                            .build()
                        )
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)
                        .base_pipeline_handle(vk::Pipeline::null())
                        .base_pipeline_index(-1)
                        .build(),
                ],
                None
            ).unwrap().pop().unwrap();
        Self {
            device,
            pipeline_layout,
            pipeline,
            storage_desc_set,
            uniform_desc_set,
            render_pass
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
                            .application_name(&CStr::from_bytes_with_nul_unchecked(b"Dust Application\0"))
                            .application_version(0)
                            .engine_name(&CStr::from_bytes_with_nul_unchecked(b"Dust Engine\0"))
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

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
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
                        })
                        .push_next(&mut vk::PhysicalDevice16BitStorageFeatures::builder()
                            .storage_buffer16_bit_access(true)
                            .build())
                        .push_next(&mut vk::PhysicalDevice8BitStorageFeatures::builder()
                            .uniform_and_storage_buffer8_bit_access(true)
                            .build()),
                    None,
                ).unwrap();
            let graphics_queue = device.get_device_queue(graphics_queue_family.0, 0);
            let transfer_binding_queue = device.get_device_queue(transfer_binding_queue_family.0, 0);

            let shared_buffer = SharedBuffer::new(
                device.clone(),
                graphics_queue,
                graphics_queue_family.0,
                &CUBE_POSITIONS,
                &CUBE_INDICES,
                &memory_properties,
            );
            let swapchain_config = Swapchain::get_config(
                physical_device,
                surface,
                surface_loader
            );
            let raytracer = RayTracer::new(
                device.clone(),
                &swapchain_config,
            );
            raytracer.bind_shared_buffer(&shared_buffer);
            let swapchain = Swapchain::new(
                &instance,
                device.clone(),
                raytracer.render_pass,
                surface,
                swapchain_config,
                graphics_queue_family.0,
                graphics_queue,
                |device, command_buffer| {
                    shared_buffer.record_cmd_buffer_copy_buffer(command_buffer);
                    device
                        .cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            raytracer.pipeline,
                        );
                    device
                        .cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            raytracer.pipeline_layout,
                            0,
                            &[
                                raytracer.uniform_desc_set,
                                raytracer.storage_desc_set,
                            ],
                            &[]
                        );
                    device
                        .cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[shared_buffer.buffer],
                            &[0]
                        );
                    device
                        .cmd_bind_index_buffer(
                            command_buffer,
                            shared_buffer.buffer,
                            std::mem::size_of_val(&CUBE_POSITIONS) as u64,
                            vk::IndexType::UINT16
                        );
                },
                |device, command_buffer| {
                    device.cmd_draw_indexed(
                        command_buffer,
                        CUBE_INDICES.len() as u32,
                        1,
                        0,
                        0,
                        0
                    );
                },
            );
            Self {
                device,
                swapchain,
                shared_buffer,
                raytracer,
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
