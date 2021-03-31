use crate::shared_buffer::SharedBuffer;
use crate::swapchain::{SwapchainConfig, RenderPassProvider};
use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CStr;
use std::io::Cursor;
use ash::vk::RenderPass;

pub const CUBE_INDICES: [u16; 14] = [3, 7, 1, 5, 4, 7, 6, 3, 2, 1, 0, 4, 2, 6];
pub const CUBE_POSITIONS: [(f32, f32, f32); 8] = [
    (-1.0, -1.0, -1.0), // 0
    (-1.0, -1.0, 1.0),  // 1
    (-1.0, 1.0, -1.0),  // 2
    (-1.0, 1.0, 1.0),   // 3
    (1.0, -1.0, -1.0),  // 4
    (1.0, -1.0, 1.0),   // 5
    (1.0, 1.0, -1.0),   // 6
    (1.0, 1.0, 1.0),    // 7
];

pub struct RayTracer {
    device: ash::Device,
    pub storage_desc_set: vk::DescriptorSet,
    pub uniform_desc_set: vk::DescriptorSet,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,

    pub shared_buffer: SharedBuffer,
}

impl RayTracer {
    pub unsafe fn new(
        device: ash::Device,
        shared_buffer: SharedBuffer,
        node_pool_buffer: vk::Buffer,
        swapchain_config: &SwapchainConfig
    ) -> Self {
        shared_buffer.write_vertex_index(
            &CUBE_POSITIONS,
            &CUBE_INDICES,
        );
        let render_pass = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[vk::AttachmentDescription::builder()
                        .format(swapchain_config.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .build()])
                    .subpasses(&[vk::SubpassDescription::builder()
                        .color_attachments(&[vk::AttachmentReference {
                            attachment: 0,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }])
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .build()])
                    .build(),
                None,
            )
            .unwrap();
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(2)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                        },
                    ]),
                None,
            )
            .unwrap();
        let uniform_desc_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                    .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
                        .descriptor_count(1)
                        .build()]),
                None,
            )
            .unwrap();
        let storage_desc_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                    .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .descriptor_count(1)
                        .build()]),
                None,
            )
            .unwrap();
        let mut desc_sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[uniform_desc_layout, storage_desc_layout])
                    .build(),
            )
            .unwrap();
        assert_eq!(desc_sets.len(), 2);
        let storage_desc_set = desc_sets.pop().unwrap();
        let uniform_desc_set = desc_sets.pop().unwrap();

        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[uniform_desc_layout, storage_desc_layout]),
                None,
            )
            .unwrap();
        let vertex_shader_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code(
                        &ash::util::read_spv(&mut Cursor::new(
                            &include_bytes!(concat!(env!("OUT_DIR"), "/shaders/ray.vert.spv"))[..],
                        ))
                        .unwrap(),
                    )
                    .build(),
                None,
            )
            .unwrap();
        let fragment_shader_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code(
                        &ash::util::read_spv(&mut Cursor::new(
                            &include_bytes!(concat!(env!("OUT_DIR"), "/shaders/ray.frag.spv"))[..],
                        ))
                        .unwrap(),
                    )
                    .build(),
                None,
            )
            .unwrap();
        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::builder()
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
                    .vertex_input_state(
                        &vk::PipelineVertexInputStateCreateInfo::builder()
                            .vertex_attribute_descriptions(&[vk::VertexInputAttributeDescription {
                                location: 0,
                                binding: 0,
                                format: vk::Format::R32G32B32_SFLOAT,
                                offset: 0,
                            }])
                            .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                                binding: 0,
                                stride: std::mem::size_of::<[f32; 3]>() as u32,
                                input_rate: vk::VertexInputRate::VERTEX,
                            }])
                            .build(),
                    )
                    .input_assembly_state(
                        &vk::PipelineInputAssemblyStateCreateInfo::builder()
                            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
                            .primitive_restart_enable(false)
                            .build(),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::builder()
                            .viewports(&[vk::Viewport {
                                x: 0.0,
                                y: 0.0,
                                width: swapchain_config.extent.width as f32,
                                height: -(swapchain_config.extent.height as f32),
                                min_depth: 0.0,
                                max_depth: 1.0,
                            }])
                            .scissors(&[vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: swapchain_config.extent,
                            }])
                            .build(),
                    )
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::builder()
                            .depth_clamp_enable(false)
                            .rasterizer_discard_enable(false)
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(vk::CullModeFlags::NONE)
                            .front_face(vk::FrontFace::CLOCKWISE)
                            .depth_bias_enable(false)
                            .line_width(1.0)
                            .build(),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::builder()
                            .sample_shading_enable(false)
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                            .build(),
                    )
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::builder()
                            .logic_op_enable(false)
                            .attachments(&[vk::PipelineColorBlendAttachmentState::builder()
                                .color_write_mask(vk::ColorComponentFlags::all())
                                .blend_enable(false)
                                .build()])
                            .build(),
                    )
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::builder()
                            .dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ])
                            .build(),
                    )
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)
                    .base_pipeline_handle(vk::Pipeline::null())
                    .base_pipeline_index(-1)
                    .build()],
                None,
            )
            .unwrap()
            .pop()
            .unwrap();

        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::builder()
                    .dst_set(uniform_desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo {
                        buffer: shared_buffer.buffer,
                        offset: 128,
                        range: 128,
                    }])
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(storage_desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo {
                        buffer: node_pool_buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }])
                    .build(),
            ],
            &[],
        );
        Self {
            device,
            pipeline_layout,
            pipeline,
            storage_desc_set,
            uniform_desc_set,
            render_pass,
            shared_buffer
        }
    }
}

impl RenderPassProvider for RayTracer {
    unsafe fn record_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        config: &SwapchainConfig,
    ) {
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: if config.flip_y_requires_shift {
                    config.extent.height as f32
                } else {
                    0.0
                },
                width: config.extent.width as f32,
                height: -(config.extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: config.extent,
            }],
        );
        let mut clear_values = [vk::ClearValue::default(), vk::ClearValue::default()];
        clear_values[0].color.float32 = [0.0, 1.0, 0.0, 1.0];
        clear_values[1].depth_stencil = vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        };
        self.shared_buffer.record_cmd_buffer_copy_buffer(command_buffer);
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.uniform_desc_set, self.storage_desc_set],
            &[],
        );
        device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.shared_buffer.buffer],
            &[0],
        );
        device.cmd_bind_index_buffer(
            command_buffer,
            self.shared_buffer.buffer,
            std::mem::size_of_val(&CUBE_POSITIONS) as u64,
            vk::IndexType::UINT16,
        );
        device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::builder()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: config.extent,
                })
                .framebuffer(framebuffer)
                .render_pass(self.render_pass)
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
        device.cmd_draw_indexed(command_buffer, CUBE_INDICES.len() as u32, 1, 0, 0, 0);
        device.cmd_end_render_pass(command_buffer);
    }

    unsafe fn get_render_pass(&self) -> RenderPass {
        self.render_pass
    }
}
