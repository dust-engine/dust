use crate::core::svo::mesher::Mesh;
use crate::device_info::DeviceInfo;
use crate::renderer::RenderContext;
use crate::shared_buffer::{SharedBuffer, StagingStateLayout};
use crate::swapchain::{RenderPassProvider, Swapchain, SwapchainConfig, SwapchainImage};
use crate::State;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::RenderPass;
use std::ffi::CStr;
use std::io::Cursor;
use std::sync::Arc;
use vk_mem as vma;
use crate::material_repo::TextureRepoUploadState;
use smallvec::SmallVec;

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
    context: Arc<RenderContext>,
    desc_pool: vk::DescriptorPool,
    pub storage_desc_set: vk::DescriptorSet,
    storage_desc_set_layout: vk::DescriptorSetLayout,
    pub uniform_desc_set: vk::DescriptorSet,
    uniform_desc_set_layout: vk::DescriptorSetLayout,
    pub frame_desc_set: vk::DescriptorSet,
    frame_desc_set_layout: vk::DescriptorSetLayout,
    pub depth_pipeline: vk::Pipeline,
    pub ray_pipeline: vk::Pipeline,
    pub compute_pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub depth_sampler: vk::Sampler,

    pub shared_buffer: SharedBuffer,
    mesh: Option<(vk::Buffer, u64, u32)>,
    // count, size, invocation
    limits: ([u32; 3], [u32; 3], u32),
}

unsafe fn create_pipelines(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> [vk::Pipeline; 3] {
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
    let compute_shader_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder()
                .code(
                    &ash::util::read_spv(&mut Cursor::new(
                        &include_bytes!(concat!(env!("OUT_DIR"), "/shaders/ray.comp.spv"))[..],
                    ))
                        .unwrap(),
                )
                .build(),
            None,
        )
        .unwrap();
    let depth_prepass_vertex_shader_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder()
                .code(
                    &ash::util::read_spv(&mut Cursor::new(
                        &include_bytes!(concat!(env!("OUT_DIR"), "/shaders/depth.vert.spv"))[..],
                    ))
                    .unwrap(),
                )
                .build(),
            None,
        )
        .unwrap();
    let mut pipelines: [vk::Pipeline; 3] = [vk::Pipeline::null(); 3];
    let result = device.fp_v1_0().create_graphics_pipelines(
        device.handle(),
        vk::PipelineCache::null(),
        2,
        [
            vk::GraphicsPipelineCreateInfo::builder()
                .flags(vk::PipelineCreateFlags::empty())
                .stages(&[vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                    .module(depth_prepass_vertex_shader_module)
                    .build()])
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
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                        .primitive_restart_enable(false)
                        .build(),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()])
                        .build(),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .depth_clamp_enable(false)
                        .rasterizer_discard_enable(false)
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::BACK)
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
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::builder()
                        .depth_test_enable(true)
                        .depth_compare_op(vk::CompareOp::LESS)
                        .depth_write_enable(true)
                        .depth_bounds_test_enable(false)
                        .stencil_test_enable(true)
                        .front(vk::StencilOpState {
                            fail_op: vk::StencilOp::ZERO,
                            pass_op: vk::StencilOp::REPLACE,
                            depth_fail_op: vk::StencilOp::REPLACE,
                            compare_op: vk::CompareOp::ALWAYS,
                            compare_mask: 0,
                            write_mask: 1,
                            reference: 1,
                        })
                        .build(),
                )
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder()
                        .logic_op_enable(false)
                        .attachments(&[])
                        .build(),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
                        .build(),
                )
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)
                .base_pipeline_handle(vk::Pipeline::null())
                .base_pipeline_index(-1)
                .build(),
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
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()])
                        .build(),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .depth_clamp_enable(false)
                        .rasterizer_discard_enable(false)
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::BACK)
                        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::builder()
                        .depth_test_enable(false)
                        .depth_write_enable(false)
                        .depth_bounds_test_enable(false)
                        .stencil_test_enable(true)
                        .front(vk::StencilOpState {
                            fail_op: vk::StencilOp::KEEP,
                            pass_op: vk::StencilOp::KEEP,
                            depth_fail_op: vk::StencilOp::KEEP,
                            compare_op: vk::CompareOp::EQUAL,
                            compare_mask: 1,
                            write_mask: 0,
                            reference: 1,
                        })
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
                            vk::DynamicState::STENCIL_REFERENCE,
                        ])
                        .build(),
                )
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(1)
                .base_pipeline_handle(vk::Pipeline::null())
                .base_pipeline_index(-1)
                .build(),
        ]
        .as_ptr(),
        std::ptr::null(),
        pipelines.as_mut_ptr(),
    );
    assert_eq!(result, vk::Result::SUCCESS);

    let result = device.fp_v1_0()
        .create_compute_pipelines(
            device.handle(),
            vk::PipelineCache::null(),
            1,
            [
                vk::ComputePipelineCreateInfo::builder()
                .stage(
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                        .module(compute_shader_module)
                        .build()
                )
                .layout(pipeline_layout)
                .build()
            ].as_ptr(),
            std::ptr::null(),
            pipelines.as_mut_ptr().add(2)
        );
    assert_eq!(result, vk::Result::SUCCESS);

    device.destroy_shader_module(vertex_shader_module, None);
    device.destroy_shader_module(fragment_shader_module, None);
    device.destroy_shader_module(compute_shader_module, None);
    device.destroy_shader_module(depth_prepass_vertex_shader_module, None);
    pipelines
}

impl RayTracer {
    pub unsafe fn new(
        context: Arc<RenderContext>,
        allocator: &vma::Allocator,
        swapchain: &Swapchain,
        info: &DeviceInfo,
        graphics_queue: vk::Queue,
        graphics_queue_family: u32,
    ) -> Self {
        let device = &context.device;
        let mut shared_buffer = SharedBuffer::new(
            context.clone(),
            allocator,
            &info.memory_properties,
            graphics_queue,
            graphics_queue_family,
        );
        shared_buffer.copy_vertex_index(&CUBE_POSITIONS, &CUBE_INDICES);
        let render_pass = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[
                        vk::AttachmentDescription::builder()
                            .format(swapchain.config.format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .build(),
                        vk::AttachmentDescription::builder()
                            .format(vk::Format::D32_SFLOAT_S8_UINT)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
                            .stencil_store_op(vk::AttachmentStoreOp::STORE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .build(),
                    ])
                    .subpasses(&[
                        vk::SubpassDescription::builder()
                            .depth_stencil_attachment(&vk::AttachmentReference {
                                attachment: 1,
                                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            })
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .build(),
                        vk::SubpassDescription::builder()
                            .color_attachments(&[vk::AttachmentReference {
                                attachment: 0,
                                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            }])
                            .depth_stencil_attachment(&vk::AttachmentReference {
                                attachment: 1,
                                layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                            })
                            .input_attachments(&[vk::AttachmentReference {
                                attachment: 1,
                                layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                            }])
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .build(),
                    ])
                    .dependencies(&[vk::SubpassDependency {
                        src_subpass: 0, // TODO
                        dst_subpass: 1,
                        src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    }])
                    .build(),
                None,
            )
            .unwrap();
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(3)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 3,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::INPUT_ATTACHMENT,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                        },
                    ]),
                None,
            )
            .unwrap();
        let uniform_desc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
                            )
                            .descriptor_count(1)
                            .build(), // Camera
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .build(), // Lights
                    ]),
                None,
            )
            .unwrap();
        let storage_desc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .build(), // Chunk Nodes
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .build(), // Regular Materials,
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .build(), // Colored Materials
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .build(), // Textures
                    ]),
                None,
            )
            .unwrap();
        let frame_desc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty())
                    .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .descriptor_count(1)
                        .build()]),
                None,
            )
            .unwrap();
        let mut desc_sets = [vk::DescriptorSet::null(); 3];
        let result = device
            .fp_v1_0()
            .allocate_descriptor_sets(
                device.handle(),
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[
                        uniform_desc_set_layout,
                        storage_desc_set_layout,
                        frame_desc_set_layout,
                    ])
                    .build(),
                &mut desc_sets[0],
            );
        assert_eq!(result, vk::Result::SUCCESS);
        let frame_desc_set = desc_sets[2];
        let storage_desc_set = desc_sets[1];
        let uniform_desc_set = desc_sets[0];
        drop(desc_sets);

        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[
                    uniform_desc_set_layout,
                    storage_desc_set_layout,
                    frame_desc_set_layout,
                    swapchain.images_desc_set_layout,
                ]),
                None,
            )
            .unwrap();
        let pipelines = create_pipelines(device, pipeline_layout, render_pass);

        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::builder()
                    .dst_set(uniform_desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo {
                        buffer: shared_buffer.buffer,
                        offset: 0,
                        range: 192,
                    }])
                    .build(), // Camera
                vk::WriteDescriptorSet::builder()
                    .dst_set(uniform_desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo {
                        buffer: shared_buffer.buffer,
                        offset: offset_of!(StagingStateLayout, sunlight) as u64,
                        range: 64,
                    }])
                    .build(), // Lights
            ],
            &[],
        );

        let depth_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .compare_enable(false)
                    .build(),
                None,
            )
            .unwrap();
        let raytracer = Self {
            context,
            pipeline_layout,
            ray_pipeline: pipelines[1],
            depth_pipeline: pipelines[0],
            compute_pipeline: pipelines[2],
            storage_desc_set,
            storage_desc_set_layout,
            uniform_desc_set,
            render_pass,
            shared_buffer,
            desc_pool,
            uniform_desc_set_layout,
            frame_desc_set,
            mesh: None,
            frame_desc_set_layout,
            depth_sampler,
            limits: (
                info.physical_device_properties.limits.max_compute_work_group_count,
                info.physical_device_properties.limits.max_compute_work_group_size,
                info.physical_device_properties.limits.max_compute_work_group_invocations,
            )
        };
        raytracer
    }
    pub fn update(&mut self, state: &State, aspect_ratio: f32) {
        self.shared_buffer.write_camera(
            state.camera_projection,
            state.camera_transform,
            aspect_ratio,
        );
        self.shared_buffer.write_light(state.sunlight);
    }
    pub unsafe fn bind_block_allocator_buffer(&mut self, block_allocator_buffer: vk::Buffer) {
        self.context.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.storage_desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo {
                        buffer: block_allocator_buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }])
                    .build(), // Nodes
            ],
            &[],
        );
    }

    pub fn bind_mesh(&mut self, mesh: &Mesh, allocator: &vma::Allocator) {
        let vertex_size = (mesh.vertices.len() * 3 * std::mem::size_of::<f32>()) as u64;
        let index_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
        let (buffer, _allocation, allocation_info) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER)
                    .size(vertex_size + index_size)
                    .build(),
                &vma::AllocationCreateInfo {
                    usage: vma::MemoryUsage::CpuToGpu,
                    flags: vma::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
            .unwrap();
        let ptr = allocation_info.get_mapped_data();
        unsafe {
            std::ptr::copy_nonoverlapping(
                mesh.vertices.as_ptr() as *const u8,
                ptr,
                vertex_size as usize,
            );

            std::ptr::copy_nonoverlapping(
                mesh.indices.as_ptr() as *const u8,
                ptr.add(vertex_size as usize),
                index_size as usize,
            );
        }

        self.mesh = Some((buffer, vertex_size, mesh.indices.len() as u32))
    }
    pub fn bind_render_target(&mut self, render_target: &mut Swapchain) {
        unsafe {
            self.context.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.frame_desc_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .image_info(&[vk::DescriptorImageInfo {
                            sampler: self.depth_sampler,
                            image_view: render_target.depth_image.view,
                            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        }])
                        .build(), // Camera
                ],
                &[],
            );
            render_target.bind_render_pass(self);
        }
    }
    pub fn bind_material_repo(&mut self, repo: &TextureRepoUploadState) {
        let buffer_info = [
            vk::DescriptorBufferInfo {
                buffer: repo.buffer,
                offset: repo.regular_material_range.start as u64,
                range: (repo.regular_material_range.end - repo.regular_material_range.start) as u64
            },
            vk::DescriptorBufferInfo {
                buffer: repo.buffer,
                offset: repo.colored_material_range.start as u64,
                range: (repo.colored_material_range.end - repo.colored_material_range.start) as u64
            }
        ];
        let image_info = [
            vk::DescriptorImageInfo {
                sampler: repo.sampler,
                image_view: repo.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }
        ];
        let mut writes: SmallVec<[vk::WriteDescriptorSet; 3]> = SmallVec::new();
        if !repo.regular_material_range.is_empty() {
            writes.push(vk::WriteDescriptorSet::builder()
                .dst_set(self.storage_desc_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info[0..1])
                .build()); // Regular Materials
        }
        if !repo.colored_material_range.is_empty() {
            writes.push(vk::WriteDescriptorSet::builder()
                .dst_set(self.storage_desc_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info[1..2])
                .build()); // Colored Materials
        }
        writes.push(vk::WriteDescriptorSet::builder()
                        .dst_set(self.storage_desc_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info)
                        .build()); // Textures
        unsafe {
            self.context.device.update_descriptor_sets(
                writes.as_slice(),
                &[],
            );
        }
    }
}

impl RenderPassProvider for RayTracer {
    unsafe fn record_command_buffer(
        &mut self,
        device: &ash::Device,
        swapchain_image: &SwapchainImage,
        config: &SwapchainConfig,
    ) {
        let command_buffer = swapchain_image.command_buffer;
        let framebuffer = swapchain_image.framebuffer;
        let max_compute_work_group_count = self.limits.0;
        let max_compute_work_group_size = self.limits.1;
        let default_local_workgroup_size: u32 = 4;
        if max_compute_work_group_size[0] < default_local_workgroup_size || max_compute_work_group_size[1] < default_local_workgroup_size {
            panic!("Max compute work group size too small. {} {}", max_compute_work_group_size[0], max_compute_work_group_size[1]);
        }
        fn div_round_up(a: u32, b: u32) -> u32 {
            (a + b - 1) / b
        }
        let work_group_count = [
            div_round_up(config.extent.width, default_local_workgroup_size),
            div_round_up(config.extent.height, default_local_workgroup_size),
        ];
        if work_group_count[0] > max_compute_work_group_count[0] || work_group_count[1] > max_compute_work_group_count[1] {
            panic!(
                "Max compute work group count too small, {} {}",
                max_compute_work_group_count[0],
                max_compute_work_group_count[1])
        }

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
        clear_values[0].color.float32 = [1.0, 1.0, 1.0, 1.0];
        clear_values[1].depth_stencil = vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        };
        self.shared_buffer.record_cmd_buffer_copy(command_buffer);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[
                self.uniform_desc_set,
                self.storage_desc_set,
                self.frame_desc_set,
                swapchain_image.desc_set,
            ],
            &[],
        );
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipeline,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[
                vk::ImageMemoryBarrier::builder()
                    .image(swapchain_image.image)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    })
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .build()
            ]
        );
        device.cmd_dispatch(
            command_buffer,
            work_group_count[0],
            work_group_count[1],
            1,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[
                vk::ImageMemoryBarrier::builder()
                    .image(swapchain_image.image)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    })
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .build()
            ]
        );
    }

    unsafe fn get_render_pass(&self) -> RenderPass {
        self.render_pass
    }
}

impl Drop for RayTracer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(self.uniform_desc_set_layout, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.storage_desc_set_layout, None);
            self.context
                .device
                .destroy_descriptor_pool(self.desc_pool, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.ray_pipeline, None);
            self.context
                .device
                .destroy_pipeline(self.depth_pipeline, None);
            self.context
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

pub mod systems {
    use super::RayTracer;
    
    use crate::render_resources::RenderResources;
    use crate::Renderer;
    use bevy::prelude::*;

    pub fn create_raytracer(
        mut commands: Commands,
        mesh: Res<Option<dust_core::svo::mesher::Mesh>>,
        renderer: Res<Renderer>,
        mut render_resources: ResMut<RenderResources>,
    ) {
        let raytracer = unsafe {
            let mut raytracer = RayTracer::new(
                renderer.context.clone(),
                &render_resources.allocator,
                &render_resources.swapchain,
                &renderer.info,
                renderer.graphics_queue,
                renderer.graphics_queue_family,
            );
            raytracer.bind_block_allocator_buffer(render_resources.block_allocator_buffer);
            raytracer.bind_material_repo(&render_resources.texture_repo);
            if let Some(mesh) = mesh.as_ref() {
                raytracer.bind_mesh(&mesh, &render_resources.allocator);
            }
            raytracer.bind_render_target(&mut render_resources.swapchain);
            raytracer
        };
        commands.insert_resource(raytracer);
    }
}
