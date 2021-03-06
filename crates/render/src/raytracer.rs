use crate::device_info::DeviceInfo;
use crate::material_repo::TextureRepoUploadState;
use crate::renderer::RenderContext;
use crate::shared_buffer::SharedBuffer;
use crate::swapchain::{Swapchain, SwapchainConfig, SwapchainImage};
use crate::utils::div_round_up;
use crate::State;
use ash::vk;
use smallvec::SmallVec;
use std::ffi::CStr;
use std::io::Cursor;
use std::sync::Arc;
use vk_mem as vma;

struct PushConstants {
    width: u32,
    height: u32,
    aspect_ratio: f32,
    terminal_pixel_size: f32,
}

pub struct RayTracer {
    context: Arc<RenderContext>,
    desc_pool: vk::DescriptorPool,
    pub storage_desc_set: vk::DescriptorSet,
    storage_desc_set_layout: vk::DescriptorSetLayout,
    pub uniform_desc_set: vk::DescriptorSet,
    uniform_desc_set_layout: vk::DescriptorSetLayout,
    pub compute_pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub shared_buffer: SharedBuffer,
    // count, size, invocation
    limits: ([u32; 3], [u32; 3], u32),
}

unsafe fn create_pipelines(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
) -> vk::Pipeline {
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
    let mut pipeline: vk::Pipeline = vk::Pipeline::null();

    let result = device.fp_v1_0().create_compute_pipelines(
        device.handle(),
        vk::PipelineCache::null(),
        1,
        [vk::ComputePipelineCreateInfo::builder()
            .stage(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                    .module(compute_shader_module)
                    .build(),
            )
            .layout(pipeline_layout)
            .build()]
        .as_ptr(),
        std::ptr::null(),
        &mut pipeline,
    );
    assert_eq!(result, vk::Result::SUCCESS);

    device.destroy_shader_module(compute_shader_module, None);
    pipeline
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
        let shared_buffer = SharedBuffer::new(
            context.clone(),
            allocator,
            &info.memory_properties,
            graphics_queue,
            graphics_queue_family,
        );
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
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_count(1)
                            .build(), // Camera
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
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_count(1)
                            .build(), // Chunk Nodes
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_count(1)
                            .build(), // Regular Materials,
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_count(1)
                            .build(), // Colored Materials
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_count(1)
                            .build(), // Textures
                    ]),
                None,
            )
            .unwrap();
        let mut desc_sets = [vk::DescriptorSet::null(); 2];
        let result = device.fp_v1_0().allocate_descriptor_sets(
            device.handle(),
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&[uniform_desc_set_layout, storage_desc_set_layout])
                .build(),
            &mut desc_sets[0],
        );
        assert_eq!(result, vk::Result::SUCCESS);
        let storage_desc_set = desc_sets[1];
        let uniform_desc_set = desc_sets[0];
        drop(desc_sets);

        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        uniform_desc_set_layout,
                        storage_desc_set_layout,
                        swapchain.images_desc_set_layout,
                    ])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: std::mem::size_of::<PushConstants>() as u32,
                    }]),
                None,
            )
            .unwrap();
        let pipeline = create_pipelines(device, pipeline_layout);

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
                        range: std::mem::size_of::<[f32; 16]>() as u64,
                    }])
                    .build(), // Camera
            ],
            &[],
        );

        let raytracer = Self {
            context,
            pipeline_layout,
            compute_pipeline: pipeline,
            storage_desc_set,
            storage_desc_set_layout,
            uniform_desc_set,
            shared_buffer,
            desc_pool,
            uniform_desc_set_layout,
            limits: (
                info.physical_device_properties
                    .limits
                    .max_compute_work_group_count,
                info.physical_device_properties
                    .limits
                    .max_compute_work_group_size,
                info.physical_device_properties
                    .limits
                    .max_compute_work_group_invocations,
            ),
        };
        raytracer
    }
    pub fn update(&mut self, state: &State) {
        self.shared_buffer
            .write_camera(state.camera_projection, state.camera_transform);
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
    pub fn bind_render_target(&mut self, render_target: &mut Swapchain) {
        unsafe {
            render_target.bind_render_pass(self);
        }
    }
    pub fn bind_material_repo(&mut self, repo: &TextureRepoUploadState) {
        let buffer_info = [
            vk::DescriptorBufferInfo {
                buffer: repo.buffer,
                offset: repo.regular_material_range.start as u64,
                range: (repo.regular_material_range.end - repo.regular_material_range.start) as u64,
            },
            vk::DescriptorBufferInfo {
                buffer: repo.buffer,
                offset: repo.colored_material_range.start as u64,
                range: (repo.colored_material_range.end - repo.colored_material_range.start) as u64,
            },
        ];
        let image_info = [vk::DescriptorImageInfo {
            sampler: repo.sampler,
            image_view: repo.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let mut writes: SmallVec<[vk::WriteDescriptorSet; 3]> = SmallVec::new();
        if !repo.regular_material_range.is_empty() {
            writes.push(
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.storage_desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info[0..1])
                    .build(),
            ); // Regular Materials
        }
        if !repo.colored_material_range.is_empty() {
            writes.push(
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.storage_desc_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info[1..2])
                    .build(),
            ); // Colored Materials
        }
        writes.push(
            vk::WriteDescriptorSet::builder()
                .dst_set(self.storage_desc_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)
                .build(),
        ); // Textures
        unsafe {
            self.context
                .device
                .update_descriptor_sets(writes.as_slice(), &[]);
        }
    }
    pub(crate) unsafe fn record_command_buffer(
        &mut self,
        device: &ash::Device,
        swapchain_image: &SwapchainImage,
        config: &SwapchainConfig,
    ) {
        let command_buffer = swapchain_image.command_buffer;
        let max_compute_work_group_count = self.limits.0;
        let max_compute_work_group_size = self.limits.1;
        let default_local_workgroup_size: u32 = 8;
        if max_compute_work_group_size[0] < default_local_workgroup_size
            || max_compute_work_group_size[1] < default_local_workgroup_size
        {
            panic!(
                "Max compute work group size too small. {} {}",
                max_compute_work_group_size[0], max_compute_work_group_size[1]
            );
        }
        let work_group_count = [
            div_round_up(config.extent.width, default_local_workgroup_size),
            div_round_up(config.extent.height, default_local_workgroup_size),
        ];
        if work_group_count[0] > max_compute_work_group_count[0]
            || work_group_count[1] > max_compute_work_group_count[1]
        {
            panic!(
                "Max compute work group count too small, {} {}",
                max_compute_work_group_count[0], max_compute_work_group_count[1]
            )
        }

        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: config.extent.width as f32,
                height: config.extent.height as f32,
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
        let performance_factor = 4.0;
        let push_constants = PushConstants {
            width: config.extent.width,
            height: config.extent.height,
            aspect_ratio: (config.extent.width as f32) / (config.extent.height as f32),
            terminal_pixel_size: (4.0 / (config.extent.width * config.extent.height) as f32).sqrt() * performance_factor,
        };
        device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(
                &push_constants as *const PushConstants as *const u8,
                std::mem::size_of_val(&push_constants),
            ),
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
            &[vk::ImageMemoryBarrier::builder()
                .image(swapchain_image.image)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .build()],
        );
        device.cmd_dispatch(command_buffer, work_group_count[0], work_group_count[1], 1);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[vk::ImageMemoryBarrier::builder()
                .image(swapchain_image.image)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build()],
        );
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
                .destroy_pipeline(self.compute_pipeline, None);
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
        _mesh: Res<Option<dust_core::svo::mesher::Mesh>>,
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
            raytracer.bind_render_target(&mut render_resources.swapchain);
            raytracer
        };
        commands.insert_resource(raytracer);
    }
}
