use bevy_ecs::event::{Event, EventReader};
use bevy_ecs::system::lifetimeless::SRes;
use bevy_ecs::system::{Local, Resource, SystemParamItem};
use bevy_ecs::world::FromWorld;
use bevy_math::Mat4;
use bevy_transform::components::GlobalTransform;
pub use nrd_sys::*;
use rhyolite::ash::prelude::VkResult;
use rhyolite::ash::vk;
use rhyolite::debug::DebugObject;
use rhyolite::future::{
    run, use_shared_image, use_shared_state, use_state, Disposable, GPUCommandFuture, RenderData,
    RenderImage, RenderRes, SharedDeviceState, SharedDeviceStateHostContainer,
};
use rhyolite::macros::commands;
use rhyolite::{
    copy_buffer, BufferLike, HasDevice, ImageExt, ImageLike, ImageRequest, ImageView,
    ImageViewLike, ResidentImage,
};
use rhyolite_bevy::{Allocator, Device, StagingRingBuffer};
use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;

use crate::PinholeProjection;

#[derive(Resource)]
pub struct NRDPipeline {
    instance: Instance,
    pipelines: Vec<rhyolite::ComputePipeline>,
    transient_pool: Vec<TextureDesc>,
    permanent_pool: Vec<TextureDesc>,
    samplers: Vec<rhyolite::Sampler>,
    binding_offsets: SPIRVBindingOffsets,
    dimensions: (u16, u16),
}
const DENOISER_IDENTIFIER: Identifier = Identifier(0);

impl FromWorld for NRDPipeline {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let device = world.resource::<Device>();
        Self::new(device, 1920, 1080)
    }
}
impl NRDPipeline {
    pub fn new(device: &Arc<rhyolite::Device>, width: u16, height: u16) -> Self {
        let instance = Instance::new(&[DenoiserDesc {
            identifier: DENOISER_IDENTIFIER,
            denoiser: Denoiser::ReblurDiffuse,
            render_width: width,
            render_height: height,
        }])
        .unwrap();
        let library_desc = Instance::library_desc();
        let desc = instance.desc();
        assert_eq!(desc.resources_space_index, 0);
        assert_eq!(desc.constant_buffer_space_index, 0);
        assert_eq!(desc.constant_buffer_register_index, 0);
        println!("{:?}", desc.samplers());

        // Creating samplers
        let sampler_create_info = vk::SamplerCreateInfo {
            flags: vk::SamplerCreateFlags::empty(),
            mip_lod_bias: 0.0,
            max_anisotropy: 0.0,
            min_lod: 0.0,
            max_lod: 16.0,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ..Default::default()
        };
        let samplers = desc
            .samplers()
            .iter()
            .map(|sampler_desc| {
                let sampler = match sampler_desc {
                    Sampler::NearestClamp => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            mag_filter: vk::Filter::NEAREST,
                            min_filter: vk::Filter::NEAREST,
                            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                            ..sampler_create_info
                        },
                    ),
                    Sampler::NearestMirroredRepeat => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            mag_filter: vk::Filter::NEAREST,
                            min_filter: vk::Filter::NEAREST,
                            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                            ..sampler_create_info
                        },
                    ),
                    Sampler::LinearClamp => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            mag_filter: vk::Filter::LINEAR,
                            min_filter: vk::Filter::LINEAR,
                            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                            ..sampler_create_info
                        },
                    ),
                    Sampler::LinearMirroredRepeat => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            mag_filter: vk::Filter::LINEAR,
                            min_filter: vk::Filter::LINEAR,
                            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                            ..sampler_create_info
                        },
                    ),
                }
                .unwrap();
                sampler
            })
            .collect::<Vec<_>>();
        let sampler_bindings: Vec<_> = samplers
            .iter()
            .enumerate()
            .map(|(i, sampler)| vk::DescriptorSetLayoutBinding {
                binding: i as u32 + library_desc.spirv_binding_offsets.sampler_offset,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: sampler.raw(),
            })
            .collect();

        // Create pipelines, allocate descriptor sets
        let pipelines = desc
            .pipelines()
            .iter()
            .map(|pipeline_desc| {
                // TODO: Cache desc layout and pipeline layouts
                let desc_layout = rhyolite::descriptor::DescriptorSetLayout::new(
                    device.clone(),
                    &pipeline_desc
                        .resource_ranges()
                        .iter()
                        .flat_map(|resource_range| {
                            // texture bindings
                            let (offset, ty) = match resource_range.descriptor_type {
                                DescriptorType::Texture => (
                                    library_desc.spirv_binding_offsets.texture_offset,
                                    vk::DescriptorType::SAMPLED_IMAGE,
                                ),
                                DescriptorType::StorageTexture => (
                                    library_desc
                                        .spirv_binding_offsets
                                        .storage_texture_and_buffer_offset,
                                    vk::DescriptorType::STORAGE_IMAGE,
                                ),
                            };
                            (0..resource_range.descriptors_num).map(move |i| {
                                vk::DescriptorSetLayoutBinding {
                                    binding: resource_range.base_register_index + offset + i,
                                    descriptor_type: ty,
                                    descriptor_count: 1,
                                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                                    ..Default::default()
                                }
                            })
                        })
                        .chain(sampler_bindings.iter().cloned())
                        .chain(
                            // constant buffer (uniform buffer) binding
                            std::iter::once_with(|| vk::DescriptorSetLayoutBinding {
                                binding: library_desc.spirv_binding_offsets.constant_buffer_offset,
                                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                                stage_flags: vk::ShaderStageFlags::COMPUTE,
                                ..Default::default()
                            })
                            .take(
                                if pipeline_desc.has_constant_data {
                                    1
                                } else {
                                    0
                                },
                            ),
                        )
                        .collect::<Vec<_>>(),
                    vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
                )
                .unwrap();
                let pipeline_layout = rhyolite::PipelineLayout::new(
                    device.clone(),
                    vec![Arc::new(desc_layout)],
                    &[],
                    Default::default(),
                )
                .unwrap();
                let pipeline_layout = Arc::new(pipeline_layout);
                let shader_data: &[u8] = &pipeline_desc.compute_shader_spirv;
                let shader = rhyolite::shader::ShaderModule::new(
                    device.clone(),
                    bytemuck::cast_slice(shader_data),
                )
                .unwrap();
                let pipeline = rhyolite::ComputePipeline::create_with_shader_and_layout(
                    shader.specialized(
                        pipeline_desc.shader_entry_point_name(),
                        vk::ShaderStageFlags::COMPUTE,
                    ),
                    pipeline_layout.clone(),
                    vk::PipelineCreateFlags::empty(),
                    None,
                )
                .unwrap();
                pipeline
            })
            .collect();
        Self {
            pipelines,
            transient_pool: desc.transient_pool().iter().cloned().collect(),
            permanent_pool: desc.permanent_pool().iter().cloned().collect(),
            binding_offsets: library_desc.spirv_binding_offsets.clone(),

            // Retain a copy of the samplers.
            // Vulkan Docs: Only the sampler handles are copied; the sampler objects must not be destroyed before
            // the final use of the set layout and any descriptor pools and sets created using it.
            samplers,
            instance,
            dimensions: (width, height),
        }
    }
    pub fn resize(&mut self, width: u16, height: u16) {
        let instance = Instance::new(&[DenoiserDesc {
            identifier: DENOISER_IDENTIFIER,
            denoiser: Denoiser::ReblurDiffuse,
            render_width: width,
            render_height: height,
        }])
        .unwrap();
        let desc = instance.desc();
        assert_eq!(desc.pipelines().len(), self.pipelines.len());

        self.transient_pool = desc.transient_pool().iter().cloned().collect();
        self.permanent_pool = desc.permanent_pool().iter().cloned().collect();
        self.dimensions = (width, height);
        self.instance = instance;
    }
}

#[derive(Default)]
pub struct NDRPipelineLocalState {
    frame_index: u32,
    view_to_clip_matrix: [f32; 16],
    world_to_view_matrix: [f32; 16],
}
pub type NDRPipelineRenderParams = (
    SRes<Allocator>,
    SRes<StagingRingBuffer>,
    Option<SRes<ReblurSettings>>,
    SRes<bevy_time::Time>,
    Local<'static, NDRPipelineLocalState>,
    EventReader<'static, 'static, DenoiserEvent>,
);
impl NRDPipeline {
    pub fn render<'a, T: ImageViewLike + RenderData + 'a>(
        &'a mut self,
        params: SystemParamItem<'a, '_, NDRPipelineRenderParams>,
        in_motion: &'a mut RenderImage<T>,
        in_normal_roughness: &'a RenderImage<T>,
        in_viewz: &'a RenderImage<T>,
        in_radiance: &'a RenderImage<T>,
        out_radiance: &'a mut RenderImage<T>,
        camera: (&PinholeProjection, &GlobalTransform),
        dimensions: (u16, u16),
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let (allocator, staging_ring, reblur_settings, time, mut local_state, mut denoiser_events) =
            params;
        let reblur_settings = reblur_settings
            .as_ref()
            .map(|a| Cow::Borrowed(a.deref()))
            .unwrap_or_default();
        if self.dimensions != dimensions {
            self.resize(dimensions.0, dimensions.1);
        }
        let common_settings = nrd_sys::CommonSettings {
            view_to_clip_matrix: Mat4::perspective_infinite_reverse_rh(
                camera.0.fov,
                dimensions.0 as f32 / dimensions.1 as f32,
                camera.0.near,
            )
            .to_cols_array(),
            view_to_clip_matrix_prev: local_state.view_to_clip_matrix,
            world_to_view_matrix: camera.1.compute_matrix().inverse().to_cols_array(),
            world_to_view_matrix_prev: local_state.world_to_view_matrix,
            world_prev_to_world_matrix: nrd_sys::CommonSettings::default()
                .world_prev_to_world_matrix,
            motion_vector_scale: reblur_settings.common_settings.motion_vector_scale,
            camera_jitter: [0.0, 0.0],
            camera_jitter_prev: [0.0, 0.0], // TODO
            resolution_scale: [1.0, 1.0],
            resolution_scale_prev: [1.0, 1.0],
            time_delta_between_frames: time.delta().as_secs_f32() * 1000.0,
            denoising_range: reblur_settings.common_settings.denoising_range,
            disocclusion_threshold: reblur_settings.common_settings.disocclusion_threshold,
            disocclusion_threshold_alternate: reblur_settings
                .common_settings
                .disocclusion_threshold_alternate,
            split_screen: reblur_settings.common_settings.split_screen,
            debug: reblur_settings.common_settings.debug,
            input_subrect_origin: reblur_settings.common_settings.input_subrect_origin,
            frame_index: local_state.frame_index,
            accumulation_mode: {
                let mut should_reset = false;
                let mut should_clear = false;
                for event in denoiser_events.read() {
                    match event {
                        DenoiserEvent::Restart => should_reset = true,
                        DenoiserEvent::ClearAndRestart => {
                            should_clear = true;
                            should_reset = true;
                        }
                    }
                }
                match (should_reset, should_clear) {
                    (true, false) => AccumulationMode::Restart,
                    (true, true) => AccumulationMode::ClearAndRestart,
                    _ => AccumulationMode::Continue,
                }
            },
            is_motion_vector_in_world_space: reblur_settings
                .common_settings
                .is_motion_vector_in_world_space,
            is_history_confidence_available: reblur_settings
                .common_settings
                .is_history_confidence_available,
            is_disocclusion_threshold_mix_available: reblur_settings
                .common_settings
                .is_disocclusion_threshold_mix_available,
            is_base_color_metalness_available: reblur_settings
                .common_settings
                .is_base_color_metalness_available,
            enable_validation: reblur_settings.common_settings.enable_validation,
        };
        self.instance.set_common_settings(&common_settings).unwrap();
        self.instance
            .set_denoiser_settings(DENOISER_IDENTIFIER, &reblur_settings.reblur_settings)
            .unwrap();
        {
            // update local state
            local_state.frame_index += 1;
            local_state.view_to_clip_matrix = common_settings.view_to_clip_matrix;
            local_state.world_to_view_matrix = common_settings.world_to_view_matrix;
        }

        // An offset into the `resources` array. Increments inside the iterator

        commands! { move
            let dispatches = self
            .instance
            .get_compute_dispatches(&[DENOISER_IDENTIFIER])
            .unwrap();
            let mut constant_buffer_size: u32 = 0;
            let uniform_alignment = allocator.device().physical_device().properties().limits.min_uniform_buffer_offset_alignment as u32;
            for dispatch in dispatches.iter() {
                constant_buffer_size += dispatch.constant_buffer().len() as u32;
                constant_buffer_size = constant_buffer_size.next_multiple_of(uniform_alignment);
            }

            let mut const_buffer = use_shared_state(using!(), |_| {
                allocator.create_device_buffer_uninit(
                    constant_buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    uniform_alignment
                ).unwrap().with_name("NRD Constant Buffer").unwrap()
            }, |old| (old.size() as u32) < constant_buffer_size);
            let mut current_buffer_offset: usize = 0;
            let mut constant_buffer_staging_data = staging_ring.allocate(constant_buffer_size as u64).unwrap();
            for dispatch in dispatches.iter() {
                let new_buffer = dispatch.constant_buffer();
                if new_buffer.is_empty() {
                    continue;
                }
                constant_buffer_staging_data[current_buffer_offset .. current_buffer_offset + new_buffer.len()].copy_from_slice(new_buffer);
                current_buffer_offset += new_buffer.len();
                current_buffer_offset = current_buffer_offset.next_multiple_of(uniform_alignment as usize);
            }
            let constant_buffer_staging_data = RenderRes::new(constant_buffer_staging_data);
            copy_buffer(&constant_buffer_staging_data, &mut const_buffer).await;


            let transient_pool: &mut Vec<Option<SharedDeviceStateHostContainer<ImageView<ResidentImage>>>> = use_state(
                using!(),
                || std::iter::repeat_with(|| None).take(self.transient_pool.len()).collect(),
                |_| {},
            );
            let mut transient_images: Vec<Option<RenderImage<SharedDeviceState<ImageView<ResidentImage>>>>> = std::iter::repeat_with(|| None).take(self.transient_pool.len()).collect();
            let permanent_pool: &mut Vec<Option<SharedDeviceStateHostContainer<ImageView<ResidentImage>>>> = use_state(
                using!(),
                || std::iter::repeat_with(|| None).take(self.permanent_pool.len()).collect(),
                |_| {},
            );
            let mut permanent_images: Vec<Option<RenderImage<SharedDeviceState<ImageView<ResidentImage>>>>> = std::iter::repeat_with(|| None).take(self.permanent_pool.len()).collect();
            let mut sampled_image_writes: Vec<vk::DescriptorImageInfo> = Vec::new();
            let mut storage_image_writes: Vec<vk::DescriptorImageInfo> = Vec::new();
            current_buffer_offset = 0;
            for dispatch in dispatches.iter() {
                let pipeline = &self.pipelines[dispatch.pipeline_index as usize];
                let layout  = pipeline.raw_layout();
                let pipeline = pipeline.raw();

                enum ImgAccess {
                    Transient(u16),
                    Permanent(u16),
                    External(ResourceType),
                }
                let mut img_to_access = Vec::new();

                for resource in dispatch.resources() {
                    let image_view = match resource.ty {
                        ResourceType::TRANSIENT_POOL => {
                            let texture_desc = &self.transient_pool[resource.index_in_pool as usize];
                            let img = transient_images[resource.index_in_pool as usize].get_or_insert_with(|| {
                                use_shared_image(
                                    &mut transient_pool[resource.index_in_pool as usize],
                                    |_| {
                                        (
                                            create_image(texture_desc, &allocator, &format!("Transient Pool Image {}", resource.index_in_pool)).unwrap(),
                                            vk::ImageLayout::UNDEFINED
                                        )
                                    },
                                    |old| old.extent() != vk::Extent3D { width: texture_desc.width as u32, height: texture_desc.height as u32, depth: 1 },
                                )
                            });
                            let view = img.inner().raw_image_view();
                            img_to_access.push((ImgAccess::Transient(resource.index_in_pool), resource.state_needed));
                            view
                        },
                        ResourceType::PERMANENT_POOL => {
                            let texture_desc = &self.permanent_pool[resource.index_in_pool as usize];
                            let img = permanent_images[resource.index_in_pool as usize].get_or_insert_with(|| {
                                use_shared_image(
                                    &mut permanent_pool[resource.index_in_pool as usize],
                                    |_| {
                                        (
                                            create_image(texture_desc, &allocator, &format!("Permanent Pool Image {}", resource.index_in_pool)).unwrap(),
                                            vk::ImageLayout::UNDEFINED
                                        )
                                    },
                                    |old| old.extent() != vk::Extent3D { width: texture_desc.width as u32, height: texture_desc.height as u32, depth: 1 },
                                )
                            });
                            let view = img.inner().raw_image_view();
                            img_to_access.push((ImgAccess::Permanent(resource.index_in_pool), resource.state_needed));
                            view
                        },
                        _ => {
                            img_to_access.push((ImgAccess::External(resource.ty), resource.state_needed));
                            match resource.ty {
                                ResourceType::IN_MV => in_motion.inner().raw_image_view(),
                                ResourceType::IN_NORMAL_ROUGHNESS => in_normal_roughness.inner().raw_image_view(),
                                ResourceType::IN_VIEWZ => in_viewz.inner().raw_image_view(),
                                ResourceType::IN_DIFF_RADIANCE_HITDIST => in_radiance.inner().raw_image_view(),
                                ResourceType::OUT_DIFF_RADIANCE_HITDIST => out_radiance.inner().raw_image_view(),
                                _ => panic!()
                            }
                        }
                    };
                    match resource.state_needed {
                        DescriptorType::Texture => {
                            sampled_image_writes.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view,
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            });
                        }
                        DescriptorType::StorageTexture => {
                            storage_image_writes.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view,
                                image_layout: vk::ImageLayout::GENERAL,
                            });
                        }
                    };
                }


                let buffer_writes = vk::DescriptorBufferInfo {
                    buffer: const_buffer.inner().raw_buffer(),
                    offset: current_buffer_offset as u64,
                    range: dispatch.constant_buffer().len() as u64
                };
                let mut desc_writes = arrayvec::ArrayVec::<vk::WriteDescriptorSet, 3>::new();
                if !sampled_image_writes.is_empty() {
                    desc_writes.push(vk::WriteDescriptorSet {
                        dst_binding: self.binding_offsets.texture_offset,
                        dst_array_element: 0,
                        descriptor_count: sampled_image_writes.len() as u32,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: sampled_image_writes.as_ptr(),
                        ..vk::WriteDescriptorSet::default()
                    });
                }
                if !storage_image_writes.is_empty() {
                    desc_writes.push(vk::WriteDescriptorSet {
                        dst_binding: self.binding_offsets.storage_texture_and_buffer_offset,
                        dst_array_element: 0,
                        descriptor_count: storage_image_writes.len() as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: storage_image_writes.as_ptr(),
                        ..vk::WriteDescriptorSet::default()
                    });
                }
                if !dispatch.constant_buffer().is_empty() {
                    desc_writes.push(vk::WriteDescriptorSet {
                        dst_binding: self.binding_offsets.constant_buffer_offset,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_writes,
                        ..vk::WriteDescriptorSet::default()
                    });
                }

                run(|ctx, command_buffer| unsafe {
                    let device = ctx.device();
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    device.push_descriptor_loader().cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &desc_writes,
                    );
                    device.cmd_dispatch(
                        command_buffer,
                        dispatch.grid_width as u32,
                        dispatch.grid_height as u32,
                        1,
                    );
                }, |ctx| {
                    if !dispatch.constant_buffer().is_empty() {
                        ctx.read(&const_buffer, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::UNIFORM_READ);
                    }
                    for (img_ty, state_needed) in img_to_access.iter_mut() {
                        match img_ty {
                            ImgAccess::Transient(index_in_pool) => {
                                let img = transient_images[*index_in_pool as usize].as_mut().unwrap();
                                if matches!(state_needed, DescriptorType::StorageTexture) {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
                                    ctx.write_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
                                } else {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                                };
                            },
                            ImgAccess::Permanent(index_in_pool) => {
                                let img = permanent_images[*index_in_pool as usize].as_mut().unwrap();
                                if matches!(state_needed, DescriptorType::StorageTexture) {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
                                    ctx.write_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
                                } else {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                                };
                            },
                            ImgAccess::External(img_ty) => {
                                let img = match img_ty {
                                    ResourceType::IN_MV => in_motion,
                                    ResourceType::IN_NORMAL_ROUGHNESS => in_normal_roughness,
                                    ResourceType::IN_VIEWZ => in_viewz,
                                    ResourceType::IN_DIFF_RADIANCE_HITDIST => in_radiance,
                                    ResourceType::OUT_DIFF_RADIANCE_HITDIST => out_radiance,
                                    _ => panic!()
                                };

                                if matches!(state_needed, DescriptorType::StorageTexture) {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
                                    match img_ty {
                                        ResourceType::OUT_DIFF_RADIANCE_HITDIST => {
                                            ctx.write_image(out_radiance, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
                                        },
                                        ResourceType::IN_MV => {
                                            ctx.write_image(in_motion, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
                                        },
                                        _ => ()
                                    }
                                } else {
                                    ctx.read_image(img, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                                }
                            },
                        }
                    }
                }).await;

                sampled_image_writes.clear();
                storage_image_writes.clear();
                if !dispatch.constant_buffer().is_empty() {
                    current_buffer_offset += dispatch.constant_buffer().len();
                    current_buffer_offset = current_buffer_offset.next_multiple_of(uniform_alignment as usize);
                }
            }
            retain!((constant_buffer_staging_data, const_buffer, transient_images, permanent_images));
        }
    }
}

fn nrd_image_format_to_vk(ty: Format) -> vk::Format {
    match ty {
        Format::R8_UNORM => vk::Format::R8_UNORM,
        Format::R8_SNORM => vk::Format::R8_SNORM,
        Format::R8_UINT => vk::Format::R8_UINT,
        Format::R8_SINT => vk::Format::R8_SINT,
        Format::RG8_UNORM => vk::Format::R8G8_UNORM,
        Format::RG8_SNORM => vk::Format::R8G8_SNORM,
        Format::RG8_UINT => vk::Format::R8G8_UINT,
        Format::RG8_SINT => vk::Format::R8G8_SINT,
        Format::RGBA8_UNORM => vk::Format::R8G8B8A8_UNORM,
        Format::RGBA8_SNORM => vk::Format::R8G8B8A8_SNORM,
        Format::RGBA8_UINT => vk::Format::R8G8B8A8_UINT,
        Format::RGBA8_SINT => vk::Format::R8G8B8A8_SINT,
        Format::RGBA8_SRGB => vk::Format::R8G8B8A8_UNORM,
        Format::R16_UNORM => vk::Format::R16_UNORM,
        Format::R16_SNORM => vk::Format::R16_SNORM,
        Format::R16_UINT => vk::Format::R16_UINT,
        Format::R16_SINT => vk::Format::R16_SINT,
        Format::R16_SFLOAT => vk::Format::R16_SFLOAT,
        Format::RG16_UNORM => vk::Format::R16G16_UNORM,
        Format::RG16_SNORM => vk::Format::R16G16_SNORM,
        Format::RG16_UINT => vk::Format::R16G16_UINT,
        Format::RG16_SINT => vk::Format::R16G16_SINT,
        Format::RG16_SFLOAT => vk::Format::R16G16_SFLOAT,
        Format::RGBA16_UNORM => vk::Format::R16G16B16A16_UNORM,
        Format::RGBA16_SNORM => vk::Format::R16G16B16A16_SNORM,
        Format::RGBA16_UINT => vk::Format::R16G16B16A16_UINT,
        Format::RGBA16_SINT => vk::Format::R16G16B16A16_SINT,
        Format::RGBA16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
        Format::R32_UINT => vk::Format::R32_UINT,
        Format::R32_SINT => vk::Format::R32_SINT,
        Format::R32_SFLOAT => vk::Format::R32_SFLOAT,
        Format::RG32_UINT => vk::Format::R32G32_UINT,
        Format::RG32_SINT => vk::Format::R32G32_SINT,
        Format::RG32_SFLOAT => vk::Format::R32G32_SFLOAT,
        Format::RGB32_UINT => vk::Format::R32G32B32_UINT,
        Format::RGB32_SINT => vk::Format::R32G32B32_SINT,
        Format::RGB32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
        Format::RGBA32_UINT => vk::Format::R32G32B32A32_UINT,
        Format::RGBA32_SINT => vk::Format::R32G32B32A32_SINT,
        Format::RGBA32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
        Format::R10_G10_B10_A2_UNORM => vk::Format::A2B10G10R10_UNORM_PACK32,
        Format::R10_G10_B10_A2_UINT => vk::Format::A2B10G10R10_UINT_PACK32,
        Format::R11_G11_B10_UFLOAT => vk::Format::B10G11R11_UFLOAT_PACK32,
        Format::R9_G9_B9_E5_UFLOAT => vk::Format::E5B9G9R9_UFLOAT_PACK32,
    }
}

fn create_image(
    texture_desc: &TextureDesc,
    allocator: &Allocator,
    name: &str,
) -> VkResult<ImageView<ResidentImage>> {
    let image = allocator
        .create_device_image_uninit(&ImageRequest {
            image_type: vk::ImageType::TYPE_2D,
            format: nrd_image_format_to_vk(texture_desc.format),
            extent: vk::Extent3D {
                width: texture_desc.width as u32,
                height: texture_desc.height as u32,
                depth: 1,
            },
            mip_levels: texture_desc.mip_num as u32,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        })?
        .with_name(name)?
        .with_2d_view()?
        .with_name(&format!("{} View", name))?;
    Ok(image)
}

#[derive(Clone)]
pub struct CommonSettings {
    // used as "IN_MV * motionVectorScale" (use .z = 0 for 2D screen-space motion)
    pub motion_vector_scale: [f32; 3],

    // (units) > 0 - use TLAS or tracing range (max value = NRD_FP16_MAX / NRD_FP16_VIEWZ_SCALE - 1 = 524031)
    pub denoising_range: f32,

    // (normalized %) - if relative distance difference is greater than threshold, history gets reset (0.5-2.5% works well)
    pub disocclusion_threshold: f32,

    // (normalized %) - alternative disocclusion threshold, which is mixed to based on IN_DISOCCLUSION_THRESHOLD_MIX
    pub disocclusion_threshold_alternate: f32,

    // [0; 1] - enables "noisy input / denoised output" comparison
    pub split_screen: f32,

    // For internal needs
    pub debug: f32,

    // (pixels) - data rectangle origin in ALL input textures
    pub input_subrect_origin: [u32; 2],

    // If "true" IN_MV is 3D motion in world-space (0 should be everywhere if the scene is static),
    // otherwise it's 2D (+ optional Z delta) screen-space motion (0 should be everywhere if the camera doesn't move) (recommended value = true)
    pub is_motion_vector_in_world_space: bool,

    // If "true" IN_DIFF_CONFIDENCE and IN_SPEC_CONFIDENCE are available
    pub is_history_confidence_available: bool,

    // If "true" IN_DISOCCLUSION_THRESHOLD_MIX is available
    pub is_disocclusion_threshold_mix_available: bool,

    // If "true" IN_BASECOLOR_METALNESS is available
    pub is_base_color_metalness_available: bool,

    // Enables debug overlay in OUT_VALIDATION, requires "InstanceCreationDesc::allowValidation = true"
    pub enable_validation: bool,
}
impl Default for CommonSettings {
    fn default() -> Self {
        let default = nrd_sys::CommonSettings::default();
        Self {
            motion_vector_scale: [1.0, 1.0, 1.0],
            denoising_range: default.denoising_range,
            disocclusion_threshold: default.disocclusion_threshold,
            disocclusion_threshold_alternate: default.disocclusion_threshold_alternate,
            split_screen: default.split_screen,
            debug: default.debug,
            input_subrect_origin: default.input_subrect_origin,
            is_motion_vector_in_world_space: true,
            is_history_confidence_available: default.is_history_confidence_available,
            is_disocclusion_threshold_mix_available: default
                .is_disocclusion_threshold_mix_available,
            is_base_color_metalness_available: default.is_base_color_metalness_available,
            enable_validation: default.enable_validation,
        }
    }
}

#[derive(Event, Clone, Copy)]
pub enum DenoiserEvent {
    // Discards history and resets accumulation
    Restart,

    // Like RESTART, but additionally clears resources from potential garbage
    ClearAndRestart,
}

#[derive(Resource, Clone)]
pub struct ReblurSettings {
    pub common_settings: CommonSettings,
    pub reblur_settings: nrd_sys::ReblurSettings,
}

impl Default for ReblurSettings {
    fn default() -> Self {
        Self {
            common_settings: CommonSettings {
                ..Default::default()
            },
            reblur_settings: nrd_sys::ReblurSettings {
                antilag_settings: nrd_sys::ReblurAntilagSettings {
                    luminance_sigma_scale: 2.0,
                    luminance_antilag_power: 0.8,
                    hit_distance_antilag_power: 0.1,
                    ..Default::default()
                },
                ..Default::default()
            },
        }
    }
}
