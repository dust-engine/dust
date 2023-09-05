use std::{collections::BTreeMap, ops::Deref};

use bevy_ecs::world::FromWorld;
use nrd::TextureDesc;
use rhyolite::ash::prelude::VkResult;
use rhyolite::ash::vk;
use rhyolite::future::{
    run, use_per_frame_state, use_shared_image, use_shared_state, use_state, Disposable,
    GPUCommandFuture, RenderData, RenderImage, RenderRes, SharedDeviceStateHostContainer,
    StageContext,
};
use rhyolite::macros::commands;
use rhyolite::smallvec::{smallvec, SmallVec};
use rhyolite::{
    copy_buffer, BufferLike, HasDevice, ImageExt, ImageLike, ImageRequest, ImageView,
    ImageViewLike, ResidentImage,
};
use rhyolite_bevy::{Allocator, Device, StagingRingBuffer};
use std::sync::Arc;

use rhyolite::descriptor::{DescriptorSetLayoutBindingInfo, DescriptorSetWrite};

pub struct ReblurPipeline {
    instance: nrd::Instance,
    pipelines: Vec<(rhyolite::ComputePipeline, SmallVec<[vk::DescriptorSet; 4]>)>,
    desc_pool: rhyolite::descriptor::DescriptorPool,
    transient_pool: Vec<TextureDesc>,
    permanent_pool: Vec<TextureDesc>,
    binding_offsets: nrd::SPIRVBindingOffsets,
}

const REBLUR_IDENTIFIER: nrd::Identifier = nrd::Identifier(0);

impl ReblurPipeline {
    pub fn new(device: &Arc<rhyolite::Device>) -> Self {
        let instance = nrd::Instance::new(&[nrd::DenoiserDesc {
            identifier: REBLUR_IDENTIFIER,
            denoiser: nrd::Denoiser::ReblurDiffuse,
            render_width: 0,
            render_height: 0,
        }])
        .unwrap();
        let library_desc = nrd::Instance::library_desc();
        let desc = instance.desc();
        assert_eq!(desc.resources_space_index, 0);
        assert_eq!(desc.constant_buffer_space_index, 0);
        assert_eq!(desc.constant_buffer_register_index, 0);

        // Creating samplers
        let sampler_create_info = vk::SamplerCreateInfo {
            flags: vk::SamplerCreateFlags::empty(),
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            mip_lod_bias: 0.0,
            max_anisotropy: 1.0,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ..Default::default()
        };
        let samplers = desc
            .samplers()
            .iter()
            .map(|sampler_desc| {
                let sampler = match sampler_desc {
                    nrd::Sampler::NearestClamp => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            mag_filter: vk::Filter::NEAREST,
                            min_filter: vk::Filter::NEAREST,
                            ..sampler_create_info
                        },
                    ),
                    nrd::Sampler::NearestMirroredRepeat => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            mag_filter: vk::Filter::NEAREST,
                            min_filter: vk::Filter::NEAREST,
                            ..sampler_create_info
                        },
                    ),
                    nrd::Sampler::LinearClamp => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            mag_filter: vk::Filter::LINEAR,
                            min_filter: vk::Filter::LINEAR,
                            ..sampler_create_info
                        },
                    ),
                    nrd::Sampler::LinearMirroredRepeat => rhyolite::Sampler::new(
                        device.clone(),
                        &vk::SamplerCreateInfo {
                            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                            mag_filter: vk::Filter::LINEAR,
                            min_filter: vk::Filter::LINEAR,
                            ..sampler_create_info
                        },
                    ),
                }
                .unwrap();
                Arc::new(sampler)
            })
            .collect::<Vec<_>>();
        let sampler_bindings: Vec<_> = samplers
            .iter()
            .enumerate()
            .map(|(i, sampler)| DescriptorSetLayoutBindingInfo {
                binding: i as u32 + library_desc.spirv_binding_offsets.sampler_offset,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                immutable_samplers: smallvec![sampler.clone()],
            })
            .collect();

        // Creating descriptor pool
        let mut desc_pool = rhyolite::descriptor::DescriptorPool::new(
            device.clone(),
            desc.descriptor_pool_desc.sets_max_num,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: desc.descriptor_pool_desc.textures_max_num,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: desc.descriptor_pool_desc.constant_buffers_max_num,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: desc.descriptor_pool_desc.storage_textures_max_num,
                },
            ],
            vk::DescriptorPoolCreateFlags::empty(),
        )
        .unwrap();

        // Create pipelines, allocate descriptor sets
        let pipelines = desc
            .pipelines()
            .iter()
            .map(|pipeline_desc| {
                // TODO: Cache desc layout and pipeline layouts
                let desc_layout = rhyolite::descriptor::DescriptorSetLayout::new(
                    device.clone(),
                    pipeline_desc
                        .resource_ranges()
                        .iter()
                        .map(|resource_range| {
                            // texture bindings
                            let (offset, ty) = match resource_range.descriptor_type {
                                nrd::DescriptorType::Texture => (
                                    library_desc.spirv_binding_offsets.texture_offset,
                                    vk::DescriptorType::SAMPLED_IMAGE,
                                ),
                                nrd::DescriptorType::StorageTexture => (
                                    library_desc
                                        .spirv_binding_offsets
                                        .storage_texture_and_buffer_offset,
                                    vk::DescriptorType::STORAGE_IMAGE,
                                ),
                            };
                            DescriptorSetLayoutBindingInfo {
                                binding: resource_range.base_register_index + offset,
                                descriptor_type: ty,
                                descriptor_count: resource_range.descriptors_num,
                                stage_flags: vk::ShaderStageFlags::COMPUTE,
                                immutable_samplers: Default::default(),
                            }
                        })
                        .chain(sampler_bindings.iter().cloned())
                        .chain(
                            // constant buffer (uniform buffer) binding
                            std::iter::once_with(|| DescriptorSetLayoutBindingInfo {
                                binding: library_desc.spirv_binding_offsets.constant_buffer_offset,
                                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                                stage_flags: vk::ShaderStageFlags::COMPUTE,
                                immutable_samplers: Default::default(),
                            })
                            .take(
                                if pipeline_desc.has_constant_data {
                                    1
                                } else {
                                    0
                                },
                            ),
                        )
                        .collect(),
                    Default::default(),
                )
                .unwrap();
                let desc_set: SmallVec<[_; 4]> = (0..pipeline_desc.max_repeat_num)
                    .map(|_| desc_pool.allocate_for_set_layout(&desc_layout).unwrap())
                    .collect();
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
                (pipeline, desc_set)
            })
            .collect();
        Self {
            pipelines,
            desc_pool,
            transient_pool: desc.transient_pool().iter().cloned().collect(),
            permanent_pool: desc.permanent_pool().iter().cloned().collect(),
            binding_offsets: library_desc.spirv_binding_offsets.clone(),
            instance,
        }
    }
}

impl ReblurPipeline {
    pub fn render<'a>(
        &'a mut self,
        allocator: &'a Allocator,
        staging_ring: &'a StagingRingBuffer,
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        if false {
            *self = Self::new(allocator.device());
        }
        let dispatches = self
            .instance
            .get_compute_dispatches(&[REBLUR_IDENTIFIER])
            .unwrap();

        // An offset into the `resources` array. Increments inside the iterator

        commands! {
            let mut constant_buffer_size: u32 = 0;
            const UNIFORM_ALIGNMENT: u32 = 4 * 4;
            for dispatch in dispatches.iter() {
                constant_buffer_size += dispatch.constant_buffer().len() as u32;
                constant_buffer_size = constant_buffer_size.next_multiple_of(UNIFORM_ALIGNMENT);
            }

            let mut const_buffer = use_shared_state(using!(), |_| {
                allocator.create_device_buffer_uninit(
                    constant_buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    UNIFORM_ALIGNMENT
                ).unwrap()
            }, |old| (old.size() as u32) < constant_buffer_size);
            let mut current_buffer_offset: usize = 0;
            let mut constant_buffer_staging_data = staging_ring.allocate(constant_buffer_size as u64).unwrap();
            for dispatch in dispatches.iter() {
                let new_buffer = dispatch.constant_buffer();
                constant_buffer_staging_data[current_buffer_offset .. current_buffer_offset + new_buffer.len()].copy_from_slice(new_buffer);
                current_buffer_offset += new_buffer.len();
                current_buffer_offset = current_buffer_offset.next_multiple_of(UNIFORM_ALIGNMENT as usize);
            }
            let constant_buffer_staging_data = RenderRes::new(constant_buffer_staging_data);
            copy_buffer(&constant_buffer_staging_data, &mut const_buffer).await;


            let transient_images: &mut Vec<Option<SharedDeviceStateHostContainer<ImageView<ResidentImage>>>> = use_state(
                using!(),
                || std::iter::repeat_with(|| None).take(self.transient_pool.len()).collect(),
                |_| {},
            );
            let permanent_images: &mut Vec<Option<SharedDeviceStateHostContainer<ImageView<ResidentImage>>>> = use_state(
                using!(),
                || std::iter::repeat_with(|| None).take(self.permanent_pool.len()).collect(),
                |_| {},
            );
            let mut pipeline_desc_sets = vec![0_u32; self.pipelines.len()].into_boxed_slice();
            let mut sampled_image_writes: Vec<vk::DescriptorImageInfo> = Vec::new();
            let mut storage_image_writes: Vec<vk::DescriptorImageInfo> = Vec::new();
            current_buffer_offset = 0;
            for dispatch in dispatches.iter() {
                let (pipeline, desc_set) = &self.pipelines[dispatch.pipeline_index as usize];
                let layout  = pipeline.raw_layout();
                let pipeline = pipeline.raw();
                let desc_set_index = &mut pipeline_desc_sets[dispatch.pipeline_index as usize];
                let desc_set = desc_set[*desc_set_index as usize];
                *desc_set_index += 1;

                current_buffer_offset += dispatch.constant_buffer().len();
                current_buffer_offset = current_buffer_offset.next_multiple_of(UNIFORM_ALIGNMENT as usize);


                let mut img_to_access = Vec::new();
                let mut img_to_access_readwrite = Vec::new();
                for resource in dispatch.resources() {
                    let has_write = matches!(resource.state_needed, nrd::DescriptorType::StorageTexture);
                    let image_view = match resource.ty {
                        nrd::ResourceType::TRANSIENT_POOL => {
                            let texture_desc = &self.transient_pool[resource.index_in_pool as usize];
                            let img = use_shared_image(
                                &mut transient_images[resource.index_in_pool as usize],
                                |_| {
                                    (
                                        create_image(texture_desc, allocator).unwrap(),
                                        vk::ImageLayout::UNDEFINED
                                    )
                                },
                                |old| false // TODO: resize when needed
                            );
                            let view = img.inner().raw_image_view();
                            img_to_access.push(img);
                            img_to_access_readwrite.push(has_write);
                            view
                        },
                        nrd::ResourceType::PERMANENT_POOL => {
                            let texture_desc = &self.permanent_pool[resource.index_in_pool as usize];
                            let img = use_shared_image(
                                &mut permanent_images[resource.index_in_pool as usize],
                                |_| {
                                    (
                                        create_image(texture_desc, allocator).unwrap(),
                                        vk::ImageLayout::UNDEFINED
                                    )
                                },
                                |_| false // TODO: resize when needed
                            );
                            let view = img.inner().raw_image_view();
                            img_to_access.push(img);
                            img_to_access_readwrite.push(has_write);
                            view
                        },
                        _ => todo!()
                    };
                    match resource.state_needed {
                        nrd::DescriptorType::Texture => {
                            sampled_image_writes.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view,
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            });
                        }
                        nrd::DescriptorType::StorageTexture => {
                            storage_image_writes.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view,
                                image_layout: vk::ImageLayout::GENERAL,
                            });
                        }
                    };
                }

                let desc_writes = (
                    DescriptorSetWrite::sampled_images(
                        desc_set,
                        self.binding_offsets.sampler_offset,
                        0,
                        &sampled_image_writes,
                    ),
                    DescriptorSetWrite::storage_images(
                        desc_set,
                        self.binding_offsets.storage_texture_and_buffer_offset,
                        0,
                        &storage_image_writes,
                    ),
                );
                if dispatch.constant_buffer().is_empty() {
                    self.desc_pool.device().write_descriptor_sets(desc_writes.into());
                } else {
                    self.desc_pool.device().write_descriptor_sets([
                        DescriptorSetWrite::uniform_buffers(
                            desc_set,
                            self.binding_offsets.constant_buffer_offset,
                            0,
                            &[],
                            false
                        ),
                        desc_writes.0,
                        desc_writes.1
                    ]);
                }
                sampled_image_writes.clear();
                storage_image_writes.clear();

                run(|ctx, command_buffer| unsafe {
                    let device = ctx.device();
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[desc_set],
                        &[]
                    );
                    device.cmd_dispatch(
                        command_buffer,
                        dispatch.grid_width as u32,
                        dispatch.grid_height as u32,
                        1,
                    );
                }, |ctx| {
                    fn record_accesses<T: RenderData + ImageLike>(res: &mut RenderImage<T>, has_write: bool, ctx: &mut StageContext) {
                        if has_write {
                            ctx.read_image(res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
                            ctx.write_image(res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
                        } else {
                            ctx.read_image(res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                        }
                    }
                    for (img, has_write) in img_to_access.iter_mut().zip(img_to_access_readwrite.iter()) {
                        record_accesses(img, *has_write, ctx);
                    }
                }).await;
                retain!(img_to_access);
            }
            retain!((constant_buffer_staging_data, const_buffer));
        }
    }
}

fn nrd_image_format_to_vk(ty: nrd::Format) -> vk::Format {
    match ty {
        nrd::Format::R8_UNORM => vk::Format::R8_UNORM,
        nrd::Format::R8_SNORM => vk::Format::R8_SNORM,
        nrd::Format::R8_UINT => vk::Format::R8_UINT,
        nrd::Format::R8_SINT => vk::Format::R8_SINT,
        nrd::Format::RG8_UNORM => vk::Format::R8G8_UNORM,
        nrd::Format::RG8_SNORM => vk::Format::R8G8_SNORM,
        nrd::Format::RG8_UINT => vk::Format::R8G8_UINT,
        nrd::Format::RG8_SINT => vk::Format::R8G8_SINT,
        nrd::Format::RGBA8_UNORM => vk::Format::R8G8B8A8_UNORM,
        nrd::Format::RGBA8_SNORM => vk::Format::R8G8B8A8_SNORM,
        nrd::Format::RGBA8_UINT => vk::Format::R8G8B8A8_UINT,
        nrd::Format::RGBA8_SINT => vk::Format::R8G8B8A8_SINT,
        nrd::Format::RGBA8_SRGB => vk::Format::R8G8B8A8_SRGB,
        nrd::Format::R16_UNORM => vk::Format::R16_UNORM,
        nrd::Format::R16_SNORM => vk::Format::R16_SNORM,
        nrd::Format::R16_UINT => vk::Format::R16_UINT,
        nrd::Format::R16_SINT => vk::Format::R16_SINT,
        nrd::Format::R16_SFLOAT => vk::Format::R16_SFLOAT,
        nrd::Format::RG16_UNORM => vk::Format::R16G16_UNORM,
        nrd::Format::RG16_SNORM => vk::Format::R16G16_SNORM,
        nrd::Format::RG16_UINT => vk::Format::R16G16_UINT,
        nrd::Format::RG16_SINT => vk::Format::R16G16_SINT,
        nrd::Format::RG16_SFLOAT => vk::Format::R16G16_SFLOAT,
        nrd::Format::RGBA16_UNORM => vk::Format::R16G16B16A16_UNORM,
        nrd::Format::RGBA16_SNORM => vk::Format::R16G16B16A16_SNORM,
        nrd::Format::RGBA16_UINT => vk::Format::R16G16B16A16_UINT,
        nrd::Format::RGBA16_SINT => vk::Format::R16G16B16A16_SINT,
        nrd::Format::RGBA16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
        nrd::Format::R32_UINT => vk::Format::R32_UINT,
        nrd::Format::R32_SINT => vk::Format::R32_SINT,
        nrd::Format::R32_SFLOAT => vk::Format::R32_SFLOAT,
        nrd::Format::RG32_UINT => vk::Format::R32G32_UINT,
        nrd::Format::RG32_SINT => vk::Format::R32G32_SINT,
        nrd::Format::RG32_SFLOAT => vk::Format::R32G32_SFLOAT,
        nrd::Format::RGB32_UINT => vk::Format::R32G32B32_UINT,
        nrd::Format::RGB32_SINT => vk::Format::R32G32B32_SINT,
        nrd::Format::RGB32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
        nrd::Format::RGBA32_UINT => vk::Format::R32G32B32A32_UINT,
        nrd::Format::RGBA32_SINT => vk::Format::R32G32B32A32_SINT,
        nrd::Format::RGBA32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
        nrd::Format::R10_G10_B10_A2_UNORM => vk::Format::A2R10G10B10_UNORM_PACK32,
        nrd::Format::R10_G10_B10_A2_UINT => vk::Format::A2R10G10B10_UINT_PACK32,
        nrd::Format::R11_G11_B10_UFLOAT => vk::Format::B10G11R11_UFLOAT_PACK32,
        nrd::Format::R9_G9_B9_E5_UFLOAT => vk::Format::E5B9G9R9_UFLOAT_PACK32,
    }
}

fn create_image(
    texture_desc: &nrd::TextureDesc,
    allocator: &Allocator,
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
        .with_2d_view()?;
    Ok(image)
}

// TODO: Properly handle texture IOs.
