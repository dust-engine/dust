use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bevy_asset::{AssetServer, Assets};
use bevy_ecs::system::{lifetimeless::SRes, Resource, SystemParamItem};
use bevy_math::{Quat, Vec3};
use bevy_transform::prelude::GlobalTransform;
use crevice::std430::AsStd430;
use rhyolite::{
    accel_struct::AccelerationStructure,
    ash::vk,
    descriptor::DescriptorPool,
    future::{
        use_per_frame_state, Disposable, DisposeContainer, GPUCommandFuture, PerFrameContainer,
        PerFrameState, RenderImage, RenderRes,
    },
    macros::{commands, set_layout},
    utils::retainer::{Retainer, RetainerHandle},
    BufferLike, HasDevice, ImageLike, ImageViewLike, Sampler,
};
use rhyolite_bevy::{Allocator, Image, SlicedImageArray};

use crate::{
    sbt::{EmptyShaderRecords, PipelineSbtManager, PipelineSbtManagerInfo, SbtManager},
    BlueNoise, PinholeProjection, RayTracingPipelineManagerSpecializedPipeline, ShaderModule,
    SpecializedShader,
};

use super::{RayTracingPipeline, RayTracingPipelineManager};

#[derive(Resource)]
pub struct StandardPipeline {
    primary_ray_pipeline: RayTracingPipelineManager,
    photon_ray_pipeline: RayTracingPipelineManager,
    hitgroup_sbt_manager: SbtManager,
    pipeline_sbt_manager: PipelineSbtManager,

    desc_pool: Retainer<DescriptorPool>,
}

impl HasDevice for StandardPipeline {
    fn device(&self) -> &Arc<rhyolite::Device> {
        self.hitgroup_sbt_manager.device()
    }
}

impl RayTracingPipeline for StandardPipeline {
    fn create_info() -> rhyolite::RayTracingPipelineLibraryCreateInfo {
        rhyolite::RayTracingPipelineLibraryCreateInfo {
            max_pipeline_ray_payload_size: 16,
            ..Default::default()
        }
    }
    fn pipeline_layout(device: &Arc<rhyolite::Device>) -> Arc<rhyolite::PipelineLayout> {
        let mut set1 = set_layout! {
            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)]
            img_output: vk::DescriptorType::STORAGE_IMAGE,

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR)]
            accel_struct: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR)]
            noise_unitvec3_cosine: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        };
        set1.bindings[2].immutable_samplers = vec![Arc::new(
            Sampler::new(
                device.clone(),
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    address_mode_w: vk::SamplerAddressMode::REPEAT,
                    unnormalized_coordinates: vk::TRUE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )];

        let set1 = set1.build(device.clone()).unwrap();
        Arc::new(
            rhyolite::PipelineLayout::new(
                device.clone(),
                vec![Arc::new(set1)],
                vk::PipelineLayoutCreateFlags::empty(),
            )
            .unwrap(),
        )
    }
    fn new(
        allocator: Allocator,
        pipeline_characteristic: super::RayTracingPipelineCharacteristics,
        asset_server: &AssetServer,
        pipeline_cache: Option<std::sync::Arc<rhyolite::PipelineCache>>,
    ) -> Self {
        let pipeline_characteristics = Arc::new(pipeline_characteristic);
        let hitgroup_sbt_manager = SbtManager::new(allocator.clone(), &pipeline_characteristics);
        let pipeline_sbt_manager = PipelineSbtManager::new(allocator.into_inner());
        Self {
            desc_pool: Retainer::new(
                DescriptorPool::for_pipeline_layouts(
                    std::iter::once(pipeline_characteristics.layout.clone()),
                    pipeline_characteristics.num_frame_in_flight,
                )
                .unwrap(),
            ),
            hitgroup_sbt_manager,
            primary_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics.clone(),
                vec![0],
                SpecializedShader::for_shader(
                    asset_server.load("primary.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![SpecializedShader::for_shader(
                    asset_server.load("miss.rmiss.spv"),
                    vk::ShaderStageFlags::MISS_KHR,
                )],
                Vec::new(),
                pipeline_cache.as_ref().cloned(),
            ),
            photon_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics,
                vec![1],
                SpecializedShader::for_shader(
                    asset_server.load("photon.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![],
                Vec::new(),
                pipeline_cache,
            ),
            pipeline_sbt_manager,
        }
    }
    fn material_instance_added<M: crate::Material<Pipeline = Self>>(
        &mut self,
        material: &M,
        params: &mut SystemParamItem<M::ShaderParameterParams>,
    ) -> crate::sbt::SbtIndex {
        self.primary_ray_pipeline.material_instance_added::<M>();
        self.photon_ray_pipeline.material_instance_added::<M>();
        self.hitgroup_sbt_manager.add_instance(material, params)
    }

    fn num_raytypes() -> u32 {
        2
    }

    fn material_instance_removed<M: crate::Material<Pipeline = Self>>(&mut self) {}
}

#[derive(AsStd430)]
struct StandardPipelineCamera {
    camera_view_col0: Vec3,
    near: f32,
    camera_view_col1: Vec3,
    far: f32,
    camera_view_col2: Vec3,
    padding: f32,
    camera_position: Vec3,
    tan_half_fov: f32,
}

#[derive(AsStd430)]
struct StandardPipelinePhotonCamera {
    camera_view_col0: Vec3,
    near: f32,
    camera_view_col1: Vec3,
    far: f32,
    camera_view_col2: Vec3,
    strength: f32,
    camera_position: Vec3,
    padding: f32,
}

impl StandardPipeline {
    pub const PRIMARY_RAYTYPE: u32 = 0;
    pub const PHOTON_RAYTYPE: u32 = 1;

    pub type RenderParams = (
        SRes<Assets<ShaderModule>>,
        SRes<Assets<SlicedImageArray>>,
        SRes<BlueNoise>,
    );
    pub fn render<'a>(
        &'a mut self,
        target: &'a mut RenderImage<impl ImageViewLike>,
        tlas: &'a RenderRes<Arc<AccelerationStructure>>,
        params: &'a mut SystemParamItem<Self::RenderParams>,
        camera: (&PinholeProjection, &GlobalTransform),
    ) -> Option<
        impl GPUCommandFuture<
                Output = (),
                RetainedState: 'static + Disposable,
                RecycledState: 'static + Default,
            > + 'a,
    > {
        let (shader_store, image_store, blue_noise) = params;
        let primary_pipeline = self.primary_ray_pipeline.get_pipeline(shader_store)?;
        let photon_pipeline = self.photon_ray_pipeline.get_pipeline(shader_store)?;
        let noise_img = image_store.get(&blue_noise.unitvec3_cosine)?;
        self.hitgroup_sbt_manager
            .specify_pipelines(&[primary_pipeline, photon_pipeline]);
        let hitgroup_sbt_buffer = self.hitgroup_sbt_manager.hitgroup_sbt_buffer()?;
        let hitgroup_stride = self.hitgroup_sbt_manager.hitgroup_stride();

        let camera = StandardPipelineCamera {
            camera_view_col0: camera.1.affine().matrix3.x_axis.into(),
            camera_view_col1: camera.1.affine().matrix3.y_axis.into(),
            camera_view_col2: camera.1.affine().matrix3.z_axis.into(),
            near: camera.0.near,
            far: camera.0.far,
            padding: 0.0,
            camera_position: camera.1.translation(),
            tan_half_fov: (camera.0.fov / 2.0).tan(),
        };

        let affine = bevy_math::Affine3A::from_scale_rotation_translation(
            Vec3::splat(500.0),
            Quat::from_rotation_x(-2.5),
            Vec3 {
                x: 0.0,
                y: 1000.0,
                z: -500.0,
            },
        );
        let photon_camera = StandardPipelinePhotonCamera {
            camera_view_col0: affine.matrix3.x_axis.into(),
            camera_view_col1: affine.matrix3.y_axis.into(),
            camera_view_col2: affine.matrix3.z_axis.into(),
            near: 0.1,
            far: 10000.0,
            padding: 0.0,
            camera_position: affine.translation.into(),
            strength: 100.0,
        };

        self.pipeline_sbt_manager
            .push_raygen(primary_pipeline, camera, 0);
        self.pipeline_sbt_manager
            .push_raygen(photon_pipeline, photon_camera, 0);
        self.pipeline_sbt_manager
            .push_miss(primary_pipeline, EmptyShaderRecords, 0);
        let pipeline_sbt_info = self.pipeline_sbt_manager.build();
        let desc_pool = &mut self.desc_pool;
        let fut = commands! { move
            let hitgroup_sbt_buffer = hitgroup_sbt_buffer.await;
            let pipeline_sbt_buffer = pipeline_sbt_info.await; // TODO: Make this join
            StandardPipelineRenderingFuture {
                accel_struct: tlas,
                target_image: target,
                primary_pipeline,
                photon_pipeline,
                desc_pool,
                hitgroup_sbt_buffer: &hitgroup_sbt_buffer,
                hitgroup_stride,
                pipeline_sbt_buffer: &pipeline_sbt_buffer,
                noise_img
            }.await;
            retain!(hitgroup_sbt_buffer);
            retain!(pipeline_sbt_buffer);
        };
        Some(fut)
    }
}

use pin_project::pin_project;

#[pin_project]
struct StandardPipelineRenderingFuture<'a, TargetImage: ImageViewLike, HitgroupBuf: BufferLike> {
    accel_struct: &'a RenderRes<Arc<AccelerationStructure>>,
    target_image: &'a mut RenderImage<TargetImage>,
    primary_pipeline: RayTracingPipelineManagerSpecializedPipeline<'a>,
    photon_pipeline: RayTracingPipelineManagerSpecializedPipeline<'a>,
    desc_pool: &'a mut Retainer<DescriptorPool>,
    hitgroup_sbt_buffer: &'a RenderRes<HitgroupBuf>,
    pipeline_sbt_buffer: &'a RenderRes<PipelineSbtManagerInfo>,
    noise_img: &'a rhyolite_bevy::SlicedImageArray,
    hitgroup_stride: usize,
}

impl<'a, TargetImage: ImageViewLike, HitgroupBuf: BufferLike> rhyolite::future::GPUCommandFuture
    for StandardPipelineRenderingFuture<'a, TargetImage, HitgroupBuf>
{
    type Output = ();

    type RetainedState = DisposeContainer<(
        Arc<rhyolite::RayTracingPipeline>,
        RetainerHandle<DescriptorPool>,
        PerFrameContainer<Vec<vk::DescriptorSet>>,
    )>;

    type RecycledState = (PerFrameState<Vec<vk::DescriptorSet>>, usize);

    fn init(
        self: std::pin::Pin<&mut Self>,
        _ctx: &mut rhyolite::future::CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        None
    }
    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut rhyolite::future::CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let (state_desc_sets, noise_texture_index) = recycled_state;
        let this = self.project();
        let noise_texture_index = {
            let index = *noise_texture_index;
            *noise_texture_index += 1;
            if *noise_texture_index as u32 >= this.noise_img.subresource_range().layer_count {
                *noise_texture_index = 0;
            }
            index
        };
        let desc_set = use_per_frame_state(state_desc_sets, || {
            this.desc_pool
                .allocate_for_pipeline_layout(this.primary_pipeline.layout())
                .unwrap()
        });
        unsafe {
            let acceleration_structure_write = vk::WriteDescriptorSetAccelerationStructureKHR {
                acceleration_structure_count: 1,
                p_acceleration_structures: &this.accel_struct.inner().raw(),
                ..Default::default()
            };
            // TODO: optimize away redundant writes
            this.primary_pipeline.device().update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: desc_set[0],
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: this.target_image.inner().raw_image_view(),
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        p_next: &acceleration_structure_write as *const _ as *const _,
                        dst_set: desc_set[0],
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: desc_set[0],
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: this.noise_img.slice(noise_texture_index).raw_image_view(),
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            );
        }
        let extent = this.target_image.inner().extent();

        ctx.record(|ctx, command_buffer| unsafe {
            let device = ctx.device();
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                this.primary_pipeline.layout().raw(),
                0,
                desc_set.as_slice(),
                &[],
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                this.photon_pipeline.pipeline().raw(),
            );
            device.rtx_loader().cmd_trace_rays(
                command_buffer,
                &this.pipeline_sbt_buffer.inner().rgen(1),
                &this.pipeline_sbt_buffer.inner().miss(),
                &vk::StridedDeviceAddressRegionKHR {
                    device_address: this.hitgroup_sbt_buffer.inner().device_address(),
                    stride: *this.hitgroup_stride as u64,
                    size: this.hitgroup_sbt_buffer.inner.size(),
                },
                &vk::StridedDeviceAddressRegionKHR::default(),
                128,
                128,
                1,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                this.primary_pipeline.pipeline().raw(),
            );
            device.rtx_loader().cmd_trace_rays(
                command_buffer,
                &this.pipeline_sbt_buffer.inner().rgen(0),
                &this.pipeline_sbt_buffer.inner().miss(),
                &vk::StridedDeviceAddressRegionKHR {
                    device_address: this.hitgroup_sbt_buffer.inner().device_address(),
                    stride: *this.hitgroup_stride as u64,
                    size: this.hitgroup_sbt_buffer.inner.size(),
                },
                &vk::StridedDeviceAddressRegionKHR::default(),
                extent.width,
                extent.height,
                extent.depth,
            );
        });
        std::task::Poll::Ready((
            (),
            DisposeContainer::new((
                this.primary_pipeline.pipeline().clone(),
                this.desc_pool.handle(),
                desc_set,
            )),
        ))
    }

    fn context(self: std::pin::Pin<&mut Self>, ctx: &mut rhyolite::future::StageContext) {
        let this = self.project();
        ctx.write_image(
            this.target_image.deref_mut(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::ImageLayout::GENERAL,
        );
        ctx.read(
            this.hitgroup_sbt_buffer.deref(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
        );
        ctx.read(
            this.pipeline_sbt_buffer.deref(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
        );
        ctx.read_others(
            this.accel_struct.deref(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }
}

// TODO: group base alignment
