use std::{ops::Deref, sync::Arc};

use bevy_asset::{AssetServer, Assets};
use bevy_ecs::system::Res;
use bevy_ecs::system::{lifetimeless::SRes, Resource, SystemParamItem};
use bevy_math::{Quat, Vec3};
use bevy_transform::prelude::GlobalTransform;
use crevice::std430::AsStd430;
use rand::Rng;
use rhyolite::future::{run, use_shared_state, use_state};
use rhyolite::{
    accel_struct::AccelerationStructure,
    ash::vk,
    descriptor::{DescriptorPool, DescriptorSetWrite, PushConstants},
    future::{
        use_per_frame_state, Disposable, DisposeContainer, GPUCommandFuture, RenderData,
        RenderImage, RenderRes,
    },
    macros::{commands, set_layout},
    utils::retainer::Retainer,
    BufferExt, BufferLike, HasDevice, ImageLike, ImageViewExt, ImageViewLike,
};
use rhyolite_bevy::StagingRingBuffer;
use rhyolite_bevy::{Allocator, SlicedImageArray};

use crate::{
    sbt::{EmptyShaderRecords, PipelineSbtManager, SbtManager},
    BlueNoise, PinholeProjection, ShaderModule, SpecializedShader,
};
use crate::{PipelineCache, Sunlight};

use super::sky::SkyModelState;
use super::{RayTracingPipeline, RayTracingPipelineManager};

#[derive(Resource)]
pub struct StandardPipeline {
    primary_ray_pipeline: RayTracingPipelineManager,
    photon_ray_pipeline: RayTracingPipelineManager,
    shadow_ray_pipeline: RayTracingPipelineManager,
    final_gather_ray_pipeline: RayTracingPipelineManager,
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
            max_pipeline_ray_payload_size: 32,
            max_pipeline_ray_hit_attribute_size: 32,
            ..Default::default()
        }
    }
    fn pipeline_layout(device: &Arc<rhyolite::Device>) -> Arc<rhyolite::PipelineLayout> {
        let set1 = set_layout! {
            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::RAYGEN_KHR)]
            img_output: vk::DescriptorType::STORAGE_IMAGE,

            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::RAYGEN_KHR)]
            img_albedo: vk::DescriptorType::STORAGE_IMAGE,
            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::RAYGEN_KHR)]
            img_normal: vk::DescriptorType::STORAGE_IMAGE,
            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::RAYGEN_KHR)]
            img_depth: vk::DescriptorType::STORAGE_IMAGE,
            #[shader(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::RAYGEN_KHR)]
            img_motion: vk::DescriptorType::STORAGE_IMAGE,

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR)]
            accel_struct: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)]
            noise_unitvec3_cosine: vk::DescriptorType::SAMPLED_IMAGE,

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR)]
            sunlight_settings: vk::DescriptorType::UNIFORM_BUFFER,
        };

        let set1 = set1.build(device.clone()).unwrap();
        Arc::new(
            rhyolite::PipelineLayout::new(
                device.clone(),
                vec![Arc::new(set1)],
                StandardPipelinePushConstant::ranges().as_slice(),
                vk::PipelineLayoutCreateFlags::empty(),
            )
            .unwrap(),
        )
    }
    fn new(
        allocator: Allocator,
        pipeline_characteristic: super::RayTracingPipelineCharacteristics,
        asset_server: &AssetServer,
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
                vec![Self::PRIMARY_RAYTYPE],
                SpecializedShader::for_shader(
                    asset_server.load("primary.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![SpecializedShader::for_shader(
                    asset_server.load("miss.rmiss.spv"),
                    vk::ShaderStageFlags::MISS_KHR,
                )],
                Vec::new(),
            ),
            photon_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics.clone(),
                vec![Self::PHOTON_RAYTYPE],
                SpecializedShader::for_shader(
                    asset_server.load("photon.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![],
                Vec::new(),
            ),
            shadow_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics.clone(),
                vec![Self::SHADOW_RAYTYPE],
                SpecializedShader::for_shader(
                    asset_server.load("shadow.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![SpecializedShader::for_shader(
                    asset_server.load("shadow.rmiss.spv"),
                    vk::ShaderStageFlags::MISS_KHR,
                )],
                Vec::new(),
            ),
            final_gather_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics,
                vec![Self::FINAL_GATHER_RAYTYPE],
                SpecializedShader::for_shader(
                    asset_server.load("final_gather.rgen.spv"),
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ),
                vec![SpecializedShader::for_shader(
                    asset_server.load("final_gather.rmiss.spv"),
                    vk::ShaderStageFlags::MISS_KHR,
                )],
                Vec::new(),
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
        self.shadow_ray_pipeline.material_instance_added::<M>();
        self.final_gather_ray_pipeline
            .material_instance_added::<M>();
        self.hitgroup_sbt_manager.add_instance(material, params)
    }

    fn num_raytypes() -> u32 {
        4
    }

    fn material_instance_removed<M: crate::Material<Pipeline = Self>>(&mut self) {}
}

#[derive(AsStd430, Clone)]
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
    padding: u32,
}
#[derive(AsStd430, Default, PushConstants)]
struct StandardPipelinePushConstant {
    #[stage(
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR
    )]
    rand: u32,
    #[stage(
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR
    )]
    frame_index: u32,
}

pub type StandardPipelineRenderParams = (
    SRes<Assets<ShaderModule>>,
    SRes<PipelineCache>,
    SRes<Allocator>,
    SRes<Sunlight>,
    SRes<StagingRingBuffer>,
);
impl StandardPipeline {
    pub const PRIMARY_RAYTYPE: u32 = 0;
    pub const PHOTON_RAYTYPE: u32 = 1;
    pub const SHADOW_RAYTYPE: u32 = 2;
    pub const FINAL_GATHER_RAYTYPE: u32 = 3;

    pub fn render<'a>(
        &'a mut self,
        target_image: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        albedo_image: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        normal_image: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        depth_image: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        motion_image: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        noise_image: &'a SlicedImageArray,
        tlas: &'a RenderRes<Arc<AccelerationStructure>>,
        params: SystemParamItem<'a, '_, StandardPipelineRenderParams>,
        camera: (&PinholeProjection, &GlobalTransform),
    ) -> Option<
        impl GPUCommandFuture<
                Output = (),
                RetainedState: 'static + Disposable,
                RecycledState: 'static + Default,
            > + 'a,
    > {
        let (shader_store, pipeline_cache, allocator, sunlight, staging_ring_buffer) = params;
        let primary_pipeline = self
            .primary_ray_pipeline
            .get_pipeline(&pipeline_cache, &shader_store)?;
        let photon_pipeline = self
            .photon_ray_pipeline
            .get_pipeline(&pipeline_cache, &shader_store)?;
        let shadow_pipeline = self
            .shadow_ray_pipeline
            .get_pipeline(&pipeline_cache, &shader_store)?;
        let final_gather_pipeline = self
            .final_gather_ray_pipeline
            .get_pipeline(&pipeline_cache, &shader_store)?;
        self.hitgroup_sbt_manager.specify_pipelines(&[
            primary_pipeline,
            photon_pipeline,
            shadow_pipeline,
            final_gather_pipeline,
        ]);
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
            padding: 0,
            camera_position: (1000.0 * sunlight.direction).into(),
            strength: 37.0,
        };

        self.pipeline_sbt_manager
            .push_raygen(primary_pipeline, camera.clone(), 0);
        self.pipeline_sbt_manager
            .push_raygen(photon_pipeline, photon_camera, 0);
        self.pipeline_sbt_manager
            .push_raygen(shadow_pipeline, camera.clone(), 0);
        self.pipeline_sbt_manager
            .push_raygen(final_gather_pipeline, camera, 0);
        self.pipeline_sbt_manager
            .push_miss(primary_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(shadow_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(final_gather_pipeline, EmptyShaderRecords, 0);
        let pipeline_sbt_info = self.pipeline_sbt_manager.build();
        let desc_pool = &mut self.desc_pool;
        let sunlight = sunlight.bake().as_std430();

        let fut = commands! { move
            let hitgroup_sbt_buffer = hitgroup_sbt_buffer.await;
            let pipeline_sbt_buffer = pipeline_sbt_info.await; // TODO: Make this join

            // TODO: Direct writes on integrated GPUs.
            let mut sunlight_buffer = use_shared_state(
                using!(),
                move |_| {
                    allocator.create_device_buffer_uninit(SkyModelState::std430_size_static() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST).unwrap()
                },
                |_| false
            );
            staging_ring_buffer.update_buffer(&mut sunlight_buffer, sunlight.as_bytes()).await;

            let frame_index = use_state(
                using!(),
                || 0,
                |a| *a += 1
            );
            let noise_texture_index = *frame_index % noise_image.subresource_range().layer_count;


            let desc_set = use_per_frame_state(using!(), || {
                desc_pool
                    .allocate_for_pipeline_layout(primary_pipeline.layout())
                    .unwrap()
            });

            primary_pipeline.device().write_descriptor_sets([
                DescriptorSetWrite::storage_images(
                    desc_set[0],
                    0,
                    0,
                    &[
                        target_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        albedo_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        normal_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        depth_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        motion_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                    ]
                ),
                DescriptorSetWrite::sampled_images(
                    desc_set[0],
                    6,
                    0,
                    &[noise_image.slice(noise_texture_index as usize).as_descriptor(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                ),
                DescriptorSetWrite::accel_structs(
                    desc_set[0],
                    5,
                    0,
                    &[tlas.inner().raw()]
                ),
                DescriptorSetWrite::uniform_buffers(
                    desc_set[0],
                    7,
                    0,
                    &[sunlight_buffer.inner().as_descriptor()],
                    false
                )
            ]);

            let extent = target_image.inner().extent();
            run(|ctx: &rhyolite::future::CommandBufferRecordContext, command_buffer: vk::CommandBuffer| unsafe {
                let device = ctx.device();
                let rand: u32 = rand::thread_rng().gen();
                device.cmd_push_constants(
                    command_buffer,
                    primary_pipeline.layout().raw(),
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    0,
                    std::slice::from_raw_parts(&rand as *const _ as *const u8, 4),
                );
                device.cmd_push_constants(
                    command_buffer,
                    primary_pipeline.layout().raw(),
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    4,
                    std::slice::from_raw_parts(frame_index as *const _ as *const u8, 4),
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    primary_pipeline.layout().raw(),
                    0,
                    desc_set.as_slice(),
                    &[],
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    photon_pipeline.pipeline().raw(),
                );
                device.rtx_loader().cmd_trace_rays(
                    command_buffer,
                    &pipeline_sbt_buffer.inner().rgen(1),
                    &pipeline_sbt_buffer.inner().miss(),
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: hitgroup_sbt_buffer.inner().device_address(),
                        stride: hitgroup_stride as u64,
                        size: hitgroup_sbt_buffer.inner.size(),
                    },
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    1024,
                    1024,
                    1,
                );
            }, |ctx: &mut rhyolite::future::StageContext| {
                ctx.read_others(
                    tlas.deref(),
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                );
                ctx.read(
                    &hitgroup_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read(
                    &pipeline_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;
            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    primary_pipeline.pipeline().raw(),
                );
                device.rtx_loader().cmd_trace_rays(
                    command_buffer,
                    &pipeline_sbt_buffer.inner().rgen(0),
                    &pipeline_sbt_buffer.inner().miss(),
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: hitgroup_sbt_buffer.inner().device_address(),
                        stride: hitgroup_stride as u64,
                        size: hitgroup_sbt_buffer.inner.size(),
                    },
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    extent.width,
                    extent.height,
                    extent.depth,
                );
            }, |ctx| {
                ctx.read_others(
                    tlas.deref(),
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                );
                ctx.read(
                    &hitgroup_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read(
                    &pipeline_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.write_image(
                    albedo_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    depth_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    normal_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    target_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;

            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    shadow_pipeline.pipeline().raw(),
                );
                device.rtx_loader().cmd_trace_rays(
                    command_buffer,
                    &pipeline_sbt_buffer.inner().rgen(2),
                    &pipeline_sbt_buffer.inner().miss(),
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: hitgroup_sbt_buffer.inner().device_address(),
                        stride: hitgroup_stride as u64,
                        size: hitgroup_sbt_buffer.inner.size(),
                    },
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    extent.width,
                    extent.height,
                    extent.depth,
                ); // TODO: Perf: Only trace rays for locations where primary ray was hit.
            }, |ctx| {
                ctx.read_others(
                    tlas.deref(),
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                );
                ctx.read(
                    &hitgroup_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read(
                    &pipeline_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.write_image(
                    target_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    depth_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    normal_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;


            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    final_gather_pipeline.pipeline().raw(),
                );
                device.rtx_loader().cmd_trace_rays(
                    command_buffer,
                    &pipeline_sbt_buffer.inner().rgen(3),
                    &pipeline_sbt_buffer.inner().miss(),
                    &vk::StridedDeviceAddressRegionKHR {
                        device_address: hitgroup_sbt_buffer.inner().device_address(),
                        stride: hitgroup_stride as u64,
                        size: hitgroup_sbt_buffer.inner.size(),
                    },
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    extent.width,
                    extent.height,
                    extent.depth,
                ); // TODO: Perf: Only trace rays for locations where primary ray was hit.
            }, |ctx| {
                ctx.read_others(
                    tlas.deref(),
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                );
                ctx.read(
                    &hitgroup_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read(
                    &pipeline_sbt_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
                );
                ctx.read_image(
                    target_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    target_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    albedo_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    depth_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    normal_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;
            retain!(hitgroup_sbt_buffer);
            retain!(pipeline_sbt_buffer);
            retain!(sunlight_buffer);
            retain!(
                DisposeContainer::new((
                    primary_pipeline.pipeline().clone(),
                    photon_pipeline.pipeline().clone(),
                    shadow_pipeline.pipeline().clone(),
                    final_gather_pipeline.pipeline().clone(),
                    desc_pool.handle(),
                    desc_set,
                )));
        };
        Some(fut)
    }
}
