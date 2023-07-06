use std::{ops::Deref, sync::Arc};

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::{AssetServer, Assets};

use bevy_ecs::prelude::{Component, Entity};
use bevy_ecs::query::{Added, Changed, Or};
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::system::lifetimeless::SResMut;
use bevy_ecs::system::{lifetimeless::SRes, Resource, SystemParamItem};
use bevy_ecs::system::{Commands, Query};
use bevy_math::{Mat4, Vec3};
use bevy_transform::prelude::GlobalTransform;

use crevice::std430::{AsStd430, Std430};
use rand::Rng;
use rhyolite::future::{
    run, use_shared_resource_flipflop, use_shared_state, use_state, GPUCommandFutureExt,
};
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
use rhyolite_bevy::{Allocator, SlicedImageArray};
use rhyolite_bevy::{RenderSystems, StagingRingBuffer};

use crate::accel_struct::instance_vec::{InstanceVecPlugin, InstanceVecStore};
use crate::{
    sbt::{EmptyShaderRecords, PipelineSbtManager, SbtManager},
    PinholeProjection, ShaderModule, SpecializedShader,
};
use crate::{PipelineCache, Renderable, Sunlight};

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

            #[shader(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR)]
            camera_settings_prev_frame: vk::DescriptorType::UNIFORM_BUFFER,
            #[shader(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR)]
            camera_settings: vk::DescriptorType::UNIFORM_BUFFER,
            #[shader(vk::ShaderStageFlags::CLOSEST_HIT_KHR)]
            instances: vk::DescriptorType::STORAGE_BUFFER,
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
                vec![SpecializedShader::for_shader(
                    asset_server.load("photon.rmiss.spv"),
                    vk::ShaderStageFlags::MISS_KHR,
                )],
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
    SResMut<InstanceVecStore<PreviousFrameGlobalTransform>>,
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
        let (
            shader_store,
            pipeline_cache,
            allocator,
            sunlight,
            mut instances_buffer,
            staging_ring_buffer,
        ) = params;
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
        let hitgroup_sbt_buffer = self.hitgroup_sbt_manager.hitgroup_sbt_buffer();
        let hitgroup_stride = self.hitgroup_sbt_manager.hitgroup_stride();
        let instances_buffer = instances_buffer.buffer.buffer();

        let camera_settings = {
            let proj = {
                let extent = target_image.inner().extent();
                Mat4::perspective_infinite_reverse_rh(
                    camera.0.fov,
                    extent.width as f32 / extent.height as f32,
                    camera.0.near,
                )
            };
            let view_proj = proj * camera.1.compute_matrix().inverse();
            CameraSettings {
                view_proj: view_proj,
                inverse_view_proj: view_proj.inverse(),
                camera_view_col0: camera.1.affine().matrix3.x_axis.into(),
                camera_view_col1: camera.1.affine().matrix3.y_axis.into(),
                camera_view_col2: camera.1.affine().matrix3.z_axis.into(),
                near: camera.0.near,
                far: camera.0.far,
                padding: 0.0,
                position_x: camera.1.translation().x,
                position_y: camera.1.translation().y,
                position_z: camera.1.translation().z,
                tan_half_fov: (camera.0.fov / 2.0).tan(),
            }
            .as_std430()
        };
        self.pipeline_sbt_manager
            .push_raygen(primary_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_raygen(photon_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_raygen(shadow_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_raygen(final_gather_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(primary_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(shadow_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(final_gather_pipeline, EmptyShaderRecords, 0);
        self.pipeline_sbt_manager
            .push_miss(photon_pipeline, EmptyShaderRecords, 0);
        let pipeline_sbt_manager = &mut self.pipeline_sbt_manager;
        let desc_pool = &mut self.desc_pool;
        let sunlight = sunlight.bake().as_std430();

        let fut = commands! { move
            let instances_buffer = instances_buffer.await;
            let (pipeline_sbt_buffer, hitgroup_sbt_buffer) = pipeline_sbt_manager.build(&staging_ring_buffer).join(hitgroup_sbt_buffer).await;

            // TODO: Direct writes on integrated GPUs.
            let mut sunlight_buffer = use_shared_state(
                using!(),
                |_| {
                    allocator.create_device_buffer_uninit(SkyModelState::std430_size_static() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, 0).unwrap()
                },
                |_| false
            );
            let (mut camera_setting_buffer, camera_setting_buffer_prev_frame) = use_shared_resource_flipflop(
                using!(),
                |_| {
                    allocator.create_device_buffer_uninit(CameraSettings::std430_size_static() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, 0).unwrap()
                },
                |_| false
            );
            staging_ring_buffer.update_buffer(&mut camera_setting_buffer, camera_settings.as_bytes()).join(
                            staging_ring_buffer.update_buffer(&mut sunlight_buffer, sunlight.as_bytes())
            ).await;

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
                    &[
                        sunlight_buffer.inner().as_descriptor(),
                        camera_setting_buffer_prev_frame.inner().as_descriptor(),
                        camera_setting_buffer.inner().as_descriptor()
                    ],
                    false
                ),
                DescriptorSetWrite::storage_buffers(
                    desc_set[0],
                    10,
                    0,
                    &[
                        instances_buffer.inner().as_descriptor(),
                    ],
                    false
                ),
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
                ctx.write_image(
                    motion_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
                ctx.read(&instances_buffer, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR, vk::AccessFlags2::SHADER_STORAGE_READ)
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
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    normal_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
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
                    vk::AccessFlags2::SHADER_STORAGE_READ,
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
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    depth_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    normal_image,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    &sunlight_buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;
            retain!((
                sunlight_buffer,
                camera_setting_buffer,
                camera_setting_buffer_prev_frame,
                pipeline_sbt_buffer,
                hitgroup_sbt_buffer,
                instances_buffer
            ));
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

#[derive(Clone, Debug, AsStd430)]
pub struct CameraSettings {
    pub view_proj: Mat4,
    pub inverse_view_proj: Mat4,
    pub camera_view_col0: Vec3,
    pub position_x: f32,
    pub camera_view_col1: Vec3,
    pub position_y: f32,
    pub camera_view_col2: Vec3,
    pub position_z: f32,
    pub tan_half_fov: f32,
    pub far: f32,
    pub near: f32,
    pub padding: f32,
}

pub struct StandardPipelinePlugin;
impl Plugin for StandardPipelinePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            PostUpdate,
            extract_global_transforms.in_set(RenderSystems::CleanUp),
        );
        app.add_plugin(
            InstanceVecPlugin::<PreviousFrameGlobalTransform, Renderable>::new(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                0,
            ),
        );
    }
}

#[derive(Component, Debug)]
pub struct PreviousFrameGlobalTransform {
    mat: GlobalTransform,
}
impl crate::accel_struct::instance_vec::InstanceVecItem for PreviousFrameGlobalTransform {
    type Data = Mat4;
    fn data(&self) -> Self::Data {
        self.mat.compute_matrix()
    }
}

pub fn extract_global_transforms(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &GlobalTransform,
            Option<&mut PreviousFrameGlobalTransform>,
        ),
        Or<(Changed<GlobalTransform>, Added<GlobalTransform>)>,
    >,
) {
    for (entity, global_transform, previous_frame_global_transform) in query.iter_mut() {
        if let Some(mut previous_frame_global_transform) = previous_frame_global_transform {
            previous_frame_global_transform.mat = global_transform.clone()
        } else {
            commands
                .entity(entity)
                .insert(PreviousFrameGlobalTransform {
                    mat: global_transform.clone(),
                });
        }
    }
}
