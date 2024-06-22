use std::{ops::DerefMut, sync::Arc};

use bevy::{
    asset::Assets,
    ecs::{
        query::With,
        system::{In, Local, Query, Res, ResMut, Resource},
        world::FromWorld,
    },
    transform::components::GlobalTransform,
};
use rhyolite::{
    ash::khr,
    ash::vk,
    commands::{CommonCommands, ResourceTransitionCommands},
    ecs::{Barriers, RenderCommands},
    pipeline::{DescriptorSetLayout, PipelineCache, PipelineLayout},
    shader::{ShaderModule, SpecializedShader},
    staging::UniformBelt,
    Access, DeferredOperationTaskPool, Device, ImageLike, SwapchainImage,
};
use rhyolite_rtx::{RayTracingPipelineBuildInfoCommon, RayTracingPipelineManager, SbtManager};

use crate::camera::{CameraUniform, PinholeProjection};

#[derive(Resource)]
pub struct PbrPipeline {
    layout: Arc<PipelineLayout>,
    pub primary: RayTracingPipelineManager,
}

impl FromWorld for PbrPipeline {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.get_resource::<rhyolite::Device>().unwrap();
        let assets = world.get_resource::<bevy::asset::AssetServer>().unwrap();
        let pipeline_cache = world.get_resource::<PipelineCache>().unwrap();

        let desc0 = DescriptorSetLayout::new(
            device.clone(),
            &playout_macro::layout!("../../../assets/shaders/headers/layout.playout", 0),
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
        )
        .unwrap();
        let layout = PipelineLayout::new(
            device.clone(),
            vec![Arc::new(desc0)],
            &[vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<[f32; 2]>() as u32,
                stage_flags: vk::ShaderStageFlags::VERTEX,
            }], // Ideally this can be specified automatically
            vk::PipelineLayoutCreateFlags::empty(),
        )
        .unwrap();
        let layout = Arc::new(layout);

        let primary = RayTracingPipelineManager::new(
            RayTracingPipelineBuildInfoCommon {
                layout: layout.clone(),
                flags: vk::PipelineCreateFlags::empty(),
                max_pipeline_ray_recursion_depth: 1,
                max_pipeline_ray_payload_size: 0,
                max_pipeline_ray_hit_attribute_size: 0,
                dynamic_states: vec![],
            },
            vec![SpecializedShader {
                stage: vk::ShaderStageFlags::RAYGEN_KHR,
                shader: assets.load("shaders/primary/primary.rgen"),
                ..Default::default()
            }],
            vec![SpecializedShader {
                stage: vk::ShaderStageFlags::MISS_KHR,
                shader: assets.load("shaders/primary/primary.rmiss"),
                ..Default::default()
            }],
            vec![],
            pipeline_cache,
        );
        Self { layout, primary }
    }
}

impl PbrPipeline {
    const PRIMARY_RAY: usize = 0;

    pub fn prepare_pipeline(
        mut this: ResMut<Self>,
        pipeline_cache: Res<PipelineCache>,
        shaders: Res<Assets<ShaderModule>>,
        pool: Res<DeferredOperationTaskPool>,
        mut sbt: ResMut<SbtManager<Self>>,
    ) {
        this.primary
            .try_build(&pipeline_cache, &shaders, &pool, &mut sbt);
    }

    pub fn trace_primary_rays_barrier(
        In(mut barriers): In<Barriers>,
        mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
        accel_struct: ResMut<rhyolite_rtx::TLASDeviceBuildStore<rhyolite_rtx::DefaultTLAS>>,
        mut hitgroup_sbt: ResMut<SbtManager<Self>>,
        device: Res<Device>,
    ) {
        let Ok(mut swapchain) = windows.get_single_mut() else {
            return;
        };
        barriers.transition(
            swapchain.deref_mut(),
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            false,
            vk::ImageLayout::GENERAL,
        );
        let Some(accel_struct) = accel_struct.into_inner().deref_mut() else {
            return;
        };
        let Some(sbt) = hitgroup_sbt.buffer_mut() else {
            return;
        };
        barriers.transition(
            accel_struct,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            },
            true,
            (),
        );
        barriers.transition(
            sbt,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: if device
                    .get_extension::<khr::ray_tracing_maintenance1::Meta>()
                    .is_ok()
                {
                    vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR
                } else {
                    vk::AccessFlags2::SHADER_READ
                },
            },
            true,
            (),
        );
    }
    pub fn trace_primary_rays(
        mut commands: RenderCommands<'u'>,
        mut this: ResMut<Self>,
        mut uniform_belt: ResMut<UniformBelt>,
        windows: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
        accel_struct: ResMut<rhyolite_rtx::TLASDeviceBuildStore<rhyolite_rtx::DefaultTLAS>>,
        hitgroup_sbt: Res<SbtManager<Self>>,
        cameras: Query<(&GlobalTransform, &PinholeProjection)>,
        mut last_frame_camera: Local<Option<CameraUniform>>,
    ) {
        let Ok(swapchain) = windows.get_single() else {
            return;
        };
        let this = &mut *this;
        let Some(pipeline) = this.primary.get_pipeline() else {
            return;
        };
        let Some(accel_struct) = accel_struct.into_inner().deref_mut() else {
            return;
        };

        let (transform, projection) = cameras.single();
        let camera = CameraUniform::from_transform_projection(
            transform,
            projection,
            swapchain.extent().x as f32 / swapchain.extent().y as f32,
        );

        let mut job = uniform_belt.start(&mut commands);
        let current_camera_uniform = job.push_item(&camera);
        let last_frame_camera_uniform =
            job.push_item(last_frame_camera.as_ref().unwrap_or(&camera));
        last_frame_camera.replace(camera);
        drop(job);

        commands.push_descriptor_set(
            &this.layout,
            0,
            &[
                vk::WriteDescriptorSet {
                    descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    dst_binding: 0,
                    descriptor_count: 1,
                    ..Default::default()
                }
                .push_next(
                    &mut vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(&[accel_struct.raw()]),
                ),
                vk::WriteDescriptorSet {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    dst_binding: 1,
                    ..Default::default()
                }
                .buffer_info(&[
                    vk::DescriptorBufferInfo {
                        buffer: last_frame_camera_uniform.buffer,
                        offset: last_frame_camera_uniform.offset,
                        range: last_frame_camera_uniform.size,
                    },
                    vk::DescriptorBufferInfo {
                        buffer: current_camera_uniform.buffer,
                        offset: current_camera_uniform.offset,
                        range: current_camera_uniform.size,
                    },
                ]),
                vk::WriteDescriptorSet {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    dst_binding: 3,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_view: swapchain.view,
                    image_layout: vk::ImageLayout::GENERAL,
                    sampler: vk::Sampler::null(),
                }]),
            ],
            vk::PipelineBindPoint::RAY_TRACING_KHR,
        );

        let mut sbt = pipeline
            .use_on(&mut commands)
            .trace_rays(&mut uniform_belt, &mut commands);

        sbt.bind_raygen(0, &());
        sbt.bind_miss([&()]);

        sbt.trace(0, swapchain.extent(), &hitgroup_sbt);
    }
}
