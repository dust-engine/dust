use std::{ops::DerefMut, sync::Arc};

use bevy::{
    asset::Assets,
    ecs::{
        query::With,
        system::{In, Local, Query, Res, ResMut, Resource},
        world::FromWorld,
    },
    utils::smallvec::SmallVec,
};
use bytemuck::{Pod, Zeroable};
use rhyolite::{
    ash::vk,
    commands::{CommonCommands, ResourceTransitionCommands},
    dispose::RenderObject,
    ecs::{Barriers, RenderCommands},
    pipeline::{CachedPipeline, DescriptorSetLayout, PipelineCache, PipelineLayout},
    shader::{ShaderModule, SpecializedShader},
    staging::UniformBelt,
    Access, DeferredOperationTaskPool, ImageLike, SwapchainImage,
};
use rhyolite_rtx::{
    RayTracingPipeline, RayTracingPipelineBuildInfoCommon, RayTracingPipelineManager, SbtManager,
};

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
            vec![],
            vec![],
            pipeline_cache,
        );
        Self { layout, primary }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
struct RayGenParams {
    color: [f32; 4],
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
    ) {
        let Ok(mut swapchain) = windows.get_single_mut() else {
            return;
        };
        let Some(accel_struct) = accel_struct.into_inner().deref_mut() else {
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
        barriers.transition(
            accel_struct,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            },
            true,
            (),
        );
    }
    pub fn trace_primary_rays(
        mut commands: RenderCommands<'u'>,
        mut this: ResMut<Self>,
        pipeline_cache: Res<PipelineCache>,
        shaders: Res<Assets<ShaderModule>>,
        pool: Res<DeferredOperationTaskPool>,
        mut uniform_belt: ResMut<UniformBelt>,
        windows: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
        accel_struct: ResMut<rhyolite_rtx::TLASDeviceBuildStore<rhyolite_rtx::DefaultTLAS>>,
        hitgroup_sbt: Res<SbtManager<Self>>,
    ) {
        let Ok(swapchain) = windows.get_single() else {
            return;
        };
        let this = &mut *this;
        let Some(pipeline) = this.primary.get_pipeline_mut().and_then(|x| x.get_mut()) else {
            return;
        };
        let Some(accel_struct) = accel_struct.into_inner().deref_mut() else {
            return;
        };

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
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    dst_binding: 1,
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

        sbt.bind_raygen(
            0,
            &RayGenParams {
                color: [0.0, 0.0, 1.0, 1.0],
            },
        );

        sbt.trace(0, swapchain.extent(), &hitgroup_sbt);
    }
}