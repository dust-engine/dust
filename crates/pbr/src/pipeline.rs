use std::sync::Arc;

use bevy::{asset::Assets, ecs::{system::{Res, ResMut, Resource}, world::FromWorld}, utils::smallvec::SmallVec};
use rhyolite::{
    ash::vk, dispose::RenderObject, ecs::RenderCommands, pipeline::{CachedPipeline, DescriptorSetLayout, PipelineCache, PipelineLayout}, shader::{ShaderModule, SpecializedShader}, DeferredOperationTaskPool
};
use rhyolite_rtx::{
    PipelineGroupManager, RayTracingPipeline, RayTracingPipelineBuildInfoCommon, RayTracingPipelineManager, SbtManager
};

#[derive(Resource)]
pub struct PbrPipeline {
    manager: PipelineGroupManager<1>,
    pipelines: SmallVec<[CachedPipeline<RenderObject<RayTracingPipeline>>; 1]>
}

impl FromWorld for PbrPipeline {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.get_resource::<rhyolite::Device>().unwrap();
        let assets = world.get_resource::<bevy::asset::AssetServer>().unwrap();
        let pipeline_cache = world.get_resource::<PipelineCache>().unwrap();

        let desc0 = DescriptorSetLayout::new(
            device.clone(),
            &playout_macro::layout!("../../../assets/shaders/headers/layout.playout", 0),
            vk::DescriptorSetLayoutCreateFlags::empty(),
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

        let manager = PipelineGroupManager::new([RayTracingPipelineManager::new(
            RayTracingPipelineBuildInfoCommon {
                layout: Arc::new(layout),
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
            },],
            vec![],
            vec![],
            pipeline_cache,
        )]);
        Self { manager, pipelines: SmallVec::new() }
    }
}

impl PbrPipeline {
    const PRIMARY_RAY: usize = 0;

    pub fn prepare_pipeline(
        mut this: ResMut<Self>,
        pipeline_cache: Res<PipelineCache>,
        shaders: Res<Assets<ShaderModule>>,
        pool: Res<DeferredOperationTaskPool>
    ) {
        if !this.pipelines.is_empty() {
            return;
        }
        this.pipelines = this.manager.build(&pipeline_cache, &shaders, &pool).unwrap_or_default();
    }
    pub fn trace_primary_rays(
        commands: RenderCommands<'c'>,
        mut this: ResMut<Self>,
        pipeline_cache: Res<PipelineCache>,
        shaders: Res<Assets<ShaderModule>>,
        pool: Res<DeferredOperationTaskPool>
    ) {
        if this.pipelines.is_empty() {
            return;
        }
        let pipeline = &mut this.pipelines[Self::PRIMARY_RAY];
        let Some(pipeline) = pipeline_cache.retrieve(pipeline, &shaders, &pool) else {
            return;
        };
        println!("We have the pipeline now");
    }
}
