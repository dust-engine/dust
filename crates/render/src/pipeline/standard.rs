use std::sync::Arc;

use bevy_ecs::system::Resource;

use super::{RayTracingPipeline, RayTracingPipelineManager};

#[derive(Resource)]
pub struct StandardPipeline {
    primary_ray_pipeline: RayTracingPipelineManager,
}

impl RayTracingPipeline for StandardPipeline {
    fn new(
        pipeline_characteristic: super::RayTracingPipelineCharacteristics,
        pipeline_cache: Option<std::sync::Arc<rhyolite::PipelineCache>>,
    ) -> Self {
        let pipeline_characteristics = Arc::new(pipeline_characteristic);
        Self {
            primary_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics,
                pipeline_cache,
            ),
        }
    }
    fn material_instance_added<M: crate::Material<Pipeline = Self>>(
        &mut self,
        _material: &M,
    ) -> crate::sbt::SbtIndex {
        self.primary_ray_pipeline.material_instance_added::<M>();
        todo!()
    }

    fn num_raytypes() -> u32 {
        1
    }

    fn material_instance_removed<M: crate::Material<Pipeline = Self>>(&mut self) {}
}
