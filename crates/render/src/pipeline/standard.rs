use std::sync::Arc;

use bevy_ecs::system::Resource;
use rhyolite::{macros::set_layout, ash::vk};
use rhyolite_bevy::Allocator;

use crate::sbt::SbtManager;

use super::{RayTracingPipeline, RayTracingPipelineManager};

#[derive(Resource)]
pub struct StandardPipeline {
    primary_ray_pipeline: RayTracingPipelineManager,
    sbt_manager: SbtManager,
}

impl RayTracingPipeline for StandardPipeline {
    fn pipeline_layout(device: &Arc<rhyolite::Device>) -> Arc<rhyolite::PipelineLayout> {
        let set1 = set_layout! {
            img_output: vk::DescriptorType::SAMPLED_IMAGE,
            accel_struct: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }.build(device.clone()).unwrap();
        Arc::new(rhyolite::PipelineLayout::new(
            device.clone(),
            vec![
                Arc::new(set1)
            ],
            vk::PipelineLayoutCreateFlags::empty()
        ).unwrap())
    }
    fn new(
        allocator: Allocator,
        pipeline_characteristic: super::RayTracingPipelineCharacteristics,
        pipeline_cache: Option<std::sync::Arc<rhyolite::PipelineCache>>,
    ) -> Self {
        let pipeline_characteristics = Arc::new(pipeline_characteristic);
        let sbt_manager = SbtManager::new(allocator, &pipeline_characteristics);
        Self {
            sbt_manager,
            primary_ray_pipeline: RayTracingPipelineManager::new(
                pipeline_characteristics,
                pipeline_cache,
            ),
        }
    }
    fn material_instance_added<M: crate::Material<Pipeline = Self>>(
        &mut self,
        material: &M,
    ) -> crate::sbt::SbtIndex {
        self.primary_ray_pipeline.material_instance_added::<M>();
        self.sbt_manager.add_instance(material)
    }

    fn num_raytypes() -> u32 {
        1
    }

    fn material_instance_removed<M: crate::Material<Pipeline = Self>>(&mut self) {}
}
