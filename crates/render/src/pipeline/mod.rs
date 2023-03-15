use std::{collections::HashMap, sync::Arc};

use bevy_ecs::system::Resource;
use rhyolite::PipelineLayout;
mod builder;
mod manager;

use crate::{material::Material, sbt::SbtIndex, shader::SpecializedShader};
pub use builder::RayTracingPipelineBuilder;
struct RayTracingPipelineCharacteristicsMaterialInfo {
    ty: rhyolite::RayTracingHitGroupType,
    /// Pipeline library containing n hitgroups, where n = number of ray types.
    shaders: Vec<(
        Option<SpecializedShader>,
        Option<SpecializedShader>,
        Option<SpecializedShader>,
    )>,
}
impl RayTracingPipelineCharacteristics {
    pub fn material_count(&self) -> usize {
        self.material_to_index.len()
    }
}

pub struct RayTracingPipelineCharacteristics {
    layout: Arc<PipelineLayout>,
    material_to_index: HashMap<std::any::TypeId, usize>,
    materials: Vec<RayTracingPipelineCharacteristicsMaterialInfo>,
    /// Raygen shaders, miss shaders, callable shaders.
    shaders: Vec<SpecializedShader>,

    num_raytype: u32,
    create_info: rhyolite::RayTracingPipelineLibraryCreateInfo,
}

/// Generally contains one or more RayTracingPipelineManager,
/// and one SbtManager
pub trait RayTracingPipeline: Send + Sync + 'static + Resource {
    fn num_raytypes() -> u32 {
        1
    }

    fn material_instance_added<M: Material<Pipeline = Self>>(&mut self, _material: &M) -> SbtIndex {
        todo!("For each subpipeline that needs to use the material, call material_instance_added");
        todo!("For each subpipeline, call get_pipeline");
        // map from (material, raytype) to hitgroup index using the pipeline objects.
        // hitgroup index needs to be adjusted by subpipeline
        todo!("Call material instance add(material.parameters, hitgroup_index) on sbtmanager")
    }
    fn material_instance_removed<M: Material<Pipeline = Self>>(&mut self) {}

    fn new(
        pipeline_characteristic: RayTracingPipelineCharacteristics,
        pipeline_cache: Option<Arc<rhyolite::PipelineCache>>,
    ) -> Self;
}
