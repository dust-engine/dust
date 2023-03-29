use std::{collections::HashMap, sync::Arc, alloc::Layout};

use bevy_ecs::{system::Resource, prelude::Component};
use rhyolite::PipelineLayout;
use rhyolite_bevy::Allocator;
mod builder;
mod manager;
mod plugin;
mod standard;

use crate::{material::Material, sbt::{SbtIndex, SbtManager}, shader::SpecializedShader, Renderable};
pub use builder::RayTracingPipelineBuilder;
pub use manager::{RayTracingPipelineManager, RayTracingPipelineManagerSpecializedPipeline};

pub use standard::StandardPipeline;
pub use plugin::RayTracingPipelinePlugin;

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
    pub sbt_param_layout: Layout,
    material_to_index: HashMap<std::any::TypeId, usize>,
    materials: Vec<RayTracingPipelineCharacteristicsMaterialInfo>,
    /// Raygen shaders, miss shaders, callable shaders.
    shaders: Vec<SpecializedShader>,

    pub num_raytype: u32,
    create_info: rhyolite::RayTracingPipelineLibraryCreateInfo,
}

/// Generally contains one or more RayTracingPipelineManager,
/// and one SbtManager
pub trait RayTracingPipeline: Send + Sync + 'static + Resource {
    /// A marker type for entities applicable to this pipeline.
    /// SbtIndex will only be inserted for entities with the marker component.
    type Marker: Component = Renderable;
    fn num_raytypes() -> u32 {
        1
    }

    fn pipeline_layout(device: &Arc<rhyolite::Device>) -> Arc<PipelineLayout>;
    /// Assuming that the ray tracing pipeline contains a number of sub-pipelines,
    /// each managed by a RayTracingPipelineManager,
    /// implementations generally need to do the following:
    /// 1. For each sub-pipeline rendering the material, call material_instance_added
    /// 2. map from (material, raytype) to hitgroup index using the pipeline objects.
    /// hitgroup index needs to be adjusted by subpipeline
    /// 3. Call material instance add(material.parameters, hitgroup_index) on sbtmanager
    fn material_instance_added<M: Material<Pipeline = Self>>(&mut self, material: &M) -> SbtIndex;
    fn material_instance_removed<M: Material<Pipeline = Self>>(&mut self) {}

    fn new(
        allocator: Allocator,
        pipeline_characteristic: RayTracingPipelineCharacteristics,
        pipeline_cache: Option<Arc<rhyolite::PipelineCache>>,
    ) -> Self;
}
