use std::{alloc::Layout, collections::HashMap, sync::Arc};

use bevy_asset::AssetServer;
use bevy_ecs::{
    prelude::Component,
    system::{Resource, SystemParamItem},
};
use rhyolite::PipelineLayout;
use rhyolite_bevy::Allocator;
mod builder;
mod manager;
mod plugin;
mod standard;

use crate::{material::Material, sbt::SbtIndex, shader::SpecializedShader, Renderable};
pub use builder::RayTracingPipelineBuilder;
pub use manager::{RayTracingPipelineManager, RayTracingPipelineManagerSpecializedPipeline};

pub use plugin::RayTracingPipelinePlugin;
pub use standard::StandardPipeline;

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
    pub num_frame_in_flight: u32,
    layout: Arc<PipelineLayout>,
    pub sbt_param_layout: Layout,
    material_to_index: HashMap<std::any::TypeId, usize>,
    materials: Vec<RayTracingPipelineCharacteristicsMaterialInfo>,

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
    fn material_instance_added<M: Material<Pipeline = Self>>(
        &mut self,
        material: &M,
        params: &mut SystemParamItem<M::ShaderParameterParams>,
    ) -> SbtIndex;
    fn material_instance_removed<M: Material<Pipeline = Self>>(&mut self) {}

    fn create_info() -> rhyolite::RayTracingPipelineLibraryCreateInfo {
        Default::default()
    }

    fn new(
        allocator: Allocator,
        pipeline_characteristic: RayTracingPipelineCharacteristics,
        asset_server: &AssetServer,
        pipeline_cache: Option<Arc<rhyolite::PipelineCache>>,
    ) -> Self;
}
