use std::{alloc::Layout, collections::HashMap, marker::PhantomData, sync::Arc};

use bevy_asset::AssetServer;
use bevy_ecs::{system::Resource, world::World};
use rhyolite::{HasDevice, PipelineCache};
use rhyolite_bevy::Allocator;

use crate::material::Material;

use super::{
    RayTracingPipeline, RayTracingPipelineCharacteristics,
    RayTracingPipelineCharacteristicsMaterialInfo,
};

#[derive(Resource)]
pub struct RayTracingPipelineBuilder<P: RayTracingPipeline> {
    allocator: rhyolite_bevy::Allocator,
    layout_size: usize,
    layout_align: usize,

    material_to_index: HashMap<std::any::TypeId, usize>,
    materials: Vec<RayTracingPipelineCharacteristicsMaterialInfo>,
    /// Raygen shaders, miss shaders, callable shaders.
    _marker: PhantomData<P>,
}
impl<P: RayTracingPipeline> RayTracingPipelineBuilder<P> {
    pub fn new(world: &World) -> Self {
        RayTracingPipelineBuilder {
            allocator: world.resource::<Allocator>().clone(),
            layout_align: 0,
            layout_size: 0,
            material_to_index: Default::default(),
            materials: Vec::new(),
            _marker: PhantomData,
        }
    }
    pub fn register_material<M: Material<Pipeline = P>>(&mut self, asset_server: &AssetServer) {
        let new_material_entry_layout = Layout::new::<M::ShaderParameters>();
        self.layout_size = self.layout_size.max(new_material_entry_layout.size());
        self.layout_align = self.layout_align.max(new_material_entry_layout.align());
        let id = self.materials.len();
        self.material_to_index
            .insert(std::any::TypeId::of::<M>(), id);
        self.materials
            .push(RayTracingPipelineCharacteristicsMaterialInfo {
                ty: M::TYPE,
                shaders: (0..P::num_raytypes())
                    .map(|ray_type| {
                        let rchit = M::rchit_shader(ray_type, asset_server);
                        let rint = M::intersection_shader(ray_type, asset_server);
                        let rahit = M::rahit_shader(ray_type, asset_server);
                        (rchit, rint, rahit)
                    })
                    .collect(),
            });
    }
    pub fn build(
        self,
        num_frame_in_flight: u32,
        asset_server: &AssetServer,
        pipeline_cache: Option<Arc<PipelineCache>>,
    ) -> P {
        let pipeline_layout = P::pipeline_layout(self.allocator.device());
        let characteristics = RayTracingPipelineCharacteristics {
            num_frame_in_flight,
            layout: pipeline_layout,
            sbt_param_layout: Layout::from_size_align(self.layout_size, self.layout_align)
                .unwrap_or(Layout::new::<()>()),
            material_to_index: self.material_to_index,
            materials: self.materials,
            num_raytype: P::num_raytypes(),
            create_info: Default::default(),
        };
        P::new(
            self.allocator,
            characteristics,
            asset_server,
            pipeline_cache,
        )
    }
}
