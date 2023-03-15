use std::{alloc::Layout, collections::HashMap, marker::PhantomData, sync::Arc};

use bevy_ecs::{system::Resource, world::World};
use rhyolite::{PipelineCache, PipelineLayout};
use rhyolite_bevy::Allocator;

use crate::{material::Material, shader::SpecializedShader};

use super::{
    RayTracingPipeline, RayTracingPipelineCharacteristics,
    RayTracingPipelineCharacteristicsMaterialInfo,
};

#[derive(Resource)]
pub struct RayTracingPipelineBuilder<P: RayTracingPipeline> {
    allocator: rhyolite::Allocator,
    layout: Layout,

    pipeline_layout: Arc<PipelineLayout>,
    material_to_index: HashMap<std::any::TypeId, usize>,
    materials: Vec<RayTracingPipelineCharacteristicsMaterialInfo>,
    /// Raygen shaders, miss shaders, callable shaders.
    shaders: Vec<SpecializedShader>,
    _marker: PhantomData<P>,
}
impl<P: RayTracingPipeline> RayTracingPipelineBuilder<P> {
    pub fn new(world: &World, layout: Arc<PipelineLayout>) -> Self {
        RayTracingPipelineBuilder {
            allocator: world.resource::<Allocator>().clone().into_inner(),
            layout: Layout::from_size_align(0, 0).unwrap(),
            pipeline_layout: layout,
            material_to_index: Default::default(),
            materials: Vec::new(),
            shaders: Vec::new(),
            _marker: PhantomData,
        }
    }
    pub fn register_material<M: Material<Pipeline = P>>(&mut self) {
        let new_material_entry_layout = Layout::new::<M::ShaderParameters>();
        self.layout = Layout::from_size_align(
            self.layout.size().max(new_material_entry_layout.size()),
            self.layout.align().max(new_material_entry_layout.align()),
        )
        .unwrap();
        let id = self.materials.len();
        self.material_to_index
            .insert(std::any::TypeId::of::<M>(), id);
        self.materials
            .push(RayTracingPipelineCharacteristicsMaterialInfo {
                ty: M::TYPE,
                shaders: (0..P::num_raytypes())
                    .map(|ray_type| {
                        let rchit = M::rchit_shader(ray_type);
                        let rint = M::intersection_shader(ray_type);
                        let rahit = M::intersection_shader(ray_type);
                        (rchit, rint, rahit)
                    })
                    .collect(),
            })
    }
    fn build(self, pipeline_cache: Option<Arc<PipelineCache>>) -> P {
        let characteristics = RayTracingPipelineCharacteristics {
            layout: self.pipeline_layout,
            material_to_index: self.material_to_index,
            materials: self.materials,
            shaders: self.shaders,
            num_raytype: P::num_raytypes(),
            create_info: Default::default(),
        };
        P::new(characteristics, pipeline_cache)
    }
}
