use std::{alloc::Layout, marker::PhantomData};

use bevy::{asset::AssetServer, ecs::system::Resource};
use rhyolite::pipeline::sbt::SbtManager;

use crate::material::{Material, SbtMarker};

pub trait RayTracingPipeline: Resource + Sized {
    const NUM_RAYTYPES: usize;
    fn manager(&self) -> &SbtManager<SbtMarker<Self>, { Self::NUM_RAYTYPES }>;
    fn manager_mut(&mut self) -> &mut SbtManager<SbtMarker<Self>, { Self::NUM_RAYTYPES }>;
}

#[derive(Resource)]
pub struct RayTracingPipelineBuilder<P: RayTracingPipeline> {
    layout_size: usize,
    layout_align: usize,
    _marker: PhantomData<P>,
}

impl<P: RayTracingPipeline> RayTracingPipelineBuilder<P> {
    pub fn register_material<M: Material<Pipeline = P>>(&mut self, asset_server: &AssetServer) {
        let new_material_entry_layout = Layout::new::<M::ShaderParameters>();
        self.layout_size = self.layout_size.max(new_material_entry_layout.size());
        self.layout_align = self.layout_align.max(new_material_entry_layout.align());
    }
}
