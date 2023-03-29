use std::marker::PhantomData;

use bevy_app::Plugin;

use crate::{RayTracingPipeline, RayTracingPipelineBuilder};


pub struct RayTracingPipelinePlugin<P: RayTracingPipeline> {
    _marker: PhantomData<P>,
}
impl<P: RayTracingPipeline> Default for RayTracingPipelinePlugin<P> {
    fn default() -> Self {
        Self {
            _marker: PhantomData
        }
    }
}
impl<P: RayTracingPipeline> Plugin for RayTracingPipelinePlugin<P> {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RayTracingPipelineBuilder::<P>::new(&app.world));
    }
    fn setup(&self, app: &mut bevy_app::App) {
        let builder = app
            .world
            .remove_resource::<RayTracingPipelineBuilder<P>>()
            .unwrap();
        let pipeline_cache: Option<&rhyolite_bevy::PipelineCache> = app.world.get_resource();
        let pipeline_cache = pipeline_cache.map(|a| a.inner().clone());
        let pipeline = builder.build(pipeline_cache);
        app.insert_resource(pipeline);
    }
}
