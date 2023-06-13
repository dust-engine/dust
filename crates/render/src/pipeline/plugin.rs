use std::marker::PhantomData;

use bevy_app::Plugin;
use bevy_asset::{AssetEvent, AssetServer};
use bevy_ecs::{prelude::EventReader, system::ResMut};

use crate::{RayTracingPipeline, RayTracingPipelineBuilder, ShaderModule};

pub struct RayTracingPipelinePlugin<P: RayTracingPipeline> {
    _marker: PhantomData<P>,
}
impl<P: RayTracingPipeline> Default for RayTracingPipelinePlugin<P> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<P: RayTracingPipeline> Plugin for RayTracingPipelinePlugin<P> {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RayTracingPipelineBuilder::<P>::new(&app.world));
    }
    fn cleanup(&self, app: &mut bevy_app::App) {
        let builder = app
            .world
            .remove_resource::<RayTracingPipelineBuilder<P>>()
            .unwrap();
        let queues = app.world.resource::<rhyolite_bevy::Queues>();
        let asset_server = app.world.resource::<AssetServer>();
        let pipeline = builder.build(queues.num_frame_in_flight(), asset_server);
        app.insert_resource(pipeline);
    }
}
