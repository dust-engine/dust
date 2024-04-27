mod pipeline;

use bevy::{app::{App, Plugin, PostUpdate, Startup}, ecs::schedule::IntoSystemConfigs};
pub use pipeline::*;


pub struct PbrRendererPlugin;
impl Plugin for PbrRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, PbrPipeline::prepare_pipeline);
        app.add_systems(PostUpdate, PbrPipeline::trace_primary_rays.after(PbrPipeline::prepare_pipeline));
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<PbrPipeline>();
    }
}
