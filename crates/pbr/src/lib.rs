mod pipeline;

use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    ecs::{query::With, schedule::IntoSystemConfigs},
    window::PrimaryWindow,
};
pub use pipeline::*;
use rhyolite::{acquire_swapchain_image, ecs::IntoRenderSystemConfigs, present, RhyoliteApp};

pub struct PbrRendererPlugin;
impl Plugin for PbrRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, PbrPipeline::prepare_pipeline);
        app.add_systems(
            PostUpdate,
            PbrPipeline::trace_primary_rays
                .after(PbrPipeline::prepare_pipeline)
                .after(acquire_swapchain_image::<With<PrimaryWindow>>)
                .after(rhyolite_rtx::build_tlas::<rhyolite_rtx::DefaultTLAS>)
                .before(present)
                .with_barriers(PbrPipeline::trace_primary_rays_barrier),
        );

        app.add_device_extension::<rhyolite::ash::khr::push_descriptor::Meta>()
            .unwrap();
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<PbrPipeline>();
    }
}
