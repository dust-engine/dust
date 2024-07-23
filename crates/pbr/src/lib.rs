#![feature(array_try_map)]

pub mod camera;
mod pipeline;

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{query::With, schedule::IntoSystemConfigs},
    window::PrimaryWindow,
};
pub use pipeline::*;
use rhyolite::{acquire_swapchain_image, ecs::IntoRenderSystemConfigs, present, RhyoliteApp};

#[cfg(feature = "gizmos")]
mod gizmos;

pub struct PbrRendererPlugin;
impl Plugin for PbrRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, PbrPipeline::prepare_pipeline);
        app.add_systems(PostUpdate, {
            let s = PbrPipeline::trace_primary_rays
                .after(PbrPipeline::prepare_pipeline)
                .after(acquire_swapchain_image::<With<PrimaryWindow>>)
                .after(rhyolite_rtx::build_tlas::<rhyolite_rtx::DefaultTLAS>)
                .before(present)
                .with_barriers(PbrPipeline::trace_primary_rays_barrier);

            // Draw the gizmos after the ray tracing renderer
            #[cfg(feature = "gizmos")]
            let s = s.before(rhyolite_gizmos::GizmoSystemSet);
            s
        });

        app.add_device_extension::<rhyolite::ash::khr::push_descriptor::Meta>()
            .unwrap();

        #[cfg(feature = "gizmos")]
        {
            rhyolite_gizmos::add_draw_delegate::<gizmos::PbrRendererDelegate>(app);
        }
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<PbrPipeline>();
    }
}
