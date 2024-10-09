#![feature(array_try_map)]

pub mod camera;
mod pipeline;

use bevy::{
    ecs::{query::With, schedule::IntoSystemConfigs},
    prelude::*,
    window::PrimaryWindow,
};
pub use pipeline::*;
use rhyolite::{acquire_swapchain_image, ecs::IntoRenderSystemConfigs, present, RhyoliteApp};

#[cfg(feature = "gizmos")]
mod gizmos;

#[derive(SystemSet, Hash, Debug, Clone, Eq, PartialEq)]
pub struct PbrRendererSystemSet;

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
                .in_set(PbrRendererSystemSet)
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

        {
            // Config swapchain
            use rhyolite::{ash::vk, SwapchainConfig};
            let world = app.world_mut();
            let primary_window = world
                .query_filtered::<Entity, With<PrimaryWindow>>()
                .iter(world)
                .next()
                .unwrap();
            let mut primary_window = world.entity_mut(primary_window);
            if !primary_window.contains::<SwapchainConfig>() {
                primary_window.insert(SwapchainConfig::default());
            }
            let mut swapchain_config = primary_window.get_mut::<SwapchainConfig>().unwrap();
            swapchain_config.image_usage |= vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE;
            swapchain_config.srgb_format = false;
        }
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<PbrPipeline>();
    }
}
