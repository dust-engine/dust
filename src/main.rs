#![feature(impl_trait_in_fn_trait_return)]

use bevy::prelude::*;
use rhyolite::{ash::vk, tracking::Access};
use rhyolite_bevy::{DefaultRenderSet, RenderSetSharedStateWrapper, swapchain::SwapchainImage};
fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins)
        .add_plugins(rhyolite_bevy::SurfacePlugin::default())
        .add_plugins(rhyolite_bevy::DebugUtilsPlugin::default())
        .add_plugins(rhyolite_bevy::RhyolitePlugin::default())
        .add_plugins(rhyolite_bevy::swapchain::SwapchainPlugin);

    // Dust plugins
    app.add_plugins(dust_vox::VoxPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(rhyolite_bevy::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.add_systems(Startup, startup_system);
    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));
    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let scene: Handle<Scene> = asset_server.load("castle.vox");
    commands.spawn(SceneRoot(scene));
}

fn clear(
    mut image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderSetSharedStateWrapper,
) {
    let Ok(mut image) = image.single_mut() else {
        return;
    };
    let image = image.write(
        vk::PipelineStageFlags2::CLEAR,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    state.record(|encoder| {
        let mut image = image.map(|x| encoder.lock(&x, vk::PipelineStageFlags2::CLEAR));
        let image = encoder.use_image_resource(
            &mut image,
            Access::CLEAR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder.clear_color_image(
            image,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 1.0, 0.0],
            },
        );
    });
}
