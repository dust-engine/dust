use bevy::asset::{AssetServer, Handle};
use bevy::ecs::schedule::IntoSystemConfigs;
use bevy::scene::{Scene, SceneBundle};
use dust_vox::VoxPlugin;
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate, Startup};
use bevy::ecs::system::{Commands, In, Query, Res};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::commands::{CommonCommands, ResourceTransitionCommands};
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::ecs::{Barriers, IntoRenderSystemConfigs, RenderCommands};
use rhyolite::{
    acquire_swapchain_image, present, Access, RhyolitePlugin, SurfacePlugin, SwapchainConfig,
    SwapchainImage, SwapchainPlugin,
};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins.set::<bevy::asset::AssetPlugin>(
        bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        },
    ))
    .add_plugins(SurfacePlugin::default())
    .add_plugins(DebugUtilsPlugin::default())
    .add_plugins(RhyolitePlugin::default())
    .add_plugins(SwapchainPlugin::default());

    
    app.add_plugins(dust_pbr::PbrRendererPlugin);

    app.add_plugins(VoxPlugin);

    app.add_systems(Startup, startup_system);

    let primary_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            srgb_format: false,
            ..Default::default()
        });

    app.run();
}


fn startup_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene: Handle<Scene> = asset_server.load("castle.vox");
    commands.spawn(SceneBundle {
        scene,
        ..Default::default()
    });
}
