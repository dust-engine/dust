#![feature(impl_trait_in_fn_trait_return)]

mod flycam;

use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use pumicite::{Allocator, ash::vk, swapchain::SwapchainColorMode, tracking::Access};

use crate::flycam::{FlyCamera, FlyCameraPlugin};
fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(dust_app::DustApp)
        .add_plugins(bevy::DefaultPlugins)
        .add_plugins(bevy_pumicite::SurfacePlugin::default())
        .add_plugins(bevy_pumicite::DebugUtilsPlugin::default())
        .add_plugins(dust_dlss::DLSSPlugin)
        .add_plugins(bevy_pumicite::PumicitePlugin::default())
        .add_plugins(bevy_pumicite::swapchain::SwapchainPlugin);

    app.add_plugins(FlyCameraPlugin);

    // Dust plugins
    app.add_plugins(dust_pbr::PbrRenderPlugin)
        .add_plugins(dust_vox::VoxPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert((
            dust_pbr::camera::Camera::default(),
            GlobalTransform::default(),
            Transform::from_translation(Vec3::new(122.0, 300.61, 54.45)),
            FlyCamera::default(),
        ))
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            color_mode: SwapchainColorMode::HDR,
            ..Default::default()
        });

    app.add_systems(Startup, startup_system.after(CreateDevice));

    app.run();
}

fn startup_system(
    mut commands: Commands,
    asset_server: Res<bevy::asset::AssetServer>,
    allocator: Res<Allocator>,
    mut geometries: ResMut<Assets<dust_vox::VoxGeometry>>,
    mut materials: ResMut<Assets<dust_vox::VoxMaterial>>,
    mut palettes: ResMut<Assets<dust_vox::VoxPalette>>,
) {
    let scene: Handle<Scene> = asset_server.load("bazel://dust/assets/castle.vox");
    commands.spawn(SceneRoot(scene));
    return;
    /*

    let mut geometry = dust_vox::VoxGeometry::new(allocator.clone(), 1.0);
    let mut material = dust_vox::VoxMaterial::new(allocator.clone());
    let mut accessor = geometry.tree.accessor_mut(&mut material);
    accessor.set(UVec3::new(8, 9, 10), 123);
    accessor.end();

    let model = commands.spawn(VoxModel {
        geometry: geometries.add(geometry),
        material: materials.add(material),
        palette: palettes.add(dust_vox::VoxPalette::colorful()),
        sbt_index: u32::MAX,
    }).id();
    commands.spawn(VoxInstanceBundle {
        transform: Default::default(),
        global_transform: Default::default(),
        instance: VoxInstance { model },
    });
    */
}
