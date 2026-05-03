#![feature(impl_trait_in_fn_trait_return)]

mod flycam;

use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use pumicite::{ash::vk, swapchain::SwapchainColorMode};

use crate::flycam::{FlyCamera, FlyCameraPlugin};

#[derive(Component)]
struct MovingTeapot {
    origin: Vec3,
    radius: f32,
    height: f32,
    angular_speed: f32,
    spin_speed: f32,
}

fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(dust_app::DustApp)
        .add_plugins(bevy::DefaultPlugins)
        .add_plugins(bevy_pumicite::SurfacePlugin::default())
        .add_plugins(bevy_pumicite::DebugUtilsPlugin::default())
        .add_plugins(dust_denoiser::dlss::DLSSPlugin)
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

    app.add_systems(Startup, startup_system.after(CreateDevice))
        .add_systems(Update, animate_teapot_system);

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let scene: Handle<Scene> = asset_server.load("bazel://dust/assets/castle.vox");
    commands.spawn(SceneRoot(scene));

    let teapot_origin = Vec3::new(122.0, 260.0, 180.0);
    let mut teapot_transform = Transform::from_translation(teapot_origin);
    teapot_transform.scale = Vec3::splat(1.5);

    let teapot: Handle<Scene> = asset_server.load("bazel://dust/assets/teapot.vox");
    commands.spawn((
        SceneRoot(teapot),
        teapot_transform,
        GlobalTransform::default(),
        MovingTeapot {
            origin: teapot_origin,
            radius: 56.0,
            height: 18.0,
            angular_speed: 0.6,
            spin_speed: 1.4,
        },
    ));
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

fn animate_teapot_system(time: Res<Time>, mut teapots: Query<(&MovingTeapot, &mut Transform)>) {
    let elapsed = time.elapsed_secs();

    for (teapot, mut transform) in teapots.iter_mut() {
        let angle = elapsed * teapot.angular_speed;
        transform.translation = teapot.origin
            + Vec3::new(
                angle.cos() * teapot.radius,
                (angle * 1.7).sin() * teapot.height,
                angle.sin() * teapot.radius,
            );
        transform.rotation = Quat::from_rotation_y(elapsed * teapot.spin_speed);
    }
}
