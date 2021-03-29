use crate::fly_camera::{FlyCamera, FlyCameraPlugin};
use bevy::prelude::*;
use bevy_dust::{Octree, RaytracerCameraBundle, Voxel};

mod fly_camera;

fn main() {
    App::build()
        .insert_resource(bevy::log::LogSettings {
            filter: "wgpu=error".to_string(),
            level: bevy::utils::tracing::Level::DEBUG,
        })
        .insert_resource(bevy::window::WindowDescriptor {
            width: 1920.0,
            height: 1080.0,
            scale_factor_override: Some(1.0),
            title: "Dust Engine".to_string(),
            mode: bevy::window::WindowMode::Windowed,
            ..Default::default()
        })
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(bevy_dust::DustPlugin::default())
        .add_plugin(fly_camera::FlyCameraPlugin)
        .add_startup_system(setup.system())
        .run();
}

fn setup(mut commands: Commands) {

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation.z = 2.0;
    commands
        .spawn()
        .insert_bundle(bundle)
        .insert(FlyCamera::default());
}
