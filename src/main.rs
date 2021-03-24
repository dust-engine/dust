use bevy::prelude::*;
use bevy_dust::{RaytracerCameraBundle, Octree, Voxel};

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
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut octree: ResMut<Octree>,
) {
    let monument = dot_vox::load("assets/monu9.vox").unwrap();
    let model = &monument.models[0];
    for voxel in &model.voxels {
        octree.set(
            voxel.x as u32,
            voxel.z as u32,
            voxel.y as u32,
            256,
            Voxel::with_id(voxel.i as u16),
        );
    }

    commands.spawn(RaytracerCameraBundle::default());
}
