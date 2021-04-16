use crate::fly_camera::FlyCamera;
use bevy::prelude::*;
use dust_core::svo::mesher::MarchingCubeMeshBuilder;
use dust_core::{Octree, SunLight, Voxel};
use dust_render::{RaytracerCameraBundle, RenderResources};
use std::io::BufWriter;
use std::ops::DerefMut;
use dust_core::svo::octree::supertree::{Supertree, OctreeLoader};
use dust_core::svo::alloc::BlockAllocator;
use std::sync::Arc;
use dust_core::svo::ArenaAllocator;

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
        .add_plugin(dust_render::DustPlugin::default())
        .add_plugin(fly_camera::FlyCameraPlugin)
        .add_startup_system(setup_from_oct_file.system())
        .add_system(run.system())
        .run();
}

fn setup_from_oct_file(
    mut commands: Commands,
) {

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(50.0, 6.0, 50.0);
    bundle
        .transform
        .look_at(Vec3::new(100.0, 0.0, 120.0), Vec3::Y);
    commands
        .spawn()
        .insert_bundle(bundle)
        .insert(FlyCamera::default());
}

fn run(mut sunlight: ResMut<SunLight>, time: Res<Time>) {
    let (sin, cos) = (time.seconds_since_startup() * 2.0).sin_cos();
    sunlight.dir = Vec3::new(sin as f32 * 10.0, -15.0, cos as f32 * 10.0).normalize();
}
