use crate::fly_camera::FlyCamera;
use bevy::prelude::*;
use bevy_dust::{Octree, RaytracerCameraBundle, SunLight, Voxel};

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
        .add_system(run.system())
        .run();
}

fn setup(mut commands: Commands, mut octree: ResMut<Octree>) {
    let monument = dot_vox::load("assets/monu9.vox").unwrap();
    let model = &monument.models[0];
    let mut octree_mutator = octree.get_random_mutator();

    for v in model.voxels.iter() {
        octree_mutator.set(
            v.x as u32,
            v.z as u32,
            v.y as u32,
            512,
            Voxel::with_id(v.i as u16),
        );
    }
    //octree_mutator.set(0, 0, 0, 512, Voxel::with_id(3));
    octree_mutator.commit();

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(4.2, 4.4, 4.0);
    commands
        .spawn()
        .insert_bundle(bundle)
        .insert(FlyCamera::default());
}

fn run(mut sunlight: ResMut<SunLight>, time: Res<Time>) {
    let (sin, cos) = (time.seconds_since_startup() * 2.0).sin_cos();
    sunlight.dir = Vec3::new(sin as f32 * 10.0, -3.0, cos as f32 * 10.0).normalize();
}
