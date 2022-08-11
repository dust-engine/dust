#![feature(generic_const_exprs)]
use bevy_asset::{AssetServer, Assets, Handle};
use bevy_ecs::prelude::*;

use bevy_ecs::system::{Commands, Res};

use bevy_transform::prelude::{GlobalTransform, Transform};

use dust_format_vox::{PaletteMaterial, VoxGeometry};
use dust_render::camera::PerspectiveCamera;

use dust_render::renderable::Renderable;

use glam::{UVec3, Vec3};

use dust_render::renderer::Renderer as DefaultRenderer;

fn main() {
    let mut app = bevy_app::App::new();

    app.insert_resource(bevy_window::WindowDescriptor {
        title: "I am a window!".to_string(),
        width: 1280.,
        height: 800.,
        scale_factor_override: Some(1.0),
        ..Default::default()
    })
    .insert_resource(bevy_asset::AssetServerSettings {
        watch_for_changes: true,
        ..Default::default()
    })
    .add_plugin(bevy_core::CorePlugin::default())
    .add_plugin(bevy_transform::TransformPlugin::default())
    .add_plugin(bevy_input::InputPlugin::default())
    .add_plugin(bevy_window::WindowPlugin::default())
    .add_plugin(bevy_asset::AssetPlugin::default())
    //.add_plugin(dust_raytrace::DustPlugin::default())
    //add_plugin(bevy::scene::ScenePlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    //.add_plugin(flycamera::FlyCameraPlugin)
    //.add_plugin(fps_counter::FPSCounterPlugin)
    .add_plugin(dust_render::RenderPlugin::default())
    .add_plugin(dust_format_vox::VoxPlugin::default())
    .add_plugin(dust_render::pipeline::RayTracingRendererPlugin::<
        DefaultRenderer,
    >::default())
    .add_plugin(dust_render::material::MaterialPlugin::<PaletteMaterial>::default())
    .add_plugin(smooth_bevy_cameras::LookTransformPlugin)
    .add_plugin(smooth_bevy_cameras::controllers::fps::FpsCameraPlugin::default())
    .add_startup_system(setup);
    app.run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    _geometry: ResMut<Assets<VoxGeometry>>,
) {
    let mut test_geometry = VoxGeometry::new(1.0);
    test_geometry.set(UVec3::new(0, 0, 0), Some(true));
    test_geometry.set(UVec3::new(3, 3, 3), Some(true));
    test_geometry.set(UVec3::new(3, 3, 4), Some(true));
    test_geometry.set(UVec3::new(3, 3, 5), Some(true));
    //let handle = geometry.add(test_geometry);
    let handle: Handle<VoxGeometry> = asset_server.load("../assets/castle.vox");
    let material_handle: Handle<PaletteMaterial> =
        asset_server.load("../assets/castle.vox#material");
    //let material_handle: Handle<DensityMaterial> = asset_server.load("../assets/test.bmp");
    commands
        .spawn()
        .insert(Renderable::default())
        .insert(Transform::default())
        .insert(GlobalTransform::default())
        .insert(handle)
        .insert(material_handle);

    commands
        .spawn()
        .insert(PerspectiveCamera::default())
        .insert(Transform::default())
        .insert(GlobalTransform::default())
        .insert_bundle(smooth_bevy_cameras::controllers::fps::FpsCameraBundle::new(
            smooth_bevy_cameras::controllers::fps::FpsCameraController::default(),
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
}
