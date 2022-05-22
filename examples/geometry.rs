use bevy_asset::{AssetServer, Handle};
use bevy_ecs::prelude::*;

use bevy_ecs::system::{Commands, Res};

use bevy_input::keyboard::KeyboardInput;
use bevy_input::ButtonState;
use bevy_transform::prelude::{GlobalTransform, Transform};

use dust_render::renderable::Renderable;

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
    .add_plugin(dust_format_explicit_aabbs::ExplicitAABBPlugin::default())
    .add_plugin(dust_render::pipeline::RayTracingRendererPlugin::<
        DefaultRenderer,
    >::default())
    .add_startup_system(setup)
    .add_system(print_keyboard_event_system);
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<dust_format_explicit_aabbs::AABBGeometry> =
        asset_server.load("../assets/out.aabb");
    commands
        .spawn()
        .insert(Renderable::default())
        .insert(Transform::default())
        .insert(GlobalTransform::default())
        .insert(handle);
}

fn print_keyboard_event_system(
    mut commands: Commands,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    query: Query<(
        Entity,
        &Renderable,
        &Handle<dust_format_explicit_aabbs::AABBGeometry>,
    )>,
) {
    for event in keyboard_input_events.iter() {
        match event {
            KeyboardInput {
                state: ButtonState::Pressed,
                ..
            } => {
                let (entity, _, _) = query.iter().next().unwrap();
                commands
                    .entity(entity)
                    .remove::<Handle<dust_format_explicit_aabbs::AABBGeometry>>();
            }
            _ => {}
        }
    }
}
