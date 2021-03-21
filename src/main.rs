use bevy::prelude::*;
use bevy::app::Events;
use bevy::window::{WindowCreated, WindowResized, WindowScaleFactorChanged};
use bevy::winit::WinitWindows;
use dust_render::{Renderer, Config};
use dust_render::hal::Instance;
use std::borrow::Cow;

fn main() {
    App::build()
        .insert_resource(bevy::log::LogSettings {
            filter: "wgpu=error".to_string(),
            level: bevy::utils::tracing::Level::TRACE
        })
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_startup_system(setup.system())
        .add_system(game_update.system())
        .add_system(render_update.exclusive_system())
        .insert_resource(Renderer::new(Config {
            name: Cow::Borrowed("dust engine"),
            version: 0
        }))
        .run();
}

fn setup(
    mut commands: Commands,
) {
}

fn game_update(
    mut window_created_events: EventReader<WindowCreated>,
    mut window_resized_events: EventReader<WindowResized>,
    winit_windows: Res<WinitWindows>,
    mut renderer: ResMut<Renderer>
) {
    for window_created_event in window_created_events.iter() {
        let window = winit_windows.get_window(window_created_event.id).unwrap();
        let surface = unsafe {
            renderer.instance.create_surface(window).unwrap()
        };
        renderer.set_surface(surface);
    }
}

fn render_update(world: &mut World) {
    let mut renderer = world.get_resource_mut::<Renderer>().unwrap();
    renderer.update();
}