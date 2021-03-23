use bevy::prelude::*;

use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;
use dust_render::hal::Instance;
use dust_render::{CameraProjection, Config, Renderer};
use std::borrow::Cow;

#[derive(Default)]
pub struct DustPlugin;

impl Plugin for DustPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system(setup.system())
            .add_system(game_update.system())
            .insert_resource(Renderer::new(Config {
                name: Cow::Borrowed("dust engine"),
                version: 0,
            }));
    }
}

#[derive(Bundle, Default)]
pub struct RaytracerCameraBundle {
    pub camera_projection: CameraProjection,
    pub global_transform: GlobalTransform,
    pub transform: Transform,
}

fn setup(mut commands: Commands) {
    commands.spawn(RaytracerCameraBundle::default());
}

fn game_update(
    mut window_created_events: EventReader<WindowCreated>,
    mut window_resized_events: EventReader<WindowResized>,
    windows: Res<Windows>,
    winit_windows: Res<WinitWindows>,
    mut renderer: ResMut<Renderer>,
    mut query: Query<(&mut CameraProjection, &GlobalTransform)>,
) {
    let (mut camera_projection, global_transform) = query
        .single_mut()
        .expect("Expecting an entity with RaytracerCameraBundle");

    for window_created_event in window_created_events.iter() {
        let window = windows.get(window_created_event.id).unwrap();
        let aspect_ratio = window.width() / window.height();
        let winit_window = winit_windows.get_window(window_created_event.id).unwrap();
        camera_projection.aspect_ratio = aspect_ratio;
        let surface = unsafe { renderer.instance.create_surface(winit_window).unwrap() };
        renderer.set_surface(surface);
    }
    for window_resized_event in window_resized_events.iter() {
        let aspect_ratio = window_resized_event.width / window_resized_event.height;
        camera_projection.aspect_ratio = aspect_ratio;
        renderer.on_resize();
    }
    let camera_transform = glam::TransformRT {
        rotation: global_transform.rotation,
        translation: global_transform.translation,
    };
    renderer.update(&dust_render::State {
        camera_projection: &*camera_projection,
        camera_transform: &camera_transform,
    });
}
