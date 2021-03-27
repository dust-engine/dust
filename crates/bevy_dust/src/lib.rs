use bevy::prelude::*;

use bevy::app::Events;
use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;
use dust_render::{CameraProjection, Renderer};

pub use dust_render::Octree;
pub use dust_render::Voxel;

#[derive(Default)]
pub struct DustPlugin;

impl Plugin for DustPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system_to_stage(StartupStage::PreStartup, setup.exclusive_system())
            .add_system(world_update.system());
    }
}

#[derive(Bundle, Default)]
pub struct RaytracerCameraBundle {
    pub camera_projection: CameraProjection,
    pub global_transform: GlobalTransform,
    pub transform: Transform,
}

fn setup(
    mut commands: Commands,
    mut window_created_events: EventReader<WindowCreated>,
    winit_windows: Res<WinitWindows>,
) {
    let window_id = window_created_events
        .iter()
        .next()
        .map(|event| event.id)
        .unwrap();

    let winit_window = winit_windows.get_window(window_id).unwrap();
    let (renderer, block_allocator) = Renderer::new(winit_window);
    let arena_allocator: svo::ArenaAllocator<Voxel> =
        svo::alloc::ArenaAllocator::new(block_allocator);
    let octree: Octree = svo::octree::Octree::new(arena_allocator);
    commands.insert_resource(octree);
    commands.insert_resource(renderer);
}

fn world_update(
    mut window_resized_events: EventReader<WindowResized>,
    mut renderer: ResMut<Renderer>,
    mut query: Query<(&mut CameraProjection, &GlobalTransform)>,
) {
    let (mut camera_projection, global_transform) = query
        .single_mut()
        .expect("Expecting an entity with RaytracerCameraBundle");

    for window_resized_event in window_resized_events.iter() {
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
