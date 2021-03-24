use bevy::prelude::*;

use bevy::app::Events;
use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;
use dust_render::{CameraProjection, Renderer};

mod voxel;
pub use voxel::Voxel;

pub type Octree = svo::octree::Octree<Voxel>;

#[derive(Default)]
pub struct DustPlugin;

impl Plugin for DustPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system(setup.exclusive_system())
            .add_system(world_update.system());
    }
}

#[derive(Bundle, Default)]
pub struct RaytracerCameraBundle {
    pub camera_projection: CameraProjection,
    pub global_transform: GlobalTransform,
    pub transform: Transform,
}

fn setup(world: &mut World) {
    let window_created_events = world.get_resource_mut::<Events<WindowCreated>>().unwrap();
    let mut window_created_events_reader = window_created_events.get_reader();
    let window_id = window_created_events_reader
        .iter(&window_created_events)
        .next()
        .map(|event| event.id)
        .unwrap();

    // Update camera projection
    let windows = world.get_resource::<Windows>().unwrap();
    let window = windows.get(window_id).unwrap();
    let aspect_ratio = window.width() / window.height();
    let mut query = world.query::<&mut CameraProjection>();
    for mut camera_projection in query.iter_mut(world) {
        camera_projection.aspect_ratio = aspect_ratio;
    }

    let winit_windows = world.get_resource::<WinitWindows>().unwrap();
    let winit_window = winit_windows.get_window(window_id).unwrap();
    let mut renderer = Renderer::new(winit_window);
    let block_allocator = renderer.create_block_allocator().unwrap();
    let arena_allocator: svo::ArenaAllocator<Voxel> =
        svo::alloc::ArenaAllocator::new(block_allocator);
    let octree: Octree = svo::octree::Octree::new(arena_allocator);
    world.insert_resource(octree);
    world.insert_resource(renderer);
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
