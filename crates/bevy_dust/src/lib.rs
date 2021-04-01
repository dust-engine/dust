use bevy::prelude::*;

use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;
use dust_render::{CameraProjection, Renderer};

pub use dust_render::Octree;
pub use dust_render::SunLight;
pub use dust_render::Voxel;

use std::borrow::BorrowMut;
use svo::alloc::CHUNK_SIZE;
use svo::ArenaAllocator;

#[derive(Default)]
pub struct DustPlugin;

impl Plugin for DustPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(SunLight::new(Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO))
            .add_startup_system_to_stage(StartupStage::PreStartup, setup.exclusive_system())
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
    let mut renderer = dust_render::renderer::Renderer::new(winit_window);
    renderer.create_raytracer(); // More like "Enter Raytracing Mode"
    let (block_allocator, block_allocator_buffer) =
        renderer.create_block_allocator(CHUNK_SIZE as u64);
    unsafe {
        let raytracer = renderer.raytracer.as_mut().unwrap();
        raytracer.bind_block_allocator_buffer(block_allocator_buffer);
        renderer.swapchain.bind_render_pass(raytracer);
    }
    let arena = ArenaAllocator::new(block_allocator);
    let octree = Octree::new(arena);

    // Insert the octree before the renderer so that the octree would be dropped first.
    // This isn't safe at all... TODO: manually drop these resources on state exit.
    commands.insert_resource(octree);
    commands.insert_resource(renderer);
}

fn world_update(
    mut window_resized_events: EventReader<WindowResized>,
    mut renderer: ResMut<Renderer>,
    sunlight: Res<SunLight>,
    mut query: Query<(&mut CameraProjection, &GlobalTransform)>,
) {
    let renderer: &mut Renderer = renderer.borrow_mut();
    let (camera_projection, global_transform) = query
        .single_mut()
        .expect("Expecting an entity with RaytracerCameraBundle");

    for _window_resized_event in window_resized_events.iter() {
        renderer.resize();
    }
    let camera_transform = glam::TransformRT {
        rotation: global_transform.rotation,
        translation: global_transform.translation,
    };
    renderer.update(&dust_render::State {
        camera_projection: &*camera_projection,
        camera_transform: &camera_transform,
        sunlight: &sunlight,
    });
}
