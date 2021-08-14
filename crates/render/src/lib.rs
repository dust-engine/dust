#![feature(array_methods)]
#![feature(array_map)]
#![feature(backtrace)]
#![feature(maybe_uninit_extra)]
#[macro_use]
extern crate memoffset;

#[macro_use]
extern crate log;

mod block_alloc;
mod device_info;
mod material;
mod material_repo;
mod raytracer;
mod render_resources;
pub mod renderer;
mod shared_buffer;
pub mod swapchain;
mod utils;

use dust_core::CameraProjection;
use dust_core::SunLight;
use glam::TransformRT;

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
    pub sunlight: &'a SunLight,
}

pub use renderer::Renderer;

use bevy::prelude::*;
pub use dust_core as core;

use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;
use dust_core::Octree;

use crate::raytracer::RayTracer;
use crate::render_resources::RenderResources;
use crate::swapchain::Swapchain;
use bevy::app::AppExit;

use dust_core::svo::ArenaAllocator;
use std::borrow::BorrowMut;

#[derive(Default)]
pub struct DustPlugin;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum RenderState {
    InGame,
    Delegated,
}

impl Plugin for DustPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SunLight::new(Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO))
            .insert_resource::<Option<dust_core::svo::mesher::Mesh>>(None)
            .add_state(RenderState::InGame)
            .add_startup_system_to_stage(StartupStage::PreStartup, setup.system())
            .add_system_set(
                SystemSet::on_enter(RenderState::InGame)
                    .with_system(raytracer::systems::create_raytracer.system()),
            )
            .add_system_set(
                SystemSet::on_update(RenderState::InGame).with_system(world_update.system()),
            )
            .add_system_to_stage(CoreStage::Last, world_cleanup.system());
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
    unsafe {
        let renderer = Renderer::new(winit_window);
        let render_resources = RenderResources::new(&renderer);
        let arena = ArenaAllocator::new(render_resources.block_allocator.clone());
        let octree = Octree::new(arena);

        commands.insert_resource(octree);
        commands.insert_resource(renderer);
        commands.insert_resource(render_resources);
    };
}

fn world_update(
    mut window_resized_events: EventReader<WindowResized>,
    mut octree: ResMut<Octree>,
    mut renderer: ResMut<Renderer>,
    mut render_resources: ResMut<RenderResources>,
    mut raytracer: ResMut<RayTracer>,
    sunlight: Res<SunLight>,
    mut query: Query<(&mut CameraProjection, &GlobalTransform)>,
) {
    let renderer: &mut Renderer = renderer.borrow_mut();
    let render_resources: &mut RenderResources = &mut *render_resources;
    let (camera_projection, global_transform) = query
        .single_mut()
        .expect("Expecting an entity with RaytracerCameraBundle");

    if window_resized_events.iter().next().is_some() {
        unsafe {
            renderer.context.device.device_wait_idle().unwrap();
            let config = Swapchain::get_config(
                renderer.physical_device,
                renderer.context.surface,
                &renderer.context.surface_loader,
            );

            render_resources
                .swapchain
                .recreate(&render_resources.allocator, config);
            raytracer.bind_render_target(&mut render_resources.swapchain);
        }
    }
    let camera_transform = glam::TransformRT {
        rotation: global_transform.rotation,
        translation: global_transform.translation,
    };
    let state = State {
        camera_projection: &*camera_projection,
        camera_transform: &camera_transform,
        sunlight: &sunlight,
    };
    unsafe {
        octree.flush();
        raytracer.update(&state);
        render_resources.swapchain.render_frame();
    }
}

fn world_cleanup(
    mut commands: Commands,
    mut app_exit_events: EventReader<AppExit>,
    renderer: Res<Renderer>,
) {
    if app_exit_events.iter().next().is_some() {
        unsafe {
            renderer.context.device.device_wait_idle().unwrap();
        }
        commands.remove_resource::<RayTracer>();
        commands.remove_resource::<Octree>();
        commands.remove_resource::<RenderResources>();
        commands.remove_resource::<Renderer>();
    }
}
