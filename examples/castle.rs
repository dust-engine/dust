#![feature(generators)]
#![feature(int_roundings)]
use bevy_app::{App, Plugin, Startup};
use bevy_asset::{AssetServer, Assets};
use bevy_ecs::prelude::*;
use bevy_window::{PrimaryWindow, Window};
use dust_render::{ShaderModule, StandardPipeline, TLASStore};
use pin_project::pin_project;
use rhyolite::ash::vk;
use rhyolite::clear_image;
use rhyolite::future::{
    use_per_frame_state, DisposeContainer, GPUCommandFutureExt, PerFrameContainer, PerFrameState,
    RenderImage,
};
use rhyolite::macros::glsl_reflected;
use rhyolite::utils::retainer::{Retainer, RetainerHandle};
use rhyolite::{
    copy_buffer_to_image,
    macros::{commands, gpu},
    ImageExt, QueueType,
};
use rhyolite::{cstr, ComputePipeline, HasDevice, ImageLike, ImageRequest, ImageViewLike};
use rhyolite_bevy::{
    Allocator, Device, Queues, QueuesRouter, RenderSystems, Swapchain, SwapchainConfigExt,
};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

fn main() {
    let mut app = App::new();
    app.add_plugin(bevy_log::LogPlugin::default())
        .add_plugin(bevy_core::TaskPoolPlugin::default())
        .add_plugin(bevy_core::TypeRegistrationPlugin::default())
        .add_plugin(bevy_core::FrameCountPlugin::default())
        .add_plugin(bevy_transform::TransformPlugin::default())
        .add_plugin(bevy_hierarchy::HierarchyPlugin::default())
        .add_plugin(bevy_diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy_input::InputPlugin::default())
        .add_plugin(bevy_window::WindowPlugin::default())
        .add_plugin(bevy_a11y::AccessibilityPlugin)
        .add_plugin(bevy_winit::WinitPlugin::default())
        .add_plugin(bevy_asset::AssetPlugin::default())
        .add_plugin(dust_render::RenderPlugin::default())
        .add_plugin(bevy_time::TimePlugin::default())
        .add_plugin(bevy_scene::ScenePlugin::default())
        .add_plugin(bevy_diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugin(bevy_diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(RenderSystem);
    let main_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(main_window)
        .insert(SwapchainConfigExt {
            image_format: vk::Format::B8G8R8A8_UNORM,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        });

    app.add_plugin(dust_vox::VoxPlugin);

    app.add_systems(Startup, setup);

    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(bevy_scene::SceneBundle {
        scene: asset_server.load("castle.vox"),
        ..Default::default()
    });
}

struct RenderSystem;
impl Plugin for RenderSystem {
    fn build(&self, app: &mut App) {
        let sys =
            |mut queues: ResMut<Queues>,
             queue_router: Res<QueuesRouter>,
             mut tlas_store: ResMut<TLASStore>,
             allocator: Res<Allocator>,
             mut pipeline: ResMut<StandardPipeline>,
             shaders: Res<Assets<ShaderModule>>,
             mut recycled_state: Local<_>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let Some((_, mut swapchain)) = windows.iter_mut().next() else {
            return;
        };
                let accel_struct = tlas_store.accel_struct();
                let graphics_queue = queue_router.of_type(QueueType::Graphics);
                let swapchain_image = swapchain.acquire_next_image(queues.current_frame());
                let future = gpu! {
                    let mut swapchain_image = swapchain_image.await;
                    commands! {
                        let mut rendered = false;
                        if let Some(accel_struct) = accel_struct {
                            println!("Has tlas");
                            let accel_struct = accel_struct.await;
                            if let Some(render) = pipeline.render(&mut swapchain_image, &accel_struct, shaders.deref()) {
                                println!("has pipeline");
                                render.await;
                                rendered = true;
                            }
                            retain!(accel_struct);
                        }
                        if !rendered {
                            clear_image(&mut swapchain_image, vk::ClearColorValue {
                                float32: [0.0, 1.0, 0.0, 0.0]
                            }).await;
                        }
                    }.schedule_on_queue(graphics_queue).await;
                    swapchain_image.present().await;
                };

                queues.submit(future, &mut *recycled_state);
            };
        app.add_system(sys.in_set(RenderSystems::Render));
    }
}
