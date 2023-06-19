#![feature(generators)]

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_window::{PrimaryWindow, Window};
use rhyolite::ash::vk;
use rhyolite::future::GPUCommandFutureExt;
use rhyolite::{
    copy_buffer, copy_buffer_to_image,
    future::RenderRes,
    macros::{commands, gpu},
    ImageExt, QueueType,
};
use rhyolite_bevy::{
    Allocator, Queues, QueuesRouter, RenderSystems, Swapchain, SwapchainConfigExt,
};

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
        .add_plugin(rhyolite_bevy::RenderPlugin::default())
        .add_plugin(bevy_time::TimePlugin::default())
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
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            ..Default::default()
        });

    app.run();
}

struct MyImage {
    image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    image_size: (u32, u32),
}
impl FromWorld for MyImage {
    fn from_world(_world: &mut World) -> Self {
        use image::GenericImageView;
        let res = reqwest::blocking::get(
            "https://github.githubassets.com/images/modules/signup/gc_banner_dark.png",
        )
        .unwrap()
        .bytes()
        .unwrap();
        let res = std::io::Cursor::new(res);
        let image = image::load(res, image::ImageFormat::Png).unwrap();
        let image_size = image.dimensions();
        let image = image.into_rgba8();
        Self { image, image_size }
    }
}

struct RenderSystem;
impl Plugin for RenderSystem {
    fn build(&self, app: &mut App) {
        let sys =
            |mut queues: ResMut<Queues>,
             queue_router: Res<QueuesRouter>,
             allocator: Res<Allocator>,
             image: Local<MyImage>,
             mut recycled_state: Local<_>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let Some((_, mut swapchain)) = windows.iter_mut().next() else {
            return;
        };

                let _transfer_queue = queue_router.of_type(QueueType::Transfer);
                let graphics_queue = queue_router.of_type(QueueType::Graphics);
                let image_buffer = allocator
                    .create_device_buffer_with_data(
                        image.image.as_raw(),
                        vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .unwrap();
                let intermediate_buffer = allocator
                    .create_device_buffer_uninit(
                        image.image.as_raw().len() as u64,
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .unwrap();
                let intermediate_buffer2 = allocator
                    .create_device_buffer_uninit(
                        image.image.as_raw().len() as u64,
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .unwrap();

                let swapchain_image = swapchain.acquire_next_image(queues.current_frame());
                let image_size = image.image_size;
                let future = gpu! { move
                    let swapchain_image = swapchain_image.await;

                    let mut swapchain_image_region = swapchain_image.map(|i| {
                        i.crop(vk::Extent3D {
                            width: image_size.0,
                            height: image_size.1,
                            depth: 1
                        }, Default::default())
                    });
                    let mut intermediate_buffer = RenderRes::new(intermediate_buffer);
                    commands! {
                        let image_buffer = image_buffer.await;
                        copy_buffer(&image_buffer, &mut intermediate_buffer).await;
                        retain!(image_buffer);
                    }.schedule_on_queue(graphics_queue).await;


                    let mut intermediate_buffer2 = RenderRes::new(intermediate_buffer2);
                    commands! {
                        copy_buffer(&intermediate_buffer, &mut intermediate_buffer2).await;
                        copy_buffer_to_image(&intermediate_buffer2, &mut swapchain_image_region, vk::ImageLayout::TRANSFER_DST_OPTIMAL).await;
                    }.schedule_on_queue(graphics_queue).await;
                    let swapchain_image = swapchain_image_region.map(|image| image.into_inner());

                    swapchain_image.present().await;
                    retain!(intermediate_buffer);
                    retain!(intermediate_buffer2);
                };

                queues.submit(future, &mut *recycled_state);
            };
        app.add_system(sys.in_set(RenderSystems::Render));
    }
}
