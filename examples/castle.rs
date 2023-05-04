#![feature(generators)]
#![feature(int_roundings)]
use bevy_app::{App, Plugin, Startup};
use bevy_asset::{AssetServer};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use bevy_transform::prelude::{GlobalTransform, Transform};
use bevy_window::{PrimaryWindow, Window};
use dust_render::{PinholeProjection, StandardPipeline, TLASStore, ToneMappingPipeline};

use rhyolite::ash::vk;
use rhyolite::{clear_image, ImageRequest, ImageLike, ImageExt};
use rhyolite::future::{GPUCommandFutureExt, RenderImage};

use rhyolite::{
    macros::{commands, gpu},
    QueueType,
};

use rhyolite_bevy::{
    Image, Queues, QueuesRouter, RenderSystems, Swapchain, SwapchainConfigExt,
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
        .add_plugin(bevy_asset::AssetPlugin {
            watch_for_changes: true,
            ..Default::default()
        })
        .add_plugin(dust_render::RenderPlugin::default())
        .add_plugin(bevy_time::TimePlugin::default())
        .add_plugin(bevy_scene::ScenePlugin::default())
        .add_plugin(bevy_diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugin(smooth_bevy_cameras::LookTransformPlugin)
        .add_plugin(smooth_bevy_cameras::controllers::fps::FpsCameraPlugin::default())
        .add_plugin(bevy_diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(RenderSystem)
        .add_systems(bevy_app::Update, print_position)
        .init_resource::<ToneMappingPipeline>();
    let main_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(main_window)
        .insert(SwapchainConfigExt {
            image_format: vk::Format::A2B10G10R10_UNORM_PACK32,
            image_color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        });

    app.add_plugin(dust_vox::VoxPlugin);

    app.add_systems(Startup, setup);

    app.run();
}

#[derive(Resource)]
pub struct NoiseResource {
    noise: bevy_asset::Handle<Image>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(NoiseResource {
        noise: asset_server.load("stbn_unitvec3_cosine_2Dx1D_128x128x64.png"),
    });
    commands.spawn(bevy_scene::SceneBundle {
        scene: asset_server.load("castle.vox"),
        ..Default::default()
    });
    commands
        .spawn(PinholeProjection::default())
        .insert(GlobalTransform::default())
        .insert(Transform::default())
        .insert(MainCamera)
        .insert(smooth_bevy_cameras::controllers::fps::FpsCameraBundle::new(
            smooth_bevy_cameras::controllers::fps::FpsCameraController {
                translate_sensitivity: 100.0,
                ..Default::default()
            },
            glam::Vec3::new(122.0, 166.61, 54.45),
            glam::Vec3::new(0., 0., 0.),
            glam::Vec3::Y,
        ));
}

#[derive(Component)]
struct MainCamera;

struct RenderSystem;
impl Plugin for RenderSystem {
    fn build(&self, app: &mut App) {
        let sys =
            |mut queues: ResMut<Queues>,
             queue_router: Res<QueuesRouter>,
             mut tlas_store: ResMut<TLASStore>,
             allocator: Res<rhyolite_bevy::Allocator>,
             mut pipeline: ResMut<StandardPipeline>,
             mut recycled_state: Local<_>,
             mut tone_mapping_pipeline: ResMut<ToneMappingPipeline>,
             mut render_params: SystemParamItem<StandardPipeline::RenderParams>,
             cameras: Query<(&PinholeProjection, &GlobalTransform), With<MainCamera>>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let Some((_, mut swapchain)) = windows.iter_mut().next() else {
                    return;
                };
                let Some(camera) = cameras.iter().next() else {
                    return;
                };
                let accel_struct = tlas_store.accel_struct();
                let graphics_queue = queue_router.of_type(QueueType::Graphics);
                let swapchain_image = swapchain.acquire_next_image(queues.current_frame());
                let future = gpu! {
                    let mut swapchain_image = swapchain_image.await;
                    commands! {
                        let albedo_image = rhyolite::future::use_shared_state(using!(), |_| {
                            allocator
                            .create_device_image_uninit(
                                &ImageRequest {
                                    format: vk::Format::A2B10G10R10_UNORM_PACK32, // TODO: try bgr?
                                    usage: vk::ImageUsageFlags::STORAGE,
                                    extent: swapchain_image.inner().extent(),
                                    ..Default::default()
                                }
                            ).unwrap().as_2d_view().unwrap()
                        }, |image| swapchain_image.inner().extent() != image.extent());
                        let mut albedo_image = RenderImage::new(albedo_image, vk::ImageLayout::UNDEFINED);

                        
                        let depth_image = rhyolite::future::use_shared_state(using!(), |_| {
                            allocator
                            .create_device_image_uninit(
                                &ImageRequest {
                                    format: vk::Format::R32_SFLOAT,
                                    usage: vk::ImageUsageFlags::STORAGE,
                                    extent: swapchain_image.inner().extent(),
                                    ..Default::default()
                                }
                            ).unwrap().as_2d_view().unwrap()
                        }, |image| swapchain_image.inner().extent() != image.extent());
                        let mut depth_image = RenderImage::new(depth_image, vk::ImageLayout::UNDEFINED);

                        
                        let normal_image = rhyolite::future::use_shared_state(using!(), |_| {
                            allocator
                            .create_device_image_uninit(
                                &ImageRequest {
                                    format: vk::Format::R16G16B16A16_SNORM,
                                    usage: vk::ImageUsageFlags::STORAGE,
                                    extent: swapchain_image.inner().extent(),
                                    ..Default::default()
                                }
                            ).unwrap().as_2d_view().unwrap()
                        }, |image| swapchain_image.inner().extent() != image.extent());
                        let mut normal_image = RenderImage::new(normal_image, vk::ImageLayout::UNDEFINED);


                        let radiance_image = rhyolite::future::use_shared_state(using!(), |_| {
                            allocator
                            .create_device_image_uninit(
                                &ImageRequest {
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    usage: vk::ImageUsageFlags::STORAGE,
                                    extent: swapchain_image.inner().extent(),
                                    ..Default::default()
                                }
                            ).unwrap().as_2d_view().unwrap()
                        }, |image| swapchain_image.inner().extent() != image.extent());
                        let mut radiance_image = RenderImage::new(radiance_image, vk::ImageLayout::UNDEFINED);


                        let mut rendered = false;
                        if let Some(accel_struct) = accel_struct {
                            let accel_struct = accel_struct.await;
                            clear_image(&mut depth_image, vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0]
                            }).await;
                            clear_image(&mut radiance_image, vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0]
                            }).await;
                            if let Some(render) = pipeline.render(
                                &mut radiance_image,
                                &mut albedo_image,
                                &mut normal_image,
                                &mut depth_image,
                                &accel_struct,
                                &mut render_params,
                                camera
                            ) {
                                render.await;
                                rendered = true;
                            }
                            if rendered {
                                tone_mapping_pipeline.render(&radiance_image, &mut swapchain_image).await;
                            }
                            retain!(accel_struct);
                        }
                        retain!((radiance_image, albedo_image, normal_image, depth_image));
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

fn print_position(a: Query<&GlobalTransform, With<MainCamera>>) {
    let transform = a.iter().next().unwrap();
    //println!("{:?}", transform);
}