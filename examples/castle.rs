#![feature(generators)]
#![feature(int_roundings)]
use bevy_app::{App, Plugin, Startup};
use bevy_asset::{AssetServer, Assets};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use bevy_transform::prelude::{GlobalTransform, Transform};
use bevy_window::{PrimaryWindow, Window};
use dust_render::{
    AutoExposurePipeline, BlueNoise, ExposureSettings, PinholeProjection, StandardPipeline,
    TLASStore, ToneMappingPipeline, Sunlight,
};

use glam::{Vec3A, Vec3};
use rhyolite::ash::vk;
use rhyolite::future::GPUCommandFutureExt;
use rhyolite::{clear_image, BufferExt, ImageExt, ImageLike, ImageRequest};

use rhyolite::{
    macros::{commands, gpu},
    QueueType,
};

use rhyolite_bevy::{
    Image, Queues, QueuesRouter, RenderSystems, SlicedImageArray, Swapchain, SwapchainConfigExt,
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
        .add_plugin(bevy_window::WindowPlugin {
            primary_window: Some(Window {
                title: "Dust Engine: Castle".into(),
                resolution: (1280., 720.).into(),
                ..Default::default()
            }),
            ..Default::default()
        })
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
        .init_resource::<ToneMappingPipeline>()
        .init_resource::<AutoExposurePipeline>()
        .init_resource::<ExposureSettings>();
    let main_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(main_window)
        .insert(SwapchainConfigExt {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            image_format: Some(vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            }),
            required_feature_flags: vk::FormatFeatureFlags::TRANSFER_DST
                | vk::FormatFeatureFlags::STORAGE_IMAGE,
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
            glam::Vec3::new(122.0, 300.61, 54.45),
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
             mut auto_exposure_pipeline: ResMut<AutoExposurePipeline>,
             mut auto_exposure_params: SystemParamItem<AutoExposurePipeline::RenderParams>,
             mut recycled_state: Local<_>,
             mut tone_mapping_pipeline: ResMut<ToneMappingPipeline>,
             mut render_params: SystemParamItem<StandardPipeline::RenderParams>,
             cameras: Query<(&PinholeProjection, &GlobalTransform), With<MainCamera>>,
             blue_noise: Res<BlueNoise>,
             img_slices: Res<Assets<SlicedImageArray>>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let Some((_, mut swapchain)) = windows.iter_mut().next() else {
                    return;
                };
                let Some(camera) = cameras.iter().next() else {
                    return;
                };
                let Some(blue_noise) = img_slices.get(&blue_noise.unitvec3_cosine) else {
                    return;
                };
                let accel_struct = tlas_store.accel_struct();
                let graphics_queue = queue_router.of_type(QueueType::Graphics);
                let swapchain_image = swapchain.acquire_next_image(queues.current_frame());

                let future = gpu! {
                    let mut swapchain_image = swapchain_image.await;
                    commands! {
                        let mut albedo_image = rhyolite::future::use_shared_image(using!(), |_| {
                            (
                                allocator
                                    .create_device_image_uninit(
                                        &ImageRequest {
                                            format: vk::Format::A2B10G10R10_UNORM_PACK32, // TODO: try bgr?
                                            usage: vk::ImageUsageFlags::STORAGE,
                                            extent: swapchain_image.inner().extent(),
                                            ..Default::default()
                                        }
                                    ).unwrap().as_2d_view().unwrap(),
                                vk::ImageLayout::UNDEFINED
                            )
                        }, |image| swapchain_image.inner().extent() != image.extent());

                        let mut depth_image = rhyolite::future::use_shared_image(using!(), |_| {
                            (
                                allocator
                                    .create_device_image_uninit(
                                        &ImageRequest {
                                            format: vk::Format::R32_SFLOAT,
                                            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                                            extent: swapchain_image.inner().extent(),
                                            ..Default::default()
                                        }
                                    ).unwrap().as_2d_view().unwrap(),
                                vk::ImageLayout::UNDEFINED,
                            )
                        }, |image| swapchain_image.inner().extent() != image.extent());


                        let mut normal_image = rhyolite::future::use_shared_image(using!(), |_| {
                            (
                                allocator
                                    .create_device_image_uninit(
                                        &ImageRequest {
                                            format: vk::Format::R16G16B16A16_SNORM,
                                            usage: vk::ImageUsageFlags::STORAGE,
                                            extent: swapchain_image.inner().extent(),
                                            ..Default::default()
                                        }
                                    ).unwrap().as_2d_view().unwrap(),
                                vk::ImageLayout::UNDEFINED,
                            )
                        }, |image| swapchain_image.inner().extent() != image.extent());


                        let mut radiance_image = rhyolite::future::use_shared_image(using!(), |_| {
                            (
                                allocator
                                    .create_device_image_uninit(
                                        &ImageRequest {
                                            format: vk::Format::R32G32B32A32_SFLOAT,
                                            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                                            extent: swapchain_image.inner().extent(),
                                            ..Default::default()
                                        }
                                    ).unwrap().as_2d_view().unwrap(),
                                vk::ImageLayout::UNDEFINED,
                            )
                        }, |image| swapchain_image.inner().extent() != image.extent());


                        let mut rendered = false;
                        if let Some(accel_struct) = accel_struct {
                            let accel_struct = accel_struct.await;
                            if let Some(render) = pipeline.render(
                                &mut radiance_image,
                                &mut albedo_image,
                                &mut normal_image,
                                &mut depth_image,
                                &blue_noise,
                                &accel_struct,
                                render_params,
                                camera,
                            ) {
                                render.await;
                                rendered = true;
                            }
                            if rendered {
                                let exposure = auto_exposure_pipeline.render(&radiance_image, &auto_exposure_params).await;
                                let exposure_avg = exposure.map(|exposure| exposure.slice(4 * 256, 4));
                                let color_space = swapchain_image.inner().color_space().clone();
                                tone_mapping_pipeline.render(
                                    &radiance_image,
                                    &albedo_image,
                                    &mut swapchain_image,
                                    &exposure_avg,
                                    &color_space
                                ).await;
                                retain!(exposure_avg);
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

fn print_position(
    mut sunlight: ResMut<Sunlight>,
    mut angle: Local<f32>,
    a: Query<&GlobalTransform, With<MainCamera>>
) {
    *angle += 0.001;
    //sunlight.direction = glam::Mat3A::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), *angle) * Vec3A::new(0.0, 0.0, 1.0);
    let _transform = a.iter().next().unwrap();
    //println!("{:?}", transform);
}
