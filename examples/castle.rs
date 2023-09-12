#![feature(generators)]
#![feature(int_roundings)]
use std::ops::DerefMut;

use bevy_app::{App, Plugin, Startup, Update};
use bevy_asset::{AssetServer, Assets};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use bevy_input::mouse::MouseWheel;
use bevy_input::prelude::{KeyCode, MouseButton};
use bevy_time::Time;
use bevy_transform::prelude::{GlobalTransform, Transform};
use bevy_window::{PrimaryWindow, Window, WindowResolution};
use dust_render::nrd::{DenoiserEvent, NRDPipeline};
use dust_render::use_gbuffer;
use dust_render::{
    pipeline::CachablePipeline, AutoExposurePipeline, AutoExposurePipelineRenderParams, BlueNoise,
    ExposureSettings, PinholeProjection, StandardPipeline, StandardPipelineRenderParams, Sunlight,
    TLASStore, ToneMappingPipeline, ToneMappingPipelineRenderParams,
};

use glam::{UVec2, Vec3, Vec3A};
use rhyolite::ash::vk;
use rhyolite::future::GPUCommandFutureExt;
use rhyolite::{
    clear_image, cstr, BufferExt, ImageArraySliceView, ImageArraySlicedViews, ImageExt, ImageLike,
    ImageRequest,
};

use rhyolite::debug::DebugObject;
use rhyolite::{
    macros::{commands, gpu},
    QueueType,
};

use rhyolite_bevy::{
    Image, Queues, QueuesRouter, RenderSystems, SlicedImageArray, Swapchain, SwapchainConfigExt,
};

fn main() {
    let mut app = App::new();

    #[cfg(feature = "sentry")]
    app.add_plugin(dust_sentry::SentryPlugin);
    #[cfg(not(feature = "sentry"))]
    app.add_plugins(bevy_log::LogPlugin::default());

    app.add_plugins(bevy_core::TaskPoolPlugin::default())
        .add_plugins(bevy_core::TypeRegistrationPlugin::default())
        .add_plugins(bevy_core::FrameCountPlugin::default())
        .add_plugins(bevy_transform::TransformPlugin::default())
        .add_plugins(bevy_hierarchy::HierarchyPlugin::default())
        .add_plugins(bevy_diagnostic::DiagnosticsPlugin::default())
        .add_plugins(bevy_input::InputPlugin::default())
        .add_plugins(bevy_window::WindowPlugin {
            primary_window: Some(Window {
                title: "Dust Renderer: Castle".into(),
                resolution: WindowResolution::new(1920.0, 1080.0).with_scale_factor_override(1.0),
                ..Default::default()
            }),
            ..Default::default()
        })
        .add_plugins(bevy_a11y::AccessibilityPlugin)
        .add_plugins(bevy_winit::WinitPlugin::default())
        .add_plugins(bevy_asset::AssetPlugin::processed_dev().watch_for_changes())
        .add_plugins(dust_render::RenderPlugin::default())
        .add_plugins(bevy_time::TimePlugin::default())
        .add_plugins(bevy_scene::ScenePlugin::default())
        .add_plugins(bevy_diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugins(smooth_bevy_cameras::LookTransformPlugin)
        .add_plugins(smooth_bevy_cameras::controllers::fps::FpsCameraPlugin::default())
        .add_plugins(bevy_diagnostic::LogDiagnosticsPlugin::default())
        .add_plugins(RenderSystem)
        .add_systems(bevy_app::Update, print_position)
        .add_systems(bevy_app::Update, cursor_grab_system)
        .init_resource::<ToneMappingPipeline>()
        .init_resource::<AutoExposurePipeline>()
        .init_resource::<ExposureSettings>()
        .init_resource::<NRDPipeline>()
        .add_event::<DenoiserEvent>();
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
            required_feature_flags: vk::FormatFeatureFlags::TRANSFER_DST
                | vk::FormatFeatureFlags::STORAGE_IMAGE,
            hdr: false,
            ..Default::default()
        });

    app.add_plugins(dust_vox::VoxPlugin);

    app.add_systems(Startup, setup);
    app.add_systems(Update, teapot_move_system);

    app.run();
}

#[derive(Component)]
struct TeaPot;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(bevy_scene::SceneBundle {
        scene: asset_server.load("castle.vox"),
        ..Default::default()
    });
    commands
        .spawn(bevy_scene::SceneBundle {
            scene: asset_server.load("teapot.vox"),
            ..Default::default()
        })
        .insert(TeaPot);
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
             pipelines: (
                ResMut<StandardPipeline>,
                ResMut<AutoExposurePipeline>,
                ResMut<ToneMappingPipeline>,
                ResMut<dust_render::pipeline::nrd::NRDPipeline>,
            ),
             mut recycled_state: Local<_>,
             pipeline_params: (
                SystemParamItem<StandardPipelineRenderParams>,
                SystemParamItem<AutoExposurePipelineRenderParams>,
                SystemParamItem<ToneMappingPipelineRenderParams>,
                SystemParamItem<dust_render::pipeline::nrd::NDRPipelineRenderParams>,
            ),
             cameras: Query<(&PinholeProjection, &GlobalTransform), With<MainCamera>>,
             blue_noise: Res<BlueNoise>,
             img_slices: Res<Assets<SlicedImageArray>>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let (
                    mut ray_tracing_pipeline,
                    mut auto_exposure_pipeline,
                    mut tone_mapping_pipeline,
                    mut nrd_pipeline,
                ) = pipelines;
                let (
                    ray_tracing_pipeline_params,
                    auto_exposure_pipeline_params,
                    tone_mapping_pipeline_params,
                    nrd_pipeline_params,
                ) = pipeline_params;
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
                        let mut gbuffer = use_gbuffer(
                            using!(),
                            &allocator,
                            UVec2::new(swapchain_image.inner().extent().width, swapchain_image.inner().extent().height),
                        );

                        let accel_struct = accel_struct.await;
                        if let Some(render) = ray_tracing_pipeline.render(
                            &mut gbuffer,
                            &blue_noise,
                            &accel_struct,
                            ray_tracing_pipeline_params,
                            camera,
                        ) {
                            render.await;
                        }
                        {
                            let size = swapchain_image.inner().extent();
                            nrd_pipeline.render(
                                nrd_pipeline_params,
                                &mut gbuffer.motion,
                                &gbuffer.normal,
                                &gbuffer.depth,
                                &gbuffer.radiance,
                                &mut gbuffer.denoised_radiance,
                                camera,
                                (size.width as u16, size.height as u16)
                            ).await;
                        }

                        let exposure = auto_exposure_pipeline.render(&gbuffer.denoised_radiance, &auto_exposure_pipeline_params).await;
                        let exposure_avg = exposure.map(|exposure| exposure.slice(4 * 256, 4));
                        let color_space = swapchain_image.inner().color_space().clone();
                        tone_mapping_pipeline.render(
                            &gbuffer.denoised_radiance,
                            &gbuffer.albedo,
                            &mut swapchain_image,
                            &exposure_avg,
                            &color_space,
                            &tone_mapping_pipeline_params
                        ).await;

                        retain!(exposure_avg);
                        retain!(accel_struct);
                        retain!(gbuffer);
                        if !swapchain_image.touched() {
                            clear_image(&mut swapchain_image, vk::ClearColorValue {
                                float32: [0.0, 1.0, 0.0, 0.0]
                            }).await;
                        }
                    }.schedule_on_queue(graphics_queue).await;
                    swapchain_image.present().await;
                };
                queues.submit(future, &mut *recycled_state);
            };
        app.add_systems(Update, sys.in_set(RenderSystems::Render));
    }
}

fn print_position(
    mut sunlight: ResMut<Sunlight>,
    mut state: Local<(f32, f32)>,
    mut events: EventReader<MouseWheel>,
    time: Res<Time>,
    a: Query<&GlobalTransform, With<MainCamera>>,
) {
    let (current, target) = &mut state.deref_mut();
    for event in events.read() {
        let delta = match event.unit {
            bevy_input::mouse::MouseScrollUnit::Line => event.y * 30.0,
            bevy_input::mouse::MouseScrollUnit::Pixel => event.y,
        };
        *target += delta;
    }
    *current += 0.5 * (*target - *current);

    let _calculated_angle =
        ((time.elapsed_seconds() * 0.4).cos() + 1.0) * std::f32::consts::FRAC_PI_2;
    sunlight.direction = glam::Mat3A::from_axis_angle(
        Vec3::new(1.0, 0.0, 0.0),
        *current * 0.001 - std::f32::consts::FRAC_PI_2,
    ) * Vec3A::new(0.0, 0.0, 1.0);
    sunlight.direction =
        glam::Mat3A::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.2) * sunlight.direction;
    let _transform = a.iter().next().unwrap();
}

fn cursor_grab_system(
    btn: Res<bevy_input::Input<MouseButton>>,
    key: Res<bevy_input::Input<KeyCode>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let mut window = windows.iter_mut().next().unwrap();

    if btn.just_pressed(MouseButton::Left) {
        window.cursor.grab_mode = bevy_window::CursorGrabMode::Locked;
        window.cursor.visible = false;
    }

    if key.just_pressed(KeyCode::Escape) {
        window.cursor.grab_mode = bevy_window::CursorGrabMode::None;
        window.cursor.visible = true;
    }
}

fn teapot_move_system(time: Res<Time>, mut query: Query<&mut Transform, With<TeaPot>>) {
    for mut teapot in query.iter_mut() {
        *teapot =
            Transform::from_translation(Vec3::new(time.elapsed_seconds().sin() * 50.0, 200.0, 0.0));
    }
}
