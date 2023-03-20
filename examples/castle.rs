#![feature(generators)]
#![feature(int_roundings)]
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_window::{PrimaryWindow, Window};
use pin_project::pin_project;
use rhyolite::ash::vk;
use rhyolite::descriptor::DescriptorPool;
use rhyolite::future::{
    use_per_frame_state, Dispose, GPUCommandFutureExt, PerFrameContainer, PerFrameState,
    RenderImage,
};
use rhyolite::macros::glsl_reflected;
use rhyolite::utils::retainer::{Retainer, RetainerHandle};
use rhyolite::{
    copy_buffer_to_image,
    macros::{commands, gpu},
    ImageExt, QueueType,
};
use rhyolite::{
    cstr, ComputePipeline, ComputePipelineCreateInfo, HasDevice, ImageLike, ImageRequest,
    ImageViewLike,
};
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
        .add_plugin(rhyolite_bevy::RenderPlugin::default())
        .add_plugin(bevy_time::TimePlugin::default())
        .add_plugin(bevy_scene::ScenePlugin::default())
        .add_plugin(bevy_diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugin(bevy_diagnostic::LogDiagnosticsPlugin::default());
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

    app.add_startup_system(setup);

    app.run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>
) {
    commands.spawn(bevy_scene::SceneBundle {
        scene: asset_server.load("castle.vox"),
        ..Default::default()
    });
}