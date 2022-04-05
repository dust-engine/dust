use ash::vk;
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::system::{Res, ResMut, Commands};
use bevy_reflect::TypeUuid;
use dust_render::RenderStage;
use dustash::{command::recorder::CommandRecorder, ray_tracing::sbt::SpecializationInfo};
// First, define our geometry

fn main() {
    tracing_subscriber::fmt::init();
    let mut app = bevy_app::App::new();

    app.insert_resource(bevy_window::WindowDescriptor {
        title: "I am a window!".to_string(),
        width: 1280.,
        height: 800.,
        scale_factor_override: Some(1.0),
        ..Default::default()
    })
    .add_plugin(bevy_core::CorePlugin::default())
    .add_plugin(bevy_asset::AssetPlugin::default())
    .add_plugin(bevy_transform::TransformPlugin::default())
    .add_plugin(bevy_input::InputPlugin::default())
    .add_plugin(bevy_window::WindowPlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    .add_plugin(dust_render::RenderPlugin::default())
    .add_plugin(dust_format_explicit_aabbs::ExplicitAABBPrimitivesPlugin::default())
    .add_startup_system(startup);

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default())
            .add_system_to_stage(RenderStage::Render, main_window_render_function);
    }
    app.run();
}


fn startup(
    mut commands: Commands,
    asset_server: Res<AssetServer>
) {
    let handle: Handle<dust_format_explicit_aabbs::AABBGeometry> = asset_server.load("../assets/out.aabb");
    commands.spawn().insert(handle);
}

fn main_window_render_function(
    mut buffer: ResMut<dust_render::swapchain::SwapchainCmdBufferState>,
    windows: Res<dust_render::swapchain::Windows>,
) {
    let current_frame = windows.primary().unwrap().current_image().unwrap();
    buffer.record(vk::CommandBufferUsageFlags::empty(), |recorder| {
        let color_value = vk::ClearColorValue {
            float32: [1.0, 0.0, 0.0, 1.0],
        };
        recorder.simple_pipeline_barrier2(&dustash::command::sync2::PipelineBarrier::new(
            None,
            &[],
            &[dustash::command::sync2::ImageBarrier {
                memory_barrier: dustash::command::sync2::MemoryBarrier {
                    prev_accesses: &[],
                    next_accesses: &[dustash::command::sync2::AccessType::ClearWrite],
                },
                discard_contents: true,
                image: current_frame.image,
                ..Default::default()
            }],
            vk::DependencyFlags::BY_REGION,
        ));
        recorder.clear_color_image(
            current_frame.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &color_value,
            &[vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        );

        recorder.simple_pipeline_barrier2(&dustash::command::sync2::PipelineBarrier::new(
            None,
            &[],
            &[dustash::command::sync2::ImageBarrier {
                memory_barrier: dustash::command::sync2::MemoryBarrier {
                    prev_accesses: &[dustash::command::sync2::AccessType::ClearWrite],
                    next_accesses: &[dustash::command::sync2::AccessType::Present],
                },
                image: current_frame.image,
                ..Default::default()
            }],
            vk::DependencyFlags::BY_REGION,
        ));
    });
}
