use std::sync::Arc;

use ash::vk;
use bevy_ecs::{
    prelude::{FromWorld, World},
    system::{Local, Res, ResMut},
};
use dust_render::{swapchain::Windows, RenderStage};
use dustash::sync::GPUFuture;
use dustash::{
    command::{pool::CommandPool, recorder::CommandExecutable},
    frames::{PerFrame, PerFrameResource},
    queue::{QueueType, Queues},
    sync::CommandsFuture,
    Device,
};

fn main() {
    let mut app = bevy_app::App::new();

    app.insert_resource(bevy_window::WindowDescriptor {
        title: "I am a window!".to_string(),
        width: 1280.,
        height: 800.,
        scale_factor_override: Some(1.0),
        ..Default::default()
    })
    .add_plugin(bevy_core::CorePlugin::default())
    .add_plugin(bevy_transform::TransformPlugin::default())
    .add_plugin(bevy_input::InputPlugin::default())
    .add_plugin(bevy_window::WindowPlugin::default())
    //.add_plugin(bevy::asset::AssetPlugin::default())
    //.add_plugin(dust_raytrace::DustPlugin::default())
    //add_plugin(bevy::scene::ScenePlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    //.add_plugin(flycamera::FlyCameraPlugin)
    //.add_plugin(fps_counter::FPSCounterPlugin)
    .add_plugin(dust_render::RenderPlugin::default());

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default())
            .add_system_to_stage(RenderStage::Render, render);
    }
    app.run();
}

struct RenderPerImageState {
    cmd_exec: Arc<CommandExecutable>,
}

struct RenderState {
    command_pool: Arc<CommandPool>,
}

impl FromWorld for RenderState {
    fn from_world(world: &mut World) -> Self {
        let device: &Arc<Device> = world.resource();
        let queues: &Queues = world.resource();
        let pool = CommandPool::new(
            device.clone(),
            vk::CommandPoolCreateFlags::empty(),
            queues.of_type(QueueType::Graphics).family_index(),
        )
        .unwrap();
        RenderState {
            command_pool: Arc::new(pool),
        }
    }
}

impl PerFrameResource for RenderPerImageState {
    const PER_IMAGE: bool = true;
}

fn render(
    mut windows: ResMut<Windows>,
    state: Local<RenderState>,
    queues: Res<Arc<Queues>>,
    mut per_image_state: Local<PerFrame<RenderPerImageState>>,
) {
    let current_frame = windows.primary_mut().unwrap().current_image_mut().unwrap();
    let cmd_exec = per_image_state
        .get_or_else(current_frame, || {
            let buf = state.command_pool.allocate_one().unwrap();
            let mut builder = buf.start(vk::CommandBufferUsageFlags::empty()).unwrap();
            builder.record(|mut recorder| {
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
            let cmd_exec = builder.end().unwrap();
            RenderPerImageState {
                cmd_exec: Arc::new(cmd_exec),
            }
        })
        .cmd_exec
        .clone();

    current_frame
        .then(
            CommandsFuture::new(queues.clone(), queues.index_of_type(QueueType::Graphics))
                .then_command_exec(cmd_exec)
                .stage(vk::PipelineStageFlags2::CLEAR),
        )
        .then_present(current_frame);
}
