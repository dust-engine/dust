use std::sync::Arc;

use ash::vk;
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_ecs::{
    prelude::{FromWorld, World},
    system::{Commands, Local, Res, ResMut},
};

use bevy_input::keyboard::KeyboardInput;
use bevy_input::ButtonState;
use bevy_transform::prelude::{GlobalTransform, Transform};
use dust_render::pipeline::{PipelineIndex, RayTracingPipelineBuildJob};
use dust_render::{renderable::Renderable, swapchain::Windows, RenderStage};
use dustash::sync::GPUFuture;
use dustash::{
    command::{pool::CommandPool, recorder::CommandExecutable},
    frames::{PerFrame, PerFrameResource},
    queue::{QueueType, Queues},
    sync::CommandsFuture,
    Device,
};

#[derive(Default)]
struct DefaultRenderer;
impl dust_render::pipeline::RayTracingRenderer for DefaultRenderer {
    fn build(
        &self,
        index: PipelineIndex,
        asset_server: &AssetServer,
    ) -> RayTracingPipelineBuildJob {
        todo!()
    }

    fn all_pipelines(&self) -> Vec<dust_render::pipeline::PipelineIndex> {
        todo!()
    }
}

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
    .add_plugin(bevy_asset::AssetPlugin::default())
    //.add_plugin(dust_raytrace::DustPlugin::default())
    //add_plugin(bevy::scene::ScenePlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    //.add_plugin(flycamera::FlyCameraPlugin)
    //.add_plugin(fps_counter::FPSCounterPlugin)
    .add_plugin(dust_render::RenderPlugin::default())
    .add_plugin(dust_format_explicit_aabbs::ExplicitAABBPlugin::default())
    .add_plugin(dust_render::pipeline::RayTracingRendererPlugin::<
        DefaultRenderer,
    >::default())
    .add_startup_system(setup)
    .add_system(print_keyboard_event_system);

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default())
            .add_system_to_stage(RenderStage::Render, render);
    }
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<dust_format_explicit_aabbs::AABBGeometry> =
        asset_server.load("../assets/out.aabb");
    commands
        .spawn()
        .insert(Renderable::default())
        .insert(Transform::default())
        .insert(GlobalTransform::default())
        .insert(handle);
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
    queues: Res<Queues>,
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
            CommandsFuture::new(&queues, queues.index_of_type(QueueType::Graphics))
                .then_command_exec(cmd_exec)
                .stage(vk::PipelineStageFlags2::CLEAR),
        )
        .then_present(current_frame);
}

fn print_keyboard_event_system(
    mut commands: Commands,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    query: Query<(
        Entity,
        &Renderable,
        &Handle<dust_format_explicit_aabbs::AABBGeometry>,
    )>,
) {
    for event in keyboard_input_events.iter() {
        match event {
            KeyboardInput {
                state: ButtonState::Pressed,
                ..
            } => {
                let (entity, _, _) = query.iter().next().unwrap();
                commands
                    .entity(entity)
                    .remove::<Handle<dust_format_explicit_aabbs::AABBGeometry>>();
                println!("{:?}", event);
            }
            _ => {}
        }
    }
}
