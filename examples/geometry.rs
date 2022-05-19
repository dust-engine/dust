use std::os::raw::c_void;
use std::sync::Arc;

use ash::vk;
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_ecs::system::lifetimeless::{SRes, SResMut};
use bevy_ecs::system::SystemParamItem;
use bevy_ecs::{
    prelude::{FromWorld, World},
    system::{Commands, Local, Res, ResMut},
};

use bevy_input::keyboard::KeyboardInput;
use bevy_input::ButtonState;
use bevy_transform::prelude::{GlobalTransform, Transform};
use dust_render::accel_struct::tlas::TLASStore;
use dust_render::pipeline::{PipelineIndex, RayTracingPipelineBuildJob};
use dust_render::shader::SpecializedShader;
use dust_render::{renderable::Renderable, swapchain::Windows, RenderStage};
use dustash::descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout};
use dustash::sync::GPUFuture;
use dustash::{
    command::{pool::CommandPool, recorder::CommandExecutable},
    frames::{PerFrame, PerFrameResource},
    queue::{QueueType, Queues},
    ray_tracing::pipeline::PipelineLayout,
    sync::CommandsFuture,
    Device,
};

#[derive(Clone)]
struct DefaultRenderer {
    pipeline_layout: Arc<PipelineLayout>,
}
const PRIMARY_RAY_PIPELINE: PipelineIndex = PipelineIndex::new(0);
impl dust_render::pipeline::RayTracingRenderer for DefaultRenderer {
    fn new(app: &mut bevy_app::App) -> Self {
        let render_app = app.sub_app_mut(dust_render::RenderApp);
        let device = render_app
            .world
            .get_resource::<Arc<Device>>()
            .unwrap()
            .clone();
        render_app.world.init_resource::<RenderState>();
        let render_state = render_app.world.get_resource::<RenderState>().unwrap();
        let pipeline_layout = PipelineLayout::new(
            device,
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[render_state.desc_layout.raw()])
                .build(),
        )
        .unwrap();
        DefaultRenderer {
            pipeline_layout: Arc::new(pipeline_layout),
        }
    }
    fn build(
        &self,
        index: PipelineIndex,
        asset_server: &AssetServer,
    ) -> RayTracingPipelineBuildJob {
        match index {
            PRIMARY_RAY_PIPELINE => RayTracingPipelineBuildJob {
                pipeline_layout: self.pipeline_layout.clone(),
                raygen_shader: SpecializedShader {
                    shader: asset_server.load("primary.rgen.spv"),
                    specialization: None,
                },
                miss_shaders: vec![],
                callable_shaders: vec![],
                max_recursion_depth: 1,
            },
            _ => unreachable!(),
        }
    }

    fn all_pipelines(&self) -> &[PipelineIndex] {
        &[PRIMARY_RAY_PIPELINE]
    }

    type RenderParam = (
        SResMut<Windows>,
        SRes<Arc<Device>>,
        SResMut<RenderState>,
        SRes<Arc<Queues>>,
        Local<'static, PerFrame<RenderPerImageState>>,
        Local<'static, PerFrame<RenderPerFrameState>>,
        SResMut<dust_render::pipeline::PipelineCache>,
        SResMut<TLASStore>,
    );
    fn render(&self, params: &mut SystemParamItem<Self::RenderParam>) {
        let (
            windows,
            device,
            state,
            queues,
            per_image_state,
            per_frame_state,
            pipeline_cache,
            tlas_store,
        ) = params;
        let current_frame = windows.primary_mut().unwrap().current_image_mut().unwrap();

        let per_frame_state = per_frame_state.get_or_else(current_frame, false, || {
            let desc_set = state
                .desc_pool
                .allocate(std::iter::once(&state.desc_layout))
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            RenderPerFrameState { desc_set }
        });
        if let Some(tlas) = tlas_store.tlas.as_ref() {
            unsafe {
                let as_write = vk::WriteDescriptorSetAccelerationStructureKHR {
                    acceleration_structure_count: 1,
                    p_acceleration_structures: &tlas.raw(),
                    ..Default::default()
                };
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: per_frame_state.desc_set.raw(),
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        p_next: &as_write as *const _ as *const c_void,
                        ..Default::default()
                    }],
                    &[],
                );
            }
        }

        let mut sbt_upload_future = pipeline_cache.sbt_upload_future.take().unwrap();
        let tlas_updated_future = tlas_store.tlas_build_future.take();

        let cmd_exec = per_image_state
            .get_or_else(current_frame, pipeline_cache.pipelines_updated, || {
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::builder()
                            .dst_set(per_frame_state.desc_set.raw())
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&[vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: current_frame.image_view,
                                image_layout: vk::ImageLayout::GENERAL,
                            }])
                            .build()],
                        &[],
                    );
                }
                let buf = state.command_pool.allocate_one().unwrap();
                let mut builder = buf.start(vk::CommandBufferUsageFlags::empty()).unwrap();
                builder.record(|mut recorder| {
                    recorder.simple_pipeline_barrier2(
                        &dustash::command::sync2::PipelineBarrier::new(
                            None,
                            &[],
                            &[dustash::command::sync2::ImageBarrier {
                                memory_barrier: dustash::command::sync2::MemoryBarrier {
                                    prev_accesses: &[],
                                    next_accesses: &[
                                        dustash::command::sync2::AccessType::RayTracingShaderWrite,
                                    ],
                                },
                                discard_contents: true,
                                image: current_frame.image,
                                ..Default::default()
                            }],
                            vk::DependencyFlags::BY_REGION,
                        ),
                    );
                    recorder.bind_descriptor_set(
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        &self.pipeline_layout,
                        0,
                        &[per_frame_state.desc_set.raw()],
                        &[],
                    );
                    if let Some(sbt) = pipeline_cache.sbts.get(0).unwrap_or(&None) {
                        // We must have already created the pipeline before we're able to create an SBT.
                        recorder.bind_raytracing_pipeline(
                            pipeline_cache.pipelines[0].as_ref().unwrap(),
                        );
                        recorder.trace_rays(
                            sbt,
                            current_frame.image_extent.width,
                            current_frame.image_extent.height,
                            1,
                        );
                    }

                    recorder.simple_pipeline_barrier2(
                        &dustash::command::sync2::PipelineBarrier::new(
                            None,
                            &[],
                            &[dustash::command::sync2::ImageBarrier {
                                memory_barrier: dustash::command::sync2::MemoryBarrier {
                                    prev_accesses: &[
                                        dustash::command::sync2::AccessType::RayTracingShaderWrite,
                                    ],
                                    next_accesses: &[dustash::command::sync2::AccessType::Present],
                                },
                                image: current_frame.image,
                                ..Default::default()
                            }],
                            vk::DependencyFlags::BY_REGION,
                        ),
                    );
                });
                let cmd_exec = builder.end().unwrap();
                RenderPerImageState {
                    cmd_exec: Arc::new(cmd_exec),
                }
            })
            .cmd_exec
            .clone();

        let mut ray_tracing_future =
            CommandsFuture::new(queues.clone(), queues.index_of_type(QueueType::Graphics));
        ray_tracing_future.then_command_exec(cmd_exec);

        // rtx depends on acquired swapchain image
        current_frame
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));
        sbt_upload_future
            .stage(vk::PipelineStageFlags2::COPY)
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));

        ray_tracing_future
            .stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .then_present(current_frame);
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
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default());
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
    desc_pool: Arc<DescriptorPool>,
    desc_layout: DescriptorSetLayout,
}

impl FromWorld for RenderState {
    fn from_world(world: &mut World) -> Self {
        let device: &Arc<Device> = world.resource();
        let queues: &Arc<Queues> = world.resource();
        let pool = CommandPool::new(
            device.clone(),
            vk::CommandPoolCreateFlags::empty(),
            queues.of_type(QueueType::Graphics).family_index(),
        )
        .unwrap();

        let desc_pool = {
            let num_frame_in_flight = 3;
            DescriptorPool::new(
                device.clone(),
                &vk::DescriptorPoolCreateInfo::builder()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(num_frame_in_flight)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            descriptor_count: num_frame_in_flight,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: num_frame_in_flight,
                        },
                    ])
                    .build(),
            )
        }
        .unwrap();
        let desc_layout = DescriptorSetLayout::new(
            device.clone(),
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        ..Default::default()
                    },
                ])
                .build(),
        )
        .unwrap();
        RenderState {
            command_pool: Arc::new(pool),
            desc_pool: Arc::new(desc_pool),
            desc_layout,
        }
    }
}

impl PerFrameResource for RenderPerImageState {
    const PER_IMAGE: bool = true;
}

struct RenderPerFrameState {
    desc_set: DescriptorSet,
}

impl PerFrameResource for RenderPerFrameState {
    const PER_IMAGE: bool = false;
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
