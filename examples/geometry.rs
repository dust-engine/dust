use std::sync::Arc;

use ash::vk;
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::{
    prelude::FromWorld,
    system::{Res, ResMut},
};
use bevy_reflect::TypeUuid;
use dust_render::{geometry::GeometryPrimitiveArray, RenderStage};
use dustash::{
    command::{
        pool::{CommandBuffer, CommandPool},
        recorder::{CommandExecutable, CommandRecorder},
    },
    frames::AcquiredFrame,
    queue::{QueueType, Queues, SemaphoreOp},
    ray_tracing::sbt::SpecializationInfo,
    Device,
};
// First, define our geometry

#[derive(TypeUuid)]
#[uuid = "75a9a733-04d7-4abb-8600-9a7d24ff0597"]
struct AABBGeometry {
    primitives: Box<[ash::vk::AabbPositionsKHR]>,
}

struct AABBGeometryGPUAsset {}
impl dust_render::geometry::GeometryPrimitiveArray for AABBGeometryGPUAsset {
    fn rebuild_blas(
        &self,
        command_recorder: &mut CommandRecorder,
    ) -> dustash::accel_struct::AccelerationStructure {
        todo!()
    }
}

enum AABBGeometryChangeSet {
    Rebuild(Box<[ash::vk::AabbPositionsKHR]>),
    None,
}
impl dust_render::geometry::GeometryChangeSet<AABBGeometryGPUAsset> for AABBGeometryChangeSet {
    type Param = ();

    fn into_primitives(
        self,
        command_recorder: &mut CommandRecorder,
        params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> (
        AABBGeometryGPUAsset,
        Vec<dust_render::geometry::GeometryBLASBuildDependency>,
    ) {
        todo!()
    }

    fn apply_on(
        self,
        primitives: &mut AABBGeometryGPUAsset,
        command_recorder: &mut CommandRecorder,
        params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> Option<Vec<dust_render::geometry::GeometryBLASBuildDependency>> {
        todo!()
    }
}

impl dust_render::geometry::Geometry for AABBGeometry {
    type Primitives = AABBGeometryGPUAsset;

    type ChangeSet = AABBGeometryChangeSet;

    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(asset_server: &AssetServer) -> Handle<dust_render::shader::Shader> {
        todo!()
    }

    fn specialization() -> SpecializationInfo {
        todo!()
    }

    fn generate_changes(&self) -> Self::ChangeSet {
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
    .add_plugin(bevy_asset::AssetPlugin::default())
    .add_plugin(bevy_transform::TransformPlugin::default())
    .add_plugin(bevy_input::InputPlugin::default())
    .add_plugin(bevy_window::WindowPlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    .add_plugin(dust_render::RenderPlugin::default())
    .add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default());

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default())
            .init_resource::<RenderState>()
            .add_system_to_stage(RenderStage::Render, default_render_function);
    }
    app.run();
}

enum RenderCommandBufferState {
    Recorded(Arc<CommandExecutable>),
    Original(CommandBuffer),
    None,
}
struct RenderState {
    // The command pool used for rendering to the swapchain
    command_pool: Arc<CommandPool>,
    command_buffers: Vec<RenderCommandBufferState>,
}
impl FromWorld for RenderState {
    fn from_world(world: &mut bevy_ecs::prelude::World) -> Self {
        let device = world.get_resource::<Arc<Device>>().unwrap();
        let queues = world.get_resource::<Queues>().unwrap();
        let pool = CommandPool::new(
            device.clone(),
            vk::CommandPoolCreateFlags::empty(),
            queues
                .of_type(dustash::queue::QueueType::Graphics)
                .family_index(),
        )
        .unwrap();
        let pool = Arc::new(pool);
        Self {
            command_pool: pool,
            command_buffers: Vec::new(),
        }
    }
}

fn default_render_function(
    windows: Res<dust_render::swapchain::Windows>,
    queues: Res<Queues>,
    mut state: ResMut<RenderState>,
) {
    let num_images = windows.primary().unwrap().frames().num_images();
    let current_image = windows.primary().unwrap().current_image().unwrap();

    if num_images != state.command_buffers.len() {
        let buffers = state.command_pool.allocate_n(num_images as u32).unwrap();
        state.command_buffers = buffers
            .into_iter()
            .map(|b| RenderCommandBufferState::Original(b))
            .collect()
    }
    let buffer_state = std::mem::replace(
        &mut state.command_buffers[current_image.image_index as usize],
        RenderCommandBufferState::None,
    );
    let f = |cr: &mut CommandRecorder| default_render_function_record(cr, current_image);
    let exec = match (buffer_state, current_image.invalidate_images) {
        (RenderCommandBufferState::Original(buffer), _) => buffer
            .record(vk::CommandBufferUsageFlags::empty(), f)
            .map(Arc::new)
            .unwrap(),
        (RenderCommandBufferState::Recorded(exec), true) => {
            drop(exec);
            let buffer = state.command_pool.allocate_one().unwrap();
            buffer
                .record(vk::CommandBufferUsageFlags::empty(), f)
                .map(Arc::new)
                .unwrap()
        }
        (RenderCommandBufferState::Recorded(exec), false) => exec,
        (RenderCommandBufferState::None, _) => panic!(),
    };
    state.command_buffers[current_image.image_index as usize] =
        RenderCommandBufferState::Recorded(exec.clone());
    queues
        .of_type(QueueType::Graphics)
        .submit(
            Box::new([SemaphoreOp {
                semaphore: current_image.acquire_ready_semaphore.clone(),
                stage_mask: vk::PipelineStageFlags2::CLEAR,
                value: 0,
            }]),
            Box::new([exec]),
            Box::new([SemaphoreOp {
                semaphore: current_image.render_complete_semaphore.clone(),
                stage_mask: vk::PipelineStageFlags2::CLEAR,
                value: 0,
            }]),
        )
        .fence(current_image.complete_fence.clone());
}

fn default_render_function_record(recorder: &mut CommandRecorder, current_frame: &AcquiredFrame) {
    println!("Recorded render func");
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
}
