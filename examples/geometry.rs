use std::sync::Arc;

use ash::vk;
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::{
    prelude::FromWorld,
    system::{IntoChainSystem, IntoSystem, Local, Res, ResMut},
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
    .add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default());

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default())
            .add_system_to_stage(RenderStage::Render, main_window_render_function);
    }
    app.run();
}

fn main_window_render_function(
    mut buffer: ResMut<dust_render::swapchain::SwapchainCmdBufferState>,
    windows: Res<dust_render::swapchain::Windows>,
) {
    let current_frame = windows.primary().unwrap().current_image().unwrap();
    buffer.record(vk::CommandBufferUsageFlags::empty(), |recorder| {
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
    });
}
