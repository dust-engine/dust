use std::{ffi::c_void, sync::Arc};

use crate::{Device, Queues, render_asset::{RawBuffer, RenderAssetStore, RenderAssetPlugin, RawBufferLoader}};
use bevy_asset::{AssetServer, Handle, Assets, AddAsset};
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{SQuery, SRes, SResMut},
        SystemParamItem,
    },
};
use ash::vk;
use dustash::{
    command::{pool::CommandPool, recorder::CommandExecutable},
    descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
    frames::PerFrame,
    queue::QueueType,
    pipeline::PipelineLayout,
    sync::{CommandsFuture, GPUFuture}, resources::alloc::{MemBuffer, BufferRequest},
};
use vk_mem::AllocationCreateFlags;

use crate::{
    accel_struct::tlas::TLASStore,
    camera::{ExtractedCamera, PerspectiveCameraParameters},
    pipeline::{PipelineIndex, RayTracingPipelineBuildJob},
    shader::SpecializedShader,
    swapchain::Windows,
};

#[derive(Resource)]
pub struct ProceduralSky {
    turbidity: f32,
    albedo: glam::Vec3,
    direction: glam::Vec3,
}
impl Default for ProceduralSky {
    fn default() -> Self {
        Self { turbidity:3.0, albedo: glam::Vec3::new(0.3, 0.3, 0.3), direction: glam::Vec3::new(10000.0, 5000.0, 10000.0) }
    }
}

pub struct RenderPerFrameState {
    cmd_exec: Option<Arc<CommandExecutable>>,
    desc_set: DescriptorSet,
    pipeline_generation: u64,
    sky_buffer: Arc<MemBuffer>,
}

#[derive(Resource)]
pub struct RenderState {
    command_pool: Arc<CommandPool>,
    desc_pool: Option<Arc<DescriptorPool>>,
    desc_pool_num_frames: u32,
}

pub struct PushConstants {
    camera_params: PerspectiveCameraParameters,
}

impl FromWorld for RenderState {
    fn from_world(world: &mut World) -> Self {
        let device: &crate::Device = world.resource();
        let queues: &crate::Queues = world.resource();
        let pool = CommandPool::new(
            device.0.clone(),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queues.of_type(QueueType::Compute).family_index(),
        )
        .unwrap();
        RenderState {
            command_pool: Arc::new(pool),
            desc_pool: None,
            desc_pool_num_frames: 0,
        }
    }
}

#[derive(Clone, Resource)]
pub struct Renderer {
    heitz_bluenoise: Handle<RawBuffer>,
}
const PRIMARY_RAY_PIPELINE: PipelineIndex = PipelineIndex::new(0);
impl crate::pipeline::RayTracingRenderer for Renderer {
    fn new(app: &mut bevy_app::App) -> Self {
        app.add_plugin(RenderAssetPlugin::<RawBuffer>::default())
            .add_asset_loader(RawBufferLoader::default());
        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app.world.init_resource::<RenderState>();
        let device = render_app.world.resource::<crate::Device>().0.clone();
        let render_state = render_app.world.get_resource::<RenderState>().unwrap();
        let material_descriptor_vec = render_app
            .world
            .get_resource::<crate::render_asset::BindlessGPUAssetDescriptors>()
            .unwrap();

        let asset_server = app.world.resource::<AssetServer>();
        Renderer {
            heitz_bluenoise: asset_server.load("heitz_spp64.bin")
        }
    }
    fn build(
        &self,
        index: PipelineIndex,
        asset_server: &AssetServer,
    ) -> RayTracingPipelineBuildJob {
        match index {
            PRIMARY_RAY_PIPELINE => RayTracingPipelineBuildJob {
                raygen_shader: SpecializedShader::new(asset_server.load("primary.rgen.spv")),
                miss_shaders: vec![SpecializedShader::new(asset_server.load("sky.rmiss.spv"))],
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
        SRes<Device>,
        SResMut<RenderState>,
        SRes<Queues>,
        Local<'static, PerFrame<RenderPerFrameState>>,
        SResMut<crate::pipeline::PipelineCache>,
        SResMut<TLASStore>,
        SRes<crate::render_asset::BindlessGPUAssetDescriptors>,
        SRes<RenderAssetStore<RawBuffer>>,
        SRes<crate::Allocator>,
        Local<'static, ProceduralSky>,
        SQuery<bevy_ecs::system::lifetimeless::Read<ExtractedCamera>>,
    );
    fn render(&self, params: &mut SystemParamItem<Self::RenderParam>) {
        let (
            windows,
            device,
            state,
            queues,
            per_frame_state,
            pipeline_cache,
            tlas_store,
            material_descriptor_vec,
            raw_buffers,
            allocator,
            sky,
            cameras,
        ) = params;
    }
}



fn trace_rays(
    sbt: i32, // Future with resource
    tlas: i32, // Future with resource
    camera: i32, // for push constants
    image: i32, // Future with resource
    skybuffer: i32,
    heitz_buffer: i32, // Asset
) {

}
