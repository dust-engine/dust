use std::{ops::Deref, sync::Arc};

use bevy_asset::{AssetServer, Assets};
use bevy_ecs::{
    system::{lifetimeless::SRes, Resource, SystemParamItem},
    world::{FromWorld, World},
};
use crevice::std430::AsStd430;
use rand::Rng;
use rhyolite::{
    ash::vk,
    descriptor::{DescriptorPool, DescriptorSetWrite, PushConstants},
    future::{
        run, use_per_frame_state, use_state, DisposeContainer, GPUCommandFuture, RenderData,
        RenderImage, RenderRes,
    },
    macros::set_layout,
    utils::retainer::Retainer,
    BufferExt, BufferLike, ComputePipeline, HasDevice, ImageViewExt, ImageViewLike, PipelineLayout,
};
use rhyolite::{future::Disposable, macros::commands};
use rhyolite_bevy::Queues;

use crate::{CachedPipeline, PipelineCache, ShaderModule, SpecializedShader};

#[derive(Resource)]
pub struct SvgfPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: CachedPipeline<ComputePipeline>,
    desc_pool: Retainer<DescriptorPool>,
}

#[derive(AsStd430, Default, PushConstants)]
struct SvgfPushConstant {
    #[stage(vk::ShaderStageFlags::COMPUTE)]
    rand: u32,
    #[stage(vk::ShaderStageFlags::COMPUTE)]
    frame_index: u32,
}

impl FromWorld for SvgfPipeline {
    /// The color input should be specified in a linear color space with primaries as specified by `scene_color_space.primaries()`.
    /// The output will be in the color space as specified in `output_color_space`, with the transfer function applied.
    fn from_world(world: &mut World) -> Self {
        let queues: &Queues = world.resource();
        let num_frame_in_flight = queues.num_frame_in_flight();
        let device = queues.device().clone();

        let set = set_layout! {
            #[shader(vk::ShaderStageFlags::COMPUTE)]
            illuminance: vk::DescriptorType::STORAGE_IMAGE,
            #[shader(vk::ShaderStageFlags::COMPUTE)]
            reservoirs: vk::DescriptorType::STORAGE_BUFFER,
        }
        .build(device.clone())
        .unwrap();
        let layout = Arc::new(
            rhyolite::PipelineLayout::new(
                device.clone(),
                vec![Arc::new(set)],
                SvgfPushConstant::ranges().as_slice(),
                vk::PipelineLayoutCreateFlags::empty(),
            )
            .unwrap(),
        );

        let desc_pool = DescriptorPool::for_pipeline_layouts(
            std::iter::once(layout.deref()),
            num_frame_in_flight,
        )
        .unwrap();

        let asset_server: &AssetServer = world.resource();
        let shader = asset_server.load("restir_spatial.comp");
        let pipeline_cache: &PipelineCache = world.resource();
        let pipeline = pipeline_cache.add_compute_pipeline(
            layout.clone(),
            SpecializedShader::for_shader(shader, vk::ShaderStageFlags::COMPUTE).into(),
        );
        SvgfPipeline {
            layout,
            pipeline,
            desc_pool: Retainer::new(desc_pool),
        }
    }
}

pub type SvgfPipelineRenderParams = (SRes<PipelineCache>, SRes<Assets<ShaderModule>>);
impl SvgfPipeline {
    pub fn render<'a>(
        &'a mut self,
        illuminance: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        reservoirs: &'a RenderRes<impl BufferLike + RenderData>,
        params: &'a SystemParamItem<SvgfPipelineRenderParams>,
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let (pipeline_cache, shader_assets) = params;
        let desc_pool = &mut self.desc_pool;
        let pipeline = &mut self.pipeline;
        commands! { move
            let Some(pipeline) = pipeline_cache.retrieve(pipeline, shader_assets) else {
                return;
            };
            let desc_set = use_per_frame_state(using!(), || {
                desc_pool
                    .allocate_for_pipeline_layout(pipeline.layout())
                    .unwrap()
            });
            pipeline.device().write_descriptor_sets([
                DescriptorSetWrite::storage_images(
                    desc_set[0],
                    0,
                    0,
                    &[
                        illuminance.inner().as_descriptor(vk::ImageLayout::GENERAL),
                    ]
                ),
                DescriptorSetWrite::storage_buffers(
                    desc_set[0],
                    1,
                    0,
                    &[
                        reservoirs.inner().as_descriptor(),
                    ],
                    false
                ),
            ]);
            let extent = illuminance.inner().extent();

            let frame_index = use_state(
                using!(),
                || 0,
                |a| *a += 1
            );
            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.raw(),
                );
                let rand: u32 = rand::thread_rng().gen();
                device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout().raw(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    std::slice::from_raw_parts(&rand as *const _ as *const u8, 4),
                );
                device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout().raw(),
                    vk::ShaderStageFlags::COMPUTE,
                    4,
                    std::slice::from_raw_parts(frame_index as *const _ as *const u8, 4),
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.raw_layout(),
                    0,
                    desc_set.as_slice(),
                    &[],
                );
                device.cmd_dispatch(
                    command_buffer,
                    extent.width.div_ceil(8),
                    extent.height.div_ceil(8),
                    extent.depth,
                );
            }, |ctx| {
                ctx.read_image(
                    illuminance,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    illuminance,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    reservoirs,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
            }).await;
            retain!(
                DisposeContainer::new((pipeline.clone(), desc_pool.handle(), desc_set)));
        }
    }
}
