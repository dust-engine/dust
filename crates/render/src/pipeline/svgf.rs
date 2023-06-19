use std::{collections::HashMap, ops::Deref, sync::Arc};

use bevy_asset::{AssetServer, Assets};
use bevy_ecs::{
    system::{
        lifetimeless::{SRes, SResMut},
        Resource, SystemParamItem,
    },
    world::{FromWorld, World},
};
use rhyolite::{
    ash::vk,
    cstr,
    descriptor::{DescriptorPool, DescriptorSetWrite},
    future::{
        run, use_per_frame_state, DisposeContainer, GPUCommandFuture, PerFrameContainer,
        PerFrameState, RenderData, RenderImage, RenderRes,
    },
    macros::{glsl_reflected, set_layout},
    utils::retainer::{Retainer, RetainerHandle},
    BufferExt, BufferLike, ComputePipeline, HasDevice, ImageViewExt, ImageViewLike, PipelineLayout,
    Sampler,
};
use rhyolite::{
    future::Disposable,
    macros::commands,
    utils::format::{ColorSpace, ColorSpaceType},
};
use rhyolite_bevy::Queues;

use crate::{CachedPipeline, PipelineCache, ShaderModule, SpecializedShader};

#[derive(Resource)]
pub struct SvgfPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: CachedPipeline<ComputePipeline>,
    desc_pool: Retainer<DescriptorPool>,
    sampler: Sampler,
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
            prev_illuminance: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
        .build(device.clone())
        .unwrap();
        let layout = Arc::new(
            rhyolite::PipelineLayout::new(
                device.clone(),
                vec![Arc::new(set)],
                Default::default(),
                vk::PipelineLayoutCreateFlags::empty(),
            )
            .unwrap(),
        );

        let desc_pool = DescriptorPool::for_pipeline_layouts(
            std::iter::once(layout.deref()),
            num_frame_in_flight,
        )
        .unwrap();

        let sampler = Sampler::new(
            device.clone(),
            &vk::SamplerCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        let asset_server: &AssetServer = world.resource();
        let shader = asset_server.load("asvgf/temporal.comp.spv");
        let pipeline_cache: &PipelineCache = world.resource();
        let pipeline = pipeline_cache.add_compute_pipeline(
            layout.clone(),
            SpecializedShader::for_shader(shader, vk::ShaderStageFlags::COMPUTE).into(),
        );
        SvgfPipeline {
            layout,
            pipeline,
            desc_pool: Retainer::new(desc_pool),
            sampler,
        }
    }
}

pub type SvgfPipelineRenderParams = (
    SRes<AssetServer>,
    SRes<PipelineCache>,
    SRes<Assets<ShaderModule>>,
);
impl SvgfPipeline {
    pub fn render<'a>(
        &'a mut self,
        illuminance: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        prev_illuminance: &'a RenderImage<impl ImageViewLike + RenderData>,
        params: &'a SystemParamItem<SvgfPipelineRenderParams>,
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let (asset_server, pipeline_cache, shader_assets) = params;
        let desc_pool = &mut self.desc_pool;
        let pipeline = &mut self.pipeline;
        let sampler = &self.sampler;
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
                DescriptorSetWrite::combined_image_samplers(
                    desc_set[0],
                    1,
                    0,
                    &[
                        prev_illuminance.inner().as_descriptor_with_sampler(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, sampler),
                    ],
                ),
            ]);
            let extent = illuminance.inner().extent();
            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.raw(),
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
                ctx.read_image(
                    prev_illuminance,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }).await;
            retain!(
                DisposeContainer::new((pipeline.clone(), desc_pool.handle(), desc_set)));
        }
    }
}
