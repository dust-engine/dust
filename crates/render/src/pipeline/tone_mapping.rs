use std::{ops::Deref, sync::Arc};

use bevy_asset::{AssetServer, Assets};
use bevy_ecs::{
    system::{lifetimeless::SRes, Resource, SystemParamItem},
    world::{FromWorld, World},
};
use rhyolite::{
    ash::vk,
    descriptor::{DescriptorPool, DescriptorSetLayout, DescriptorSetWrite},
    future::{
        run, use_per_frame_state, DisposeContainer, GPUCommandFuture, RenderData, RenderImage,
        RenderRes,
    },
    utils::{
        format::{ColorSpacePrimaries, ColorSpaceTransferFunction},
        retainer::Retainer,
    },
    BufferExt, BufferLike, ComputePipeline, HasDevice, ImageViewExt, ImageViewLike, PipelineLayout,
};
use rhyolite::{future::Disposable, macros::commands, utils::format::ColorSpace};
use rhyolite_bevy::Queues;

use crate::{CachedPipeline, PipelineCache, ShaderModule, SpecializedShader};

#[derive(Resource)]
pub struct ToneMappingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: Option<CachedPipeline<ComputePipeline>>,
    display_color_space_transfer_fn: ColorSpaceTransferFunction,
    desc_pool: Retainer<DescriptorPool>,
    scene_color_space: ColorSpacePrimaries,
}

impl FromWorld for ToneMappingPipeline {
    /// The color input should be specified in a linear color space with primaries as specified by `scene_color_space.primaries()`.
    /// The output will be in the color space as specified in `output_color_space`, with the transfer function applied.
    fn from_world(world: &mut World) -> Self {
        let queues: &Queues = world.resource();
        let num_frame_in_flight = queues.num_frame_in_flight();
        let device = queues.device().clone();

        let set0 = playout_macro::layout!("../../../../assets/shaders/tone_map.playout", 0);
        let set0 = DescriptorSetLayout::new(device.clone(), &set0, Default::default()).unwrap();
        let layout = Arc::new(
            rhyolite::PipelineLayout::new(
                device.clone(),
                vec![Arc::new(set0)],
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
        ToneMappingPipeline {
            layout,
            pipeline: None,
            display_color_space_transfer_fn: ColorSpaceTransferFunction::LINEAR,
            desc_pool: Retainer::new(desc_pool),
            scene_color_space: ColorSpacePrimaries::ACES_AP1,
        }
    }
}

pub type ToneMappingPipelineRenderParams = (
    SRes<AssetServer>,
    SRes<PipelineCache>,
    SRes<Assets<ShaderModule>>,
);
impl ToneMappingPipeline {
    pub fn render<'a>(
        &'a mut self,
        src: &'a RenderImage<impl ImageViewLike + RenderData>,
        albedo: &'a RenderImage<impl ImageViewLike + RenderData>,
        dst: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        exposure: &'a RenderRes<impl BufferLike + RenderData>,
        output_color_space: &ColorSpace,
        params: &'a SystemParamItem<ToneMappingPipelineRenderParams>,
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let (asset_server, pipeline_cache, shader_assets) = params;
        if output_color_space.transfer_function != self.display_color_space_transfer_fn {
            // Will need to recreate the pipeline if the transfer function changes,
            // since the transfer function was baked into the shader with a specialization constant.
            self.pipeline = None;
            self.display_color_space_transfer_fn = output_color_space.transfer_function;
        }
        let pipeline = self.pipeline.get_or_insert_with(|| {
            let mat = self
                .scene_color_space
                .to_color_space(&output_color_space.primaries);
            let transfer_function = output_color_space.transfer_function as u32;

            let shader = asset_server.load("shaders/tone_map.comp");
            pipeline_cache.add_compute_pipeline(
                self.layout.clone(),
                SpecializedShader::for_shader(shader, vk::ShaderStageFlags::COMPUTE)
                    .with_const(0, transfer_function)
                    .with_const(1, mat.x_axis.x)
                    .with_const(2, mat.x_axis.y)
                    .with_const(3, mat.x_axis.z)
                    .with_const(4, mat.y_axis.x)
                    .with_const(5, mat.y_axis.y)
                    .with_const(6, mat.y_axis.z)
                    .with_const(7, mat.z_axis.x)
                    .with_const(8, mat.z_axis.y)
                    .with_const(9, mat.z_axis.z)
                    .into(),
            )
        });
        let desc_pool = &mut self.desc_pool;
        commands! { move
            let Some(pipeline) = pipeline_cache.retrieve(pipeline, shader_assets) else {
                return;
            };
            let desc_set = use_per_frame_state(using!(), || {
                desc_pool
                    .allocate_for_pipeline_layout(pipeline.layout())
                    .unwrap()
            });
            pipeline.device().write_descriptor_sets(&[
                DescriptorSetWrite::storage_images(
                    desc_set[0],
                    0,
                    0,
                    &[
                        src.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        albedo.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        dst.inner().as_descriptor(vk::ImageLayout::GENERAL),
                    ]
                ),
                DescriptorSetWrite::uniform_buffers(
                    desc_set[0],
                    3,
                    0,
                    &[
                        exposure.inner().as_descriptor(),
                    ],
                    false
                ),
            ]);
            let extent = src.inner().extent();
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
                    src,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read_image(
                    albedo,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL,
                );
                ctx.write_image(
                    dst,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
                ctx.read(
                    exposure,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::UNIFORM_READ,
                );
            }).await;
            retain!(
                DisposeContainer::new((pipeline.clone(), desc_pool.handle(), desc_set)));
        }
    }
}
