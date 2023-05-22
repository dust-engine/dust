use std::{
    collections::HashMap,
    ops::Deref,
    sync::Arc,
};

use bevy_ecs::{
    system::Resource,
    world::{FromWorld, World},
};
use rhyolite::{
    ash::vk,
    cstr,
    descriptor::{DescriptorPool, DescriptorSetWrite},
    future::{
        run, use_per_frame_state, DisposeContainer, GPUCommandFuture, PerFrameContainer,
        PerFrameState, RenderData, RenderImage,
    },
    macros::{glsl_reflected, set_layout},
    utils::retainer::{Retainer, RetainerHandle},
    ComputePipeline, HasDevice, ImageViewExt, ImageViewLike, PipelineLayout,
};
use rhyolite::{
    future::Disposable,
    macros::commands,
    utils::format::{ColorSpace, ColorSpaceType},
};
use rhyolite_bevy::Queues;

#[derive(Resource)]
pub struct ToneMappingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: HashMap<ColorSpace, Arc<ComputePipeline>>,
    desc_pool: Retainer<DescriptorPool>,
    scene_color_space: ColorSpaceType,
}

impl FromWorld for ToneMappingPipeline {
    /// The color input should be specified in a linear color space with primaries as specified by `scene_color_space.primaries()`.
    /// The output will be in the color space as specified in `output_color_space`, with the transfer function applied.
    fn from_world(world: &mut World) -> Self {
        let queues: &Queues = world.resource();
        let num_frame_in_flight = queues.num_frame_in_flight();
        let device = queues.device().clone();

        let set = set_layout! {
            #[shader(vk::ShaderStageFlags::COMPUTE)]
            src_img: vk::DescriptorType::STORAGE_IMAGE,

            #[shader(vk::ShaderStageFlags::COMPUTE)]
            dst_img: vk::DescriptorType::STORAGE_IMAGE,
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
        ToneMappingPipeline {
            layout,
            pipeline: HashMap::new(),
            desc_pool: Retainer::new(desc_pool),
            scene_color_space: ColorSpaceType::sRGB, // The default scene color space.
        }
    }
}
impl ToneMappingPipeline {
    pub fn render<'a>(
        &'a mut self,
        src: &'a RenderImage<impl ImageViewLike + RenderData>,
        mut dst: &'a mut RenderImage<impl ImageViewLike + RenderData>,
        output_color_space: &ColorSpace,
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let pipeline = self
            .pipeline
            .entry(output_color_space.clone())
            .or_insert_with(|| {
                let device = self.desc_pool.device().clone();
                let mat = self
                    .scene_color_space
                    .primaries()
                    .to_color_space(&output_color_space.primaries());
                let transfer_function = output_color_space.transfer_function() as u32;

                let shader = glsl_reflected!("tone_map.comp");
                let module = shader.build(device).unwrap();
                let pipeline = ComputePipeline::create_with_reflected_shader_and_layout(
                    module
                        .specialized(cstr!("main"))
                        .with_const(0, transfer_function)
                        .with_const(1, mat.x_axis.x)
                        .with_const(2, mat.x_axis.y)
                        .with_const(3, mat.x_axis.z)
                        .with_const(4, mat.y_axis.x)
                        .with_const(5, mat.y_axis.y)
                        .with_const(6, mat.y_axis.z)
                        .with_const(7, mat.z_axis.x)
                        .with_const(8, mat.z_axis.y)
                        .with_const(9, mat.z_axis.z),
                    Default::default(),
                    self.layout.clone(),
                )
                .unwrap();
                Arc::new(pipeline)
            })
            .clone();
        let desc_pool = &mut self.desc_pool;
        commands! { move
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
                        src.inner().as_descriptor(vk::ImageLayout::GENERAL),
                        dst.inner().as_descriptor(vk::ImageLayout::GENERAL),
                    ]
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
                ctx.write_image(
                    dst,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                );
            }).await;
            retain!(
                DisposeContainer::new((pipeline.clone(), desc_pool.handle(), desc_set)));
        }
    }
}
