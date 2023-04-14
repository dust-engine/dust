use std::{sync::Arc, ops::{Deref, DerefMut}};

use bevy_ecs::{world::{FromWorld, World}, system::Resource};
use pin_project::pin_project;
use rhyolite::{ComputePipeline, utils::retainer::{Retainer, RetainerHandle}, descriptor::DescriptorPool, macros::glsl_reflected, cstr, ImageViewLike, future::{RenderImage, GPUCommandFuture, DisposeContainer, PerFrameContainer, PerFrameState, use_per_frame_state}, ash::vk, HasDevice};
use rhyolite_bevy::{Queues, Device};

#[derive(Resource)]
pub struct ToneMappingPipeline {
    pipeline: Arc<ComputePipeline>,
    desc_pool: Retainer<DescriptorPool>,
}


impl FromWorld for ToneMappingPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = glsl_reflected!("tone_map.comp");
        let num_frame_in_flight = world.resource::<Queues>().num_frame_in_flight();
        let module = shader
            .build(world.resource::<Device>().inner().clone())
            .unwrap();
        let pipeline = ComputePipeline::create_with_reflected_shader(
            module.specialized(cstr!("main")),
            Default::default(),
        )
        .unwrap();
        let desc_pool = DescriptorPool::for_pipeline_layouts(
            std::iter::once(pipeline.layout().deref()),
            num_frame_in_flight,
        )
        .unwrap();
    ToneMappingPipeline {
            pipeline: Arc::new(pipeline),
            desc_pool: Retainer::new(desc_pool),
        }
    }
}
impl ToneMappingPipeline {
    pub fn render<'a,
    S: ImageViewLike, SRef: Deref<Target = RenderImage<S>>,
    T: ImageViewLike, TRef: DerefMut<Target = RenderImage<T>>
    >(&mut self, src: SRef, dst: TRef) -> ToneMappingFuture<S, SRef, T, TRef>{
        ToneMappingFuture {
            src_img: src,
            dst_img: dst,
            pipeline: self
        }
    }
}

#[pin_project]
pub struct ToneMappingFuture<'a,
S: ImageViewLike, SRef: Deref<Target = RenderImage<S>>,
T: ImageViewLike, TRef: DerefMut<Target = RenderImage<T>>
> {
    src_img: SRef,
    dst_img: TRef,
    pipeline: &'a mut ToneMappingPipeline
}

impl<'a,
S: ImageViewLike, SRef: Deref<Target = RenderImage<S>>,
T: ImageViewLike, TRef: DerefMut<Target = RenderImage<T>>
> GPUCommandFuture for ToneMappingFuture<'a, S, SRef, T, TRef> {
    type Output = ();

    type RetainedState = DisposeContainer<(
        Arc<ComputePipeline>,
        RetainerHandle<DescriptorPool>,
        PerFrameContainer<Vec<vk::DescriptorSet>>,
    )>;

    type RecycledState = PerFrameState<Vec<vk::DescriptorSet>>;

    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut rhyolite::future::CommandBufferRecordContext,
        state_desc_sets: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert_eq!(this.src_img.inner().extent(), this.dst_img.inner().extent());
        let extent = this.src_img.inner().extent();

        let desc_set = use_per_frame_state(state_desc_sets, || {
            this.pipeline
                .desc_pool
                .allocate_for_pipeline_layout(this.pipeline.pipeline.layout())
                .unwrap()
        });
        unsafe {
            // TODO: optimize away redundant writes
            let image_infos = [
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: this.src_img.inner().raw_image_view(),
                    image_layout: vk::ImageLayout::GENERAL,
                },
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: this.dst_img.inner().raw_image_view(),
                    image_layout: vk::ImageLayout::GENERAL,
                },
            ];
            this.pipeline.pipeline.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: desc_set[0],
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: image_infos.as_ptr(),
                    ..Default::default()
                }],
                &[],
            );
        }

        ctx.record(|ctx, command_buffer| unsafe {
            let device = ctx.device();
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                this.pipeline.pipeline.raw(),
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                this.pipeline.pipeline.raw_layout(),
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
        });
        std::task::Poll::Ready((
            (),
            DisposeContainer::new((
                this.pipeline.pipeline.clone(),
                this.pipeline.desc_pool.handle(),
                desc_set,
            )),
        ))
    }

    fn context(self: std::pin::Pin<&mut Self>, ctx: &mut rhyolite::future::StageContext) {
        let this = self.project();
        ctx.read_image(
            this.src_img.deref(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            vk::ImageLayout::GENERAL,
        );
        ctx.write_image(
            this.dst_img.deref_mut(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::ImageLayout::GENERAL,
        );
    }
}