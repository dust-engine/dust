use std::{
    ops::{Deref, DerefMut},
    sync::Arc, collections::HashMap,
};

use bevy_ecs::{
    system::Resource,
    world::{FromWorld, World},
};
use pin_project::pin_project;
use rhyolite::{
    ash::vk,
    cstr,
    descriptor::DescriptorPool,
    future::{
        use_per_frame_state, DisposeContainer, GPUCommandFuture, PerFrameContainer, PerFrameState,
        RenderImage,
    },
    macros::{glsl_reflected, set_layout},
    utils::retainer::{Retainer, RetainerHandle},
    ComputePipeline, HasDevice, ImageViewLike, PipelineLayout,
};
use rhyolite_bevy::{Device, Queues};
use rhyolite::utils::format::{ColorSpace, ColorSpaceType};

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
        }.build(device.clone()).unwrap();
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
    pub fn render<
        'a,
        S: ImageViewLike,
        SRef: Deref<Target = RenderImage<S>>,
        T: ImageViewLike,
        TRef: DerefMut<Target = RenderImage<T>>,
    >(
        &mut self,
        src: SRef,
        dst: TRef,
        output_color_space: &ColorSpace,
    ) -> ToneMappingFuture<S, SRef, T, TRef> {
        let pipeline = self.pipeline
        .entry(output_color_space.clone())
        .or_insert_with(|| {
            let device = self.desc_pool.device().clone();
            let mat = self.scene_color_space.primaries().to_color_space(&output_color_space.primaries());
            let transfer_function = output_color_space.transfer_function() as u32;
            
            let shader = glsl_reflected!("tone_map.comp");
            let module = shader
                .build(device)
                .unwrap();
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
                self.layout.clone()
            )
            .unwrap();
            Arc::new(pipeline)
        });

        ToneMappingFuture {
            src_img: src,
            dst_img: dst,
            pipeline,
            desc_pool: &mut self.desc_pool
        }
    }
}

#[pin_project]
pub struct ToneMappingFuture<
    'a,
    S: ImageViewLike,
    SRef: Deref<Target = RenderImage<S>>,
    T: ImageViewLike,
    TRef: DerefMut<Target = RenderImage<T>>,
> {
    src_img: SRef,
    dst_img: TRef,
    desc_pool: &'a mut Retainer<DescriptorPool>,
    pipeline: &'a mut Arc<ComputePipeline>,
}

impl<
        'a,
        S: ImageViewLike,
        SRef: Deref<Target = RenderImage<S>>,
        T: ImageViewLike,
        TRef: DerefMut<Target = RenderImage<T>>,
    > GPUCommandFuture for ToneMappingFuture<'a, S, SRef, T, TRef>
{
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
            this
                .desc_pool
                .allocate_for_pipeline_layout(this.pipeline.layout())
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
            this.pipeline.device().update_descriptor_sets(
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
                this.pipeline.raw(),
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                this.pipeline.raw_layout(),
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
                this.pipeline.clone(),
                this.desc_pool.handle(),
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
