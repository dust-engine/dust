#![feature(generators)]
#![feature(int_roundings)]
use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_window::{PrimaryWindow, Window};
use pin_project::pin_project;
use rhyolite::ash::vk;
use rhyolite::descriptor::DescriptorPool;
use rhyolite::future::{
    use_per_frame_state, DisposeContainer, GPUCommandFutureExt, PerFrameContainer, PerFrameState,
    RenderData, RenderImage,
};
use rhyolite::macros::glsl_reflected;
use rhyolite::utils::retainer::{Retainer, RetainerHandle};
use rhyolite::{
    copy_buffer_to_image,
    macros::{commands, gpu},
    ImageExt, QueueType,
};
use rhyolite::{cstr, ComputePipeline, HasDevice, ImageLike, ImageRequest, ImageViewLike};
use rhyolite_bevy::{
    Allocator, Device, Queues, QueuesRouter, RenderSystems, Swapchain, SwapchainConfigExt,
};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

fn main() {
    let mut app = App::new();
    app.add_plugin(bevy_log::LogPlugin::default())
        .add_plugin(bevy_core::TaskPoolPlugin::default())
        .add_plugin(bevy_core::TypeRegistrationPlugin::default())
        .add_plugin(bevy_core::FrameCountPlugin::default())
        .add_plugin(bevy_transform::TransformPlugin::default())
        .add_plugin(bevy_hierarchy::HierarchyPlugin::default())
        .add_plugin(bevy_diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy_input::InputPlugin::default())
        .add_plugin(bevy_window::WindowPlugin::default())
        .add_plugin(bevy_a11y::AccessibilityPlugin)
        .add_plugin(bevy_winit::WinitPlugin::default())
        .add_plugin(rhyolite_bevy::RenderPlugin::default())
        .add_plugin(bevy_time::TimePlugin::default())
        .add_plugin(bevy_diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugin(bevy_diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(RenderSystem);

    let main_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(main_window)
        .insert(SwapchainConfigExt {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            required_feature_flags: vk::FormatFeatureFlags::TRANSFER_DST_KHR
                | vk::FormatFeatureFlags::STORAGE_IMAGE,
            ..Default::default()
        });

    app.run();
}

struct MyImage {
    image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    image_size: (u32, u32),
}
impl FromWorld for MyImage {
    fn from_world(_world: &mut World) -> Self {
        use image::GenericImageView;
        let res = reqwest::blocking::get(
            "https://github.githubassets.com/images/modules/signup/gc_banner_dark.png",
        )
        .unwrap()
        .bytes()
        .unwrap();
        let res = std::io::Cursor::new(res);
        let image = image::load(res, image::ImageFormat::Png).unwrap();
        let image_size = image.dimensions();
        let image = image.into_rgba8();
        Self { image, image_size }
    }
}

struct GaussianBlurPipeline {
    pipeline: Arc<ComputePipeline>,
    desc_pool: Retainer<DescriptorPool>,
}

impl FromWorld for GaussianBlurPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = glsl_reflected!("filter.comp");
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
        GaussianBlurPipeline {
            pipeline: Arc::new(pipeline),
            desc_pool: Retainer::new(desc_pool),
        }
    }
}
impl GaussianBlurPipeline {
    pub fn apply<
        'a,
        S: ImageViewLike + RenderData,
        T: ImageViewLike + RenderData,
        SRef: Deref<Target = RenderImage<S>>,
        TRef: DerefMut<Target = RenderImage<T>>,
    >(
        &'a mut self,
        src_img: SRef,
        tmp_img: TRef,
    ) -> GaussianBlur<'a, S, T, SRef, TRef> {
        GaussianBlur {
            src_img,
            tmp_img,
            pipeline: self,
        }
    }
}

#[pin_project]
struct GaussianBlur<
    'a,
    S: ImageViewLike + RenderData,
    T: ImageViewLike + RenderData,
    SRef: Deref<Target = RenderImage<S>>,
    TRef: Deref<Target = RenderImage<T>>,
> {
    src_img: SRef,
    tmp_img: TRef,
    pipeline: &'a mut GaussianBlurPipeline,
}
impl<
        'a,
        S: ImageViewLike + RenderData,
        T: ImageViewLike + RenderData,
        SRef: Deref<Target = RenderImage<S>>,
        TRef: DerefMut<Target = RenderImage<T>>,
    > rhyolite::future::GPUCommandFuture for GaussianBlur<'a, S, T, SRef, TRef>
{
    type Output = ();

    type RetainedState = DisposeContainer<(
        Arc<ComputePipeline>,
        RetainerHandle<DescriptorPool>,
        PerFrameContainer<Vec<vk::DescriptorSet>>,
    )>;

    type RecycledState = (PerFrameState<Vec<vk::DescriptorSet>>, u32);

    fn init(
        self: std::pin::Pin<&mut Self>,
        _ctx: &mut rhyolite::future::CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        None
    }
    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut rhyolite::future::CommandBufferRecordContext,
        (state_desc_sets, state_kernel_size): &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert_eq!(this.src_img.inner().extent(), this.tmp_img.inner().extent());
        let extent = this.src_img.inner().extent();

        let desc_set = use_per_frame_state(state_desc_sets, || {
            this.pipeline
                .desc_pool
                .allocate_for_pipeline_layout(this.pipeline.pipeline.layout())
                .unwrap()
        });

        let kernel_size = *state_kernel_size;
        *state_kernel_size += 1;
        if *state_kernel_size > 32 {
            *state_kernel_size = 0;
        }
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
                    image_view: this.tmp_img.inner().raw_image_view(),
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

            let kernel_size: [u8; 4] = unsafe { std::mem::transmute(kernel_size) };
            device.cmd_push_constants(
                command_buffer,
                this.pipeline.pipeline.raw_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                kernel_size.as_slice(),
            );
            device.cmd_dispatch(
                command_buffer,
                extent.width.div_ceil(128),
                extent.height,
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
            this.tmp_img.deref_mut(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::ImageLayout::GENERAL,
        );
    }
}

struct RenderSystem;
impl Plugin for RenderSystem {
    fn build(&self, app: &mut App) {
        let sys =
            |mut queues: ResMut<Queues>,
             queue_router: Res<QueuesRouter>,
             allocator: Res<Allocator>,
             image: Local<MyImage>,
             mut pipeline: Local<GaussianBlurPipeline>,
             mut recycled_state: Local<_>,
             mut windows: Query<(&Window, &mut Swapchain), With<PrimaryWindow>>| {
                let Some((_, mut swapchain)) = windows.iter_mut().next() else {
            return;
        };
                let _transfer_queue = queue_router.of_type(QueueType::Transfer);
                let graphics_queue = queue_router.of_type(QueueType::Graphics);
                let image_buffer = allocator
                    .create_device_buffer_with_data(
                        image.image.as_raw(),
                        vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .unwrap();

                let swapchain_image = swapchain.acquire_next_image(queues.current_frame());
                let image_size = image.image_size;
                let future = gpu! {
                    let image_size = image_size;
                    let mut swapchain_image = swapchain_image.await;

                    let intermediate_image = rhyolite::future::use_shared_image(using!(), |_| {
                        (
                            allocator
                                .create_device_image_uninit(
                                    &ImageRequest {
                                        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                                        extent: swapchain_image.inner().extent(),
                                        ..Default::default()
                                    }
                                ).unwrap().as_2d_view().unwrap(),
                            vk::ImageLayout::UNDEFINED,
                        )
                    }, |image| swapchain_image.inner().extent() != image.extent());
                    let tmp_image = intermediate_image;

                    let mut tmp_image_region = tmp_image.map(|i| {
                        i.crop(vk::Extent3D {
                            width: image_size.0,
                            height: image_size.1,
                            depth: 1
                        }, Default::default())
                    });
                    commands! {
                        let image_buffer = image_buffer.await;
                        copy_buffer_to_image(&image_buffer, &mut tmp_image_region, vk::ImageLayout::TRANSFER_DST_OPTIMAL).await;
                        retain!(image_buffer);
                    }.schedule_on_queue(graphics_queue).await;
                    let tmp_image = tmp_image_region.map(|image| image.into_inner());

                    commands! {
                        pipeline.apply(&tmp_image, &mut swapchain_image).await;
                        retain!(tmp_image);
                    }.schedule_on_queue(graphics_queue).await;

                    swapchain_image.present().await;
                };

                queues.submit(future, &mut *recycled_state);
            };
        app.add_system(sys.in_set(RenderSystems::Render));
    }
}
