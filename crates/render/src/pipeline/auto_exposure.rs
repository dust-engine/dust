use std::{ops::Deref, sync::Arc};

use bevy_ecs::{
    system::{lifetimeless::SRes, Res, ResMut, Resource, SystemParamItem},
    world::{FromWorld, World},
};
use rhyolite::future::run;
use rhyolite::future::GPUCommandFutureExt;
use rhyolite::BufferExt;
use rhyolite::{
    ash::vk,
    cstr,
    descriptor::{DescriptorPool, DescriptorSetWrite},
    fill_buffer,
    future::{
        use_per_frame_state, use_shared_state, Disposable, DisposeContainer, GPUCommandFuture,
        PerFrameContainer, PerFrameState, RenderData, RenderImage, RenderRes, SharedDeviceState,
        SharedDeviceStateHostContainer,
    },
    macros::{commands, glsl_reflected, set_layout},
    utils::retainer::Retainer,
    ComputePipeline, HasDevice, ImageExt, ImageLike, ImageViewExt, ImageViewLike, PipelineLayout,
    ResidentBuffer,
};
use rhyolite_bevy::{Allocator, Device, Queues};
struct AutoExposureBuffer {
    histogram: [f32; 256],
    avg: f32,
}

#[derive(Resource)]
pub struct AutoExposurePipeline {
    layout: Arc<PipelineLayout>,
    pipeline: Arc<ComputePipeline>,
    avg_pipeline: Arc<ComputePipeline>,
    desc_pool: Retainer<DescriptorPool>,

    buffer: SharedDeviceStateHostContainer<ResidentBuffer>,
}

impl FromWorld for AutoExposurePipeline {
    /// The color input should be specified in a linear color space with primaries as specified by `scene_color_space.primaries()`.
    /// The output will be in the color space as specified in `output_color_space`, with the transfer function applied.
    fn from_world(world: &mut World) -> Self {
        let queues: &Queues = world.resource();
        let num_frame_in_flight = queues.num_frame_in_flight();
        let device = queues.device().clone();

        let set = set_layout! {
            #[shader(vk::ShaderStageFlags::COMPUTE)]
            illuminance_image: vk::DescriptorType::STORAGE_IMAGE,

            #[shader(vk::ShaderStageFlags::COMPUTE)]
            params: [vk::DescriptorType::INLINE_UNIFORM_BLOCK; 12],

            #[shader(vk::ShaderStageFlags::COMPUTE)]
            histogram: vk::DescriptorType::STORAGE_BUFFER,
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

        let shader = glsl_reflected!("auto_exposure.comp");
        let module = shader.build(device.clone()).unwrap();
        let pipeline = ComputePipeline::create_with_shader_and_layout(
            module.specialized(cstr!("main")).into(),
            layout.clone(),
            Default::default(),
            None,
        )
        .unwrap();

        let shader = glsl_reflected!("auto_exposure_avg.comp");
        let module = shader.build(device).unwrap();
        let avg_pipeline = ComputePipeline::create_with_shader_and_layout(
            module.specialized(cstr!("main")).into(),
            layout.clone(),
            Default::default(),
            None,
        )
        .unwrap();

        let allocator = world.resource::<Allocator>();
        let buffer = allocator
            .create_device_buffer_uninit(
                std::mem::size_of::<AutoExposureBuffer>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap();
        AutoExposurePipeline {
            layout,
            pipeline: Arc::new(pipeline),
            desc_pool: Retainer::new(desc_pool),
            avg_pipeline: Arc::new(avg_pipeline),
            buffer: SharedDeviceStateHostContainer::new(buffer),
        }
    }
}

impl AutoExposurePipeline {
    pub type RenderParams = (SRes<Allocator>, SRes<ExposureSettings>);

    pub fn render<'a>(
        &'a mut self,
        illuminance_image: &'a RenderImage<impl ImageViewLike + RenderData>,
        params: &'a SystemParamItem<Self::RenderParams>,
    ) -> impl GPUCommandFuture<
        Output = RenderRes<SharedDeviceState<ResidentBuffer>>,
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        let (allocator, settings) = params;
        commands! {
            let size = illuminance_image.inner().extent();
            let mut buffer = use_shared_state(using!(), |_| {
                let buffer = allocator.create_device_buffer_uninit(
                    std::mem::size_of::<AutoExposureBuffer>() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
                ).unwrap();
                buffer
            }, |_| false);
            if !buffer.inner().reused() {
                fill_buffer(&mut buffer, 0).await;
            }


            let desc_set = use_per_frame_state(using!(), || {
                self.desc_pool
                    .allocate_for_pipeline_layout(self.pipeline.layout())
                    .unwrap()
            });
            self.pipeline.device().write_descriptor_sets([
                DescriptorSetWrite::storage_images(
                    desc_set[0],
                    0,
                    0,
                    &[
                        illuminance_image.inner().as_descriptor(vk::ImageLayout::GENERAL),
                    ]
                ),
                DescriptorSetWrite::inline_uniform_block(
                    desc_set[0],
                    1,
                    0,
                    unsafe {
                        &[
                            std::mem::transmute(settings.min_log_luminance),
                            std::mem::transmute(settings.max_log_luminance - settings.min_log_luminance),
                            std::mem::transmute(settings.time_coefficient),
                        ]
                    }
                ),
                DescriptorSetWrite::storage_buffers(
                    desc_set[0],
                    2,
                    0,
                    &[
                        buffer.inner().as_descriptor()
                    ],
                    false
                ),
            ]);

            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.layout().raw(),
                    0,
                    &desc_set,
                    &[]
                );
                device.cmd_dispatch(
                    command_buffer,
                    size.width.div_ceil(16),
                    size.height.div_ceil(16),
                    1
                );
            }, |ctx| {
                ctx.read_image(
                    illuminance_image,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::ImageLayout::GENERAL
                );
                ctx.read(
                    &buffer,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
                ctx.write(
                    &mut buffer,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );

            }).await;
            run(|ctx, command_buffer| unsafe {
                let device = ctx.device();
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.avg_pipeline.raw());
                device.cmd_dispatch(
                    command_buffer,
                    1,
                    1,
                    1
                );
            }, |ctx| {
                ctx.read(
                    &buffer,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
                ctx.write(
                    &mut buffer,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
            }).await;
            retain!(DisposeContainer::new((desc_set, self.desc_pool.handle(), self.pipeline.clone())));
            buffer
        }
    }
}

#[derive(Resource)]
pub struct ExposureSettings {
    pub min_log_luminance: f32,
    pub max_log_luminance: f32,
    /// A value between 0 and 1. The higher the value, the faster the exposure will adapt to changes.
    pub time_coefficient: f32,
    pub default: f32,
    pub current: f32,
}
impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            min_log_luminance: -6.0,
            max_log_luminance: 8.5,
            default: 0.0,
            time_coefficient: 0.2,
            current: 0.0,
        }
    }
}
