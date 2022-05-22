use std::{ffi::c_void, sync::Arc};

use bevy_asset::AssetServer;
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
};
use dustash::{
    command::{pool::CommandPool, recorder::CommandExecutable},
    descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
    frames::PerFrame,
    queue::{QueueType, Queues},
    ray_tracing::pipeline::PipelineLayout,
    sync::{CommandsFuture, GPUFuture},
    Device,
};

use crate::{
    accel_struct::tlas::TLASStore,
    pipeline::{PipelineIndex, RayTracingPipelineBuildJob},
    shader::SpecializedShader,
    swapchain::{Windows, Window},
};
use ash::vk;

pub struct RenderPerImageState {
    cmd_exec: Arc<CommandExecutable>,
    desc_set: DescriptorSet,
    pipeline_generation: u64,
}

pub struct RenderState {
    command_pool: Arc<CommandPool>,
    desc_pool: Option<Arc<DescriptorPool>>,
    desc_pool_num_frames: u32,
    desc_layout: DescriptorSetLayout,
}

impl FromWorld for RenderState {
    fn from_world(world: &mut World) -> Self {
        let device: &Arc<Device> = world.resource();
        let queues: &Arc<Queues> = world.resource();
        let pool = CommandPool::new(
            device.clone(),
            vk::CommandPoolCreateFlags::empty(),
            queues.of_type(QueueType::Graphics).family_index(),
        )
        .unwrap();

        let desc_layout = DescriptorSetLayout::new(
            device.clone(),
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        ..Default::default()
                    },
                ])
                .build(),
        )
        .unwrap();
        RenderState {
            command_pool: Arc::new(pool),
            desc_pool: None,
            desc_pool_num_frames: 0,
            desc_layout,
        }
    }
}

#[derive(Clone)]
pub struct Renderer {
    pipeline_layout: Arc<PipelineLayout>,
}
const PRIMARY_RAY_PIPELINE: PipelineIndex = PipelineIndex::new(0);
impl crate::pipeline::RayTracingRenderer for Renderer {
    fn new(app: &mut bevy_app::App) -> Self {
        let render_app = app.sub_app_mut(crate::RenderApp);
        let device = render_app
            .world
            .get_resource::<Arc<Device>>()
            .unwrap()
            .clone();
        render_app.world.init_resource::<RenderState>();
        let render_state = render_app.world.get_resource::<RenderState>().unwrap();
        let pipeline_layout = PipelineLayout::new(
            device,
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[render_state.desc_layout.raw()])
                .build(),
        )
        .unwrap();
        Renderer {
            pipeline_layout: Arc::new(pipeline_layout),
        }
    }
    fn build(
        &self,
        index: PipelineIndex,
        asset_server: &AssetServer,
    ) -> RayTracingPipelineBuildJob {
        match index {
            PRIMARY_RAY_PIPELINE => RayTracingPipelineBuildJob {
                pipeline_layout: self.pipeline_layout.clone(),
                raygen_shader: SpecializedShader {
                    shader: asset_server.load("primary.rgen.spv"),
                    specialization: None,
                },
                miss_shaders: vec![],
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
        SRes<Arc<Device>>,
        SResMut<RenderState>,
        SRes<Arc<Queues>>,
        Local<'static, PerFrame<RenderPerImageState, true>>,
        SResMut<crate::pipeline::PipelineCache>,
        SResMut<TLASStore>,
    );
    fn render(&self, params: &mut SystemParamItem<Self::RenderParam>) {
        let (windows, device, state, queues, per_image_state, pipeline_cache, tlas_store) = params;
        {
            let num_swapchain_images = windows.primary().unwrap().frames().num_images() as u32;
            if state.desc_pool.is_none() || state.desc_pool_num_frames != num_swapchain_images {
                // Update descriptor pool
                // We need one descriptor for each image.
                state.desc_pool = Some(Arc::new(DescriptorPool::new(
                    device.clone(),
                    &vk::DescriptorPoolCreateInfo::builder()
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                        .max_sets(num_swapchain_images)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                                descriptor_count: num_swapchain_images,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: num_swapchain_images,
                            },
                        ])
                        .build(),
                ).unwrap()));
            }
        };
        let current_frame = windows.primary_mut().unwrap().current_image_mut().unwrap();

        let mut sbt_upload_future = pipeline_cache.sbt_upload_future.take().unwrap();
        let tlas_updated_future = tlas_store.tlas_build_future.take();

        let per_image_state = per_image_state.get_or_else(
            current_frame,
            |original| pipeline_cache.generation != original.pipeline_generation,
            |original| {
                let desc_set = original.map_or_else(
                    || {
                        state
                            .desc_pool
                            .as_ref()
                            .unwrap()
                            .allocate(std::iter::once(&state.desc_layout))
                            .unwrap()
                            .into_iter()
                            .next()
                            .unwrap()
                    },
                    |a| a.desc_set,
                );
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::builder()
                            .dst_set(desc_set.raw())
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&[vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: current_frame.image_view,
                                image_layout: vk::ImageLayout::GENERAL,
                            }])
                            .build()],
                        &[],
                    );
                    println!(
                        "{:?} -> {:?} {:?}",
                        desc_set.raw(),
                        current_frame.image_view,
                        current_frame.image
                    );
                }
                let buf = state.command_pool.allocate_one().unwrap();
                let mut builder = buf.start(vk::CommandBufferUsageFlags::empty()).unwrap();
                builder.record(|mut recorder| {
                    recorder.simple_pipeline_barrier2(
                        &dustash::command::sync2::PipelineBarrier::new(
                            None,
                            &[],
                            &[dustash::command::sync2::ImageBarrier {
                                memory_barrier: dustash::command::sync2::MemoryBarrier {
                                    prev_accesses: &[],
                                    next_accesses: &[
                                        dustash::command::sync2::AccessType::RayTracingShaderWrite,
                                    ],
                                },
                                discard_contents: true,
                                image: current_frame.image,
                                ..Default::default()
                            }],
                            vk::DependencyFlags::BY_REGION,
                        ),
                    );
                    recorder.bind_descriptor_set(
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        &self.pipeline_layout,
                        0,
                        &[desc_set.raw()],
                        &[],
                    );
                    if let Some(sbt) = pipeline_cache.sbts.get(0).unwrap_or(&None) {
                        // We must have already created the pipeline before we're able to create an SBT.
                        recorder.bind_raytracing_pipeline(
                            pipeline_cache.pipelines[0].as_ref().unwrap(),
                        );
                        recorder.trace_rays(
                            sbt,
                            current_frame.image_extent.width,
                            current_frame.image_extent.height,
                            1,
                        );
                    }

                    recorder.simple_pipeline_barrier2(
                        &dustash::command::sync2::PipelineBarrier::new(
                            None,
                            &[],
                            &[dustash::command::sync2::ImageBarrier {
                                memory_barrier: dustash::command::sync2::MemoryBarrier {
                                    prev_accesses: &[
                                        dustash::command::sync2::AccessType::RayTracingShaderWrite,
                                    ],
                                    next_accesses: &[dustash::command::sync2::AccessType::Present],
                                },
                                image: current_frame.image,
                                ..Default::default()
                            }],
                            vk::DependencyFlags::BY_REGION,
                        ),
                    );
                });
                let cmd_exec = builder.end().unwrap();
                RenderPerImageState {
                    cmd_exec: Arc::new(cmd_exec),
                    desc_set,
                    pipeline_generation: pipeline_cache.generation,
                }
            },
        );
        let cmd_exec = per_image_state.cmd_exec.clone();
        if let Some(tlas) = tlas_store.tlas.as_ref() {
            // Update Acceleration Structure descriptor set
            unsafe {
                let as_write = vk::WriteDescriptorSetAccelerationStructureKHR {
                    acceleration_structure_count: 1,
                    p_acceleration_structures: &tlas.raw(),
                    ..Default::default()
                };
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: per_image_state.desc_set.raw(),
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        p_next: &as_write as *const _ as *const c_void,
                        ..Default::default()
                    }],
                    &[],
                );
            }
        }
        let mut ray_tracing_future =
            CommandsFuture::new(queues.clone(), queues.index_of_type(QueueType::Graphics));
        ray_tracing_future.then_command_exec(cmd_exec);

        // rtx depends on acquired swapchain image
        current_frame
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));
        // rtx depends on sbt upload
        sbt_upload_future
            .stage(vk::PipelineStageFlags2::COPY)
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));

        // After rtx completes, swapchain present.
        ray_tracing_future
            .stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .then_present(current_frame);
    }
}
