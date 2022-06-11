use std::{ffi::c_void, sync::Arc};

use bevy_asset::AssetServer;
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{SQuery, SRes, SResMut},
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
    camera::{ExtractedCamera, PerspectiveCameraParameters},
    pipeline::{PipelineIndex, RayTracingPipelineBuildJob},
    shader::SpecializedShader,
    swapchain::Windows,
};
use ash::vk;

pub struct RenderPerFrameState {
    cmd_exec: Option<Arc<CommandExecutable>>,
    desc_set: DescriptorSet,
    pipeline_generation: u64,
}

pub struct RenderState {
    command_pool: Arc<CommandPool>,
    desc_pool: Option<Arc<DescriptorPool>>,
    desc_pool_num_frames: u32,
    desc_layout: DescriptorSetLayout,
}

pub struct PushConstants {
    camera_params: PerspectiveCameraParameters,
}

impl FromWorld for RenderState {
    fn from_world(world: &mut World) -> Self {
        let device: &Arc<Device> = world.resource();
        let queues: &Arc<Queues> = world.resource();
        let pool = CommandPool::new(
            device.clone(),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queues.of_type(QueueType::Compute).family_index(),
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
        let material_descriptor_vec = render_app
            .world
            .get_resource::<crate::render_asset::BindlessGPUAssetDescriptors>()
            .unwrap();
        let pipeline_layout = PipelineLayout::new(
            device,
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[
                    render_state.desc_layout.raw(),
                    material_descriptor_vec.descriptor_vec.raw_layout(),
                ])
                .push_constant_ranges(&[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                    offset: 0,
                    size: std::mem::size_of::<PushConstants>() as u32,
                }])
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
                miss_shaders: vec![SpecializedShader {
                    shader: asset_server.load("sky.rmiss.spv"),
                    specialization: None,
                }],
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
        Local<'static, PerFrame<RenderPerFrameState>>,
        SResMut<crate::pipeline::PipelineCache>,
        SResMut<TLASStore>,
        SRes<crate::render_asset::BindlessGPUAssetDescriptors>,
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
            cameras,
        ) = params;

        {
            // Update descriptor pool
            let num_swapchain_images = windows.primary().unwrap().frames().num_images() as u32;
            if state.desc_pool.is_none() || state.desc_pool_num_frames != num_swapchain_images {
                // Update descriptor pool
                // We need one descriptor for each image.
                state.desc_pool = Some(Arc::new(
                    DescriptorPool::new(
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
                    )
                    .unwrap(),
                ));
            }
        };

        let mut sbt_upload_future = pipeline_cache.sbt_upload_future.take().unwrap();
        let tlas_updated_future = tlas_store.tlas_build_future.take();

        let camera = {
            let mut camera_iter = cameras.iter();
            let camera = camera_iter.next();
            assert!(
                camera_iter.next().is_none(),
                "Only supports one camera for now"
            );
            camera
        };

        let push_constants = PushConstants {
            camera_params: camera.unwrap().params.clone(),
        };
        let current_frame = windows.primary_mut().unwrap().current_image_mut().unwrap();

        // Allocate descriptor set and bind swapchain image
        let per_frame_state = per_frame_state.get_or_else(
            current_frame,
            |_original| false,
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
                RenderPerFrameState {
                    cmd_exec: None,
                    desc_set,
                    pipeline_generation: pipeline_cache.generation,
                }
            },
        );

        // Update per-frame descriptor sets
        // We have one descriptor set per frame to ensure that by the time we get to this point,
        // the GPU is already done using `per_frame_state.desc_set`. We'd have to use UPDATE_AFTER_BIND
        // if we just have one global descriptor set for all frames.
        if let Some(tlas) = tlas_store.tlas.as_ref() {
            // Update Acceleration Structure descriptor set
            unsafe {
                let raw = tlas.raw();
                let as_write = vk::WriteDescriptorSetAccelerationStructureKHR {
                    acceleration_structure_count: 1,
                    p_acceleration_structures: &raw,
                    ..Default::default()
                };
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: per_frame_state.desc_set.raw(),
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

        // Record command buffer
        let buf = per_frame_state.cmd_exec.take().map_or_else(
            || state.command_pool.allocate_one().unwrap(),
            |exec| Arc::try_unwrap(exec).unwrap().reset(false),
        );
        let mut builder = buf.start(vk::CommandBufferUsageFlags::empty()).unwrap();
        builder.record(|mut recorder| {
            recorder.simple_pipeline_barrier2(&dustash::command::sync2::PipelineBarrier::new(
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
            ));
            recorder.bind_descriptor_set(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                &self.pipeline_layout,
                0,
                &[
                    per_frame_state.desc_set.raw(),
                    material_descriptor_vec.descriptor_vec.raw(),
                ],
                &[],
            );
            recorder.push_constants(
                &self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                unsafe {
                    std::slice::from_raw_parts(
                        &push_constants as *const _ as *const u8,
                        std::mem::size_of::<PushConstants>(),
                    )
                },
            );
            if let Some(sbt) = pipeline_cache.sbts.get(0).unwrap_or(&None) {
                if tlas_store.tlas.is_some() {
                    // We must have already created the pipeline before we're able to create an SBT.
                    recorder
                        .bind_raytracing_pipeline(pipeline_cache.pipelines[0].as_ref().unwrap());
                    recorder.trace_rays(
                        sbt,
                        current_frame.image_extent.width,
                        current_frame.image_extent.height,
                        1,
                    );
                }
            }

            recorder.simple_pipeline_barrier2(&dustash::command::sync2::PipelineBarrier::new(
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
            ));
        });
        let cmd_exec = Arc::new(builder.end().unwrap());
        per_frame_state.cmd_exec = Some(cmd_exec.clone());

        let mut ray_tracing_future =
            CommandsFuture::new(queues.clone(), queues.index_of_type(QueueType::Compute));
        ray_tracing_future.then_command_exec(cmd_exec);

        // rtx depends on acquired swapchain image
        current_frame
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));
        // rtx depends on sbt upload
        sbt_upload_future
            .stage(vk::PipelineStageFlags2::COPY)
            .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));

        if let Some(mut future) = tlas_updated_future {
            future
                .stage(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .then(ray_tracing_future.stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR));
        }

        // After rtx completes, swapchain present.
        ray_tracing_future
            .stage(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .then_present(current_frame);
    }
}
