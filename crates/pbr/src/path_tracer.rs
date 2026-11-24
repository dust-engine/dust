//! The path tracer: lights the scene by brute force (path_tracer.slang).
//!
//! While [`RenderQualitySettings::path_tracing`] is on, it runs instead of
//! [`HybridLightingSet`]: its paths start at the surfaces the primary rays hit,
//! and it adds their light to the HDR output.
//!
//! With the radiance cache ([`crate::sharc`]), its paths end in the cache, which
//! two passes before them bring up to date. Without, it's the reference, a
//! debugging tool to compare the renderer against: its paths are noisy, and the
//! reference accumulator averages them.

use std::ops::Deref;
use std::sync::Arc;

use bevy::prelude::*;
use bevy_pumicite::rtx::tlas::TLAS;
use bevy_pumicite::rtx::{RayTracingPipeline, RtxPipelineManager};
use bevy_pumicite::staging::{BufferInitializer, UniformRingBuffer};
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::command::CommandEncoder;
use pumicite::pipeline::Pipeline;
use pumicite::rtx::ShaderBindingTable;
use pumicite::tracking::Access;

use crate::camera::Camera;
use crate::sharc::{self, SharcConstants, SharcParams, SharcSettings};
use crate::sky::AtmosphereLUTs;
use crate::{
    HdrRenderTarget, HdrRenderTargetViewsExt, HybridLightingSet, JitterState, PbrInstanceData,
    PbrPipelineParams, PbrRenderSet, RenderQualitySettings, build_camera_uniform,
};

/// A ray-tracing pass of the path tracer.
pub struct PathTracerPass {
    pub pipeline: Handle<RayTracingPipeline>,
    /// Only in the frames that run the pass.
    pub sbt: Option<ShaderBindingTable>,
}

/// The path tracer's passes. Each runs a permutation of path_tracer.slang.
#[derive(Resource)]
pub struct PathTracer {
    /// Lights the frame without the radiance cache.
    pub reference: PathTracerPass,
    /// Adds the light that a few paths find to the radiance cache.
    pub sharc_update: PathTracerPass,
    /// Lights the frame with paths that end in the radiance cache.
    pub sharc_query: PathTracerPass,
}

impl PathTracer {
    /// Every pass, for crates with geometry: they add a library with their hit
    /// groups to each pass's pipeline, and push the groups into its `sbt` each
    /// frame, in the order they push them into the other pipelines' tables.
    pub fn passes_mut(&mut self) -> [&mut PathTracerPass; 3] {
        [
            &mut self.reference,
            &mut self.sharc_update,
            &mut self.sharc_query,
        ]
    }
}

pub(crate) struct PathTracerPlugin;

impl Plugin for PathTracerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(sharc::SharcPlugin);
        app.add_systems(Startup, setup.after(CreateDevice));
        app.configure_sets(
            PostUpdate,
            HybridLightingSet.run_if(|settings: Res<RenderQualitySettings>| !settings.path_tracing),
        );
        app.add_systems(
            PostUpdate,
            (
                create_sbts.before(PbrRenderSet),
                path_trace_pass
                    .run_if(|settings: Res<RenderQualitySettings>| settings.path_tracing)
                    .in_set(DefaultRenderSet)
                    .after(crate::render)
                    .after(create_sbts)
                    .before(crate::tonemap::autoexposure_pass),
            ),
        );
    }
}

pub fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
) {
    let mut pass = |library: &'static str| {
        let pipeline = pipeline_manager.add_pipeline();
        pipeline_manager.add_library_for_pipeline(&pipeline, asset_server.load(library));
        PathTracerPass {
            pipeline,
            sbt: None,
        }
    };
    commands.insert_resource(PathTracer {
        reference: pass("bazel://dust/crates/pbr/shaders/path_tracer.rtx.pipeline.bin"),
        sharc_update: pass(
            "bazel://dust/crates/pbr/shaders/path_tracer_sharc_update.rtx.pipeline.bin",
        ),
        sharc_query: pass(
            "bazel://dust/crates/pbr/shaders/path_tracer_sharc_query.rtx.pipeline.bin",
        ),
    });
}

/// Creates the shader binding tables of the passes that run this frame. The other
/// passes get none.
fn create_sbts(
    mut path_tracer: ResMut<PathTracer>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    settings: Res<RenderQualitySettings>,
    sharc: Res<SharcSettings>,
) {
    let path_tracer = &mut *path_tracer;
    if !settings.path_tracing {
        path_tracer.reference.sbt = None;
        path_tracer.sharc_update.sbt = None;
        path_tracer.sharc_query.sbt = None;
    } else if sharc.enabled {
        path_tracer.reference.sbt = None;
        path_tracer.sharc_update.create_sbt(&pipelines);
        path_tracer.sharc_query.create_sbt(&pipelines);
    } else {
        path_tracer.reference.create_sbt(&pipelines);
        path_tracer.sharc_update.sbt = None;
        path_tracer.sharc_query.sbt = None;
    }
}

/// `PathTracerPush` in path_tracer.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathTracerPush {
    sharc: SharcConstants,
    bounces: vk::DeviceAddress,
    frame_index: u32,
    max_bounces: u32,
}

/// Lights the frame, after updating and resolving the radiance cache if the
/// paths are to end in it.
pub(crate) fn path_trace_pass(
    mut ctx: SubmissionState,
    mut path_tracer: ResMut<PathTracer>,
    settings: Res<RenderQualitySettings>,
    mut sharc: SharcParams,
    tlas: Res<TLAS<PbrInstanceData>>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    jitter: Res<JitterState>,
    mut frame_index: Local<u32>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    let path_tracer = &mut *path_tracer;
    let sharc_settings = *sharc.settings;
    let lighting = if sharc_settings.enabled {
        &mut path_tracer.sharc_query
    } else {
        &mut path_tracer.reference
    };
    let Some(lighting) = lighting.prepare(&pipelines) else {
        return;
    };
    // The cache, and the passes that bring it up to date.
    let mut cache_passes = None;
    if sharc_settings.enabled {
        let resolve = sharc
            .compute_pipelines
            .get(&sharc.state.resolve_pipeline)
            .map(|pipeline| pipeline.clone().into_inner());
        let state = &mut *sharc.state;
        let (Some(update), Some(resolve), Some(cache)) = (
            path_tracer.sharc_update.prepare(&pipelines),
            resolve,
            &mut state.cache,
        ) else {
            return;
        };
        cache_passes = Some((update, resolve, cache, &mut state.bounces));
    }

    // The primary rays' jitter: the paths' first positions are rebuilt from
    // their depth.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let camera_position = transform.translation();
    let mut push = PathTracerPush {
        // Set once the buffers' addresses are known.
        sharc: SharcConstants::zeroed(),
        bounces: 0,
        frame_index: *frame_index,
        max_bounces: settings.max_bounces,
    };
    *frame_index = frame_index.wrapping_add(1);
    ctx.record(move |encoder, _| {
        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.use_image_resource(
            views.hdr_output(),
            &mut hdr.state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.albedo(),
            &mut hdr.albedo_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.normal(),
            &mut hdr.normal_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            transmittance_view,
            &mut atmosphere_luts.transmittance.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            sky_view,
            &mut atmosphere_luts.sky_view.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        let mut descriptors = PbrPipelineParams::new();
        descriptors
            .scene_bvh(tlas)
            .uniforms(uniform)
            .output_texture(views.hdr_output().full_view())
            .gbuffer_albedo_srgb(views.albedo().deref_inner().srgb_view())
            .gbuffer_normal_texture(views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(views.depth().full_view())
            .sky_atmosphere_params(atmo_buffer)
            .sky_transmittance_lut(transmittance_view.deref_inner().full_view())
            .sky_sky_view_lut(sky_view.deref_inner().full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler);
        // `as_slice` returns only the descriptors that were set since its last
        // call. Calling it once here gives every pass below all of them.
        let descriptors = descriptors.as_slice();

        if let Some((update, resolve, cache, bounces)) = cache_passes {
            let buffer = encoder.lock(
                &cache.buffer,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::CLEAR,
            );
            push.sharc = SharcConstants::new(
                &sharc_settings,
                buffer.device_address(),
                camera_position,
                cache
                    .previous_camera_position
                    .replace(camera_position)
                    .unwrap_or(camera_position),
            );
            if std::mem::replace(&mut cache.needs_clear, false) {
                encoder.use_buffer_resource(buffer, &mut cache.state, Access::CLEAR);
                encoder.emit_barriers();
                encoder.fill_buffer(buffer, 0, buffer.size(), 0);
            }

            let read_write = |stage| Access {
                stage,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            };
            encoder.use_buffer_resource(
                buffer,
                &mut cache.state,
                read_write(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR),
            );
            encoder.timing_scope(profiler.as_deref_mut(), "sharc update", |encoder| {
                let block = sharc_settings.update_block;
                let extent = UVec2::new(
                    hdr.render_extent.x.div_ceil(block),
                    hdr.render_extent.y.div_ceil(block),
                );
                trace(encoder, &mut uploader, update, descriptors, &push, extent);
            });

            encoder.use_buffer_resource(
                buffer,
                &mut cache.state,
                read_write(vk::PipelineStageFlags2::COMPUTE_SHADER),
            );
            encoder.emit_barriers();
            let resolve = encoder.retain(resolve).deref_inner();
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, resolve);
            encoder.push_constants(
                resolve.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push.sharc),
            );
            encoder.timing_scope(profiler.as_deref_mut(), "sharc resolve", |encoder| {
                encoder.dispatch(UVec3::new(sharc::ENTRIES / sharc::RESOLVE_GROUP_SIZE, 1, 1));
            });

            encoder.use_buffer_resource(
                buffer,
                &mut cache.state,
                Access {
                    stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    access: vk::AccessFlags2::SHADER_STORAGE_READ,
                },
            );
            if let Some(bounces) = bounces {
                let buffer = encoder.lock(
                    &bounces.buffer,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                );
                encoder.use_buffer_resource(buffer, &mut bounces.state, Access::RTX_WRITE);
                push.bounces = buffer.device_address();
            }
        }

        encoder.timing_scope(profiler.as_deref_mut(), "path trace", |encoder| {
            trace(
                encoder,
                &mut uploader,
                lighting,
                descriptors,
                &push,
                hdr.render_extent,
            );
        });
    });
}

impl PathTracerPass {
    /// Creates this frame's table, in the memory of last frame's. There is none
    /// until the pipeline has loaded.
    fn create_sbt(&mut self, pipelines: &Assets<RayTracingPipeline>) {
        let last_frame = self.sbt.take();
        self.sbt = pipelines
            .get(&self.pipeline)
            .map(|pipeline| pipeline.create_sbt(last_frame));
    }

    /// The pipeline, and the table with the pass's own shaders pushed. `None`
    /// while the pipeline is loading, or if the frame doesn't run the pass.
    fn prepare(
        &mut self,
        pipelines: &Assets<RayTracingPipeline>,
    ) -> Option<(Arc<Pipeline>, &ShaderBindingTable)> {
        let pipeline = pipelines.get(&self.pipeline)?.deref().clone();
        let sbt = self.sbt.as_mut()?;
        sbt.push_raygen(0, 0, |_| {});
        sbt.push_miss(0, 1, |_| {});
        Some((pipeline, sbt))
    }
}

/// Records a pass over `extent`, and the barriers pending before it.
fn trace<'a>(
    encoder: &mut CommandEncoder<'a>,
    uploader: &mut BufferInitializer,
    (pipeline, sbt): (Arc<Pipeline>, &ShaderBindingTable),
    descriptors: &[vk::WriteDescriptorSet<'_>],
    push: &PathTracerPush,
    extent: UVec2,
) {
    let sbt_buffer =
        uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
            sbt.write_buffer(slice);
        });
    encoder.memory_barrier(
        Access::COPY_WRITE,
        Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
        },
    );
    encoder.emit_barriers();
    let pipeline = encoder.retain(pipeline).deref_inner();
    encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);
    encoder.push_descriptor_set(
        vk::PipelineBindPoint::RAY_TRACING_KHR,
        pipeline.layout(),
        0,
        descriptors,
    );
    encoder.push_constants(
        pipeline.layout(),
        vk::ShaderStageFlags::RAYGEN_KHR,
        0,
        bytemuck::bytes_of(push),
    );
    encoder.trace_rays(sbt, 0, sbt_buffer, extent.extend(1));
}
