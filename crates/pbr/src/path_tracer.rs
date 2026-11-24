//! The reference path tracer: a debugging tool that lights the scene by brute
//! force, to compare the renderer against (path_tracer.slang).
//!
//! While [`RenderQualitySettings::path_tracing`] is on, it runs instead of
//! [`HybridLightingSet`]: its paths start at the surfaces the primary rays hit,
//! and it adds their light to the HDR output. They are noisy: average them with
//! the reference accumulator.

use std::ops::Deref;

use bevy::prelude::*;
use bevy_pumicite::rtx::tlas::TLAS;
use bevy_pumicite::rtx::{RayTracingPipeline, RtxPipelineManager};
use bevy_pumicite::staging::{BufferInitializer, UniformRingBuffer};
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::rtx::ShaderBindingTable;
use pumicite::tracking::Access;

use crate::camera::Camera;
use crate::sky::AtmosphereLUTs;
use crate::{
    HdrRenderTarget, HdrRenderTargetViewsExt, HybridLightingSet, JitterState, PbrInstanceData,
    PbrPipelineParams, PbrRenderSet, RenderQualitySettings, build_camera_uniform,
};

#[derive(Resource)]
pub struct PathTracer {
    /// Crates with geometry add a library with their hit groups to it, and
    /// push the groups into `sbt` each frame, in the order they push them into
    /// the other pipelines' tables.
    pub pipeline: Handle<RayTracingPipeline>,
    pub sbt: Option<ShaderBindingTable>,
}

pub(crate) struct PathTracerPlugin;

impl Plugin for PathTracerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup.after(CreateDevice));
        app.configure_sets(
            PostUpdate,
            HybridLightingSet.run_if(|settings: Res<RenderQualitySettings>| !settings.path_tracing),
        );
        app.add_systems(
            PostUpdate,
            (
                create_sbt.before(PbrRenderSet),
                path_trace_pass
                    .run_if(|settings: Res<RenderQualitySettings>| settings.path_tracing)
                    .in_set(DefaultRenderSet)
                    .after(crate::render)
                    .after(create_sbt)
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
    let pipeline = pipeline_manager.add_pipeline();
    pipeline_manager.add_library_for_pipeline(
        &pipeline,
        asset_server.load("bazel://dust/crates/pbr/shaders/path_tracer.rtx.pipeline.bin"),
    );
    commands.insert_resource(PathTracer {
        pipeline,
        sbt: None,
    });
}

fn create_sbt(mut path_tracer: ResMut<PathTracer>, pipelines: Res<Assets<RayTracingPipeline>>) {
    let reusing = path_tracer.sbt.take();
    path_tracer.sbt = pipelines
        .get(&path_tracer.pipeline)
        .map(|pipeline| pipeline.create_sbt(reusing));
}

/// `PathTracerPush` in path_tracer.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathTracerPush {
    frame_index: u32,
    max_bounces: u32,
}

fn path_trace_pass(
    mut ctx: SubmissionState,
    mut path_tracer: ResMut<PathTracer>,
    settings: Res<RenderQualitySettings>,
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
    let Some(pipeline) = pipelines
        .get(&path_tracer.pipeline)
        .map(|pipeline| pipeline.deref().clone())
    else {
        return;
    };
    let Some(sbt) = path_tracer.sbt.as_mut() else {
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
    sbt.push_raygen(0, 0, |_| {});
    sbt.push_miss(0, 1, |_| {});
    // The primary rays' jitter: the paths' first positions are rebuilt from
    // their depth.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let push = PathTracerPush {
        frame_index: *frame_index,
        max_bounces: settings.max_bounces,
    };
    *frame_index = frame_index.wrapping_add(1);
    ctx.record(move |encoder, _| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                sbt.write_buffer(slice);
            });
        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();

        let pipeline = encoder.retain(pipeline).deref_inner();
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

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
        encoder.memory_barrier(
            Access::COPY_WRITE,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            },
        );
        encoder.emit_barriers();

        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            PbrPipelineParams::new()
                .scene_bvh(tlas)
                .uniforms(uniform)
                .output_texture(views.hdr_output().full_view())
                .gbuffer_albedo_srgb(views.albedo().deref_inner().srgb_view())
                .gbuffer_normal_texture(views.normal().deref_inner().full_view())
                .gbuffer_depth_texture(views.depth().full_view())
                .sky_atmosphere_params(atmo_buffer)
                .sky_transmittance_lut(transmittance_view.deref_inner().full_view())
                .sky_sky_view_lut(sky_view.deref_inner().full_view())
                .sky_linear_sampler(&atmosphere_luts.sampler)
                .as_slice(),
        );
        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&push),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "path trace", |encoder| {
            encoder.trace_rays(
                sbt,
                0,
                sbt_buffer,
                UVec3 {
                    x: hdr.render_extent.x,
                    y: hdr.render_extent.y,
                    z: 1,
                },
            );
        });
    });
}
