//! SHARC integration: hash-grid debug overlay (Phase 1) + Update/Resolve/Query
//! pipeline (Phase 2/3, AGENTS.md §1).
//!
//! When `SharcConfig::enabled` is true the renderer's indirect-lighting integrator
//! is replaced by:
//!
//!     SHARC Update (sparse RT)  →  barrier  →
//!     SHARC Resolve (compute)   →  barrier  →
//!     SHARC Query  (full-res RT, writes HDR additively)
//!
//! When `enabled` is false the original `final_gather_pass` runs as the fallback
//! (AGENTS.md §1). The Phase 1 colored-hash debug overlay remains painted by
//! `final_gather` when SHARC is off, controlled by `SharcDebugState`.

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use pumicite::ash::vk;
use pumicite::buffer::{Buffer, BufferLike};
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::rtx::{
    RayTracingPipeline, RtxPipelineManager,
    tlas::TLAS,
};
use bevy_pumicite::shader::RayTracingPipelineLibrary;
use bevy_pumicite::staging::{BufferInitializer, UniformRingBuffer};
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use std::ops::Deref;

use crate::camera::Camera;
use crate::sky::AtmosphereLUTs;
use crate::{
    HdrRenderTarget, PbrInstanceData, PbrPipelineParams, PbrRenderSet,
    build_camera_uniform,
};

// ─── Phase 1: hash-grid debug overlay (unchanged behavior) ─────────────────

/// Tweakables for the hash-grid colored-hash debug overlay (Phase 1). Painted
/// by `final_gather` when SHARC is OFF — see `FinalGatherPush` for the wire
/// format.
#[derive(Resource, Clone, Copy)]
pub struct SharcDebugState {
    pub enabled: bool,
    pub logarithm_base: f32,
    pub scene_scale: f32,
    pub level_bias: f32,
    pub brightness: f32,
}

impl Default for SharcDebugState {
    fn default() -> Self {
        Self {
            enabled: false,
            logarithm_base: 2.0,
            scene_scale: 32.0,
            level_bias: 0.0,
            brightness: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FinalGatherPush {
    pub frame_index: u32,
    pub sharc_debug_enabled: u32,
    pub sharc_logarithm_base: f32,
    pub sharc_scene_scale: f32,
    pub sharc_level_bias: f32,
    pub sharc_brightness: f32,
}

impl FinalGatherPush {
    pub(crate) fn new(frame_index: u32, state: &SharcDebugState) -> Self {
        Self {
            frame_index,
            sharc_debug_enabled: state.enabled as u32,
            sharc_logarithm_base: state.logarithm_base,
            sharc_scene_scale: state.scene_scale,
            sharc_level_bias: state.level_bias,
            sharc_brightness: state.brightness,
        }
    }
}

// ─── Phase 2/3: SHARC Update / Resolve / Query pipeline ────────────────────

/// Tunable knobs that drive the SHARC passes (AGENTS.md §10).
#[derive(Resource, Clone, Copy)]
pub struct SharcConfig {
    pub enabled: bool,
    pub reset_pending: bool,
    pub entries_num: u32,
    pub downscale_factor: u32,
    pub accumulation_frame_num: u32,
    pub stale_frame_num: u32,
    pub scene_scale: f32,
    pub roughness_min: f32,
    pub radiance_scale: f32,
    pub debug_mode: u32,
}

impl Default for SharcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reset_pending: true,
            entries_num: 1 << 20, // 40 MiB total
            downscale_factor: 5,
            accumulation_frame_num: 32,
            stale_frame_num: 64,
            scene_scale: 32.0,
            roughness_min: 0.4,
            radiance_scale: 1.0e3,
            debug_mode: 0,
        }
    }
}

/// CPU mirror of `SharcShaderConstants` in `sharc_pt.slang`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SharcConstantsRaw {
    pub camera_position: [f32; 3],
    pub scene_scale: f32,
    pub camera_position_prev: [f32; 3],
    pub radiance_scale: f32,
    pub roughness_min: f32,
    pub accumulation_frame_num: u32,
    pub stale_frame_num: u32,
    pub entries_num: u32,
    pub frame_index: u32,
    pub downscale_factor: u32,
    pub debug_mode: u32,
    pub _pad: u32,
}

#[derive(Resource, Default)]
struct SharcFrameState {
    frame_index: u32,
    prev_camera_position: Vec3,
    prev_has_value: bool,
}

#[derive(Resource)]
pub struct SharcResources {
    pub entries_num: u32,
    pub hash_entries: GPUMutex<Buffer>,
    pub accumulation: GPUMutex<Buffer>,
    pub resolved: GPUMutex<Buffer>,
    pub hash_entries_state: ResourceState,
    pub accumulation_state: ResourceState,
    pub resolved_state: ResourceState,
    needs_clear: bool,
}

impl SharcResources {
    const STRIDE_HASH_ENTRIES: u64 = 8;
    const STRIDE_ACCUMULATION: u64 = 16;
    const STRIDE_RESOLVED: u64 = 16;
}

fn create_sharc_resources(allocator: &Allocator, entries_num: u32) -> SharcResources {
    let n = entries_num as u64;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let hash_entries = Buffer::new_private(
        allocator.clone(),
        n * SharcResources::STRIDE_HASH_ENTRIES,
        16,
        usage,
    )
    .expect("SHARC hash entries buffer");
    let accumulation = Buffer::new_private(
        allocator.clone(),
        n * SharcResources::STRIDE_ACCUMULATION,
        16,
        usage,
    )
    .expect("SHARC accumulation buffer");
    let resolved = Buffer::new_private(
        allocator.clone(),
        n * SharcResources::STRIDE_RESOLVED,
        16,
        usage,
    )
    .expect("SHARC resolved buffer");
    SharcResources {
        entries_num,
        hash_entries: GPUMutex::new(hash_entries),
        accumulation: GPUMutex::new(accumulation),
        resolved: GPUMutex::new(resolved),
        hash_entries_state: ResourceState::default(),
        accumulation_state: ResourceState::default(),
        resolved_state: ResourceState::default(),
        needs_clear: true,
    }
}

#[derive(Resource)]
pub struct SharcPipelines {
    pub update_pipeline: Handle<RayTracingPipeline>,
    pub query_pipeline: Handle<RayTracingPipeline>,
    pub resolve_pipeline: Handle<ComputePipeline>,
    pub update_sbt: Option<pumicite::rtx::ShaderBindingTable>,
    pub query_sbt: Option<pumicite::rtx::ShaderBindingTable>,
}

// ─── Plugin / setup ───────────────────────────────────────────────────────

pub struct SharcDebugPlugin;

impl Plugin for SharcDebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SharcDebugState>();
        app.init_resource::<SharcConfig>();
        app.init_resource::<SharcFrameState>();
        app.add_systems(EguiPrimaryContextPass, sharc_debug_ui);

        app.add_systems(Startup, setup_sharc.after(CreateDevice).after(crate::setup));
        app.add_systems(
            PostUpdate,
            (
                create_sharc_sbts.before(PbrRenderSet),
                clear_sharc_buffers
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(crate::shadow_pass),
                sharc_update_pass
                    .in_set(DefaultRenderSet)
                    .after(clear_sharc_buffers)
                    .after(create_sharc_sbts),
                sharc_resolve_pass
                    .in_set(DefaultRenderSet)
                    .after(sharc_update_pass),
                sharc_query_pass
                    .in_set(DefaultRenderSet)
                    .after(sharc_resolve_pass)
                    // Query writes the indirect contribution into the HDR
                    // target that DLSS reads as `pInColor`. Without this
                    // ordering Bevy is free to run DLSS first, denoise an
                    // HDR that hasn't been touched by Query yet, and tonemap
                    // the result — making Query's writes invisible.
                    .before(crate::dlss_evaluate),
            ),
        );
    }
}

pub fn setup_sharc(
    mut commands: Commands,
    allocator: Res<Allocator>,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
    config: Res<SharcConfig>,
) {
    let resources = create_sharc_resources(&allocator, config.entries_num);
    commands.insert_resource(resources);

    let update_library: Handle<RayTracingPipelineLibrary> = asset_server.load(
        "bazel://dust/crates/pbr/shaders/sharc/sharc_update.rtx.pipeline.bin",
    );
    let query_library: Handle<RayTracingPipelineLibrary> = asset_server.load(
        "bazel://dust/crates/pbr/shaders/sharc/sharc_query.rtx.pipeline.bin",
    );
    let resolve_pipeline: Handle<ComputePipeline> = asset_server
        .load("bazel://dust/crates/pbr/shaders/sharc/sharc_resolve.comp.pipeline.bin");

    commands.insert_resource(SharcPipelines {
        update_pipeline: pipeline_manager.add_pipeline(update_library),
        query_pipeline: pipeline_manager.add_pipeline(query_library),
        resolve_pipeline,
        update_sbt: None,
        query_sbt: None,
    });
}

fn create_sharc_sbts(
    mut state: ResMut<SharcPipelines>,
    pipelines: Res<Assets<RayTracingPipeline>>,
) {
    if let Some(pipeline) = pipelines.get(&state.update_pipeline) {
        state.update_sbt = Some(pipeline.create_sbt(state.update_sbt.take()));
    } else {
        state.update_sbt = None;
    }
    if let Some(pipeline) = pipelines.get(&state.query_pipeline) {
        state.query_sbt = Some(pipeline.create_sbt(state.query_sbt.take()));
    } else {
        state.query_sbt = None;
    }
}

// ─── Clear ─────────────────────────────────────────────────────────────────

fn clear_sharc_buffers(
    mut ctx: SubmissionState,
    mut resources: Option<ResMut<SharcResources>>,
    mut config: ResMut<SharcConfig>,
) {
    let Some(resources) = resources.as_deref_mut() else { return };
    if !resources.needs_clear && !config.reset_pending {
        return;
    }
    ctx.record(|encoder| {
        let hash_buf = encoder.lock(&resources.hash_entries, vk::PipelineStageFlags2::COPY);
        let accum_buf = encoder.lock(&resources.accumulation, vk::PipelineStageFlags2::COPY);
        let resolved_buf = encoder.lock(&resources.resolved, vk::PipelineStageFlags2::COPY);

        encoder.use_buffer_resource(
            hash_buf,
            &mut resources.hash_entries_state,
            Access::CLEAR,
        );
        encoder.use_buffer_resource(
            accum_buf,
            &mut resources.accumulation_state,
            Access::CLEAR,
        );
        encoder.use_buffer_resource(
            resolved_buf,
            &mut resources.resolved_state,
            Access::CLEAR,
        );
        encoder.emit_barriers();

        let device = encoder.device().clone();
        let cmd_buffer = encoder.buffer().vk_handle();
        unsafe {
            device.cmd_fill_buffer(cmd_buffer, hash_buf.vk_handle(), 0, hash_buf.size(), 0);
            device.cmd_fill_buffer(cmd_buffer, accum_buf.vk_handle(), 0, accum_buf.size(), 0);
            device.cmd_fill_buffer(
                cmd_buffer,
                resolved_buf.vk_handle(),
                0,
                resolved_buf.size(),
                0,
            );
        }
    });
    resources.needs_clear = false;
    config.reset_pending = false;
}

// ─── Per-frame constants ───────────────────────────────────────────────────

fn build_sharc_constants(
    config: &SharcConfig,
    frame_state: &SharcFrameState,
    camera_origin: Vec3,
) -> SharcConstantsRaw {
    let prev = if frame_state.prev_has_value {
        frame_state.prev_camera_position
    } else {
        camera_origin
    };
    SharcConstantsRaw {
        camera_position: camera_origin.to_array(),
        scene_scale: config.scene_scale,
        camera_position_prev: prev.to_array(),
        radiance_scale: config.radiance_scale,
        roughness_min: config.roughness_min,
        accumulation_frame_num: config.accumulation_frame_num,
        stale_frame_num: config.stale_frame_num,
        entries_num: config.entries_num,
        frame_index: frame_state.frame_index,
        downscale_factor: config.downscale_factor,
        debug_mode: config.debug_mode,
        _pad: 0,
    }
}

// ─── Update / Query (RT) ───────────────────────────────────────────────────

fn sharc_update_pass(
    mut ctx: SubmissionState,
    mut sharc_pipelines: ResMut<SharcPipelines>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    config: Res<SharcConfig>,
    mut frame_state: ResMut<SharcFrameState>,
    mut sharc_resources: Option<ResMut<SharcResources>>,
    tlas: Res<TLAS<PbrInstanceData>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    jitter: Res<crate::JitterState>,
) {
    if !config.enabled {
        return;
    }
    let Some(resources) = sharc_resources.as_deref_mut() else { return };
    let Ok((camera, transform)) = swapchain_images.single() else { return };
    let Some(pipeline) = pipelines
        .get(&sharc_pipelines.update_pipeline)
        .map(|p| p.deref().clone())
    else {
        return;
    };
    let Some(sbt) = sharc_pipelines.update_sbt.as_mut() else { return };
    let Some(per_instance_mutex) = tlas.tlas_per_instance_data.as_ref() else { return };
    let Some(tlas) = tlas.get() else { return };
    let Some(hdr) = hdr_target.as_mut() else { return };

    sbt.push_raygen(0, |_| {});
    sbt.push_miss(0, |_| {});
    sbt.push_miss(1, |_| {});

    let camera_origin = transform.translation();
    // Reuse the frame's primary-RT jitter for worldPos reconstruction — the
    // depth G-buffer was written by the jittered primary ray, so using the
    // same jitter here makes the reconstructed worldPos coincide with the
    // actual hit point Update is supposed to populate.
    let cam_uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let sharc_constants = build_sharc_constants(&config, &frame_state, camera_origin);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };

    // Advance the canonical frame state — Update is the gate for "SHARC ran
    // this frame", so `cameraPositionPrev` is tied to it (AGENTS.md §10).
    frame_state.prev_camera_position = camera_origin;
    frame_state.prev_has_value = true;
    let frame_index = frame_state.frame_index;
    frame_state.frame_index = frame_state.frame_index.wrapping_add(1);

    let downscale = config.downscale_factor.max(1);
    let dispatch = UVec3::new(
        hdr.extent.x.div_ceil(downscale),
        hdr.extent.y.div_ceil(downscale),
        1,
    );

    record_sharc_rt(
        &mut ctx,
        pipeline,
        sbt,
        cam_uniform,
        sharc_constants,
        atmosphere_uniform_buffer,
        per_instance_mutex,
        tlas,
        hdr,
        resources,
        atmosphere_luts.as_mut(),
        &mut uploader,
        &mut uniform_ring_buffer,
        frame_index,
        dispatch,
    );
}

fn sharc_query_pass(
    mut ctx: SubmissionState,
    mut sharc_pipelines: ResMut<SharcPipelines>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    config: Res<SharcConfig>,
    frame_state: Res<SharcFrameState>,
    mut sharc_resources: Option<ResMut<SharcResources>>,
    tlas: Res<TLAS<PbrInstanceData>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    jitter: Res<crate::JitterState>,
) {
    if !config.enabled {
        return;
    }
    let Some(resources) = sharc_resources.as_deref_mut() else { return };
    let Ok((camera, transform)) = swapchain_images.single() else { return };
    let Some(pipeline) = pipelines
        .get(&sharc_pipelines.query_pipeline)
        .map(|p| p.deref().clone())
    else {
        return;
    };
    let Some(sbt) = sharc_pipelines.query_sbt.as_mut() else { return };
    let Some(per_instance_mutex) = tlas.tlas_per_instance_data.as_ref() else { return };
    let Some(tlas) = tlas.get() else { return };
    let Some(hdr) = hdr_target.as_mut() else { return };

    sbt.push_raygen(0, |_| {});
    sbt.push_miss(0, |_| {});
    sbt.push_miss(1, |_| {});

    let camera_origin = transform.translation();
    // Same jitter as the primary RT pass — matches the depth G-buffer's ray.
    let cam_uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let sharc_constants = build_sharc_constants(&config, &frame_state, camera_origin);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };

    let dispatch = UVec3::new(hdr.extent.x, hdr.extent.y, 1);
    let frame_index = frame_state.frame_index.wrapping_sub(1);
    record_sharc_rt(
        &mut ctx,
        pipeline,
        sbt,
        cam_uniform,
        sharc_constants,
        atmosphere_uniform_buffer,
        per_instance_mutex,
        tlas,
        hdr,
        resources,
        atmosphere_luts.as_mut(),
        &mut uploader,
        &mut uniform_ring_buffer,
        frame_index,
        dispatch,
    );
}

// Shared body for the Update and Query passes.
fn record_sharc_rt(
    ctx: &mut SubmissionState,
    pipeline: std::sync::Arc<pumicite::pipeline::Pipeline>,
    sbt: &mut pumicite::rtx::ShaderBindingTable,
    cam_uniform: crate::Uniforms,
    sharc_constants: SharcConstantsRaw,
    atmosphere_uniform_buffer: pumicite::buffer::RingBufferSuballocation,
    per_instance_mutex: &GPUMutex<pumicite::buffer::RingBufferSuballocation>,
    tlas: &GPUMutex<bevy_pumicite::rtx::tlas::TLASInner>,
    hdr: &mut HdrRenderTarget,
    resources: &mut SharcResources,
    atmosphere_luts: &mut AtmosphereLUTs,
    uploader: &mut BufferInitializer,
    uniform_ring_buffer: &mut UniformRingBuffer,
    frame_index: u32,
    dispatch: UVec3,
) {
    ctx.record(|encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                slice.copy_from_slice(sbt.buffer())
            });
        let cam_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&cam_uniform));
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&sharc_constants));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas_locked = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let hash_buf = encoder.lock(
            &resources.hash_entries,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let accum_buf = encoder.lock(
            &resources.accumulation,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let resolved_buf = encoder.lock(
            &resources.resolved,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        encoder.use_image_resource(
            render_target_views.hdr_output.image(),
            &mut hdr.state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.albedo.image(),
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
            render_target_views.normal.image(),
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
            render_target_views.depth.image(),
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

        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.use_image_resource(
            transmittance_view.image(),
            &mut atmosphere_luts.transmittance.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            sky_view.image(),
            &mut atmosphere_luts.sky_view.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );

        let rt_rw = Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        };
        encoder.use_buffer_resource(hash_buf, &mut resources.hash_entries_state, rt_rw);
        encoder.use_buffer_resource(accum_buf, &mut resources.accumulation_state, rt_rw);
        encoder.use_buffer_resource(resolved_buf, &mut resources.resolved_state, rt_rw);

        encoder.memory_barrier(
            Access::COPY_WRITE,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            },
        );

        encoder.emit_barriers();

        // Single push descriptor set: bindings 0..13 via the PbrPipelineParams
        // codegen helper, 14..17 SHARC bindings appended. Vulkan permits only
        // one push-descriptor set per pipeline layout (see sharc.playout.ron),
        // and `params` is pinned to set 0 in pbr_common.slang so this lines up
        // with what Slang emits.
        let mut params = PbrPipelineParams::new();
        params
            .scene_bvh(tlas_locked)
            .uniforms(cam_buffer)
            .output_texture(&render_target_views.hdr_output)
            .gbuffer_albedo_linear(render_target_views.albedo.linear_view())
            .gbuffer_albedo_srgb(render_target_views.albedo.srgb_view())
            .gbuffer_normal_texture(&render_target_views.normal)
            .gbuffer_depth_texture(&render_target_views.depth)
            .gbuffer_occlusion_texture(render_target_views.sdr_target.linear_view())
            .gbuffer_motion_vector_texture(&render_target_views.motion_vectors)
            .sky_atmosphere_params(atmo_buffer)
            .sky_transmittance_lut(transmittance_view)
            .sky_sky_view_lut(sky_view)
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .per_instance_data(per_instance_buf);

        let hash_info = [vk::DescriptorBufferInfo {
            buffer: hash_buf.vk_handle(),
            offset: 0,
            range: hash_buf.size(),
        }];
        let accum_info = [vk::DescriptorBufferInfo {
            buffer: accum_buf.vk_handle(),
            offset: 0,
            range: accum_buf.size(),
        }];
        let resolved_info = [vk::DescriptorBufferInfo {
            buffer: resolved_buf.vk_handle(),
            offset: 0,
            range: resolved_buf.size(),
        }];
        let cb_info = [vk::DescriptorBufferInfo {
            buffer: constants_buffer.vk_handle(),
            offset: constants_buffer.offset(),
            range: constants_buffer.size(),
        }];
        let mut writes: Vec<vk::WriteDescriptorSet> = params.as_slice().to_vec();
        writes.extend_from_slice(&[
            vk::WriteDescriptorSet {
                dst_binding: 14,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&hash_info),
            vk::WriteDescriptorSet {
                dst_binding: 15,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&accum_info),
            vk::WriteDescriptorSet {
                dst_binding: 16,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&resolved_info),
            vk::WriteDescriptorSet {
                dst_binding: 17,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ..Default::default()
            }
            .buffer_info(&cb_info),
        ]);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            &writes,
        );

        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&frame_index),
        );

        encoder.trace_rays(sbt, 0, sbt_buffer, dispatch);
    });
}

// ─── Resolve (compute) ─────────────────────────────────────────────────────

fn sharc_resolve_pass(
    mut ctx: SubmissionState,
    sharc_pipelines: Res<SharcPipelines>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    config: Res<SharcConfig>,
    frame_state: Res<SharcFrameState>,
    mut sharc_resources: Option<ResMut<SharcResources>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<&GlobalTransform, With<bevy::window::PrimaryWindow>>,
) {
    if !config.enabled {
        return;
    }
    let Some(resources) = sharc_resources.as_deref_mut() else { return };
    let Ok(transform) = swapchain_images.single() else { return };
    let Some(pipeline) = compute_pipelines.get(&sharc_pipelines.resolve_pipeline) else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();

    let camera_origin = transform.translation();
    let constants = build_sharc_constants(&config, &frame_state, camera_origin);
    let entries_num = config.entries_num;

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));

        let hash_buf =
            encoder.lock(&resources.hash_entries, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let accum_buf =
            encoder.lock(&resources.accumulation, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let resolved_buf =
            encoder.lock(&resources.resolved, vk::PipelineStageFlags2::COMPUTE_SHADER);

        let rw = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        };
        encoder.use_buffer_resource(hash_buf, &mut resources.hash_entries_state, rw);
        encoder.use_buffer_resource(accum_buf, &mut resources.accumulation_state, rw);
        encoder.use_buffer_resource(resolved_buf, &mut resources.resolved_state, rw);
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline.clone());
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        let cb_info = [vk::DescriptorBufferInfo {
            buffer: constants_buffer.vk_handle(),
            offset: constants_buffer.offset(),
            range: constants_buffer.size(),
        }];
        let hash_info = [vk::DescriptorBufferInfo {
            buffer: hash_buf.vk_handle(),
            offset: 0,
            range: hash_buf.size(),
        }];
        let accum_info = [vk::DescriptorBufferInfo {
            buffer: accum_buf.vk_handle(),
            offset: 0,
            range: accum_buf.size(),
        }];
        let resolved_info = [vk::DescriptorBufferInfo {
            buffer: resolved_buf.vk_handle(),
            offset: 0,
            range: resolved_buf.size(),
        }];
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&cb_info),
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&hash_info),
                vk::WriteDescriptorSet {
                    dst_binding: 2,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&accum_info),
                vk::WriteDescriptorSet {
                    dst_binding: 3,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&resolved_info),
            ],
        );

        encoder.dispatch(UVec3::new(entries_num.div_ceil(256), 1, 1));
    });
}

// ─── egui UI ───────────────────────────────────────────────────────────────

fn sharc_debug_ui(
    mut contexts: EguiContexts,
    mut debug_state: ResMut<SharcDebugState>,
    mut config: ResMut<SharcConfig>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Window::new("SHARC")
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.collapsing("Pipeline", |ui| {
                ui.checkbox(&mut config.enabled, "Enabled (Update / Resolve / Query)");
                if ui.button("Reset cache").clicked() {
                    config.reset_pending = true;
                }

                // Query-pass diagnostic overlays — mirror the switch in
                // sharc_pt.slang. Mode 0 is normal rendering.
                let mut mode = config.debug_mode;
                egui::ComboBox::from_label("Debug mode")
                    .selected_text(match mode {
                        0 => "0: off (normal SHARC render)",
                        1 => "1: solid red (Query ran?)",
                        2 => "2: cache occupancy heatmap",
                        3 => "3: colored hash at primary",
                        4 => "4: cached radiance at primary",
                        _ => "?",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut mode, 0, "0: off (normal SHARC render)");
                        ui.selectable_value(&mut mode, 1, "1: solid red (Query ran?)");
                        ui.selectable_value(&mut mode, 2, "2: cache occupancy heatmap");
                        ui.selectable_value(&mut mode, 3, "3: colored hash at primary");
                        ui.selectable_value(&mut mode, 4, "4: cached radiance at primary");
                    });
                config.debug_mode = mode;

                ui.add(
                    egui::Slider::new(&mut config.downscale_factor, 1..=10)
                        .text("Downscale factor"),
                );
                ui.add(
                    egui::Slider::new(&mut config.accumulation_frame_num, 0..=128)
                        .text("Accumulation frames"),
                );
                ui.add(
                    egui::Slider::new(&mut config.stale_frame_num, 0..=256)
                        .text("Stale frames"),
                );
                ui.add(
                    egui::Slider::new(&mut config.scene_scale, 1.0..=200.0).text("Scene scale"),
                );
                ui.add(
                    egui::Slider::new(&mut config.roughness_min, 0.0..=1.0)
                        .text("Min roughness (Update)"),
                );
                ui.add(
                    egui::Slider::new(&mut config.radiance_scale, 1.0..=1.0e4)
                        .logarithmic(true)
                        .text("Radiance scale"),
                );
            });

            ui.collapsing("Hash-grid debug overlay (off when SHARC enabled)", |ui| {
                ui.checkbox(&mut debug_state.enabled, "Overlay");
                ui.add(
                    egui::Slider::new(&mut debug_state.scene_scale, 1.0..=500.0)
                        .logarithmic(true)
                        .text("Scene scale"),
                );
                ui.add(
                    egui::Slider::new(&mut debug_state.logarithm_base, 1.1..=4.0)
                        .text("Logarithm base"),
                );
                ui.add(
                    egui::Slider::new(&mut debug_state.level_bias, -4.0..=4.0)
                        .text("Level bias"),
                );
                ui.add(
                    egui::Slider::new(&mut debug_state.brightness, 0.0..=10.0).text("Brightness"),
                );
            });
        });
}
