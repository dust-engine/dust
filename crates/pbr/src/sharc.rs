//! SHARC integration: hash-grid debug overlay (Phase 1) + Update/Resolve
//! pipeline (Phase 2, AGENTS.md §1).
//!
//! When `SharcConfig::enabled` is true the renderer runs the cache-population
//! passes each frame:
//!
//!     SHARC Update (sparse RT)  →  barrier  →  SHARC Resolve (compute)
//!
//! Visible indirect light continues to come from `final_gather_pass`; Update
//! keeps the cache fresh so future work can sample from it. The Phase 1
//! colored-hash debug overlay is painted by `final_gather` when SHARC is off,
//! controlled by `SharcDebugState`.

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
use dust_gfxdebug::{GpuProfiler, GpuTimerCommands};
use std::ops::Deref;

use crate::camera::Camera;
use crate::sky::AtmosphereLUTs;
use crate::{
    HdrRenderTarget, PbrInstanceData, PbrPipelineParams, PbrRenderSet,
    build_camera_uniform,
};

// ─── SHARC debug visualization ─────────────────────────────────────────────

/// Which SHARC debug view to draw. When not `Off`, the SHARC debug compute pass
/// (`sharc_debug_overlay_pass`) paints the visualization into occlusionTexture,
/// which tonemap composites over the frame, and the shadow pass is skipped (see
/// `shadow_pass`). Encoded into `SharcShaderConstants::debugMode`.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SharcDebugMode {
    /// Normal rendering — no overlay.
    #[default]
    Off,
    /// Colored hash of the grid cell each surface point lands in. Verifies grid
    /// layout / scene scale against geometry.
    ColoredHash,
    /// The resolved radiance currently cached at each surface point. Cells with
    /// no usable entry paint black.
    CachedRadiance,
    /// Hash-collision heatmap: the linear-probe bucket offset each surface
    /// point resolved to (blue = no collision … red = heavy probing). Verifies
    /// hash-grid load / collision behavior against scene scale.
    HashCollisions,
    HashOccupancy,
    /// Splat every occupied candidate-pool slot's world position onto the
    /// screen — shows where the cache is generating candidates (i.e. where it's
    /// short on samples). A scatter pass, not per-pixel.
    CandidatePool,
}

impl SharcDebugMode {
    /// Wire encoding for `SharcShaderConstants::debugMode`.
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            SharcDebugMode::Off => 0,
            SharcDebugMode::ColoredHash => 1,
            SharcDebugMode::CachedRadiance => 2,
            SharcDebugMode::HashCollisions => 3,
            SharcDebugMode::HashOccupancy => 4,
            SharcDebugMode::CandidatePool => 5,
        }
    }
}

/// Controls for the SHARC debug visualization painted by the primary pass.
#[derive(Resource, Clone, Copy)]
pub struct SharcDebugState {
    pub mode: SharcDebugMode,
    /// Multiplier applied to the visualized color before tone-mapping.
    pub brightness: f32,
}

impl SharcDebugState {
    pub fn is_active(&self) -> bool {
        self.mode != SharcDebugMode::Off
    }
}

impl Default for SharcDebugState {
    fn default() -> Self {
        Self {
            mode: SharcDebugMode::Off,
            brightness: 1.0,
        }
    }
}

/// Push constants for the final-gather ray-gen — just the per-frame sampling
/// seed. (The SHARC debug overlay used to ride along here; it now lives in the
/// primary pass via `SharcShaderConstants::debugMode`.)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FinalGatherPush {
    pub frame_index: u32,
}

impl FinalGatherPush {
    pub(crate) fn new(frame_index: u32) -> Self {
        Self { frame_index }
    }
}

// ─── Phase 2: SHARC Update / Resolve pipeline ──────────────────────────────

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
    /// Size of the candidate pool — each slot is one Update-pass dispatch
    /// invocation. The Update cascade runs a per-slot WRS-of-1, so this is
    /// also the effective Update ray-count budget.
    pub pool_capacity: u32,
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
            pool_capacity: 1 << 16, // 64K candidates × 32 B = 2 MiB per pool
        }
    }
}

/// CPU mirror of `SharcShaderConstants` in `sharc.slang`.
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
    pub pool_capacity: u32,
    /// Primary-pass debug view (`SharcDebugMode::as_u32`). The cache passes
    /// leave this 0; only `render` sets it from `SharcDebugState`.
    pub debug_mode: u32,
    pub debug_brightness: f32,
}

#[derive(Resource, Default)]
pub(crate) struct SharcFrameState {
    pub(crate) frame_index: u32,
    pub(crate) prev_camera_position: Vec3,
    pub(crate) prev_has_value: bool,
}

/// One side of the ping-ponged candidate pool. Each frame the renderer reads
/// one and writes the other; roles swap based on `SharcFrameState::frame_index`
/// parity.
///
/// Layout:
///   - `candidates`: capacity × 32 B (`Candidate { vec3 worldPos; vec3 normal; }`
///     padded to std430 stride).
///   - `keys`: capacity × 4 B (`asuint(r)` per slot — the WRS-of-1 reservoir
///     key; 0 means empty).
pub struct CandidatePool {
    pub candidates: GPUMutex<Buffer>,
    pub keys: GPUMutex<Buffer>,
    pub candidates_state: ResourceState,
    pub keys_state: ResourceState,
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
    pub pool_capacity: u32,
    /// Ping-ponged. `frame_index & 1 == 0` ⇒ pool[0] = read, pool[1] = write.
    pub pool: [CandidatePool; 2],
    needs_clear: bool,
}

impl SharcResources {
    const STRIDE_HASH_ENTRIES: u64 = 8;
    const STRIDE_ACCUMULATION: u64 = 16;
    const STRIDE_RESOLVED: u64 = 16;
    /// Std430 stride for
    /// `Candidate { vec3 worldPos; vec3 normal; vec3 albedo; }`. Each vec3
    /// is 16-aligned in std430 (12 B payload + 4 B pad), so the struct sums
    /// to 48 B with no trailing pad needed.
    const STRIDE_CANDIDATE: u64 = 48;
    const STRIDE_KEY: u64 = 4;

    /// Returns `(read_index, write_index)` into `self.pool` based on the
    /// frame parity. Frame N reads what frame N-1 wrote, and writes for
    /// frame N+1 to read.
    pub fn pool_indices(frame_index: u32) -> (usize, usize) {
        let read = (frame_index & 1) as usize;
        (read, 1 - read)
    }
}

fn create_sharc_resources(
    allocator: &Allocator,
    entries_num: u32,
    pool_capacity: u32,
) -> SharcResources {
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
    let pool = [(); 2].map(|_| {
        let cap = pool_capacity as u64;
        let candidates = Buffer::new_private(
            allocator.clone(),
            cap * SharcResources::STRIDE_CANDIDATE,
            16,
            usage,
        )
        .expect("SHARC candidate pool buffer");
        let keys = Buffer::new_private(
            allocator.clone(),
            cap * SharcResources::STRIDE_KEY,
            16,
            usage,
        )
        .expect("SHARC reservoir keys buffer");
        CandidatePool {
            candidates: GPUMutex::new(candidates),
            keys: GPUMutex::new(keys),
            candidates_state: ResourceState::default(),
            keys_state: ResourceState::default(),
        }
    });
    SharcResources {
        entries_num,
        hash_entries: GPUMutex::new(hash_entries),
        accumulation: GPUMutex::new(accumulation),
        resolved: GPUMutex::new(resolved),
        hash_entries_state: ResourceState::default(),
        accumulation_state: ResourceState::default(),
        resolved_state: ResourceState::default(),
        pool_capacity,
        pool,
        needs_clear: true,
    }
}

#[derive(Resource)]
pub struct SharcPipelines {
    pub update_pipeline: Handle<RayTracingPipeline>,
    pub resolve_pipeline: Handle<ComputePipeline>,
    /// SHARC debug overlay (all debug views). Shares the RT pipeline layout
    /// (pbr.playout) so it can bind the G-buffers + SHARC buffers through the
    /// same descriptor slots the RT pipelines use. Dispatched before tonemap
    /// when a debug mode is active (`SharcDebugState::is_active`).
    pub debug_overlay_pipeline: Handle<ComputePipeline>,
    /// Candidate-pool splat debug view — a second entry point of the debug
    /// shader, dispatched over the pool (scatter) instead of per-pixel.
    pub candidate_splat_pipeline: Handle<ComputePipeline>,
    pub update_sbt: Option<pumicite::rtx::ShaderBindingTable>,
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
                    .after(create_sharc_sbts)
                    // Update consumes this frame's read pool but ALSO writes
                    // cascade entries into the *write* pool — same buffer
                    // final_gather's CHS pushes into on cache miss.
                    // Serializing avoids both dispatches' atomic_max ops
                    // racing on the same shader storage in adjacent
                    // submissions.
                    .after(crate::final_gather_pass),
                sharc_resolve_pass
                    .in_set(DefaultRenderSet)
                    .after(sharc_update_pass),
                // Debug overlay writes the visualization into occlusionTexture
                // (== sdr_target), which tonemap then composites onto the
                // swapchain — the same path egui rides. Ordered after
                // final-gather and before Update so the candidate-pool splat
                // reads this frame's final-gather write side (Update would
                // otherwise add cascade entries and flip the ping-pong index).
                sharc_debug_overlay_pass
                    .in_set(DefaultRenderSet)
                    .after(crate::final_gather_pass)
                    .before(sharc_update_pass)
                    .before(crate::tonemap_pass),
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
    let resources = create_sharc_resources(&allocator, config.entries_num, config.pool_capacity);
    commands.insert_resource(resources);

    let update_library: Handle<RayTracingPipelineLibrary> = asset_server.load(
        "bazel://dust/crates/pbr/shaders/sharc/sharc_update.rtx.pipeline.bin",
    );
    let resolve_pipeline: Handle<ComputePipeline> = asset_server
        .load("bazel://dust/crates/pbr/shaders/sharc/sharc_resolve.comp.pipeline.bin");
    let debug_overlay_pipeline: Handle<ComputePipeline> = asset_server
        .load("bazel://dust/crates/pbr/shaders/sharc_occupancy.comp.pipeline.bin");
    let candidate_splat_pipeline: Handle<ComputePipeline> = asset_server
        .load("bazel://dust/crates/pbr/shaders/sharc_candidate_splat.comp.pipeline.bin");

    commands.insert_resource(SharcPipelines {
        update_pipeline: pipeline_manager.add_pipeline(update_library),
        resolve_pipeline,
        debug_overlay_pipeline,
        candidate_splat_pipeline,
        update_sbt: None,
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
}

// ─── Clear ─────────────────────────────────────────────────────────────────

pub(crate) fn clear_sharc_buffers(
    mut ctx: SubmissionState,
    mut resources: Option<ResMut<SharcResources>>,
    mut config: ResMut<SharcConfig>,
    frame_state: Res<SharcFrameState>,
) {
    let Some(resources) = resources.as_deref_mut() else { return };
    let full_reset = resources.needs_clear || config.reset_pending;
    // Per-frame: the side that's about to be *written* (by final_gather's
    // CHS and Update's cascade) needs its reservoir keys zeroed so the
    // WRS-of-1 atomic_max starts from "empty". The candidates buffer
    // doesn't need clearing — consumers gate on `keys[slot] != 0`.
    let (_, write_idx) = SharcResources::pool_indices(frame_state.frame_index);

    ctx.record(|encoder| {
        let device = encoder.device().clone();
        if full_reset {
            let hash_buf =
                encoder.lock(&resources.hash_entries, vk::PipelineStageFlags2::COPY);
            let accum_buf =
                encoder.lock(&resources.accumulation, vk::PipelineStageFlags2::COPY);
            let resolved_buf =
                encoder.lock(&resources.resolved, vk::PipelineStageFlags2::COPY);
            let keys_bufs = [
                encoder.lock(&resources.pool[0].keys, vk::PipelineStageFlags2::COPY),
                encoder.lock(&resources.pool[1].keys, vk::PipelineStageFlags2::COPY),
            ];
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
            // Clear both pools' keys on reset (the read side might still
            // carry stale candidates from a previous run).
            let [pool0, pool1] = &mut resources.pool;
            encoder.use_buffer_resource(keys_bufs[0], &mut pool0.keys_state, Access::CLEAR);
            encoder.use_buffer_resource(keys_bufs[1], &mut pool1.keys_state, Access::CLEAR);
            encoder.emit_barriers();
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
                for kb in &keys_bufs {
                    device.cmd_fill_buffer(cmd_buffer, kb.vk_handle(), 0, kb.size(), 0);
                }
            }
        } else {
            // Per-frame: just clear the write side's keys.
            let keys_buf = encoder.lock(
                &resources.pool[write_idx].keys,
                vk::PipelineStageFlags2::COPY,
            );
            encoder.use_buffer_resource(
                keys_buf,
                &mut resources.pool[write_idx].keys_state,
                Access::CLEAR,
            );
            encoder.emit_barriers();
            let cmd_buffer = encoder.buffer().vk_handle();
            unsafe {
                device.cmd_fill_buffer(cmd_buffer, keys_buf.vk_handle(), 0, keys_buf.size(), 0);
            }
        }
    });
    if full_reset {
        resources.needs_clear = false;
        config.reset_pending = false;
    }
}

// ─── Per-frame constants ───────────────────────────────────────────────────

pub(crate) fn build_sharc_constants(
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
        pool_capacity: config.pool_capacity,
        // Debug-off by default; `render` overrides these for the primary pass.
        debug_mode: 0,
        debug_brightness: 1.0,
    }
}

// ─── Update (RT) ───────────────────────────────────────────────────────────

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
    mut profiler: Option<ResMut<GpuProfiler>>,
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

    // Pool-driven Update: dispatch is 1D over the candidate pool. Each
    // invocation reads `keys[slot]`; empty slots no-op. This means the
    // dispatch is the upper bound on Update ray-tracing work — exactly the
    // hook we need for adaptive ray budgeting later.
    let dispatch = UVec3::new(config.pool_capacity, 1, 1);

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
        profiler.as_deref_mut(),
    );
}

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
    mut profiler: Option<&mut GpuProfiler>,
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
        // Pool ping-pong: pick which physical buffers go to read vs write
        // bindings based on this frame's parity. Update's raygen consumes
        // `read`, and its cascade pushes into `write` for next frame to
        // consume.
        let (pool_read_idx, pool_write_idx) =
            SharcResources::pool_indices(frame_index);
        let pool_read_candidates_buf = encoder.lock(
            &resources.pool[pool_read_idx].candidates,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_read_keys_buf = encoder.lock(
            &resources.pool[pool_read_idx].keys,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_write_candidates_buf = encoder.lock(
            &resources.pool[pool_write_idx].candidates,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_write_keys_buf = encoder.lock(
            &resources.pool[pool_write_idx].keys,
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
        let rt_read = Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_buffer_resource(hash_buf, &mut resources.hash_entries_state, rt_rw);
        encoder.use_buffer_resource(accum_buf, &mut resources.accumulation_state, rt_rw);
        encoder.use_buffer_resource(resolved_buf, &mut resources.resolved_state, rt_rw);
        // Split pool[0]/pool[1] mutably without violating borrow rules.
        let [pool_a, pool_b] = &mut resources.pool;
        let (pool_read, pool_write) = if pool_read_idx == 0 {
            (pool_a, pool_b)
        } else {
            (pool_b, pool_a)
        };
        encoder.use_buffer_resource(
            pool_read_candidates_buf,
            &mut pool_read.candidates_state,
            rt_read,
        );
        encoder.use_buffer_resource(
            pool_read_keys_buf,
            &mut pool_read.keys_state,
            rt_read,
        );
        encoder.use_buffer_resource(
            pool_write_candidates_buf,
            &mut pool_write.candidates_state,
            rt_rw,
        );
        encoder.use_buffer_resource(
            pool_write_keys_buf,
            &mut pool_write.keys_state,
            rt_rw,
        );

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
        let pool_read_candidates_info = [vk::DescriptorBufferInfo {
            buffer: pool_read_candidates_buf.vk_handle(),
            offset: 0,
            range: pool_read_candidates_buf.size(),
        }];
        let pool_read_keys_info = [vk::DescriptorBufferInfo {
            buffer: pool_read_keys_buf.vk_handle(),
            offset: 0,
            range: pool_read_keys_buf.size(),
        }];
        let pool_write_candidates_info = [vk::DescriptorBufferInfo {
            buffer: pool_write_candidates_buf.vk_handle(),
            offset: 0,
            range: pool_write_candidates_buf.size(),
        }];
        let pool_write_keys_info = [vk::DescriptorBufferInfo {
            buffer: pool_write_keys_buf.vk_handle(),
            offset: 0,
            range: pool_write_keys_buf.size(),
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
            vk::WriteDescriptorSet {
                dst_binding: 18,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&pool_read_candidates_info),
            vk::WriteDescriptorSet {
                dst_binding: 19,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&pool_read_keys_info),
            vk::WriteDescriptorSet {
                dst_binding: 20,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&pool_write_candidates_info),
            vk::WriteDescriptorSet {
                dst_binding: 21,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&pool_write_keys_info),
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

        encoder.timing_scope(profiler.as_deref_mut(), "SHARC rays", |encoder| {
            encoder.trace_rays(sbt, 0, sbt_buffer, dispatch);
        });
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
    mut profiler: Option<ResMut<GpuProfiler>>,
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

        encoder.timing_scope(profiler.as_deref_mut(), "SHARC resolve", |encoder| {
            encoder.dispatch(UVec3::new(entries_num.div_ceil(256), 1, 1));
        });
    });
}

// ─── Debug overlay (compute) ─────────────────────────────────────────────────

/// All SHARC debug views, in one compute pass. Runs after the RT + tonemap
/// passes when a debug mode is active, dispatching one thread per display pixel
/// and writing the visualization straight into `occlusionTexture` (the final
/// SDR image) — see `sharc_occupancy.slang`. The compute pipeline shares the RT
/// pipeline layout (pbr.playout), so it binds the G-buffers + SHARC buffers
/// through the same descriptor slots the RT pipelines use.
///
/// Per-surface views (colored hash / cached radiance / hash collisions)
/// reconstruct worldPos from the depth G-buffer, so this binds the camera
/// uniform (1), normal (5), depth (6) and the resolved cache (16) in addition
/// to the occlusion image (7), hash entries (14) and constants (17).
fn sharc_debug_overlay_pass(
    mut ctx: SubmissionState,
    sharc_pipelines: Res<SharcPipelines>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    config: Res<SharcConfig>,
    debug: Res<SharcDebugState>,
    frame_state: Res<SharcFrameState>,
    mut sharc_resources: Option<ResMut<SharcResources>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    jitter: Res<crate::JitterState>,
) {
    if !debug.is_active() {
        return;
    }
    let Some(resources) = sharc_resources.as_deref_mut() else { return };
    let Some(hdr) = hdr_target.as_mut() else { return };
    let Ok((camera, transform)) = swapchain_images.single() else { return };

    // The per-surface views reconstruct worldPos from depth, so the camera
    // uniform must carry the same jitter the primary pass used to write depth.
    let cam_uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let mut constants =
        build_sharc_constants(&config, &frame_state, transform.translation());
    constants.debug_mode = debug.mode.as_u32();
    constants.debug_brightness = debug.brightness;

    // Candidate-pool splat is a scatter over the pool (one thread per slot), not
    // a per-pixel pass, so it has its own entry point + dispatch. It samples
    // this frame's final-gather *write* side, bound to the read slots (18/19)
    // the splat shader reads through.
    if debug.mode == SharcDebugMode::CandidatePool {
        let Some(splat) = compute_pipelines.get(&sharc_pipelines.candidate_splat_pipeline)
        else {
            return;
        };
        let splat = splat.clone().into_inner();
        let (_, write_idx) = SharcResources::pool_indices(frame_state.frame_index);
        // splatCandidates is numthreads(32, 1, 1).
        let groups = config.pool_capacity.div_ceil(32);
        ctx.record(|encoder| {
            let constants_buffer =
                uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
            let cam_buffer =
                uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&cam_uniform));
            let render_target_views =
                encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
            let cand_buf = encoder.lock(
                &resources.pool[write_idx].candidates,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );
            let keys_buf = encoder.lock(
                &resources.pool[write_idx].keys,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );

            encoder.use_image_resource(
                render_target_views.sdr_target.image(),
                &mut hdr.sdr_target_state,
                Access {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                },
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                false,
            );
            let read = Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            };
            {
                let pool = &mut resources.pool[write_idx];
                encoder.use_buffer_resource(cand_buf, &mut pool.candidates_state, read);
                encoder.use_buffer_resource(keys_buf, &mut pool.keys_state, read);
            }
            encoder.emit_barriers();

            let splat = encoder.retain(splat.clone());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &splat);
            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                splat.layout(),
                0,
                &[
                    vk::WriteDescriptorSet {
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: cam_buffer.vk_handle(),
                            offset: cam_buffer.offset(),
                            range: cam_buffer.size(),
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 7,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: render_target_views.sdr_target.linear_view().vk_handle(),
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 17,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: constants_buffer.vk_handle(),
                            offset: constants_buffer.offset(),
                            range: constants_buffer.size(),
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 18,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: cand_buf.vk_handle(),
                            offset: cand_buf.offset(),
                            range: cand_buf.size(),
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 19,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: keys_buf.vk_handle(),
                            offset: keys_buf.offset(),
                            range: keys_buf.size(),
                        },
                        ..Default::default()
                    },
                ],
            );
            encoder.dispatch(UVec3::new(groups, 1, 1));
        });
        return;
    }

    let Some(pipeline) = compute_pipelines.get(&sharc_pipelines.debug_overlay_pipeline) else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();

    let dispatch = UVec3::new(
        hdr.display_extent.x.div_ceil(8),
        hdr.display_extent.y.div_ceil(8),
        1,
    );

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let cam_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&cam_uniform));

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let hash_buf = encoder.lock(
            &resources.hash_entries,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let resolved_buf = encoder.lock(
            &resources.resolved,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        // The SDR target (== occlusionTexture) already holds the tonemapped
        // scene + egui; we overwrite only the visualized pixels, so it's read-
        // modify-write storage access. Normal + depth G-buffers are read-only.
        encoder.use_image_resource(
            render_target_views.sdr_target.image(),
            &mut hdr.sdr_target_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.normal.image(),
            &mut hdr.normal_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
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
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        let read = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_buffer_resource(hash_buf, &mut resources.hash_entries_state, read);
        encoder.use_buffer_resource(resolved_buf, &mut resources.resolved_state, read);
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline.clone());
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        // Manual partial bind of the shared RT layout: only the slots the debug
        // overlay reads/writes. (The PbrPipelineParams codegen helper emits all
        // of its writes, so it can't be used for a partial set yet.)
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: cam_buffer.vk_handle(),
                        offset: cam_buffer.offset(),
                        range: cam_buffer.size(),
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 5,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views.normal.vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 6,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views.depth.vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 7,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views.sdr_target.linear_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 14,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: hash_buf.vk_handle(),
                        offset: hash_buf.offset(),
                        range: hash_buf.size(),
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 16,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: resolved_buf.vk_handle(),
                        offset: resolved_buf.offset(),
                        range: resolved_buf.size(),
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 17,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: constants_buffer.vk_handle(),
                        offset: constants_buffer.offset(),
                        range: constants_buffer.size(),
                    },
                    ..Default::default()
                },
            ],
        );
        encoder.dispatch(dispatch);
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
            // Master controls — the two knobs reached for most often.
            ui.horizontal(|ui| {
                ui.checkbox(&mut config.enabled, "Cache enabled");
                if ui.button("Reset cache").clicked() {
                    config.reset_pending = true;
                }
            });

            ui.separator();

            // Grid — shared by the live cache and the debug overlay. There is
            // exactly one scene scale: the Update/Resolve passes and the
            // colored-hash overlay both key off it. logarithm base / level
            // bias are compile-time constants (SHARC_GRID_*), shown read-only
            // so the overlay can't drift from the grid the cache uses.
            ui.label(egui::RichText::new("Grid").strong());
            ui.add(egui::Slider::new(&mut config.scene_scale, 1.0..=200.0).text("Scene scale"));

            ui.separator();

            // Cache population (Update / Resolve) knobs.
            ui.collapsing("Cache population", |ui| {
                // Pool capacity ≡ Update ray budget. The pool buffers are
                // sized to SharcConfig::default().pool_capacity at startup, so
                // keep the slider at/under that — growing past it is a no-op
                // without a reallocation.
                ui.add(
                    egui::Slider::new(&mut config.pool_capacity, 1024..=(1 << 16))
                        .logarithmic(true)
                        .text("Pool capacity (Update rays / frame)"),
                );
                ui.add(
                    egui::Slider::new(&mut config.accumulation_frame_num, 0..=128)
                        .text("Accumulation frames"),
                );
                ui.add(
                    egui::Slider::new(&mut config.stale_frame_num, 0..=256).text("Stale frames"),
                );
                ui.add(
                    egui::Slider::new(&mut config.roughness_min, 0.0..=1.0).text("Min roughness"),
                );
                ui.add(
                    egui::Slider::new(&mut config.radiance_scale, 1.0..=1.0e4)
                        .logarithmic(true)
                        .text("Radiance scale"),
                );
            });

            // Debug view — painted by the SHARC debug compute pass into
            // occlusionTexture, composited over the frame by tonemap. While a
            // mode is active the shadow pass is skipped.
            ui.collapsing("Debug view", |ui| {
                ui.radio_value(&mut debug_state.mode, SharcDebugMode::Off, "Off");
                ui.radio_value(
                    &mut debug_state.mode,
                    SharcDebugMode::ColoredHash,
                    "Colored hash (grid cells)",
                );
                ui.radio_value(
                    &mut debug_state.mode,
                    SharcDebugMode::CachedRadiance,
                    "Cached radiance",
                );
                ui.radio_value(
                    &mut debug_state.mode,
                    SharcDebugMode::HashCollisions,
                    "Hash collisions",
                );
                ui.radio_value(
                    &mut debug_state.mode,
                    SharcDebugMode::HashOccupancy,
                    "Hash occupancy",
                );
                ui.radio_value(
                    &mut debug_state.mode,
                    SharcDebugMode::CandidatePool,
                    "Candidate pool (splat)",
                );
                ui.add(
                    egui::Slider::new(&mut debug_state.brightness, 0.0..=10.0).text("Brightness"),
                );
            });
        });
}
