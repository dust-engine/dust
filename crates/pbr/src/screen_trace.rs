//! Screen-space gather trace (Lumen-style): march each pixel's gather ray
//! through the HZB, light on-screen hits from last frame's denoised color, and
//! give the misses the sky. Shader: screen_trace.slang.
//!
//! No hardware gather follows the trace any more, so nothing reads the list of
//! misses the compaction builds (`compact_traces_pass`, skipped in screen-only
//! mode) or uses [`ScreenTraceConfig::pullback_bias`].

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::{ReadbackRingBuffer, UniformRingBuffer};
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::buffer::{Buffer, BufferLike, RingBufferSuballocation};
use pumicite::debug::DebugObject;
use pumicite::image::{FullImageView, Image, ImageExt};
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice};

use crate::camera::Camera;
use crate::hzb::{HzbResources, HzbState};
use crate::sky::AtmosphereLUTs;
use crate::{HdrRenderTarget, HdrRenderTargetViewsExt, PbrPipelineParams, build_camera_uniform};

/// Debug view painted over the frame by `screen_trace_debug_pass`. Wire
/// values are the `DEBUG_*` constants in screen_trace.slang.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenTraceDebugMode {
    #[default]
    Off,
    /// Per pixel: green = screen hit, yellow = hit rejected by the history or
    /// edge tests, orange = uncertain (thin feature), red = miss, blue = trace
    /// skipped.
    TraceTypes,
    /// One HZB level as log-scaled gray (`debug_mip`).
    HzbMip,
    /// Last frame's denoised color, what the screen hits sample.
    PrevColor,
}

impl ScreenTraceDebugMode {
    fn as_u32(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::TraceTypes => 1,
            Self::HzbMip => 2,
            Self::PrevColor => 3,
        }
    }
}

/// Tunables of the screen trace; the devtools "Screen trace" page edits them.
#[derive(Resource, Clone)]
pub struct ScreenTraceConfig {
    /// Off: no gather at all, so the frame holds direct light only (A/B
    /// reference).
    pub enabled: bool,
    /// The screen trace resolves every pixel, with the sky as the fallback for
    /// rays that leave the screen (SCREEN_TRACE_FLAG_SCREEN_ONLY). Off: only
    /// accepted hits add light.
    pub screen_only: bool,
    pub max_iterations: u32,
    pub relative_depth_thickness: f32,
    pub history_depth_thickness: f32,
    /// World units. Voxel geometry is all hard edges; about half a voxel.
    pub min_world_thickness: f32,
    pub num_thickness_steps: u32,
    /// World units a hardware ray would back up from the last visible point.
    /// Unused: no hardware gather follows the trace.
    pub pullback_bias: f32,
    /// Lanes of a wave that must still be marching for the wave to go on;
    /// 0 lets every ray run to its own end.
    pub minimum_occupancy: u32,
    pub debug_mode: ScreenTraceDebugMode,
    pub debug_mip: u32,
    pub debug_scale: f32,
}

impl Default for ScreenTraceConfig {
    /// The defaults, with `DUST_SCREEN_TRACE=0|1`, `DUST_SCREEN_ONLY=0|1`,
    /// `DUST_SCREEN_TRACE_ITER`,
    /// `DUST_SCREEN_TRACE_OCCUPANCY` and `DUST_SCREEN_TRACE_DEBUG=0..3` (the
    /// `ScreenTraceDebugMode` order) honored for headless captures.
    fn default() -> Self {
        let env_u32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        Self {
            enabled: env_u32("DUST_SCREEN_TRACE").is_none_or(|v| v != 0),
            screen_only: env_u32("DUST_SCREEN_ONLY").is_none_or(|v| v != 0),
            max_iterations: env_u32("DUST_SCREEN_TRACE_ITER").unwrap_or(50),
            relative_depth_thickness: 0.02,
            history_depth_thickness: 0.005,
            min_world_thickness: 0.03,
            num_thickness_steps: 4,
            pullback_bias: 0.1,
            minimum_occupancy: env_u32("DUST_SCREEN_TRACE_OCCUPANCY").unwrap_or(0),
            debug_mode: match env_u32("DUST_SCREEN_TRACE_DEBUG") {
                Some(1) => ScreenTraceDebugMode::TraceTypes,
                Some(2) => ScreenTraceDebugMode::HzbMip,
                Some(3) => ScreenTraceDebugMode::PrevColor,
                _ => ScreenTraceDebugMode::Off,
            },
            debug_mip: env_u32("DUST_SCREEN_TRACE_DEBUG_MIP").unwrap_or(0),
            debug_scale: 1.0,
        }
    }
}

/// CPU mirror of `ScreenTraceConstants` in pbr_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScreenTraceConstantsRaw {
    hzb_uv_factor: [f32; 2],
    hzb_inv_uv_factor: [f32; 2],
    hzb_base_texel_size: [f32; 2],
    render_extent: [f32; 2],
    max_iterations: f32,
    relative_depth_thickness: f32,
    history_depth_thickness: f32,
    min_world_thickness: f32,
    num_thickness_steps: u32,
    pullback_bias: f32,
    flags: u32,
    debug_mode: u32,
    frame_index: u32,
    hzb_num_mips: u32,
    debug_mip: u32,
    debug_scale: f32,
    hzb_extent: [u32; 2],
    minimum_occupancy: u32,
    _pad: u32,
}

// `SCREEN_TRACE_FLAG_*` in pbr_common.slang.
const FLAG_ENABLED: u32 = 1 << 0;
const FLAG_HISTORY_VALID: u32 = 1 << 1;
const FLAG_SCREEN_ONLY: u32 = 1 << 2;

pub struct ScreenTraceResources {
    /// R32_UINT, render resolution: the packed trace record per pixel.
    pub trace_hit: GPUMutex<FullImageView<Image>>,
    pub trace_hit_state: ResourceState,
    /// One `u32` per pixel: the compacted list of pixels needing a ray.
    pub compacted_texels: GPUMutex<Buffer>,
    pub compacted_texels_state: ResourceState,
    /// `[count, 1, 1]`, a VkTraceRaysIndirectCommandKHR.
    pub compacted_args: GPUMutex<Buffer>,
    pub compacted_args_state: ResourceState,
    /// `[1] = [2] = 1` not written yet.
    args_initialized: bool,
    render_extent: UVec2,
}

#[derive(Resource)]
pub struct ScreenTraceState {
    pub trace_pipeline: Handle<ComputePipeline>,
    pub compact_pipeline: Handle<ComputePipeline>,
    pub debug_pipeline: Handle<ComputePipeline>,
    pub resources: Option<ScreenTraceResources>,
    /// Per-frame sampling seed of the gather ray, advanced by
    /// `screen_trace_pass`.
    pub frame_index: u32,
    /// Pixels the trace left unresolved in the most recent frame whose count
    /// has been read back (`None` until the first readback, or while disabled).
    pub hardware_rays: Option<u32>,
    /// `screen_trace_pass` recorded this frame, so the trace records are
    /// current and `compact_traces_pass` may run. False while the pipelines are
    /// still loading.
    pub traced_this_frame: bool,
}

/// The state the screen trace passes share, bundled to stay under Bevy's system
/// parameter limit.
#[derive(SystemParam)]
pub(crate) struct ScreenTraceSystemParams<'w> {
    pub state: ResMut<'w, ScreenTraceState>,
    pub config: Res<'w, ScreenTraceConfig>,
    pub hzb: ResMut<'w, HzbState>,
}

impl ScreenTraceSystemParams<'_> {
    /// Whether the trace runs this frame.
    pub fn active(&self) -> bool {
        self.config.enabled && self.state.resources.is_some() && self.hzb.resources.is_some()
    }
}

pub struct ScreenTracePlugin;

impl Plugin for ScreenTracePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenTraceConfig>();
        app.add_systems(
            Startup,
            setup_screen_trace.after(CreateDevice).after(crate::setup),
        );
        app.add_systems(
            PostUpdate,
            (
                ensure_screen_trace_resources
                    .in_set(DefaultRenderSet)
                    .after(crate::ensure_hdr_target)
                    .before(screen_trace_pass),
                screen_trace_pass
                    .in_set(DefaultRenderSet)
                    .after(crate::hzb::hzb_build_pass)
                    // Adds to the HDR output the shadow miss shader also
                    // read-modify-writes.
                    .after(crate::shadow_pass)
                    // The lighting is complete only after this pass, and the
                    // upscaler, which runs after the auto-exposure, overwrites
                    // the history color it reads.
                    .before(crate::tonemap::autoexposure_pass),
                compact_traces_pass
                    .in_set(DefaultRenderSet)
                    .after(screen_trace_pass),
                screen_trace_debug_pass
                    .in_set(DefaultRenderSet)
                    .after(crate::clear_debug_overlay)
                    .after(screen_trace_pass)
                    // Before the upscaler overwrites the denoised image, so
                    // the "previous color" view shows what the trace read.
                    .before(crate::upscaler::upscaler_evaluate)
                    .before(crate::tonemap::tonemap_pass),
            ),
        );
    }
}

fn setup_screen_trace(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(ScreenTraceState {
        trace_pipeline: asset_server
            .load("bazel://dust/crates/pbr/shaders/screen_trace.comp.pipeline.bin"),
        compact_pipeline: asset_server
            .load("bazel://dust/crates/pbr/shaders/screen_trace_compact.comp.pipeline.bin"),
        debug_pipeline: asset_server
            .load("bazel://dust/crates/pbr/shaders/screen_trace_debug.comp.pipeline.bin"),
        resources: None,
        frame_index: 0,
        hardware_rays: None,
        traced_this_frame: false,
    });
}

fn create_screen_trace_resources(
    allocator: &Allocator,
    render_extent: UVec2,
) -> ScreenTraceResources {
    let trace_hit = Image::new_device(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_UINT,
            extent: vk::Extent3D {
                width: render_extent.x,
                height: render_extent.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Screen Trace Hit Records")
    .create_full_view()
    .unwrap()
    .with_name(c"Screen Trace Hit Records View");
    let texels = render_extent.x as u64 * render_extent.y as u64;
    let compacted_texels = Buffer::new_device(
        allocator.clone(),
        texels * 4,
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
    .unwrap();
    let compacted_args = Buffer::new_device(
        allocator.clone(),
        12,
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
    .unwrap();
    ScreenTraceResources {
        trace_hit: GPUMutex::new(trace_hit),
        trace_hit_state: ResourceState::default(),
        compacted_texels: GPUMutex::new(compacted_texels),
        compacted_texels_state: ResourceState::default(),
        compacted_args: GPUMutex::new(compacted_args),
        compacted_args_state: ResourceState::default(),
        args_initialized: false,
        render_extent,
    }
}

fn ensure_screen_trace_resources(
    mut state: ResMut<ScreenTraceState>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    allocator: Res<Allocator>,
) {
    let Some(hdr) = hdr_target else {
        return;
    };
    if hdr.render_extent.x == 0 || hdr.render_extent.y == 0 {
        return;
    }
    let stale = state
        .resources
        .as_ref()
        .is_none_or(|res| res.render_extent != hdr.render_extent);
    if stale {
        state.resources = Some(create_screen_trace_resources(&allocator, hdr.render_extent));
        state.hardware_rays = None;
    }
}

pub(crate) fn build_screen_trace_constants(
    config: &ScreenTraceConfig,
    hzb: &HzbResources,
    frame_index: u32,
    history_valid: bool,
    enabled: bool,
) -> ScreenTraceConstantsRaw {
    let uv_factor = hzb.uv_factor();
    let mut flags = 0;
    if enabled {
        flags |= FLAG_ENABLED;
    }
    if history_valid {
        flags |= FLAG_HISTORY_VALID;
    }
    if config.screen_only {
        flags |= FLAG_SCREEN_ONLY;
    }
    ScreenTraceConstantsRaw {
        hzb_uv_factor: uv_factor.to_array(),
        hzb_inv_uv_factor: (1.0 / uv_factor).to_array(),
        hzb_base_texel_size: (1.0 / hzb.extent.as_vec2()).to_array(),
        render_extent: hzb.render_extent.as_vec2().to_array(),
        max_iterations: config.max_iterations as f32,
        relative_depth_thickness: config.relative_depth_thickness,
        history_depth_thickness: config.history_depth_thickness,
        min_world_thickness: config.min_world_thickness,
        num_thickness_steps: config.num_thickness_steps,
        pullback_bias: config.pullback_bias,
        flags,
        debug_mode: config.debug_mode.as_u32(),
        frame_index,
        hzb_num_mips: hzb.num_mips,
        debug_mip: config.debug_mip.min(hzb.num_mips - 1),
        debug_scale: config.debug_scale,
        hzb_extent: hzb.extent.to_array(),
        minimum_occupancy: config.minimum_occupancy,
        _pad: 0,
    }
}

fn compute_read() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ,
    }
}

fn compute_sampled_read() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    }
}

fn compute_read_write() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    }
}

/// One thread per pixel: HZB march, on-screen lighting, trace record.
fn screen_trace_pass(
    mut ctx: SubmissionState,
    mut params: ScreenTraceSystemParams,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform)>,
    jitter: Res<crate::JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
) {
    // A fresh seed every frame, whether or not the trace runs.
    params.state.frame_index = params.state.frame_index.wrapping_add(1);
    params.state.traced_this_frame = false;
    if !params.active() {
        params.state.hardware_rays = None;
        return;
    }
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Ok((camera, transform)) = swapchain_images.single() else {
        return;
    };
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.clone() else {
        return;
    };
    // Both passes or neither: `traced_this_frame` promises the compaction runs.
    if compute_pipelines
        .get(&params.state.compact_pipeline)
        .is_none()
    {
        return;
    }
    let Some(pipeline) = compute_pipelines.get(&params.state.trace_pipeline) else {
        return;
    };
    params.state.traced_this_frame = true;
    let pipeline = pipeline.clone().into_inner();
    let frame_index = params.state.frame_index;
    let constants = build_screen_trace_constants(
        &params.config,
        params.hzb.resources.as_ref().unwrap(),
        frame_index,
        hdr.history_valid,
        true,
    );
    // Same jitter as the primary pass: positions are rebuilt from its depth.
    let cam_uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let render_extent = hdr.render_extent;
    let hzb = params.hzb.resources.as_mut().unwrap();
    let res = params.state.resources.as_mut().unwrap();

    ctx.record(|encoder, _| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let cam_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&cam_uniform));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let hzb_image = encoder.lock(&hzb.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);
        // Sky, for the screen-only misses.
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();
        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        // HDR: read-modify-write (adds the screen hits' lighting).
        encoder.use_image_resource(
            views.hdr_output(),
            &mut hdr.state,
            compute_read_write(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Albedo: sampled through the sRGB view.
        encoder.use_image_resource(
            views.albedo(),
            &mut hdr.albedo_state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.normal(),
            &mut hdr.normal_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.motion_vectors(),
            &mut hdr.motion_vectors_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // History: last frame's depth and denoised color, sampled.
        encoder.use_image_resource(
            views.depth_prev(),
            &mut hdr.depth_prev_state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.hdr_denoised_output(),
            &mut hdr.hdr_denoised_target_state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            hzb_image,
            &mut hzb.state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..hzb.num_mips,
            0..1,
            false,
        );
        encoder.use_image_resource(
            trace_hit,
            &mut res.trace_hit_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        encoder.use_image_resource(
            transmittance_view,
            &mut atmosphere_luts.transmittance.state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            sky_view,
            &mut atmosphere_luts.sky_view.state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        let mut p = PbrPipelineParams::new();
        p.uniforms(cam_buffer)
            .output_texture(views.hdr_output().full_view())
            .gbuffer_albedo_srgb(views.albedo().deref_inner().srgb_view())
            .gbuffer_normal_texture(views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(views.depth().full_view())
            .gbuffer_motion_vector_texture(views.motion_vectors().full_view())
            .hzb_texture(hzb_image.deref_inner().full_view())
            .prev_scene_color(views.hdr_denoised_output().full_view())
            .sky_atmosphere_params(atmo_buffer)
            .sky_transmittance_lut(transmittance_view.deref_inner().full_view())
            .sky_sky_view_lut(sky_view.deref_inner().full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .prev_depth_texture(views.depth_prev().full_view())
            .screen_trace(constants_buffer)
            .trace_hit(trace_hit.full_view());
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            p.as_slice(),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "screen trace", |encoder| {
            encoder.dispatch(UVec3::new(
                render_extent.x.div_ceil(8),
                render_extent.y.div_ceil(8),
                1,
            ));
        });
    });
}

/// Builds the list of pixels whose trace missed and its indirect arguments;
/// also reads back an earlier frame's count. The frame state is the host copy
/// of the count.
fn compact_traces_pass(
    mut ctx: SubmissionState<RingBufferSuballocation>,
    mut params: ScreenTraceSystemParams,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut readback_ring_buffer: ResMut<ReadbackRingBuffer>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    if !params.state.traced_this_frame {
        return;
    }
    // Nothing traces the list.
    if params.config.screen_only {
        params.state.hardware_rays = None;
        return;
    }
    let Some(hdr) = hdr_target else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&params.state.compact_pipeline) else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();
    let frame_index = params.state.frame_index;
    let constants = build_screen_trace_constants(
        &params.config,
        params.hzb.resources.as_ref().unwrap(),
        frame_index,
        hdr.history_valid,
        true,
    );
    let state = &mut *params.state;
    let res = state.resources.as_mut().unwrap();
    let render_extent = res.render_extent;
    let init_args = !std::mem::replace(&mut res.args_initialized, true);

    // Room for the count alone: a copy stops at the end of the smaller buffer.
    let copy = readback_ring_buffer.allocate_buffer(4, 16);
    if let Some(bytes) = ctx
        .set_frame_state(copy)
        .as_ref()
        .and_then(|copy| copy.as_slice())
    {
        state.hardware_rays = Some(u32::from_le_bytes(bytes[0..4].try_into().unwrap()));
    }

    ctx.record(|encoder, copy| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let texels = encoder.lock(
            &res.compacted_texels,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let args = encoder.lock(&res.compacted_args, vk::PipelineStageFlags2::COMPUTE_SHADER);

        // Zero the count (and write the constant height / depth once).
        encoder.use_buffer_resource(args, &mut res.compacted_args_state, Access::CLEAR);
        encoder.emit_barriers();
        let device = encoder.device().clone();
        let cmd = encoder.buffer().vk_handle();
        unsafe {
            device.cmd_fill_buffer(cmd, args.vk_handle(), 0, 4, 0);
            if init_args {
                device.cmd_fill_buffer(cmd, args.vk_handle(), 4, 8, 1);
            }
        }

        encoder.use_image_resource(
            trace_hit,
            &mut res.trace_hit_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_buffer_resource(
            texels,
            &mut res.compacted_texels_state,
            Access::COMPUTE_WRITE,
        );
        encoder.use_buffer_resource(args, &mut res.compacted_args_state, compute_read_write());
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        let mut p = PbrPipelineParams::new();
        p.screen_trace(constants_buffer)
            .trace_hit(trace_hit.full_view())
            .compacted_trace_texels(texels)
            .compacted_trace_args(args);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            p.as_slice(),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "trace compaction", |encoder| {
            encoder.dispatch(UVec3::new(
                render_extent.x.div_ceil(8),
                render_extent.y.div_ceil(8),
                1,
            ));
        });

        // Snapshot the count for the statistic.
        encoder.use_buffer_resource(args, &mut res.compacted_args_state, Access::COPY_READ);
        encoder.emit_barriers();
        ReadbackRingBuffer::read_back(encoder, args, copy);
    });
}

/// Paints the selected debug view into the overlay layer.
pub(crate) fn screen_trace_debug_pass(
    mut ctx: SubmissionState,
    mut params: ScreenTraceSystemParams,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    atmosphere_luts: Res<AtmosphereLUTs>,
) {
    if params.config.debug_mode == ScreenTraceDebugMode::Off {
        return;
    }
    if params.state.resources.is_none() || params.hzb.resources.is_none() {
        return;
    }
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&params.state.debug_pipeline) else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();
    let constants = build_screen_trace_constants(
        &params.config,
        params.hzb.resources.as_ref().unwrap(),
        params.state.frame_index,
        hdr.history_valid,
        params.config.enabled,
    );
    let display_extent = hdr.display_extent;
    let hzb = params.hzb.resources.as_mut().unwrap();
    let res = params.state.resources.as_mut().unwrap();

    ctx.record(|encoder, _| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let hzb_image = encoder.lock(&hzb.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);

        encoder.use_image_resource(
            views.debug_overlay(),
            &mut hdr.debug_overlay_state,
            compute_read_write(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            trace_hit,
            &mut res.trace_hit_state,
            compute_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            hzb_image,
            &mut hzb.state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..hzb.num_mips,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.hdr_denoised_output(),
            &mut hdr.hdr_denoised_target_state,
            compute_sampled_read(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        let mut p = PbrPipelineParams::new();
        p.gbuffer_depth_texture(views.depth().full_view())
            .gbuffer_debug_overlay_texture(views.debug_overlay().deref_inner().full_view())
            .hzb_texture(hzb_image.deref_inner().full_view())
            .prev_scene_color(views.hdr_denoised_output().full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .screen_trace(constants_buffer)
            .trace_hit(trace_hit.full_view());
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            p.as_slice(),
        );
        encoder.dispatch(UVec3::new(
            display_extent.x.div_ceil(8),
            display_extent.y.div_ceil(8),
            1,
        ));
    });
}
