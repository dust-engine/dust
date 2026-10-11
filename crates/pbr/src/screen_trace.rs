//! Screen-space gather trace (Lumen-style): march each pixel's gather ray
//! through the HZB, light on-screen hits from last frame's denoised color,
//! and hand only the misses to the hardware gather through a compacted
//! indirect dispatch. Shaders: screen_trace.slang, final_gather.slang.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_gfxdebug::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::buffer::{Buffer, BufferLike};
use pumicite::debug::DebugObject;
use pumicite::image::{FullImageView, Image, ImageExt};
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

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
    /// edge tests, orange = uncertain (thin feature), red = miss (hardware
    /// ray), blue = trace skipped.
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

/// Tunables of the screen trace; the egui window "Screen trace" edits them.
#[derive(Resource, Clone)]
pub struct ScreenTraceConfig {
    /// Off: every surface pixel traces a hardware ray from the surface, as
    /// before the screen trace existed (A/B reference).
    pub enabled: bool,
    pub max_iterations: u32,
    pub relative_depth_thickness: f32,
    pub history_depth_thickness: f32,
    /// World units. Voxel geometry is all hard edges; about half a voxel.
    pub min_world_thickness: f32,
    pub num_thickness_steps: u32,
    /// World units the hardware ray backs up from the last visible point.
    pub pullback_bias: f32,
    /// Lanes of a wave that must still be marching for the wave to go on;
    /// 0 lets every ray run to its own end.
    pub minimum_occupancy: u32,
    pub debug_mode: ScreenTraceDebugMode,
    pub debug_mip: u32,
    pub debug_scale: f32,
}

impl Default for ScreenTraceConfig {
    /// The defaults, with `DUST_SCREEN_TRACE=0|1`, `DUST_SCREEN_TRACE_ITER`,
    /// `DUST_SCREEN_TRACE_OCCUPANCY` and `DUST_SCREEN_TRACE_DEBUG=0..3` (the
    /// `ScreenTraceDebugMode` order) honored for headless captures.
    fn default() -> Self {
        let env_u32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        Self {
            enabled: env_u32("DUST_SCREEN_TRACE").is_none_or(|v| v != 0),
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
/// Number of host-visible copies of the compaction count kept in flight.
const READBACK_SLOTS: usize = 3;

pub struct ScreenTraceResources {
    /// R32_UINT, render resolution: the packed trace record per pixel.
    pub trace_hit: GPUMutex<FullImageView<Image>>,
    pub trace_hit_state: ResourceState,
    /// One `u32` per pixel: the compacted list of pixels needing a ray.
    pub compacted_texels: GPUMutex<Buffer>,
    pub compacted_texels_state: ResourceState,
    /// `[count, 1, 1]`, the gather's VkTraceRaysIndirectCommandKHR.
    pub compacted_args: GPUMutex<Buffer>,
    pub compacted_args_state: ResourceState,
    /// Host copies of `compacted_args`, for the hit-rate statistic.
    readback: [GPUMutex<Buffer>; READBACK_SLOTS],
    readback_state: [ResourceState; READBACK_SLOTS],
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
    /// `screen_trace_pass` and shared with `final_gather_pass`.
    pub frame_index: u32,
    /// Hardware rays traced in the most recent frame whose count has been
    /// read back (`None` until the first readback, or while disabled).
    pub hardware_rays: Option<u32>,
    /// `screen_trace_pass` recorded this frame, so the trace records are
    /// current and `compact_traces_pass` will run: the gather may dispatch
    /// over the compacted list. False while the pipelines are still loading.
    pub traced_this_frame: bool,
}

/// What `final_gather_pass` needs from this module, bundled to stay under
/// Bevy's system parameter limit.
#[derive(SystemParam)]
pub(crate) struct ScreenTraceSystemParams<'w> {
    pub state: ResMut<'w, ScreenTraceState>,
    pub config: Res<'w, ScreenTraceConfig>,
    pub hzb: ResMut<'w, HzbState>,
}

impl ScreenTraceSystemParams<'_> {
    /// Whether this frame's gather resumes from screen-trace records.
    pub fn active(&self) -> bool {
        self.config.enabled && self.state.resources.is_some() && self.hzb.resources.is_some()
    }
}

pub struct ScreenTracePlugin;

impl Plugin for ScreenTracePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenTraceConfig>();
        app.add_systems(EguiPrimaryContextPass, screen_trace_ui);
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
                    .after(crate::shadow_pass),
                compact_traces_pass
                    .in_set(DefaultRenderSet)
                    .after(screen_trace_pass)
                    .before(crate::final_gather_pass),
                screen_trace_debug_pass
                    .in_set(DefaultRenderSet)
                    .after(crate::final_gather_pass)
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
    let readback = [(); READBACK_SLOTS].map(|_| {
        GPUMutex::new(
            Buffer::new_dynamic(
                allocator.clone(),
                12,
                16,
                vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap(),
        )
    });
    ScreenTraceResources {
        trace_hit: GPUMutex::new(trace_hit),
        trace_hit_state: ResourceState::default(),
        compacted_texels: GPUMutex::new(compacted_texels),
        compacted_texels_state: ResourceState::default(),
        compacted_args: GPUMutex::new(compacted_args),
        compacted_args_state: ResourceState::default(),
        readback,
        readback_state: Default::default(),
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
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    jitter: Res<crate::JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
    atmosphere_luts: Res<AtmosphereLUTs>,
) {
    // A fresh seed every frame, whether or not the trace runs: the hardware
    // gather reads it too.
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
    // Both passes or neither: the gather must not consume a list the
    // compaction never wrote.
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

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let cam_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&cam_uniform));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let hzb_image = encoder.lock(&hzb.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);

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

/// Builds the list of pixels whose trace missed and the indirect arguments
/// of the gather dispatch; also reads back an earlier frame's count.
fn compact_traces_pass(
    mut ctx: SubmissionState,
    mut params: ScreenTraceSystemParams,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    if !params.state.traced_this_frame {
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

    // The slot recorded READBACK_SLOTS frames ago is normally complete; if
    // the GPU still holds it, skip this frame's statistic rather than stall.
    let slot = frame_index as usize % READBACK_SLOTS;
    if let Some(buffer) = res.readback[slot].try_deref_mut() {
        if let Some(bytes) = buffer.as_slice() {
            let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            if count <= render_extent.x * render_extent.y {
                state.hardware_rays = Some(count);
            }
        }
    }

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let texels = encoder.lock(
            &res.compacted_texels,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let args = encoder.lock(&res.compacted_args, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let readback = encoder.lock(&res.readback[slot], vk::PipelineStageFlags2::COPY);

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
        encoder.use_buffer_resource(readback, &mut res.readback_state[slot], Access::COPY_WRITE);
        encoder.emit_barriers();
        encoder.copy_buffer(args, readback);
    });
}

/// Paints the selected debug view into the overlay layer.
fn screen_trace_debug_pass(
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

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let hzb_image = encoder.lock(&hzb.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let trace_hit = encoder.lock(&res.trace_hit, vk::PipelineStageFlags2::COMPUTE_SHADER);

        encoder.use_image_resource(
            views.sdr_target(),
            &mut hdr.sdr_target_state,
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
            .gbuffer_occlusion_texture(views.sdr_target().deref_inner().full_view())
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

fn screen_trace_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<ScreenTraceConfig>,
    state: Option<Res<ScreenTraceState>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Window::new("Screen trace")
        .default_width(300.0)
        // Right of the SHARC / atmosphere stack so it is visible by default.
        .default_pos([330.0, 20.0])
        .show(ctx, |ui| {
            ui.checkbox(&mut config.enabled, "Screen-space gather trace");
            if let Some(state) = state.as_deref() {
                match (state.hardware_rays, hdr_target.as_deref()) {
                    (Some(rays), Some(hdr)) if config.enabled => {
                        let pixels = hdr.render_extent.x * hdr.render_extent.y;
                        ui.label(format!(
                            "Hardware rays: {rays} ({:.1}% of pixels)",
                            100.0 * rays as f32 / pixels.max(1) as f32
                        ));
                    }
                    _ => {
                        ui.label("Hardware rays: every surface pixel");
                    }
                }
            }
            ui.separator();
            ui.add(egui::Slider::new(&mut config.max_iterations, 1..=128).text("Max iterations"));
            ui.add(
                egui::Slider::new(&mut config.relative_depth_thickness, 0.0..=0.2)
                    .text("Relative thickness"),
            );
            ui.add(
                egui::Slider::new(&mut config.min_world_thickness, 0.0..=1.0)
                    .text("Min thickness (world)"),
            );
            ui.add(
                egui::Slider::new(&mut config.history_depth_thickness, 0.0..=0.05)
                    .text("History depth thickness"),
            );
            ui.add(
                egui::Slider::new(&mut config.num_thickness_steps, 0..=16).text("Certainty steps"),
            );
            ui.add(egui::Slider::new(&mut config.pullback_bias, 0.0..=2.0).text("Pullback bias"));
            ui.add(
                egui::Slider::new(&mut config.minimum_occupancy, 0..=32)
                    .text("Min wave occupancy (0 = off)"),
            );
            ui.collapsing("Debug view", |ui| {
                ui.radio_value(&mut config.debug_mode, ScreenTraceDebugMode::Off, "Off");
                ui.radio_value(
                    &mut config.debug_mode,
                    ScreenTraceDebugMode::TraceTypes,
                    "Trace types (green hit, yellow rejected, orange uncertain, red miss, blue skipped)",
                );
                ui.radio_value(&mut config.debug_mode, ScreenTraceDebugMode::HzbMip, "HZB mip");
                ui.radio_value(
                    &mut config.debug_mode,
                    ScreenTraceDebugMode::PrevColor,
                    "Previous frame color",
                );
                ui.add(egui::Slider::new(&mut config.debug_mip, 0..=11).text("Mip"));
                ui.add(
                    egui::Slider::new(&mut config.debug_scale, 0.0..=10.0).text("Color scale"),
                );
            });
        });
}
