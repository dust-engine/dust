//! Radiance cascades: diffuse global illumination from radiance probes placed in
//! world space. `rc_common.slang` explains how they work; this module records
//! their passes and owns their buffers, which the shaders reach by device
//! address.
//!
//! So far it builds the probes and shows them in the debug views; nothing lights
//! the frame from them yet.

use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::buffer::{Buffer, BufferLike};
use pumicite::command::{CommandEncoder, GPURef};
use pumicite::pipeline::Pipeline;
use pumicite::sync::GPUMutex;
use pumicite::tracking::Access;
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice};

use crate::camera::Camera;
use crate::{
    HdrRenderTarget, HdrRenderTargetViewsExt, JitterState, PbrPipelineParams, build_camera_uniform,
};

/// `RC_CASCADES` in pbr_common.slang.
pub const CASCADES: usize = 4;

/// The most [`RadianceCascadesConfig::lod_distance`] may be, in probe spacings.
/// LOD `n` ends `lod_distance * 2^n` from the camera, and its cascade 0 cells
/// are `probe_spacing * 2^n` wide, so every LOD reaches `lod_distance /
/// probe_spacing` of its own cells either side of the camera's cell. A key tells
/// 512 cells per axis apart (`rcKey`), 256 either side; the margin under 256
/// covers rounding and the grid point's offset off the surface.
const MAX_LOD_CELLS: f32 = 240.0;

// Offsets into the counters buffer, in `u32`s: `RC_COUNTER_*` in
// pbr_common.slang. Per cascade: how many probes it has, then the arguments of
// an indirect dispatch over them.
const COUNTER_PROBES: usize = 0;
const COUNTER_PROBE_ARGS: usize = COUNTER_PROBES + CASCADES;
/// The counters buffer's length, in `u32`s.
const COUNTERS: usize = COUNTER_PROBE_ARGS + 3 * CASCADES;

/// Where the arguments of the indirect dispatch over a cascade's probes are in
/// the counters buffer, in bytes.
fn probe_args_offset(cascade: usize) -> u64 {
    ((COUNTER_PROBE_ARGS + 3 * cascade) * 4) as u64
}

/// Where the compute passes' push constants (`RcPush` in rc_common.slang) start:
/// after the ray generation stage's (pbr.playout.ron).
const PUSH_CONSTANTS_OFFSET: u32 = 4;

/// What the debug view paints over the frame.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RadianceCascadesDebugMode {
    /// Nothing.
    #[default]
    Off,
    /// A color per cascade 0 cell, or white where cascade 0 had no room for the
    /// cell's probe.
    Probes,
    /// A color per LOD.
    Lod,
}

impl RadianceCascadesDebugMode {
    /// The value of `RcConstants::debugMode`: `RC_DEBUG_*` in rc_debug.slang.
    fn as_u32(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Probes => 1,
            Self::Lod => 2,
        }
    }
}

/// Settings of the radiance cascades.
#[derive(Resource, Clone)]
pub struct RadianceCascadesConfig {
    /// Builds the probes every frame. Off by default, since nothing lights the
    /// frame from them yet.
    pub enabled: bool,
    /// World units between the probes of cascade 0 at LOD 0. Lighting detail
    /// finer than this needs another technique.
    pub probe_spacing: f32,
    /// The Chebyshev distance from the camera at which LOD 0 ends, in world
    /// units. Each next LOD ends twice as far as the last, with probes twice as
    /// far apart. Divided by `probe_spacing`, it is how many cells every LOD
    /// spans either side of the camera, which sets how large they look on
    /// screen. Used up to 240 probe spacings.
    pub lod_distance: f32,
    /// How many probes cascade 0 has room for. Each cascade up has room for a
    /// third as many: its cells are twice as wide, so a surface crosses about a
    /// quarter as many. Sets the memory use.
    pub max_probes: u32,
    pub debug_mode: RadianceCascadesDebugMode,
}

impl Default for RadianceCascadesConfig {
    /// The defaults, which environment variables override for runs without the
    /// DevTools: `DUST_RC=1` enables the cascades, `DUST_RC_DEBUG` picks a debug
    /// view by its `RcConstants::debugMode` value, and `DUST_RC_SPACING` and
    /// `DUST_RC_LOD_DISTANCE` set those settings.
    fn default() -> Self {
        let env_u32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        let env_f32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
        Self {
            enabled: env_u32("DUST_RC").is_some_and(|v| v != 0),
            probe_spacing: env_f32("DUST_RC_SPACING").unwrap_or(0.125),
            lod_distance: env_f32("DUST_RC_LOD_DISTANCE").unwrap_or(16.0),
            max_probes: 1 << 16,
            debug_mode: env_u32("DUST_RC_DEBUG")
                .and_then(|value| {
                    [
                        RadianceCascadesDebugMode::Probes,
                        RadianceCascadesDebugMode::Lod,
                    ]
                    .into_iter()
                    .find(|mode| mode.as_u32() == value)
                })
                .unwrap_or_default(),
        }
    }
}

/// Where one cascade's hash table and probes are in a [`Cache`]'s buffers, which
/// hold the cascades one after another: `RcCascade` in pbr_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RcCascadeRaw {
    /// The table's first slot in `keys` and `slots`.
    table_offset: u32,
    /// The table's slot count, a power of two, minus one.
    table_mask: u32,
    /// The cascade's first probe in `probes`.
    probe_offset: u32,
    /// How many probes the cascade has room for.
    probe_capacity: u32,
}

impl RcCascadeRaw {
    fn table_size(&self) -> u32 {
        self.table_mask + 1
    }
}

/// The parameters of the radiance cascades passes, uploaded every frame:
/// `RcConstants` in pbr_common.slang, which Slang lays out by std140 rules
/// (`render_extent` starts on 8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RcConstantsRaw {
    /// [`Cache::cascades`].
    cascades: [RcCascadeRaw; CASCADES],
    /// The LODs are centered on it.
    camera_position: [f32; 3],
    /// [`RadianceCascadesConfig::probe_spacing`].
    spacing: f32,
    /// [`RadianceCascadesConfig::lod_distance`], up to [`MAX_LOD_CELLS`] probe
    /// spacings.
    lod_distance: f32,
    /// [`RadianceCascadesDebugMode::as_u32`].
    debug_mode: u32,
    /// Of the G-buffer.
    render_extent: [u32; 2],
    /// The device addresses of the [`CacheBuffers`].
    keys: u64,
    slots: u64,
    probes: u64,
    counters: u64,
}
const _: () = assert!(size_of::<RcConstantsRaw>() == 128);

/// A [`Cache`]'s buffers, rebuilt every frame. Each holds the cascades one after
/// another, at the offsets in their [`RcCascadeRaw`].
struct CacheBuffers {
    /// Per table slot: the key of the cell there (`rcKey`), or 0 if the slot is
    /// empty.
    keys: Buffer,
    /// Per table slot with a key: the index of the cell's probe, counted from
    /// the cascade's first, or `RC_INVALID` if the cascade had no room left.
    slots: Buffer,
    /// Per probe: an `RcProbe`.
    probes: Buffer,
    /// Per cascade: its probe count and the arguments of an indirect dispatch
    /// over its probes, at the `COUNTER_*` offsets.
    counters: Buffer,
}

pumicite::project_fields! {
    trait CacheBuffersExt for impl[] CacheBuffers {
        keys: Buffer,
        counters: Buffer,
    }
}

/// The probes of every cascade, for the surfaces the camera sees.
struct Cache {
    /// Where each cascade is in `buffers`. Fixed when they're created; every
    /// frame copies it into [`RcConstantsRaw::cascades`] for the shaders.
    cascades: [RcCascadeRaw; CASCADES],
    buffers: GPUMutex<CacheBuffers>,
}

/// `new_device` for what the shaders access by device address.
fn create_buffer(allocator: &Allocator, size: u64, usage: vk::BufferUsageFlags) -> Buffer {
    Buffer::new_device(
        allocator.clone(),
        size,
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST
            | usage,
    )
    .unwrap()
}

impl Cache {
    fn new(allocator: &Allocator, max_probes: u32) -> Self {
        let mut cascades = [RcCascadeRaw::zeroed(); CASCADES];
        let mut slots = 0;
        let mut probes = 0;
        for (cascade, raw) in cascades.iter_mut().enumerate() {
            // A third as many each cascade up (`RadianceCascadesConfig::max_probes`).
            let probe_capacity = max_probes / 3u32.pow(cascade as u32);
            // Four slots per probe: few keys per slot keep linear probing's walks
            // short.
            let table_size = (probe_capacity * 4).next_power_of_two();
            *raw = RcCascadeRaw {
                table_offset: slots,
                table_mask: table_size - 1,
                probe_offset: probes,
                probe_capacity,
            };
            slots += table_size;
            probes += probe_capacity;
        }
        let none = vk::BufferUsageFlags::empty();
        Self {
            cascades,
            buffers: GPUMutex::new(CacheBuffers {
                keys: create_buffer(allocator, slots as u64 * 4, none),
                slots: create_buffer(allocator, slots as u64 * 4, none),
                probes: create_buffer(allocator, probes as u64 * 4, none),
                counters: create_buffer(
                    allocator,
                    COUNTERS as u64 * 4,
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                ),
            }),
        }
    }
}

/// What the passes keep while the cascades are enabled.
struct Resources {
    primary: Cache,
    /// The [`RadianceCascadesConfig::max_probes`] `primary` was created with.
    max_probes: u32,
}

/// The compute passes; each one's pipeline is at its index in
/// [`COMPUTE_PASSES`].
#[derive(Clone, Copy)]
enum Pass {
    InsertPixels,
    Allocate,
    WriteArgs,
    InsertParents,
    Debug,
}

/// The pipeline of each [`Pass`], by file name.
const COMPUTE_PASSES: [&str; 5] = [
    "rc_insert_pixels",
    "rc_allocate",
    "rc_write_args",
    "rc_insert_parents",
    "rc_debug",
];

/// The pipelines of the radiance cascades, and what they keep while enabled.
#[derive(Resource)]
pub struct RadianceCascadesState {
    /// By [`Pass`].
    compute_pipelines: [Handle<ComputePipeline>; COMPUTE_PASSES.len()],
    /// `None` while the cascades are disabled.
    resources: Option<Resources>,
}

/// Builds the probes every frame while [`RadianceCascadesConfig::enabled`].
pub struct RadianceCascadesPlugin;

impl Plugin for RadianceCascadesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RadianceCascadesConfig>();
        app.add_systems(
            Startup,
            setup_radiance_cascades
                .after(CreateDevice)
                .after(crate::setup),
        );
        app.add_systems(
            PostUpdate,
            radiance_cascades_pass
                .in_set(DefaultRenderSet)
                // Reads the G-buffer it writes.
                .after(crate::render)
                // Paints the debug overlay once it's cleared, before tonemap
                // composites it.
                .after(crate::clear_debug_overlay)
                .before(crate::tonemap::tonemap_pass),
        );
    }
}

fn setup_radiance_cascades(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(RadianceCascadesState {
        compute_pipelines: COMPUTE_PASSES.map(|name| {
            asset_server.load(format!(
                "bazel://dust/crates/pbr/shaders/{name}.comp.pipeline.bin"
            ))
        }),
        resources: None,
    });
}

fn create_resources(allocator: &Allocator, max_probes: u32) -> Resources {
    Resources {
        primary: Cache::new(allocator, max_probes),
        max_probes,
    }
}

/// Any access by the compute passes to the cache's buffers or the debug overlay.
fn storage_read_write() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    }
}

/// Reads of the G-buffer by the compute passes.
fn storage_read() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ,
    }
}

/// Finishes a pass's writes, for any later pass to read, or use as indirect
/// arguments. A pass's atomics need no ordering among themselves.
fn pass_barrier(encoder: &mut CommandEncoder) {
    encoder.memory_barrier(
        storage_read_write(),
        storage_read_write()
            | Access {
                stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            },
    );
    encoder.emit_barriers();
}

fn fill_with_zeros(encoder: &mut CommandEncoder, buffer: vk::Buffer) {
    let device = encoder.device().clone();
    let cmd = encoder.buffer().vk_handle();
    unsafe {
        device.cmd_fill_buffer(cmd, buffer, 0, vk::WHOLE_SIZE, 0);
    }
}

/// The compute pipelines, retained by the command buffer.
struct Passes<'a> {
    pipelines: [GPURef<'a, Pipeline>; COMPUTE_PASSES.len()],
}

impl<'a> Passes<'a> {
    fn get(&self, pass: Pass) -> GPURef<'a, Pipeline> {
        self.pipelines[pass as usize]
    }

    /// Binds `descriptors` for the passes, which share one layout
    /// (pbr.playout.ron).
    fn push_descriptors(
        &self,
        encoder: &mut CommandEncoder<'a>,
        descriptors: &mut PbrPipelineParams,
    ) {
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            self.pipelines[0].layout(),
            0,
            descriptors.as_slice(),
        );
    }

    /// Runs a pass on a cascade, then makes its writes visible to the next.
    fn run(
        &self,
        encoder: &mut CommandEncoder<'a>,
        pass: Pass,
        cascade: usize,
        dispatch: impl FnOnce(&mut CommandEncoder<'a>),
    ) {
        let pipeline = self.get(pass);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::COMPUTE,
            PUSH_CONSTANTS_OFFSET,
            bytemuck::bytes_of(&(cascade as u32)),
        );
        dispatch(encoder);
        pass_barrier(encoder);
    }
}

/// Records steps 1 and 2 of `rc_common.slang`: the probes of every cascade for
/// the surfaces in `extent`, once their descriptors are bound.
fn record_build<'a>(
    encoder: &mut CommandEncoder<'a>,
    passes: &Passes<'a>,
    cache: &Cache,
    extent: UVec2,
    profiler: &mut Option<ResMut<GpuProfiler>>,
) {
    let stages = storage_read_write().stage | vk::PipelineStageFlags2::CLEAR;
    let buffers = encoder.lock(&cache.buffers, stages);

    // Once last frame's passes are done with them, the tables and counts start
    // from zero. `slots` needs no clear: it's read only where `keys` has a key,
    // and `rcAllocate` writes it for every key.
    encoder.memory_barrier(storage_read_write(), Access::CLEAR);
    encoder.emit_barriers();
    fill_with_zeros(encoder, buffers.keys().vk_handle());
    fill_with_zeros(encoder, buffers.counters().vk_handle());
    encoder.memory_barrier(Access::CLEAR, storage_read_write());
    encoder.emit_barriers();

    encoder.timing_scope(profiler.as_deref_mut(), "RC build", |encoder| {
        passes.run(encoder, Pass::InsertPixels, 0, |encoder| {
            encoder.dispatch(UVec3::new(extent.x.div_ceil(8), extent.y.div_ceil(8), 1));
        });
        for (cascade, raw) in cache.cascades.iter().enumerate() {
            if cascade > 0 {
                passes.run(encoder, Pass::InsertParents, cascade, |encoder| {
                    encoder.dispatch_indirect(buffers.counters(), probe_args_offset(cascade - 1));
                });
            }
            passes.run(encoder, Pass::Allocate, cascade, |encoder| {
                encoder.dispatch(UVec3::new(raw.table_size().div_ceil(64), 1, 1));
            });
            passes.run(encoder, Pass::WriteArgs, cascade, |encoder| {
                encoder.dispatch(UVec3::ONE);
            });
        }
    });
}

/// Builds the probes and paints the debug view.
fn radiance_cascades_pass(
    mut ctx: SubmissionState,
    config: Res<RadianceCascadesConfig>,
    mut state: ResMut<RadianceCascadesState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    allocator: Res<Allocator>,
    jitter: Res<JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let state = &mut *state;
    // Each buffer's `GPUMutex` defers its free until the GPU is done with it.
    if !config.enabled {
        state.resources = None;
        return;
    }
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let mut pipelines = Vec::with_capacity(COMPUTE_PASSES.len());
    for handle in &state.compute_pipelines {
        let Some(pipeline) = compute_pipelines.get(handle) else {
            return;
        };
        pipelines.push(pipeline.clone().into_inner());
    }

    let render_extent = hdr.render_extent;
    let resources = match &mut state.resources {
        Some(resources) if resources.max_probes == config.max_probes => resources,
        slot => slot.insert(create_resources(&allocator, config.max_probes)),
    };
    let primary = &mut resources.primary;

    // Same jitter as the primary pass: positions are rebuilt from its depth.
    let camera_uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let constants = {
        // SAFETY: only device addresses are read.
        let buffers = unsafe { primary.buffers.get_unchecked() };
        RcConstantsRaw {
            cascades: primary.cascades,
            camera_position: transform.translation().to_array(),
            spacing: config.probe_spacing,
            lod_distance: config
                .lod_distance
                .min(MAX_LOD_CELLS * config.probe_spacing),
            debug_mode: config.debug_mode.as_u32(),
            render_extent: render_extent.to_array(),
            keys: buffers.keys.device_address(),
            slots: buffers.slots.device_address(),
            probes: buffers.probes.device_address(),
            counters: buffers.counters.device_address(),
        }
    };

    ctx.record(|encoder, _| {
        let hdr = &mut **hdr;
        let passes = Passes {
            pipelines: std::array::from_fn(|i| encoder.retain(pipelines[i].clone()).deref_inner()),
        };
        let camera_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&camera_uniform));
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));

        let views = encoder.lock(&hdr.view, storage_read().stage);
        encoder.use_image_resource(
            views.normal(),
            &mut hdr.normal_state,
            storage_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            storage_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let mut descriptors = PbrPipelineParams::new();
        descriptors
            .uniforms(camera_buffer)
            .gbuffer_normal_texture(views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(views.depth().full_view())
            .rc(constants_buffer);
        passes.push_descriptors(encoder, &mut descriptors);
        record_build(encoder, &passes, primary, render_extent, &mut profiler);

        if config.debug_mode != RadianceCascadesDebugMode::Off {
            encoder.use_image_resource(
                views.debug_overlay(),
                &mut hdr.debug_overlay_state,
                storage_read_write(),
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                false,
            );
            encoder.emit_barriers();
            descriptors
                .uniforms(camera_buffer)
                .gbuffer_normal_texture(views.normal().deref_inner().full_view())
                .gbuffer_depth_texture(views.depth().full_view())
                .gbuffer_debug_overlay_texture(views.debug_overlay().deref_inner().full_view())
                .rc(constants_buffer);
            passes.push_descriptors(encoder, &mut descriptors);
            let display_extent = hdr.display_extent;
            passes.run(encoder, Pass::Debug, 0, |encoder| {
                encoder.dispatch(UVec3::new(
                    display_extent.x.div_ceil(8),
                    display_extent.y.div_ceil(8),
                    1,
                ));
            });
        }
    });
}
