//! Radiance cascades: diffuse global illumination from radiance probes placed in
//! world space. `radiance_cascades.slang` explains how they work; this module
//! records their passes and owns their buffers, which the shaders reach by
//! device address.
//!
//! So far it builds the probes and shows them in the debug views; nothing lights
//! the frame from them yet.

use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::{ReadbackRingBuffer, UniformRingBuffer};
use bevy_pumicite::{DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::buffer::{Buffer, BufferLike, RingBufferSuballocation};
use pumicite::command::{CommandEncoder, GPURef, GPURefMut};
use pumicite::pipeline::Pipeline;
use pumicite::sync::GPUMutex;
use pumicite::tracking::Access;
use pumicite::Allocator;

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

/// How many probes cascade 0 has room for. Each cascade up has room for a third
/// as many: its cells are twice as wide, so a surface crosses about a quarter
/// as many. Sets the memory use.
const MAX_PROBES: u32 = 1 << 16;

/// Where the compute passes' push constants (`RcPush` in
/// radiance_cascades.slang) start: after the ray generation stage's
/// (pbr.playout.ron).
const PUSH_CONSTANTS_OFFSET: u32 = 4;

/// What the debug view paints over the frame. Each is the value of
/// `RcConstants::debugMode` that picks it: `RC_DEBUG_*` in
/// radiance_cascades.slang.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RadianceCascadesDebugMode {
    /// Nothing.
    #[default]
    Off = 0,
    /// A color per cascade 0 cell, or white where cascade 0 had no room for the
    /// cell's probe.
    Probes = 1,
    /// A color per LOD.
    Lod = 2,
}

/// Settings of the radiance cascades, which the DevTools "Radiance cascades"
/// page edits.
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
            debug_mode: match env_u32("DUST_RC_DEBUG") {
                Some(1) => RadianceCascadesDebugMode::Probes,
                Some(2) => RadianceCascadesDebugMode::Lod,
                _ => RadianceCascadesDebugMode::Off,
            },
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
/// `RcConstants` in pbr_common.slang, which Slang lays out by std140 rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RcConstantsRaw {
    /// [`Cache::cascades`].
    cascades: [RcCascadeRaw; CASCADES],
    /// [`RadianceCascadesConfig::probe_spacing`].
    spacing: f32,
    /// [`RadianceCascadesConfig::lod_distance`], up to [`MAX_LOD_CELLS`] probe
    /// spacings.
    lod_distance: f32,
    /// [`RadianceCascadesConfig::debug_mode`].
    debug_mode: u32,
    /// std140 starts `render_extent` on 8 bytes.
    _padding: u32,
    /// Of the G-buffer.
    render_extent: [u32; 2],
    /// The device addresses of the [`CacheBuffers`].
    keys: u64,
    slots: u64,
    probes: u64,
    counters: u64,
}
const _: () = assert!(size_of::<RcConstantsRaw>() == 120);

/// A [`Cache`]'s buffers, rebuilt every frame. Each holds the cascades one after
/// another, at the offsets in their [`RcCascadeRaw`].
struct CacheBuffers {
    /// Per table slot: the key of the cell there (`rcKey`), or 0 if the slot is
    /// empty.
    keys: Buffer,
    /// Per table slot with a key: the index of the cell's probe, counted from
    /// the cascade's first, or `RC_INVALID` if the cascade had no room left.
    slots: Buffer,
    /// Per probe: the key of the cell it's for.
    probes: Buffer,
    /// Per cascade: its probe count and the arguments of an indirect dispatch
    /// over its probes, at the `COUNTER_*` offsets.
    counters: Buffer,
}

pumicite::project_fields! {
    trait CacheBuffersExt for impl[] CacheBuffers {
        keys: Buffer,
        slots: Buffer,
        probes: Buffer,
        counters: Buffer,
    }
}

/// What the DevTools show of the cache, from a recent frame's counters.
#[derive(Clone, Copy, PartialEq)]
pub struct CacheStats {
    /// How many probes each cascade has. A cascade that ran out of room shows
    /// its capacity.
    pub probes: [u32; CASCADES],
    /// How many probes each cascade has room for.
    pub capacity: [u32; CASCADES],
}

/// The probes of every cascade, for the surfaces the camera sees.
struct Cache {
    /// Where each cascade is in `buffers`. Fixed when they're created; every
    /// frame copies it into [`RcConstantsRaw::cascades`] for the shaders.
    cascades: [RcCascadeRaw; CASCADES],
    buffers: GPUMutex<CacheBuffers>,
    stats: Option<CacheStats>,
}

/// `new_device` for what the shaders access by device address, and `usage`.
fn create_buffer(allocator: &Allocator, size: u64, usage: vk::BufferUsageFlags) -> Buffer {
    Buffer::new_device(
        allocator.clone(),
        size,
        16,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | usage,
    )
    .unwrap()
}

impl Cache {
    fn new(allocator: &Allocator) -> Self {
        let mut cascades = [RcCascadeRaw::zeroed(); CASCADES];
        let mut slots = 0;
        let mut probes = 0;
        for (cascade, raw) in cascades.iter_mut().enumerate() {
            // A third as many each cascade up (`MAX_PROBES`).
            let probe_capacity = MAX_PROBES / 3u32.pow(cascade as u32);
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
        // `keys` and `counters` are cleared every frame.
        let cleared = vk::BufferUsageFlags::TRANSFER_DST;
        let none = vk::BufferUsageFlags::empty();
        Self {
            cascades,
            buffers: GPUMutex::new(CacheBuffers {
                keys: create_buffer(allocator, slots as u64 * 4, cleared),
                slots: create_buffer(allocator, slots as u64 * 4, none),
                probes: create_buffer(allocator, probes as u64 * 4, none),
                counters: create_buffer(
                    allocator,
                    COUNTERS as u64 * 4,
                    cleared
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                ),
            }),
            stats: None,
        }
    }

    /// Updates the statistics from a host copy of the probe counts that the GPU
    /// has written.
    fn read_stats(&mut self, copy: &RingBufferSuballocation) {
        let Some(bytes) = copy.as_slice() else {
            return;
        };
        self.stats = Some(CacheStats {
            probes: bytemuck::pod_read_unaligned(&bytes[..CASCADES * 4]),
            capacity: self.cascades.map(|cascade| cascade.probe_capacity),
        });
    }
}

/// The compute pipelines, one per pass, all with pbr.playout.ron's layout. Each
/// is a `P`: a [`Handle`] while it loads, then the loaded [`ComputePipeline`],
/// then the [`Pipeline`] retained by a command buffer.
struct ComputePipelines<P> {
    /// `rcInsertPixels`: inserts the cascade 0 cell of every visible surface.
    insert_pixels: P,
    /// `rcInsertParents`: inserts the parent cells of a cascade's probes into
    /// the next cascade.
    insert_parents: P,
    /// `rcAllocate`: gives the cells in a cascade's table probes.
    allocate: P,
    /// `rcWriteArgs`: clamps a cascade's probe count to its capacity and writes
    /// the arguments of the dispatches over its probes.
    write_args: P,
    /// `rcDebugOverlay`: paints the debug view.
    debug_overlay: P,
}

impl ComputePipelines<Handle<ComputePipeline>> {
    fn load(asset_server: &AssetServer) -> Self {
        let load = |name: &str| -> Handle<ComputePipeline> {
            asset_server.load(format!(
                "bazel://dust/crates/pbr/shaders/{name}.comp.pipeline.bin"
            ))
        };
        Self {
            insert_pixels: load("rc_insert_pixels"),
            insert_parents: load("rc_insert_parents"),
            allocate: load("rc_allocate"),
            write_args: load("rc_write_args"),
            debug_overlay: load("rc_debug"),
        }
    }

    /// The pipelines, once all of them have loaded.
    fn get<'a>(
        &self,
        assets: &'a Assets<ComputePipeline>,
    ) -> Option<ComputePipelines<&'a ComputePipeline>> {
        Some(ComputePipelines {
            insert_pixels: assets.get(&self.insert_pixels)?,
            insert_parents: assets.get(&self.insert_parents)?,
            allocate: assets.get(&self.allocate)?,
            write_args: assets.get(&self.write_args)?,
            debug_overlay: assets.get(&self.debug_overlay)?,
        })
    }
}

impl ComputePipelines<&ComputePipeline> {
    /// Keeps the pipelines alive until `encoder`'s command buffer has run.
    fn retain<'a>(
        &self,
        encoder: &mut CommandEncoder<'a>,
    ) -> ComputePipelines<GPURef<'a, Pipeline>> {
        let mut retain = |pipeline: &ComputePipeline| {
            encoder.retain(pipeline.clone().into_inner()).deref_inner()
        };
        ComputePipelines {
            insert_pixels: retain(self.insert_pixels),
            insert_parents: retain(self.insert_parents),
            allocate: retain(self.allocate),
            write_args: retain(self.write_args),
            debug_overlay: retain(self.debug_overlay),
        }
    }
}

/// The pipelines of the radiance cascades, and what they keep while enabled.
#[derive(Resource)]
pub struct RadianceCascadesState {
    pipelines: ComputePipelines<Handle<ComputePipeline>>,
    /// `None` while the cascades are disabled.
    cache: Option<Cache>,
}

impl RadianceCascadesState {
    /// The cache's latest statistics.
    pub fn stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().and_then(|cache| cache.stats)
    }
}

/// Builds the probes every frame while [`RadianceCascadesConfig::enabled`].
pub struct RadianceCascadesPlugin;

impl Plugin for RadianceCascadesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RadianceCascadesConfig>();
        app.add_systems(Startup, setup_radiance_cascades);
        app.add_systems(
            PostUpdate,
            radiance_cascades_pass
                .in_set(DefaultRenderSet)
                // Reads the G-buffer it writes.
                .after(crate::render)
                // Paints the debug overlay once it's cleared, before tonemap
                // composites it.
                .after(crate::clear_debug_overlay)
                // And over the screen trace's debug view: with both on, this
                // one shows, whatever order the systems would run in.
                .after(crate::screen_trace::screen_trace_debug_pass)
                .before(crate::tonemap::tonemap_pass),
        );
    }
}

fn setup_radiance_cascades(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(RadianceCascadesState {
        pipelines: ComputePipelines::load(&asset_server),
        cache: None,
    });
}

/// Runs a pass's `pipeline` on `cascade`, then makes its writes visible to the
/// passes after it.
fn run_pass<'a>(
    encoder: &mut CommandEncoder<'a>,
    pipeline: GPURef<'a, Pipeline>,
    cascade: usize,
    dispatch: impl FnOnce(&mut CommandEncoder<'a>),
) {
    encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
    encoder.push_constants(
        pipeline.layout(),
        vk::ShaderStageFlags::COMPUTE,
        PUSH_CONSTANTS_OFFSET,
        bytemuck::bytes_of(&(cascade as u32)),
    );
    dispatch(encoder);
    encoder.memory_barrier(
        Access::COMPUTE_READ | Access::COMPUTE_WRITE,
        Access::COMPUTE_READ
            | Access::COMPUTE_WRITE
            | Access {
                stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            },
    );
    encoder.emit_barriers();
}

/// Records steps 1 and 2 of `radiance_cascades.slang` into `buffers`, laid out
/// as `cascades`: the probes of every cascade for the surfaces in `extent`,
/// once their descriptors are bound.
fn record_build<'a>(
    encoder: &mut CommandEncoder<'a>,
    pipelines: &ComputePipelines<GPURef<'a, Pipeline>>,
    cascades: &[RcCascadeRaw; CASCADES],
    buffers: GPURefMut<'a, CacheBuffers>,
    extent: UVec2,
    profiler: &mut Option<ResMut<GpuProfiler>>,
) {
    // Once last frame's passes and copy are done with them, the tables and
    // counts start from zero. `slots` needs no clear: it's read only where
    // `keys` has a key, and `rcAllocate` writes it for every key.
    encoder.memory_barrier(
        Access::COMPUTE_READ | Access::COMPUTE_WRITE | Access::COPY_READ,
        Access::CLEAR,
    );
    encoder.emit_barriers();
    encoder.fill_buffer(buffers.keys(), 0);
    encoder.fill_buffer(buffers.counters(), 0);
    encoder.memory_barrier(Access::CLEAR, Access::COMPUTE_READ | Access::COMPUTE_WRITE);
    encoder.emit_barriers();

    encoder.timing_scope(profiler.as_deref_mut(), "RC build", |encoder| {
        run_pass(encoder, pipelines.insert_pixels, 0, |encoder| {
            encoder.dispatch(UVec3::new(extent.x.div_ceil(8), extent.y.div_ceil(8), 1));
        });
        for (cascade, raw) in cascades.iter().enumerate() {
            if cascade > 0 {
                run_pass(encoder, pipelines.insert_parents, cascade, |encoder| {
                    encoder.dispatch_indirect(buffers.counters(), probe_args_offset(cascade - 1));
                });
            }
            run_pass(encoder, pipelines.allocate, cascade, |encoder| {
                encoder.dispatch(UVec3::new(raw.table_size().div_ceil(64), 1, 1));
            });
            run_pass(encoder, pipelines.write_args, cascade, |encoder| {
                encoder.dispatch(UVec3::ONE);
            });
        }
    });
}

/// Copies the probe counts into `copy`, for [`Cache::read_stats`]: the copy
/// stops at the end of `copy`, and the counts come first (`COUNTER_PROBES`).
fn record_stats<'a>(
    encoder: &mut CommandEncoder<'a>,
    buffers: GPURefMut<'a, CacheBuffers>,
    copy: GPURefMut<'a, RingBufferSuballocation>,
) {
    encoder.memory_barrier(
        Access::COMPUTE_READ | Access::COMPUTE_WRITE,
        Access::COPY_READ,
    );
    encoder.emit_barriers();
    ReadbackRingBuffer::read_back(encoder, buffers.counters(), copy);
}

/// Builds the probes, copies their counts for the statistics and paints the
/// debug view. The frame state is the host copy of the counts.
fn radiance_cascades_pass(
    mut ctx: SubmissionState<RingBufferSuballocation>,
    config: Res<RadianceCascadesConfig>,
    mut state: ResMut<RadianceCascadesState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut readback_ring_buffer: ResMut<ReadbackRingBuffer>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    allocator: Res<Allocator>,
    jitter: Res<JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let state = &mut *state;
    // The cache's `GPUMutex` defers freeing its buffers until the GPU is done
    // with them.
    if !config.enabled {
        state.cache = None;
        return;
    }
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let Some(pipelines) = state.pipelines.get(&compute_pipelines) else {
        return;
    };

    let render_extent = hdr.render_extent;
    let cache = state.cache.get_or_insert_with(|| Cache::new(&allocator));
    // `set_frame_state` returns the copy from `FRAMES_IN_FLIGHT` runs ago: for
    // that many runs after a new cache is made, it holds an earlier cache's
    // counts.
    let copy = readback_ring_buffer.allocate_buffer(CASCADES as u64 * 4, 16);
    if let Some(copy) = ctx.set_frame_state(copy) {
        cache.read_stats(&copy);
    }

    // Same jitter as the primary pass: positions are rebuilt from its depth.
    let camera_uniform = build_camera_uniform(camera, transform, None, jitter.offset);

    ctx.record(|encoder, copy| {
        let hdr = &mut **hdr;
        let pipelines = pipelines.retain(encoder);
        let buffers = encoder.lock(
            &cache.buffers,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::CLEAR,
        );
        let constants = RcConstantsRaw {
            cascades: cache.cascades,
            spacing: config.probe_spacing,
            lod_distance: config
                .lod_distance
                .min(MAX_LOD_CELLS * config.probe_spacing),
            debug_mode: config.debug_mode as u32,
            _padding: 0,
            render_extent: render_extent.to_array(),
            keys: buffers.keys().device_address(),
            slots: buffers.slots().device_address(),
            probes: buffers.probes().device_address(),
            counters: buffers.counters().device_address(),
        };
        let camera_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&camera_uniform));
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));

        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        encoder.use_image_resource(
            views.normal(),
            &mut hdr.normal_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            Access::COMPUTE_READ,
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
        // Every pass's layout is pbr.playout.ron, so these stay bound for all
        // of them.
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipelines.insert_pixels.layout(),
            0,
            descriptors.as_slice(),
        );
        record_build(
            encoder,
            &pipelines,
            &cache.cascades,
            buffers,
            render_extent,
            &mut profiler,
        );
        record_stats(encoder, buffers, copy);

        if config.debug_mode != RadianceCascadesDebugMode::Off {
            encoder.use_image_resource(
                views.debug_overlay(),
                &mut hdr.debug_overlay_state,
                Access::COMPUTE_READ | Access::COMPUTE_WRITE,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                false,
            );
            encoder.emit_barriers();
            // Pushes only the overlay: the build's descriptors stay bound.
            descriptors
                .gbuffer_debug_overlay_texture(views.debug_overlay().deref_inner().full_view());
            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipelines.debug_overlay.layout(),
                0,
                descriptors.as_slice(),
            );
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipelines.debug_overlay);
            let display_extent = hdr.display_extent;
            encoder.dispatch(UVec3::new(
                display_extent.x.div_ceil(8),
                display_extent.y.div_ceil(8),
                1,
            ));
        }
    });
}
