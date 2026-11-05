//! Split Radiance Cascades (Freeman, Sannikov and Pitt-Francis): the diffuse
//! global illumination. `rc_common.slang` explains the technique; this records
//! its passes and owns its buffers, which shaders reach by device address.
//!
//! Each frame runs up to two caches through the same passes: the secondary
//! cache, for multiple bounces, whose rays leave from the points last frame's
//! rays hit, then the primary cache, whose rays leave from the visible surfaces
//! and which lights them.

use std::ops::Deref;

use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::rtx::RayTracingPipeline;
use bevy_pumicite::rtx::tlas::{TLAS, TLASInner};
use bevy_pumicite::staging::{BufferInitializer, UniformRingBuffer};
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::bindless::WithResourceHandle;
use pumicite::buffer::{Buffer, BufferLike, RingBufferSuballocation};
use pumicite::command::{CommandEncoder, GPURef, GPURefMut};
use pumicite::debug::DebugObject;
use pumicite::image::{FullImageView, Image, ImageExt};
use pumicite::pipeline::Pipeline;
use pumicite::rtx::ShaderBindingTable;
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice, Sampler};

use crate::camera::Camera;
use crate::sky::AtmosphereLUTs;
use crate::{
    HdrRenderTarget, HdrRenderTargetViews, HdrRenderTargetViewsExt, JitterState, PbrInstanceData,
    PbrPipelineParams, PbrRenderSet, PbrRenderState, build_camera_uniform,
};

pub const CASCADES: usize = 4;
/// Texels per axis of a probe's irradiance tile, with its border.
const TILE_SIZE: u32 = 8;
/// Keeps a LOD's reach under the 512 cells per axis that hash keys tell apart
/// (`rc_common.slang`): a LOD ends this many of its cascade 0 cells from the
/// camera, on either side, with room for the LOD overlap and rounding.
const MAX_LOD_CELLS: f32 = 240.0;
/// Most ray hits a frame keeps for the secondary cache.
const HIT_CAPACITY: u32 = 1 << 20;
/// Number of host-visible copies of the counters kept in flight.
const READBACK_SLOTS: usize = 3;

// `RC_FLAG_*` in pbr_common.slang.
const FLAG_SECONDARY: u32 = 1 << 0;
const FLAG_BOUNCE: u32 = 1 << 1;
const FLAG_RECORD_HITS: u32 = 1 << 2;

// `RC_COUNTER_*` and `RC_HIT_*` in pbr_common.slang, in bytes.
const COUNTERS_SIZE: u64 = 29 * 4;
const fn probe_args_offset(cascade: usize) -> u64 {
    (4 + cascade as u64 * 3) * 4
}
const fn merge_args_offset(cascade: usize) -> u64 {
    (16 + cascade as u64 * 3) * 4
}
const HIT_COUNTERS_SIZE: u64 = 7 * 4;
const HIT_BUILD_ARGS_OFFSET: u64 = 4;
const HIT_TRACE_ARGS_OFFSET: u64 = 4 * 4;

/// Debug view painted over the frame. Wire values are the `RC_DEBUG_*` constants
/// in rc_shade.slang.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RadianceCascadesDebugMode {
    #[default]
    Off,
    /// The incoming light at each surface, before its albedo.
    Irradiance,
    /// A color per cascade 0 cell; white where a cell has no probe.
    Probes,
    /// A color per LOD, darkening over the overlap with the next.
    Lod,
    /// How much of each surface's hemisphere the probes around it know about.
    Coverage,
    /// The incoming light the secondary cache holds for each surface.
    Bounce,
}

impl RadianceCascadesDebugMode {
    fn as_u32(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Irradiance => 1,
            Self::Probes => 2,
            Self::Lod => 3,
            Self::Coverage => 4,
            Self::Bounce => 5,
        }
    }
}

/// Tunables of the radiance cascades; the devtools "Radiance cascades" page edits
/// them.
#[derive(Resource, Clone)]
pub struct RadianceCascadesConfig {
    /// Off: no gather at all, so the frame holds direct light only.
    pub enabled: bool,
    /// World units between the cascade 0 probes of LOD 0. Lighting detail finer
    /// than this needs another technique.
    pub probe_spacing: f32,
    /// Length of cascade 0's interval, in probe spacings. The paper uses 1.6, so
    /// that the distance between rays at the end of the interval about equals the
    /// spacing.
    pub ray_length_scale: f32,
    /// Chebyshev distance from the camera at which LOD 0 ends, and each next LOD
    /// ends at twice the last. Over the probe spacing, this sets how large cascade
    /// 0's cells appear on screen.
    pub lod_distance: f32,
    /// The rays a cascade 0 probe gets per frame, when its cell has that many
    /// pixels. Pixels past that don't trace, and most of the rays near the camera
    /// aren't needed. Half of them leave into the surface's hemisphere of the 32
    /// directions.
    pub rays_per_probe: f32,
    /// How much one ray moves a direction's history. 1 turns history off, for
    /// lighting computed entirely within the frame.
    pub temporal_alpha: f32,
    /// Frames a probe lives on after the last ray left its cell, in either cache.
    pub max_probe_age: u32,
    /// A secondary cache, which gives surfaces reached by this frame's rays more
    /// than the direct light. Off: one bounce.
    pub multibounce: bool,
    /// Fraction of the primary cache's ray hits that spawn a secondary cache ray.
    pub bounce_ray_fraction: f32,
    /// LODs by which the secondary cache's probes are coarser than the primary's.
    pub secondary_lod_offset: u32,
    /// Capacity of the primary cache's cascade 0; each cascade up has a third of
    /// the probes. Sets the memory use, about 8 KiB per probe, and the secondary
    /// cache has a quarter of it.
    pub max_probes: u32,
    pub debug_mode: RadianceCascadesDebugMode,
    /// Multiplies the debug views of incoming light.
    pub debug_scale: f32,
}

impl Default for RadianceCascadesConfig {
    /// The paper's parameters and these defaults, with `DUST_RC=0|1`,
    /// `DUST_RC_BOUNCE=0|1`, `DUST_RC_DEBUG=0..5` (the
    /// `RadianceCascadesDebugMode` order), `DUST_RC_ALPHA`, `DUST_RC_RAYS`,
    /// `DUST_RC_SPACING` and `DUST_RC_LOD_DISTANCE` honored for headless captures.
    fn default() -> Self {
        let env_u32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        let env_f32 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
        Self {
            enabled: env_u32("DUST_RC").is_none_or(|v| v != 0),
            probe_spacing: env_f32("DUST_RC_SPACING").unwrap_or(0.125),
            ray_length_scale: 1.6,
            lod_distance: env_f32("DUST_RC_LOD_DISTANCE").unwrap_or(16.0),
            rays_per_probe: env_f32("DUST_RC_RAYS").unwrap_or(128.0),
            temporal_alpha: env_f32("DUST_RC_ALPHA").unwrap_or(0.1),
            max_probe_age: 8,
            multibounce: env_u32("DUST_RC_BOUNCE").is_none_or(|v| v != 0),
            bounce_ray_fraction: 0.25,
            secondary_lod_offset: 2,
            max_probes: 1 << 16,
            debug_mode: match env_u32("DUST_RC_DEBUG") {
                Some(1) => RadianceCascadesDebugMode::Irradiance,
                Some(2) => RadianceCascadesDebugMode::Probes,
                Some(3) => RadianceCascadesDebugMode::Lod,
                Some(4) => RadianceCascadesDebugMode::Coverage,
                Some(5) => RadianceCascadesDebugMode::Bounce,
                _ => RadianceCascadesDebugMode::Off,
            },
            debug_scale: 1.0,
        }
    }
}

/// CPU mirror of `RcCascade` in pbr_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct CascadeRaw {
    table_offset: u32,
    table_mask: u32,
    probe_offset: u32,
    probe_capacity: u32,
    data_offset: u32,
    merged_offset: u32,
    neighbor_offset: u32,
    _pad: u32,
}

/// CPU mirror of `RcIrradianceSource` in pbr_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct IrradianceSourceRaw {
    keys: u64,
    slots: u64,
    table_mask: u32,
    lod_offset: u32,
    tiles_per_row: u32,
    _pad: u32,
}

/// CPU mirror of `RcConstants` in pbr_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConstantsRaw {
    own: IrradianceSourceRaw,
    bounce: IrradianceSourceRaw,
    cascades: [CascadeRaw; CASCADES],
    camera_position: [f32; 3],
    spacing: f32,
    jitter: [f32; 2],
    base_ray_length: f32,
    lod_distance: f32,
    render_extent: [u32; 2],
    lod_offset: u32,
    temporal_alpha: f32,
    flags: u32,
    max_age: u32,
    rays_per_probe: f32,
    pixels_per_radian: f32,
    frame_index: u32,
    bounce_ray_fraction: f32,
    hit_capacity: u32,
    debug_mode: u32,
    debug_scale: f32,
    _pad: u32,
    keys: u64,
    slots: u64,
    probes: u64,
    history: u64,
    counters: u64,
    prev_keys: u64,
    prev_slots: u64,
    prev_probes: u64,
    prev_history: u64,
    prev_counters: u64,
    accum: u64,
    merged: u64,
    neighbors: u64,
    hits: u64,
    hit_counters: u64,
}
const _: () = assert!(size_of::<ConstantsRaw>() == 400);

/// Where each cascade lives in a cache's buffers.
#[derive(Clone, Copy)]
struct CacheLayout {
    cascades: [CascadeRaw; CASCADES],
    table_size: u64,
    probes: u64,
    directions: u64,
    merged: u64,
    neighbors: u64,
    tiles_per_row: u32,
}

impl CacheLayout {
    fn new(max_probes: u32) -> Self {
        let mut layout = Self {
            cascades: [CascadeRaw::default(); CASCADES],
            table_size: 0,
            probes: 0,
            directions: 0,
            merged: 0,
            neighbors: 0,
            tiles_per_row: (max_probes as f64).sqrt().ceil() as u32,
        };
        for cascade in 0..CASCADES {
            let probe_capacity = (max_probes / 3u32.pow(cascade as u32)).max(16);
            // A low load keeps linear probing's walks short.
            let table_size = (probe_capacity * 4).next_power_of_two();
            let directions = 32u32 << (2 * cascade);
            // Per direction of the cascade below; cascade 0 has its own.
            let merged = 32u32 << (2 * cascade.saturating_sub(1));
            layout.cascades[cascade] = CascadeRaw {
                table_offset: layout.table_size as u32,
                table_mask: table_size - 1,
                probe_offset: layout.probes as u32,
                probe_capacity,
                data_offset: layout.directions as u32,
                merged_offset: layout.merged as u32,
                neighbor_offset: (layout.neighbors * 8) as u32,
                _pad: 0,
            };
            layout.table_size += table_size as u64;
            layout.probes += probe_capacity as u64;
            layout.directions += (probe_capacity * directions) as u64;
            layout.merged += (probe_capacity * merged) as u64;
            if cascade + 1 < CASCADES {
                layout.neighbors += probe_capacity as u64;
            }
        }
        layout
    }
}

/// The buffers one frame fills and the next reads.
struct CacheFrame {
    keys: Buffer,
    slots: Buffer,
    probes: Buffer,
    history: Buffer,
    counters: Buffer,
}

pumicite::project_fields! {
    trait CacheFrameExt for impl[] CacheFrame {
        keys: Buffer,
        slots: Buffer,
        counters: Buffer,
    }
}

/// The buffers that hold nothing past the frame.
struct CacheScratch {
    accum: Buffer,
    merged: Buffer,
    neighbors: Buffer,
}

pumicite::project_fields! {
    trait CacheScratchExt for impl[] CacheScratch {
        accum: Buffer,
    }
}

struct Atlas {
    /// R16G16B16A16_SFLOAT tiles, one per cascade 0 probe.
    view: GPUMutex<FullImageView<Image>>,
    state: ResourceState,
}

/// What the devtools show of a cache.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct CacheStats {
    pub probes: [u32; CASCADES],
    pub capacity: [u32; CASCADES],
    pub rays: u32,
}

/// One radiance cache.
struct Cache {
    layout: CacheLayout,
    frames: [GPUMutex<CacheFrame>; 2],
    scratch: GPUMutex<CacheScratch>,
    /// One atlas per frame if something samples last frame's, otherwise one.
    atlases: Vec<Atlas>,
    /// Host copies of the counters, for the statistics.
    readback: [GPUMutex<Buffer>; READBACK_SLOTS],
    readback_state: [ResourceState; READBACK_SLOTS],
    readback_written: [bool; READBACK_SLOTS],
    stats: Option<CacheStats>,
    /// The new buffers are yet to be zeroed.
    needs_clear: bool,
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
    fn new(allocator: &Allocator, max_probes: u32, atlas_count: usize) -> Self {
        let layout = CacheLayout::new(max_probes);
        let none = vk::BufferUsageFlags::empty();
        let frames = [(); 2].map(|_| {
            GPUMutex::new(CacheFrame {
                keys: create_buffer(allocator, layout.table_size * 4, none),
                slots: create_buffer(allocator, layout.table_size * 4, none),
                probes: create_buffer(allocator, layout.probes * 24, none),
                history: create_buffer(allocator, layout.directions * 8, none),
                counters: create_buffer(
                    allocator,
                    COUNTERS_SIZE,
                    vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                ),
            })
        });
        let scratch = GPUMutex::new(CacheScratch {
            accum: create_buffer(allocator, layout.directions * 16, none),
            merged: create_buffer(allocator, layout.merged * 8, none),
            neighbors: create_buffer(allocator, layout.neighbors * 8 * 4, none),
        });
        let atlas_extent = layout.tiles_per_row * TILE_SIZE;
        let atlases = (0..atlas_count)
            .map(|_| {
                let view = Image::new_device(
                    allocator.clone(),
                    &vk::ImageCreateInfo {
                        image_type: vk::ImageType::TYPE_2D,
                        format: vk::Format::R16G16B16A16_SFLOAT,
                        extent: vk::Extent3D {
                            width: atlas_extent,
                            height: atlas_extent,
                            depth: 1,
                        },
                        mip_levels: 1,
                        array_layers: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        ..Default::default()
                    },
                )
                .unwrap()
                .with_name(c"Radiance Cascades Irradiance")
                .create_full_view()
                .unwrap()
                .with_name(c"Radiance Cascades Irradiance View");
                Atlas {
                    view: GPUMutex::new(view),
                    state: ResourceState::default(),
                }
            })
            .collect();
        let readback = [(); READBACK_SLOTS].map(|_| {
            GPUMutex::new(
                Buffer::new_dynamic(
                    allocator.clone(),
                    COUNTERS_SIZE,
                    16,
                    vk::BufferUsageFlags::TRANSFER_DST,
                )
                .unwrap(),
            )
        });
        Self {
            layout,
            frames,
            scratch,
            atlases,
            readback,
            readback_state: Default::default(),
            readback_written: [false; READBACK_SLOTS],
            stats: None,
            needs_clear: true,
        }
    }

    /// The atlas filled in frames of this parity.
    fn atlas_index(&self, frame: usize) -> usize {
        frame % self.atlases.len()
    }

    /// Cascade 0, as sampling its irradiance sees it in frames of this parity.
    fn irradiance_source(&self, frame: usize, lod_offset: u32) -> IrradianceSourceRaw {
        // SAFETY: only device addresses are read.
        let buffers = unsafe { self.frames[frame].get_unchecked() };
        IrradianceSourceRaw {
            keys: buffers.keys.device_address(),
            slots: buffers.slots.device_address(),
            table_mask: self.layout.cascades[0].table_mask,
            lod_offset,
            tiles_per_row: self.layout.tiles_per_row,
            _pad: 0,
        }
    }

    /// Updates the statistics from the copy of the counters made
    /// [`READBACK_SLOTS`] frames ago, normally complete; if the GPU still holds
    /// it, they wait for another frame rather than stall.
    fn read_stats(&mut self, slot: usize) {
        if !self.readback_written[slot] {
            return;
        }
        let Some(buffer) = self.readback[slot].try_deref_mut() else {
            return;
        };
        let Some(bytes) = buffer.as_slice() else {
            return;
        };
        let counters: &[u32] = bytemuck::cast_slice(&bytes[..COUNTERS_SIZE as usize]);
        let mut stats = CacheStats {
            rays: counters[28],
            ..Default::default()
        };
        for cascade in 0..CASCADES {
            stats.probes[cascade] = counters[cascade];
            stats.capacity[cascade] = self.layout.cascades[cascade].probe_capacity;
        }
        self.stats = Some(stats);
    }
}

/// The ray hits one frame keeps for the next frame's secondary cache.
struct HitList {
    hits: Buffer,
    counters: Buffer,
}

pumicite::project_fields! {
    trait HitListExt for impl[] HitList {
        counters: Buffer,
    }
}

struct Resources {
    primary: Cache,
    secondary: Option<Cache>,
    hit_list: GPUMutex<HitList>,
    hit_list_cleared: bool,
    render_extent: UVec2,
    max_probes: u32,
}

/// The compute pipelines, in the order of [`COMPUTE_PASSES`].
#[derive(Clone, Copy)]
enum Pass {
    InsertPixels,
    InsertHits,
    CarryOver,
    Allocate,
    WriteArgs,
    InsertParents,
    Link,
    WriteHitArgs,
    Merge,
    Irradiance,
    Shade,
    Debug,
}

const COMPUTE_PASSES: [&str; 12] = [
    "rc_insert_pixels",
    "rc_insert_hits",
    "rc_carry_over",
    "rc_allocate",
    "rc_write_args",
    "rc_insert_parents",
    "rc_link",
    "rc_write_hit_args",
    "rc_merge",
    "rc_irradiance",
    "rc_shade",
    "rc_debug",
];

#[derive(Resource)]
pub struct RadianceCascadesState {
    compute_pipelines: [Handle<ComputePipeline>; COMPUTE_PASSES.len()],
    resources: Option<Resources>,
    frame_index: u32,
}

impl RadianceCascadesState {
    /// The primary and, if there is one, the secondary cache's latest statistics.
    pub fn stats(&self) -> (Option<CacheStats>, Option<CacheStats>) {
        let Some(resources) = &self.resources else {
            return (None, None);
        };
        (
            resources.primary.stats,
            resources.secondary.as_ref().and_then(|cache| cache.stats),
        )
    }
}

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
                .after(PbrRenderSet)
                .after(bevy_pumicite::rtx::tlas::tlas_build_system::<PbrInstanceData>)
                .after(bevy_pumicite::rtx::build_rtx_pipeline_system)
                .after(crate::create_sbt)
                .after(crate::render)
                .after(crate::clear_debug_overlay)
                // Both read-modify-write the HDR output.
                .after(crate::shadow_pass)
                .after(crate::screen_trace::screen_trace_pass)
                // The lighting is complete only after this pass.
                .before(crate::tonemap::autoexposure_pass)
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
        frame_index: 0,
    });
}

fn create_resources(allocator: &Allocator, render_extent: UVec2, max_probes: u32) -> Resources {
    Resources {
        primary: Cache::new(allocator, max_probes, 1),
        secondary: None,
        hit_list: GPUMutex::new(HitList {
            hits: create_buffer(
                allocator,
                HIT_CAPACITY as u64 * 16,
                vk::BufferUsageFlags::empty(),
            ),
            counters: create_buffer(
                allocator,
                HIT_COUNTERS_SIZE,
                vk::BufferUsageFlags::INDIRECT_BUFFER,
            ),
        }),
        hit_list_cleared: false,
        render_extent,
        max_probes,
    }
}

/// Everything shaders read and write outside the G-buffer: any access by any of
/// the passes here.
fn shader_access() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    }
}

/// Reads of the G-buffer by any of the passes here.
fn shader_read() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        access: vk::AccessFlags2::SHADER_STORAGE_READ,
    }
}

/// An irradiance atlas being sampled.
fn atlas_read() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    }
}

/// An image a compute pass adds to.
fn image_read_write() -> Access {
    Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    }
}

/// Finishes a pass's writes, for any later pass to read, or use as indirect
/// arguments. A pass's atomics need no ordering among themselves.
fn pass_barrier(encoder: &mut CommandEncoder) {
    encoder.memory_barrier(
        shader_access(),
        shader_access()
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

/// The compute pipelines, kept alive by the command buffer.
struct Passes<'a> {
    pipelines: [GPURef<'a, Pipeline>; COMPUTE_PASSES.len()],
}

impl<'a> Passes<'a> {
    fn get(&self, pass: Pass) -> GPURef<'a, Pipeline> {
        self.pipelines[pass as usize]
    }

    /// Runs one pass, with `cascade` as its push constant, ready for the next.
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
            4,
            bytemuck::bytes_of(&(cascade as u32)),
        );
        dispatch(encoder);
        pass_barrier(encoder);
    }
}

/// Where a cache's rays leave from.
#[derive(Clone, Copy)]
enum Source<'a> {
    /// The visible surfaces.
    Pixels(UVec2),
    /// The ray hits kept by last frame, with these counters.
    Hits(GPURef<'a, Buffer>),
}

/// What the passes of both caches bind, for the whole command buffer.
#[derive(Clone, Copy)]
struct FrameBindings<'a, 'r> {
    camera: GPURef<'a, RingBufferSuballocation>,
    views: GPURefMut<'a, HdrRenderTargetViews>,
    tlas: GPURefMut<'a, TLASInner>,
    atmosphere: GPURef<'a, RingBufferSuballocation>,
    transmittance: GPURefMut<'a, WithResourceHandle<FullImageView<Image>>>,
    sky_view: GPURefMut<'a, WithResourceHandle<FullImageView<Image>>>,
    sampler: &'r Sampler,
    rt_pipeline: GPURef<'a, Pipeline>,
    sbt: &'r ShaderBindingTable,
    sbt_buffer: GPURef<'a, RingBufferSuballocation>,
}

impl<'a> FrameBindings<'a, '_> {
    /// What the passes that build and merge a cache read.
    fn push_compute(
        &self,
        encoder: &mut CommandEncoder<'a>,
        passes: &Passes<'a>,
        constants: GPURef<'a, RingBufferSuballocation>,
    ) {
        let mut descriptors = PbrPipelineParams::new();
        descriptors
            .uniforms(self.camera)
            .gbuffer_normal_texture(self.views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(self.views.depth().full_view())
            .rc(constants);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            passes.get(Pass::Merge).layout(),
            0,
            descriptors.as_slice(),
        );
    }
}

/// The names a cache's work goes by in the profiler.
struct TimingScopes {
    build: &'static str,
    trace: &'static str,
    merge: &'static str,
}

/// Steps 1 to 6 of `rc_common.slang` for one cache, up to its filled atlas, which
/// is returned along with the frame's buffers. `bounce_atlas` is what ray hits
/// sample, if `constants` says they do.
#[allow(clippy::too_many_arguments)]
fn record_cache<'a>(
    encoder: &mut CommandEncoder<'a>,
    passes: &Passes<'a>,
    bindings: &FrameBindings<'a, '_>,
    cache: &mut Cache,
    frame: usize,
    constants: GPURef<'a, RingBufferSuballocation>,
    source: Source<'a>,
    bounce_atlas: Option<GPURefMut<'a, FullImageView<Image>>>,
    keep_probes: bool,
    scopes: &TimingScopes,
    profiler: &mut Option<ResMut<GpuProfiler>>,
) -> (GPURefMut<'a, FullImageView<Image>>, GPURef<'a, CacheFrame>) {
    let stages = shader_read().stage | vk::PipelineStageFlags2::CLEAR;
    let frame_token = encoder.lock(&cache.frames[frame], stages).readonly();
    let prev_token = encoder.lock(&cache.frames[1 - frame], stages).readonly();
    let scratch_token = encoder.lock(&cache.scratch, stages);

    // This frame's tables start empty.
    encoder.memory_barrier(shader_access(), Access::CLEAR);
    encoder.emit_barriers();
    fill_with_zeros(encoder, frame_token.keys().vk_handle());
    fill_with_zeros(encoder, frame_token.slots().vk_handle());
    fill_with_zeros(encoder, frame_token.counters().vk_handle());
    if std::mem::take(&mut cache.needs_clear) {
        fill_with_zeros(encoder, prev_token.keys().vk_handle());
        fill_with_zeros(encoder, prev_token.counters().vk_handle());
        fill_with_zeros(encoder, scratch_token.accum().vk_handle());
    }
    encoder.memory_barrier(Access::CLEAR, shader_access());
    encoder.emit_barriers();

    bindings.push_compute(encoder, passes, constants);
    let table_sizes = cache.layout.cascades.map(|cascade| cascade.table_mask + 1);
    encoder.timing_scope(profiler.as_deref_mut(), scopes.build, |encoder| {
        if keep_probes {
            passes.run(encoder, Pass::CarryOver, 0, |encoder| {
                encoder.dispatch_indirect(prev_token.counters(), probe_args_offset(0));
            });
        }
        match source {
            Source::Pixels(extent) => passes.run(encoder, Pass::InsertPixels, 0, |encoder| {
                encoder.dispatch(UVec3::new(extent.x.div_ceil(8), extent.y.div_ceil(8), 1));
            }),
            Source::Hits(counters) => passes.run(encoder, Pass::InsertHits, 0, |encoder| {
                encoder.dispatch_indirect(counters, HIT_BUILD_ARGS_OFFSET);
            }),
        }
        for cascade in 0..CASCADES {
            if cascade > 0 {
                passes.run(encoder, Pass::InsertParents, cascade, |encoder| {
                    encoder
                        .dispatch_indirect(frame_token.counters(), probe_args_offset(cascade - 1));
                });
            }
            passes.run(encoder, Pass::Allocate, cascade, |encoder| {
                encoder.dispatch(UVec3::new(table_sizes[cascade].div_ceil(64), 1, 1));
            });
            passes.run(encoder, Pass::WriteArgs, cascade, |encoder| {
                encoder.dispatch(UVec3::ONE);
            });
        }
        for cascade in (0..CASCADES).rev() {
            passes.run(encoder, Pass::Link, cascade, |encoder| {
                encoder.dispatch_indirect(frame_token.counters(), probe_args_offset(cascade));
            });
        }
    });

    encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, bindings.rt_pipeline);
    let mut descriptors = PbrPipelineParams::new();
    descriptors
        .scene_bvh(bindings.tlas)
        .uniforms(bindings.camera)
        .gbuffer_normal_texture(bindings.views.normal().deref_inner().full_view())
        .gbuffer_depth_texture(bindings.views.depth().full_view())
        .sky_atmosphere_params(bindings.atmosphere)
        .sky_transmittance_lut(bindings.transmittance.deref_inner().full_view())
        .sky_sky_view_lut(bindings.sky_view.deref_inner().full_view())
        .sky_linear_sampler(bindings.sampler)
        .rc(constants);
    match bounce_atlas {
        Some(atlas) => descriptors.rc_bounce_atlas(atlas.full_view()),
        None => descriptors.rc_bounce_atlas_null(),
    };
    encoder.push_descriptor_set(
        vk::PipelineBindPoint::RAY_TRACING_KHR,
        bindings.rt_pipeline.layout(),
        0,
        descriptors.as_slice(),
    );
    encoder.timing_scope(
        profiler.as_deref_mut(),
        scopes.trace,
        |encoder| match source {
            Source::Pixels(extent) => {
                encoder.trace_rays(
                    bindings.sbt,
                    0,
                    bindings.sbt_buffer,
                    UVec3::new(extent.x, extent.y, 1),
                );
            }
            Source::Hits(counters) => {
                encoder.trace_rays_indirect(
                    bindings.sbt,
                    0,
                    bindings.sbt_buffer,
                    counters,
                    HIT_TRACE_ARGS_OFFSET,
                );
            }
        },
    );
    pass_barrier(encoder);

    let atlas_index = cache.atlas_index(frame);
    let atlas = &mut cache.atlases[atlas_index];
    let atlas_token = encoder.lock(&atlas.view, shader_read().stage);
    encoder.use_image_resource(
        atlas_token,
        &mut atlas.state,
        Access::COMPUTE_WRITE,
        vk::ImageLayout::GENERAL,
        0..1,
        0..1,
        true,
    );
    encoder.emit_barriers();
    bindings.push_compute(encoder, passes, constants);
    encoder.timing_scope(profiler.as_deref_mut(), scopes.merge, |encoder| {
        for cascade in (0..CASCADES).rev() {
            passes.run(encoder, Pass::Merge, cascade, |encoder| {
                encoder.dispatch_indirect(frame_token.counters(), merge_args_offset(cascade));
            });
        }
        let mut descriptors = PbrPipelineParams::new();
        descriptors
            .rc(constants)
            .rc_irradiance_output(atlas_token.full_view());
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            passes.get(Pass::Irradiance).layout(),
            0,
            descriptors.as_slice(),
        );
        passes.run(encoder, Pass::Irradiance, 0, |encoder| {
            encoder.dispatch_indirect(frame_token.counters(), probe_args_offset(0));
        });
    });
    encoder.use_image_resource(
        atlas_token,
        &mut atlas.state,
        atlas_read(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        0..1,
        0..1,
        false,
    );
    encoder.emit_barriers();
    (atlas_token, frame_token)
}

/// Copies a cache's counters for the statistics.
fn record_stats<'a>(
    encoder: &mut CommandEncoder<'a>,
    cache: &mut Cache,
    slot: usize,
    frame: GPURef<'a, CacheFrame>,
) {
    let readback = encoder.lock(&cache.readback[slot], vk::PipelineStageFlags2::COPY);
    encoder.memory_barrier(shader_access(), Access::COPY_READ);
    encoder.use_buffer_resource(
        readback,
        &mut cache.readback_state[slot],
        Access::COPY_WRITE,
    );
    encoder.emit_barriers();
    encoder.copy_buffer(frame.counters(), readback);
    cache.readback_written[slot] = true;
}

fn radiance_cascades_pass(
    mut ctx: SubmissionState,
    config: Res<RadianceCascadesConfig>,
    mut state: ResMut<RadianceCascadesState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    rt_pipelines: Res<Assets<RayTracingPipeline>>,
    mut pbr: ResMut<PbrRenderState>,
    tlas: Res<TLAS<PbrInstanceData>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
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
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.clone() else {
        return;
    };
    let pbr = &mut *pbr;
    let Some(rt_pipeline) = rt_pipelines
        .get(&pbr.radiance_cascades_pipeline)
        .map(|pipeline| pipeline.deref().clone())
    else {
        return;
    };
    let Some(sbt) = pbr.radiance_cascades_sbt.as_mut() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
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
        Some(resources)
            if resources.render_extent == render_extent
                && resources.max_probes == config.max_probes =>
        {
            resources
        }
        slot => slot.insert(create_resources(
            &allocator,
            render_extent,
            config.max_probes,
        )),
    };
    match (config.multibounce, resources.secondary.is_some()) {
        (true, false) => {
            let max_probes = (config.max_probes / 4).max(1024);
            resources.secondary = Some(Cache::new(&allocator, max_probes, 2));
        }
        (false, true) => resources.secondary = None,
        _ => {}
    }

    sbt.push_raygen(0, 0, |_| {});
    sbt.push_miss(0, 1, |_| {});

    let frame_index = state.frame_index;
    state.frame_index = frame_index.wrapping_add(1);
    let frame = (frame_index & 1) as usize;
    let prev_frame = 1 - frame;
    let readback_slot = frame_index as usize % READBACK_SLOTS;

    // Same jitter as the primary pass: positions are rebuilt from its depth.
    let camera_uniform = build_camera_uniform(camera, transform, None, jitter.offset);

    let Resources {
        primary,
        secondary,
        hit_list,
        hit_list_cleared,
        ..
    } = resources;
    primary.read_stats(readback_slot);
    if let Some(secondary) = secondary {
        secondary.read_stats(readback_slot);
    }
    let hit_addresses = {
        // SAFETY: only device addresses are read.
        let list = unsafe { hit_list.get_unchecked() };
        (list.hits.device_address(), list.counters.device_address())
    };
    let lod_offset = config.secondary_lod_offset;
    let build_constants =
        |cache: &Cache, lod_offset: u32, flags: u32, bounce: IrradianceSourceRaw| {
            // SAFETY: only device addresses are read.
            let (buffers, prev, scratch) = unsafe {
                (
                    cache.frames[frame].get_unchecked(),
                    cache.frames[prev_frame].get_unchecked(),
                    cache.scratch.get_unchecked(),
                )
            };
            ConstantsRaw {
                own: cache.irradiance_source(frame, lod_offset),
                bounce,
                cascades: cache.layout.cascades,
                camera_position: transform.translation().to_array(),
                spacing: config.probe_spacing,
                jitter: [
                    crate::upscaler::halton(frame_index % 1024 + 1, 2),
                    crate::upscaler::halton(frame_index % 1024 + 1, 3),
                ],
                base_ray_length: config.probe_spacing * config.ray_length_scale,
                lod_distance: config
                    .lod_distance
                    .min(MAX_LOD_CELLS * config.probe_spacing),
                render_extent: render_extent.to_array(),
                lod_offset,
                temporal_alpha: config.temporal_alpha,
                flags,
                max_age: config.max_probe_age,
                rays_per_probe: config.rays_per_probe,
                pixels_per_radian: render_extent.y as f32 / (2.0 * camera.tan_half_fov()),
                frame_index,
                bounce_ray_fraction: config.bounce_ray_fraction,
                hit_capacity: HIT_CAPACITY,
                debug_mode: config.debug_mode.as_u32(),
                debug_scale: config.debug_scale,
                _pad: 0,
                keys: buffers.keys.device_address(),
                slots: buffers.slots.device_address(),
                probes: buffers.probes.device_address(),
                history: buffers.history.device_address(),
                counters: buffers.counters.device_address(),
                prev_keys: prev.keys.device_address(),
                prev_slots: prev.slots.device_address(),
                prev_probes: prev.probes.device_address(),
                prev_history: prev.history.device_address(),
                prev_counters: prev.counters.device_address(),
                accum: scratch.accum.device_address(),
                merged: scratch.merged.device_address(),
                neighbors: scratch.neighbors.device_address(),
                hits: hit_addresses.0,
                hit_counters: hit_addresses.1,
            }
        };
    // What the secondary cache's rays hit is lit by its own last frame, on the
    // first frames an empty table, so that its atlas is never sampled.
    let secondary_constants = secondary.as_ref().map(|cache| {
        build_constants(
            cache,
            lod_offset,
            FLAG_SECONDARY | FLAG_BOUNCE,
            cache.irradiance_source(prev_frame, lod_offset),
        )
    });
    let primary_constants = match secondary.as_ref() {
        Some(cache) => build_constants(
            primary,
            0,
            FLAG_BOUNCE | FLAG_RECORD_HITS,
            cache.irradiance_source(frame, lod_offset),
        ),
        None => build_constants(primary, 0, 0, IrradianceSourceRaw::default()),
    };
    let keep_probes = config.max_probe_age > 0 && config.temporal_alpha < 1.0;

    ctx.record(|encoder| {
        let hdr = &mut **hdr;
        let passes = Passes {
            pipelines: std::array::from_fn(|i| encoder.retain(pipelines[i].clone()).deref_inner()),
        };
        let rt_pipeline = encoder.retain(rt_pipeline).deref_inner();
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                sbt.write_buffer(slice);
            });
        let camera_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&camera_uniform));
        let primary_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&primary_constants));
        let atmosphere_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();

        let stages = shader_read().stage;
        let tlas = encoder.lock(tlas, stages);
        let views = encoder.lock(&hdr.view, stages);
        let transmittance = encoder.lock(&atmosphere_luts.transmittance.view, stages);
        let sky_view = encoder.lock(&atmosphere_luts.sky_view.view, stages);
        let hit_list_token = encoder.lock(hit_list, stages | vk::PipelineStageFlags2::CLEAR);

        encoder.use_image_resource(
            views.normal(),
            &mut hdr.normal_state,
            shader_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            shader_read(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            transmittance,
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

        let bindings = FrameBindings {
            camera: camera_buffer.readonly(),
            views,
            tlas,
            atmosphere: atmosphere_buffer,
            transmittance,
            sky_view,
            sampler: &atmosphere_luts.sampler,
            rt_pipeline,
            sbt,
            sbt_buffer: sbt_buffer.readonly(),
        };

        // Fresh counters mean no hits, and no work, until a frame has kept some.
        if !std::mem::replace(hit_list_cleared, true) {
            encoder.memory_barrier(Access::NONE, Access::CLEAR);
            encoder.emit_barriers();
            fill_with_zeros(encoder, hit_list_token.counters().vk_handle());
            encoder.memory_barrier(Access::CLEAR, shader_access());
            encoder.emit_barriers();
        }

        let mut bounce_atlas = None;
        if let (Some(cache), Some(constants)) = (secondary.as_mut(), secondary_constants.as_ref()) {
            let constants =
                uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(constants));
            let prev_atlas_index = cache.atlas_index(prev_frame);
            let prev_atlas = &mut cache.atlases[prev_atlas_index];
            let prev_atlas_token = encoder.lock(&prev_atlas.view, stages);
            encoder.use_image_resource(
                prev_atlas_token,
                &mut prev_atlas.state,
                atlas_read(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            encoder.emit_barriers();
            let (atlas, frame_token) = record_cache(
                encoder,
                &passes,
                &bindings,
                cache,
                frame,
                constants.readonly(),
                Source::Hits(hit_list_token.counters().readonly()),
                Some(prev_atlas_token),
                keep_probes,
                &TimingScopes {
                    build: "RC secondary build",
                    trace: "RC secondary trace",
                    merge: "RC secondary merge",
                },
                &mut profiler,
            );
            bounce_atlas = Some(atlas);
            record_stats(encoder, cache, readback_slot, frame_token);
            // The list is used up; the primary cache's rays start the next.
            encoder.memory_barrier(shader_access(), Access::CLEAR);
            encoder.emit_barriers();
            fill_with_zeros(encoder, hit_list_token.counters().vk_handle());
            encoder.memory_barrier(Access::CLEAR, shader_access());
            encoder.emit_barriers();
        }

        let primary_constants = primary_buffer.readonly();
        let (atlas, frame_token) = record_cache(
            encoder,
            &passes,
            &bindings,
            primary,
            frame,
            primary_constants,
            Source::Pixels(render_extent),
            bounce_atlas,
            keep_probes,
            &TimingScopes {
                build: "RC build",
                trace: "RC trace",
                merge: "RC merge",
            },
            &mut profiler,
        );
        if secondary.is_some() {
            passes.run(encoder, Pass::WriteHitArgs, 0, |encoder| {
                encoder.dispatch(UVec3::ONE);
            });
        }
        record_stats(encoder, primary, readback_slot, frame_token);

        // Step 6's shading, into the HDR output.
        encoder.use_image_resource(
            views.hdr_output(),
            &mut hdr.state,
            image_read_write(),
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.albedo(),
            &mut hdr.albedo_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        let mut descriptors = PbrPipelineParams::new();
        descriptors
            .uniforms(camera_buffer)
            .output_texture(views.hdr_output().full_view())
            .gbuffer_albedo_srgb(views.albedo().deref_inner().srgb_view())
            .gbuffer_normal_texture(views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(views.depth().full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .rc(primary_constants)
            .rc_irradiance_atlas(atlas.full_view());
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            passes.get(Pass::Shade).layout(),
            0,
            descriptors.as_slice(),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "RC shade", |encoder| {
            passes.run(encoder, Pass::Shade, 0, |encoder| {
                encoder.dispatch(UVec3::new(
                    render_extent.x.div_ceil(8),
                    render_extent.y.div_ceil(8),
                    1,
                ));
            });
        });

        if config.debug_mode != RadianceCascadesDebugMode::Off {
            encoder.use_image_resource(
                views.debug_overlay(),
                &mut hdr.debug_overlay_state,
                image_read_write(),
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
                .gbuffer_debug_overlay_texture(views.debug_overlay().deref_inner().full_view())
                .sky_linear_sampler(&atmosphere_luts.sampler)
                .rc(primary_constants)
                .rc_irradiance_atlas(atlas.full_view());
            match bounce_atlas {
                Some(atlas) => descriptors.rc_bounce_atlas(atlas.full_view()),
                None => descriptors.rc_bounce_atlas_null(),
            };
            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                passes.get(Pass::Debug).layout(),
                0,
                descriptors.as_slice(),
            );
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
