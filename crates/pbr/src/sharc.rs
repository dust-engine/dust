//! The path tracer's radiance cache, SHaRC (sharc.slang): its settings, its
//! memory, and its debug views.
//!
//! While [`SharcSettings::enabled`] is on, a path-traced frame runs three passes
//! (see [`crate::path_tracer`]): the update adds the light that a few paths find
//! to the cache, the resolve averages it with the earlier frames', and the paths
//! that light the frame end in it.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bevy_pumicite::{CreateDevice, DefaultRenderSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use pumicite::Allocator;
use pumicite::ash::vk;
use pumicite::buffer::Buffer;
use pumicite::device::DeviceBuilder;
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};

use crate::camera::Camera;
use crate::{
    HdrRenderTarget, HdrRenderTargetViewsExt, JitterState, PbrPipelineParams,
    RenderQualitySettings, build_camera_uniform,
};

/// Cells the cache has room for, a power of two.
pub(crate) const ENTRIES: u32 = 1 << 22;

/// Entries one workgroup of sharc_resolve.slang resolves.
pub(crate) const RESOLVE_GROUP_SIZE: u32 = 256;

// Offsets of the four arrays in the cache's buffer: the keys of the hash map (8
// bytes an entry), the locks that guard its insertions (4), this frame's samples
// (16), and the average over the frames (16).
const LOCKS_OFFSET: u64 = 8 * ENTRIES as u64;
const ACCUMULATION_OFFSET: u64 = LOCKS_OFFSET + 4 * ENTRIES as u64;
const RESOLVED_OFFSET: u64 = ACCUMULATION_OFFSET + 16 * ENTRIES as u64;
const SIZE: u64 = RESOLVED_OFFSET + 16 * ENTRIES as u64;

/// Debug views of the cache, drawn over the path-traced frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SharcView {
    #[default]
    Off,
    /// The cell each pixel's surface is in, as a color.
    Grid,
    /// What the cache holds for each pixel's surface, at the frame's exposure;
    /// black where it holds nothing.
    Radiance,
    /// The hash map's entries, as squares that are green if taken. With a still
    /// camera, 10 to 20% of them should be.
    Occupancy,
    /// Green where a path's first bounce ended in the cache or the sky, red where
    /// it went on.
    Bounces,
}

impl SharcView {
    /// The `kView*` in sharc_debug.slang.
    fn as_u32(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Grid => 1,
            Self::Radiance => 2,
            Self::Occupancy => 3,
            Self::Bounces => 4,
        }
    }
}

/// The DevTools "Path tracer" page and the console's `path-tracer` edit these.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct SharcSettings {
    /// Whether the path tracer's paths end in the cache. Off, the path tracer
    /// is the reference.
    pub enabled: bool,
    /// Width of the smallest cells, in world units: a voxel's, so that the grid
    /// lines up with the voxels.
    pub cell_size: f32,
    /// Distance from the camera up to which cells are that small. They're twice
    /// as wide up to twice the distance, and so on.
    pub cell_distance: f32,
    /// The update traces a path from one pixel of each block of this many
    /// squared, a random one each frame.
    pub update_block: u32,
    /// Frames a cell averages: more make it less noisy, and slower to follow
    /// changes of the light.
    pub accumulation_frames: u32,
    /// Frames without a new sample after which a cell is dropped.
    pub stale_frames: u32,
    pub view: SharcView,
}

impl Default for SharcSettings {
    /// NVIDIA's recommendations, but for the grid's, which follows the voxels.
    fn default() -> Self {
        Self {
            enabled: true,
            // `VoxLoader`'s voxels.
            cell_size: 1.0 / 16.0,
            cell_distance: 16.0,
            update_block: 5,
            accumulation_frames: 32,
            stale_frames: 64,
            view: SharcView::Off,
        }
    }
}

/// The cache, while path-traced frames use it.
pub(crate) struct SharcCache {
    /// Every array of it, at the offsets above. The shaders use it by address.
    pub buffer: GPUMutex<Buffer>,
    pub state: ResourceState,
    /// The buffer hasn't been zeroed yet.
    pub needs_clear: bool,
    /// Where the camera was the last time the cache was updated.
    pub previous_camera_position: Option<Vec3>,
}

/// How many bounces each pixel's path took, while the view of them is on.
pub(crate) struct PathBounces {
    /// A `u32` for each pixel of render resolution, row after row. The shaders
    /// use it by address.
    pub buffer: GPUMutex<Buffer>,
    pub state: ResourceState,
    extent: UVec2,
}

#[derive(Resource)]
pub(crate) struct SharcState {
    pub resolve_pipeline: Handle<ComputePipeline>,
    debug_pipeline: Handle<ComputePipeline>,
    pub cache: Option<SharcCache>,
    pub bounces: Option<PathBounces>,
}

/// What the passes use of the cache, bundled to stay under Bevy's system
/// parameter limit.
#[derive(SystemParam)]
pub(crate) struct SharcParams<'w> {
    pub settings: Res<'w, SharcSettings>,
    pub state: ResMut<'w, SharcState>,
    pub compute_pipelines: Res<'w, Assets<ComputePipeline>>,
}

/// `SharcConstants` in sharc.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SharcConstants {
    hash_entries: u64,
    locks: u64,
    accumulation: u64,
    resolved: u64,
    camera_position: [f32; 3],
    scene_scale: f32,
    previous_camera_position: [f32; 3],
    level_bias: f32,
    entries: u32,
    accumulation_frames: u32,
    stale_frames: u32,
    update_block: u32,
}

impl SharcConstants {
    /// For a cache whose buffer is at `address`.
    pub fn new(
        settings: &SharcSettings,
        address: vk::DeviceAddress,
        camera_position: Vec3,
        previous_camera_position: Vec3,
    ) -> Self {
        Self {
            hash_entries: address,
            locks: address + LOCKS_OFFSET,
            accumulation: address + ACCUMULATION_OFFSET,
            resolved: address + RESOLVED_OFFSET,
            camera_position: camera_position.to_array(),
            // At distance d, SHaRC's level is the whole part of log2(d) +
            // level_bias, and at least 1; its cells are 2^(level - level_bias) /
            // scene_scale wide.
            scene_scale: settings.cell_distance / (2.0 * settings.cell_size),
            previous_camera_position: previous_camera_position.to_array(),
            level_bias: 2.0 - settings.cell_distance.log2(),
            entries: ENTRIES,
            accumulation_frames: settings.accumulation_frames,
            stale_frames: settings.stale_frames,
            update_block: settings.update_block,
        }
    }
}

pub(crate) struct SharcPlugin;

impl Plugin for SharcPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SharcSettings>();
        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                // The keys and the buffers' addresses, and the averages' floats.
                device_builder
                    .enable_feature::<vk::PhysicalDeviceFeatures>(|features| {
                        &mut features.shader_int64
                    })
                    .unwrap();
                device_builder
                    .enable_feature::<vk::PhysicalDeviceFloat16Int8FeaturesKHR>(|features| {
                        &mut features.shader_float16
                    })
                    .unwrap();
                device_builder
                    .enable_feature::<vk::PhysicalDevice16BitStorageFeatures>(|features| {
                        &mut features.storage_buffer16_bit_access
                    })
                    .unwrap();
            })
            .before(CreateDevice),
        );
        app.add_systems(
            Startup,
            (|mut commands: Commands, asset_server: Res<AssetServer>| {
                commands.insert_resource(SharcState {
                    resolve_pipeline: asset_server
                        .load("bazel://dust/crates/pbr/shaders/sharc_resolve.comp.pipeline.bin"),
                    debug_pipeline: asset_server
                        .load("bazel://dust/crates/pbr/shaders/sharc_debug.comp.pipeline.bin"),
                    cache: None,
                    bounces: None,
                });
            })
            .after(CreateDevice),
        );
        app.add_systems(
            PostUpdate,
            (
                ensure_buffers
                    .after(crate::ensure_hdr_target)
                    .before(crate::path_tracer::path_trace_pass),
                debug_pass
                    .run_if(|settings: Res<SharcSettings>| settings.view != SharcView::Off)
                    .in_set(DefaultRenderSet)
                    .after(crate::clear_debug_overlay)
                    .after(crate::path_tracer::path_trace_pass)
                    // It reads this frame's exposure.
                    .after(crate::tonemap::autoexposure_pass)
                    .before(crate::tonemap::tonemap_pass),
            ),
        );
    }
}

/// Keeps the cache only while path-traced frames use it, and the bounces only
/// while their view is on. The `GPUMutex`es free the buffers once the GPU is done
/// with them.
fn ensure_buffers(
    mut state: ResMut<SharcState>,
    settings: Res<SharcSettings>,
    quality: Res<RenderQualitySettings>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    allocator: Res<Allocator>,
) {
    let create_buffer = |size| {
        Buffer::new_device(
            allocator.clone(),
            size,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .unwrap()
    };
    let used = quality.path_tracing && settings.enabled;
    if used != state.cache.is_some() {
        state.cache = used.then(|| SharcCache {
            buffer: GPUMutex::new(create_buffer(SIZE)),
            state: ResourceState::default(),
            needs_clear: true,
            previous_camera_position: None,
        });
    }
    let extent = hdr_target
        .filter(|_| used && settings.view == SharcView::Bounces)
        .map(|hdr| hdr.render_extent);
    if state.bounces.as_ref().map(|bounces| bounces.extent) != extent {
        state.bounces = extent.map(|extent| PathBounces {
            buffer: GPUMutex::new(create_buffer(4 * extent.element_product().max(1) as u64)),
            state: ResourceState::default(),
            extent,
        });
    }
}

/// What the debug pass pushes: the parameters of `main` in sharc_debug.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugPush {
    sharc: SharcConstants,
    bounces: vk::DeviceAddress,
    view: u32,
    _padding: u32,
}

/// Draws the selected view into the debug overlay.
fn debug_pass(
    mut ctx: SubmissionState,
    mut sharc: SharcParams,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    jitter: Res<JitterState>,
) {
    let Some(pipeline) = sharc
        .compute_pipelines
        .get(&sharc.state.debug_pipeline)
        .map(|pipeline| pipeline.clone().into_inner())
    else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let settings = *sharc.settings;
    let state = &mut *sharc.state;
    // None, or never updated: there's nothing to draw.
    let Some(cache) = state.cache.as_mut().filter(|cache| !cache.needs_clear) else {
        return;
    };
    let bounces = &mut state.bounces;
    // The primary rays' jitter: the surfaces are rebuilt from their depth.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let camera_position = transform.translation();
    ctx.record(move |encoder, _| {
        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let cache_buffer = encoder.lock(&cache.buffer, vk::PipelineStageFlags2::COMPUTE_SHADER);
        encoder.use_buffer_resource(cache_buffer, &mut cache.state, Access::COMPUTE_READ);
        let mut push = DebugPush {
            sharc: SharcConstants::new(
                &settings,
                cache_buffer.device_address(),
                camera_position,
                camera_position,
            ),
            bounces: 0,
            view: settings.view.as_u32(),
            _padding: 0,
        };
        if let Some(bounces) = bounces {
            let buffer = encoder.lock(&bounces.buffer, vk::PipelineStageFlags2::COMPUTE_SHADER);
            encoder.use_buffer_resource(buffer, &mut bounces.state, Access::COMPUTE_READ);
            push.bounces = buffer.device_address();
        }
        encoder.use_image_resource(
            views.debug_overlay(),
            &mut hdr.debug_overlay_state,
            Access::COMPUTE_WRITE,
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
            views.albedo(),
            &mut hdr.albedo_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.exposure(),
            &mut hdr.exposure_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            PbrPipelineParams::new()
                .uniforms(uniform)
                .gbuffer_albedo_srgb(views.albedo().deref_inner().srgb_view())
                .gbuffer_normal_texture(views.normal().deref_inner().full_view())
                .gbuffer_depth_texture(views.depth().full_view())
                .gbuffer_debug_overlay_texture(views.debug_overlay().deref_inner().full_view())
                .exposure_texture(views.exposure().full_view())
                .as_slice(),
        );
        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );
        encoder.dispatch(UVec3::new(
            hdr.display_extent.x.div_ceil(8),
            hdr.display_extent.y.div_ceil(8),
            1,
        ));
    });
}
