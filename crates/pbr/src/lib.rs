pub mod camera;
pub mod sharc;
pub mod sky;
pub mod tonemap;

include!(concat!(
    env!("BAZEL_BIN"),
    "/crates/pbr/shaders/pbr_module_layout.rs"
));

use std::ops::Deref;

use tonemap::tonemap_pass;

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy_pumicite::{
    CreateDevice, DefaultRenderSet, PumiciteApp, SubmissionState,
    rtx::{
        RayTracingPipeline, RtxPipelineManager,
        tlas::{TLAS, TLASInstance, tlas_build_input_upload_system},
    },
    shader::RayTracingPipelineLibrary,
    staging::{BufferInitializer, UniformRingBuffer},
    swapchain::{SwapchainImage, SwapchainSet},
};
use bytemuck::{Pod, Zeroable};
use pumicite::{device::DeviceBuilder, image::UintImageView};
use pumicite::{
    Allocator, HasDevice,
    ash::vk::{self, TaggedStructure},
    buffer::BufferLike,
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike, SrgbImageView},
    rtx::ShaderBindingTable,
    sync::GPUMutex,
    tracking::{Access, ResourceState},
    utils::{AsVkHandle, glam_to_vk_transform},
};
use pumicite_egui::{EguiPrimaryContextPass, EguiRenderSet};

use dust_gfxdebug::{GpuProfiler, GpuTimerCommands, PerformancePanel};

use bevy_pumicite::prelude::ComputePipeline;

use crate::{
    camera::Camera,
    sky::{AtmosphereLUTs, SkyAtmosphereLUTRenderSet},
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Uniforms {
    /// row-major affine transformation matrix.
    model: vk::TransformMatrixKHR,
    tan_half_fov: f32,
    far: f32,
    near: f32,
    /// Sub-pixel jitter applied to the current frame's primary ray, in pixels.
    /// `jitter_x` is laid out where `_padding` used to be so the existing
    /// `prev_*` block stays at the same offset; `jitter_y` reuses the first
    /// slot of `_padding2`.
    jitter_x: f32,
    /// Previous-frame world-from-view matrix; used by the primary RT pass to
    /// project hit points through last frame's camera and produce screen-space
    /// motion vectors for DLSS-RR.
    prev_model: vk::TransformMatrixKHR,
    prev_tan_half_fov: f32,
    jitter_y: f32,
    _padding2: [f32; 2],
}
unsafe impl Pod for Uniforms {}
unsafe impl Zeroable for Uniforms {}

#[derive(Pod, Clone, Copy, Zeroable, bevy::reflect::TypePath)]
#[repr(C)]
pub struct PbrInstanceData {
    /// Row-major 3x4 world-from-object transform from the previous frame.
    /// Layout matches `vk::TransformMatrixKHR::matrix`. Filled by
    /// `update_prev_transforms` after the per-instance buffer is uploaded, so
    /// next frame's upload reads this frame's current transform.
    pub previous_transform: [f32; 12],
}

impl Default for PbrInstanceData {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Last frame's camera transform and FOV. Held as `Local` state on the
/// `render` system to fill the `prev_*` fields of the camera uniform.
#[derive(Default)]
struct PreviousCameraState {
    model: Option<vk::TransformMatrixKHR>,
    tan_half_fov: f32,
}

/// Build a camera uniform for the current frame.
///
/// If `prev` is `Some`, the previous-frame matrix/FOV are read from it and
/// then overwritten with the current frame's values. If `None` (or on the
/// very first call), `prev_*` mirror the current camera, which produces zero
/// motion vectors — the right behavior for passes that don't reproject and
/// for the first frame after a camera teleport / reset.
pub(crate) fn build_camera_uniform(
    camera: &Camera,
    transform: &GlobalTransform,
    prev: Option<&mut PreviousCameraState>,
    jitter: Vec2,
) -> Uniforms {
    let model = glam_to_vk_transform(transform.affine());
    let tan_half_fov = camera.tan_half_fov();
    let (prev_model, prev_tan_half_fov) = match prev {
        Some(prev) => {
            let prev_model = prev.model.unwrap_or(model);
            let prev_tan_half_fov = if prev.model.is_some() {
                prev.tan_half_fov
            } else {
                tan_half_fov
            };
            prev.model = Some(model);
            prev.tan_half_fov = tan_half_fov;
            (prev_model, prev_tan_half_fov)
        }
        None => (model, tan_half_fov),
    };
    Uniforms {
        far: camera.depth.end,
        near: camera.depth.start,
        model,
        tan_half_fov,
        jitter_x: jitter.x,
        prev_model,
        prev_tan_half_fov,
        jitter_y: jitter.y,
        _padding2: [0.0; 2],
    }
}

/// Snapshot the current world-from-object transform of every TLAS instance
/// into its `PbrInstanceData::previous_transform` slot. Runs after
/// `tlas_build_input_upload_system::<PbrInstanceData>`, so this frame's upload
/// has already consumed the value written last frame; the value we write here
/// becomes the "previous transform" for next frame's upload.
fn update_prev_transforms(
    mut instances: Query<(&GlobalTransform, &mut TLASInstance<PbrInstanceData>)>,
) {
    for (transform, mut instance) in instances.iter_mut() {
        instance.data.previous_transform = glam_to_vk_transform(transform.affine()).matrix;
    }
}

pub struct PbrRenderPlugin;
impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup.after(CreateDevice));
        app.init_resource::<JitterState>();
        app.init_resource::<RenderQualitySettings>();
        app.add_systems(
            PostUpdate,
            (
                create_sbt.before(PbrRenderSet),
                ensure_hdr_target.in_set(SwapchainSet).before(render),
                ensure_dlss_feature
                    .in_set(DefaultRenderSet)
                    .after(ensure_hdr_target)
                    .before(render),
                update_prev_transforms
                    .in_set(DefaultRenderSet)
                    .after(tlas_build_input_upload_system::<PbrInstanceData>),
                render
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(bevy_pumicite::rtx::tlas::tlas_build_system::<PbrInstanceData>)
                    .after(bevy_pumicite::rtx::build_rtx_pipeline_system)
                    .after(OccludingRenderPass)
                    .after(create_sbt),
                shadow_pass
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(render)
                    .after(create_sbt),
                final_gather_pass
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(shadow_pass)
                    .after(create_sbt)
                    // SHARC ping-pong: the write pool's keys must be cleared
                    // to 0 (so the WRS-of-1 atomic_max starts from "empty")
                    // before vox_final_gather CHS pushes Query misses.
                    .after(sharc::clear_sharc_buffers),
                dlss_evaluate
                    .in_set(DefaultRenderSet)
                    .after(ensure_dlss_feature)
                    .after(final_gather_pass),
                tonemap_pass
                    .in_set(DefaultRenderSet)
                    .after(render)
                    .after(shadow_pass)
                    .after(final_gather_pass)
                    .after(dlss_evaluate),
            ),
        );
        app.add_plugins(bevy_pumicite::rtx::RtxPipelinePlugin);

        app.add_render_set(OccludingRenderPass, start_occluding_render_pass);
        app.configure_sets(PostUpdate, OccludingRenderPass.in_set(DefaultRenderSet));
        app.configure_sets(PostUpdate, EguiRenderSet.in_set(OccludingRenderPass));

        // Build a TLAS over everything.
        app.add_plugins(bevy_pumicite::rtx::tlas::TLASBuilderPlugin::<PbrInstanceData>::default());

        app.add_plugins(sky::SkyPlugin);
        app.add_plugins(sharc::SharcDebugPlugin);
        app.configure_sets(
            PostUpdate,
            SkyAtmosphereLUTRenderSet
                .in_set(DefaultRenderSet)
                .before(render),
        );

        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                device_builder
                    .enable_extension::<pumicite::ash::ext::hdr_metadata::Meta>()
                    .ok();

                // Enable either VK_KHR_robustness2 or VK_EXT_robustness2. This allows us to write
                // null descriptors - required by pumicite cli codegen
                device_builder
                    .enable_extension::<pumicite::ash::khr::robustness2::Meta>()
                    .or_else(|_| {
                        device_builder.enable_extension::<pumicite::ash::ext::robustness2::Meta>()
                    })
                    .ok();
                device_builder
                    .enable_feature::<vk::PhysicalDeviceRobustness2FeaturesKHR>(|feature| {
                        &mut feature.null_descriptor
                    })
                    .ok();

                // SHARC requirements (AGENTS.md §14, SDK v1.8):
                //   * shaderFloat16 + 16-bit storage for the SharcPackedData
                //     resolved buffer (float16x4 per entry).
                //   * shaderBufferInt64Atomics for the 64-bit hash-key CAS used
                //     by HashGridInsert when HASH_GRID_ENABLE_64_BIT_ATOMICS = 1.
                device_builder
                    .enable_feature::<vk::PhysicalDeviceFloat16Int8FeaturesKHR>(|feature| {
                        &mut feature.shader_float16
                    })
                    .ok();
                device_builder
                    .enable_feature::<vk::PhysicalDevice16BitStorageFeatures>(|feature| {
                        &mut feature.storage_buffer16_bit_access
                    })
                    .ok();
                // Slang lowers `RWStructuredBuffer<SharcPackedData>` (which
                // contains `float16_t4`) through the older Uniform-Block
                // SPIR-V form, so storageBuffer16BitAccess alone is not
                // enough — the validator demands
                // uniformAndStorageBuffer16BitAccess too. Without this
                // feature the SHARC shader modules silently fail to create
                // (the validation error we hit on first run).
                device_builder
                    .enable_feature::<vk::PhysicalDevice16BitStorageFeatures>(|feature| {
                        &mut feature.uniform_and_storage_buffer16_bit_access
                    })
                    .ok();
                device_builder
                    .enable_extension::<pumicite::ash::khr::shader_atomic_int64::Meta>()
                    .ok();
                device_builder
                    .enable_feature::<vk::PhysicalDeviceShaderAtomicInt64Features>(|feature| {
                        &mut feature.shader_buffer_int64_atomics
                    })
                    .ok();
            })
            .before(CreateDevice),
        );
    }
}

/// All the systems preparing for PBR raytracing render must go into this set.
#[derive(SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct PbrRenderSet;

fn render(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS<PbrInstanceData>>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut prev_camera: Local<PreviousCameraState>,
    jitter: Res<JitterState>,
    // SHARC cache, queried by the primary closest hit for the debug
    // visualization (vox_pbr.slang). Bindings 14..17 are statically referenced
    // by the primary pipeline, so the resources must be bound every frame —
    // `SharcDebugPlugin` creates them at startup, so they are always present.
    mut sharc_resources: Option<ResMut<sharc::SharcResources>>,
    sharc_config: Res<sharc::SharcConfig>,
    sharc_frame_state: Res<sharc::SharcFrameState>,
    sharc_debug: Res<sharc::SharcDebugState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok((camera, transform)) = swapchain_images.single_mut() else {
        tracing::warn!("Frame not rendered; missing swapchain image");
        return;
    };
    let Some(pipeline) = pipelines.get(&state.pipeline).map(|x| x.deref().clone()) else {
        tracing::warn!("Frame not rendered; missing pipeline");
        return;
    };
    let Some(sbt) = state.sbt.as_mut() else {
        tracing::warn!("Frame not rendered; missing SBT");
        return;
    };
    let Some(per_instance_mutex) = tlas.tlas_per_instance_data.as_ref() else {
        tracing::warn!("Frame not rendered; missing per-instance data buffer");
        return;
    };
    let Some(tlas) = tlas.get() else {
        tracing::warn!("Frame not rendered; missing TLAS");
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        tracing::warn!("Frame not rendered; missing HDR target");
        return;
    };
    let Some(sharc_resources) = sharc_resources.as_deref_mut() else {
        tracing::warn!("Frame not rendered; missing SHARC resources");
        return;
    };
    sbt.push_raygen(0, 0, |_| {});
    sbt.push_miss(0, 1, |_| {});
    let uniform = build_camera_uniform(camera, transform, Some(&mut prev_camera), jitter.offset);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        tracing::warn!("Frame not rendered; missing atmosphere uniform buffer");
        return;
    };
    // SHARC constants for the primary closest hit's debug query. Debug fields
    // come from `SharcDebugState`; the rest mirrors the cache passes.
    let mut sharc_constants =
        sharc::build_sharc_constants(&sharc_config, &sharc_frame_state, transform.translation());
    sharc_constants.debug_mode = sharc_debug.mode.as_u32();
    sharc_constants.debug_brightness = sharc_debug.brightness;
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        // HDR intermediary at binding 1
        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // Lock sky atmosphere LUTs for reading in the miss shader
        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        // HDR target: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.hdr_output,
            &mut hdr.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.albedo,
            &mut hdr.albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Normal G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.normal,
            &mut hdr.normal_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Specular-albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.specular_albedo,
            &mut hdr.specular_albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Depth G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.depth,
            &mut hdr.depth_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Motion vectors G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            &render_target_views.motion_vectors,
            &mut hdr.motion_vectors_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // SDR target: read for the occlusion check against egui content, and
        // written by the primary closest-hit when a SHARC per-surface debug
        // overlay is active (it paints the visualization into this layer).
        encoder.use_image_resource(
            &render_target_views.sdr_target,
            &mut hdr.sdr_target_state,
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

        // SHARC cache buffers for the primary closest-hit debug query. Read-only
        // here; Update/Resolve declare R/W on the same buffers, so the tracker
        // inserts a barrier from last frame's Resolve. The candidate pool is not
        // touched by the query, so it is left unbound.
        let sharc_hash_buf = encoder.lock(
            &sharc_resources.hash_entries,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_accum_buf = encoder.lock(
            &sharc_resources.accumulation,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_resolved_buf = encoder.lock(
            &sharc_resources.resolved,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_constants_buf =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&sharc_constants));
        let sharc_rt_read = Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_buffer_resource(
            sharc_hash_buf,
            &mut sharc_resources.hash_entries_state,
            sharc_rt_read,
        );
        encoder.use_buffer_resource(
            sharc_accum_buf,
            &mut sharc_resources.accumulation_state,
            sharc_rt_read,
        );
        encoder.use_buffer_resource(
            sharc_resolved_buf,
            &mut sharc_resources.resolved_state,
            sharc_rt_read,
        );

        encoder.emit_barriers();

        let mut params = PbrPipelineParams::new();
        params
            .scene_bvh(tlas)
            .uniforms(uniform)
            .output_texture(render_target_views.hdr_output.full_view())
            .gbuffer_albedo_linear(render_target_views.albedo.linear_view())
            .gbuffer_albedo_srgb(render_target_views.albedo.srgb_view())
            .gbuffer_albedo_uint(render_target_views.albedo.uint_view())
            .gbuffer_normal_texture(render_target_views.normal.full_view())
            .gbuffer_depth_texture(render_target_views.depth.full_view())
            .gbuffer_occlusion_texture(render_target_views.sdr_target.linear_view())
            .gbuffer_motion_vector_texture(render_target_views.motion_vectors.full_view())
            .gbuffer_specular_albedo(render_target_views.specular_albedo.full_view())
            .sky_atmosphere_params(atmo_buffer)
            .sky_transmittance_lut(transmittance_view.full_view())
            .sky_sky_view_lut(sky_view.full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .per_instance_data(per_instance_buf)
            // Bind the SHARC buffers through the generated helper too, so their
            // binding numbers track the shader reflection automatically instead of
            // hand-numbered WriteDescriptorSets.
            .sharc_g_sharc_hash_entries(sharc_hash_buf)
            .sharc_g_sharc_accumulation(sharc_accum_buf)
            .sharc_g_sharc_resolved(sharc_resolved_buf)
            .sharc_g_sharc_constants(sharc_constants_buf);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            params.as_slice(),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "primary ray", |encoder| {
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

pub struct HdrRenderTargetViews {
    // R16G16B16A16_SFLOAT. Stores noisy raw light.
    pub hdr_output: FullImageView<Image>,
    // R8G8B8A8_UNORM. Stores sRGB UI elements.
    pub sdr_target: SrgbImageView<Image>,
    /// R8G8B8A8_SRGB. Stores albedo.
    pub albedo: UintImageView<SrgbImageView<Image>>,
    /// R16G16B16A16_SFLOAT
    pub normal: FullImageView<Image>,
    /// R32_SFLOAQT
    pub depth: FullImageView<Image>,
    /// R16G16_SFLOAT. Screen-space motion vectors in pixels
    /// (currentPixel - prevPixel). Written by the primary RT pass.
    pub motion_vectors: FullImageView<Image>,
    /// R8G8B8A8_UNORM. Stand-in specular albedo for DLSS-RR. Cleared to 0
    /// each frame in `dlss_evaluate` until the renderer produces a real
    /// specular term — voxel materials are matte today, so black is a
    /// physically reasonable placeholder.
    pub specular_albedo: FullImageView<Image>,

    // R16G16B16A16_SFLOAT. Stores denoised (and potentially upscaled) raw light.
    pub hdr_denoised_output: FullImageView<Image>,
}

#[derive(Resource)]
pub struct HdrRenderTarget {
    pub view: GPUMutex<HdrRenderTargetViews>,
    pub state: ResourceState,
    pub albedo_state: ResourceState,
    pub normal_state: ResourceState,
    pub depth_state: ResourceState,
    pub sdr_target_state: ResourceState,
    pub hdr_target_state: ResourceState,
    pub hdr_denoised_target_state: ResourceState,
    pub motion_vectors_state: ResourceState,
    pub specular_albedo_state: ResourceState,
    /// Display/output resolution (matches the swapchain). DLSS *output* targets
    /// (`hdr_denoised_output`, `sdr_target`) and the tonemap pass use this.
    pub display_extent: UVec2,
    /// Internal render resolution (≤ `extent`), resolved from the active
    /// [`RenderQualitySettings`] via NGX optimal settings. The RT passes and the
    /// DLSS *input* G-buffers are sized to this; DLSS upscales to `extent`.
    pub render_extent: UVec2,
}

/// User-facing DLSS render-quality / upscaling mode. Selects the internal
/// render resolution relative to the display resolution; the exact render
/// extent is resolved per display size via NGX's optimal-settings query
/// (`NgxContext::get_optimal_settings`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscalerQualityMode {
    /// 0.33× render scale.
    UltraPerformance,
    /// 0.5× render scale. Matches NVIDIA's published "Performance" timings.
    #[default]
    Performance,
    /// ~0.58× render scale.
    Balanced,
    /// ~0.667× render scale.
    Quality,
    /// ~0.77× render scale.
    UltraQuality,
    /// Native render resolution — antialiasing only, no upscaling.
    Dlaa,
}

impl UpscalerQualityMode {
    /// Maps to the NGX `PerfQuality` value used for both the optimal-settings
    /// query and DLSS feature creation.
    pub fn to_ngx(self) -> dust_denoiser::dlss::sys::NVSDK_NGX_PerfQuality_Value {
        use dust_denoiser::dlss::sys::NVSDK_NGX_PerfQuality_Value as Q;
        match self {
            Self::UltraPerformance => Q::UltraPerformance,
            Self::Performance => Q::MaxPerf,
            Self::Balanced => Q::Balanced,
            Self::Quality => Q::MaxQuality,
            Self::UltraQuality => Q::UltraQuality,
            Self::Dlaa => Q::DLAA,
        }
    }
}

/// Pure user-controlled render-quality settings. Holds intent only — the
/// derived render resolution lives on [`HdrRenderTarget::render_extent`], since
/// it depends on the (non-user-settable) display/swapchain resolution.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct RenderQualitySettings {
    pub quality: UpscalerQualityMode,
}

#[derive(Resource)]
pub struct PbrRenderState {
    pub pipeline: Handle<RayTracingPipeline>,
    pub shadow_pipeline: Handle<RayTracingPipeline>,
    pub final_gather_pipeline: Handle<RayTracingPipeline>,
    pub tonemap_pipeline: Handle<ComputePipeline>,
    pub sbt: Option<ShaderBindingTable>,
    pub shadow_sbt: Option<ShaderBindingTable>,
    pub final_gather_sbt: Option<ShaderBindingTable>,
}

#[derive(Default)]
struct FrameCounter(u32);

/// Sub-pixel jitter for the current frame, in pixels. Refreshed once per
/// frame by `advance_jitter` and read by `render`, `final_gather_pass` (kept
/// at zero — final gather doesn't reproject), and `dlss_evaluate`.
#[derive(Resource)]
pub(crate) struct JitterState {
    pub(crate) frame_count: u32,
    pub(crate) offset: Vec2,
}
impl Default for JitterState {
    fn default() -> Self {
        let mut state = JitterState {
            frame_count: 0,
            offset: Vec2::ZERO,
        };
        state.advance_jitter();
        state
    }
}
impl JitterState {
    /// Length of the jitter window. NVIDIA recommends ≥ 32 jitter positions; we
    /// use 64 to match the codebase's existing 64-layer blue-noise convention.
    const JITTER_SAMPLE_COUNT: u32 = 256;

    pub fn advance_jitter(&mut self) {
        /// Radical-inverse / van der Corput sample for the Halton sequence at `index`
        /// in the given `base`. Returns a value in `[0, 1)`.
        fn halton(mut index: u32, base: u32) -> f32 {
            let mut f = 1.0_f32;
            let mut r = 0.0_f32;
            let inv_base = 1.0_f32 / base as f32;
            while index > 0 {
                f *= inv_base;
                r += f * (index % base) as f32;
                index /= base;
            }
            r
        }

        let i = self.frame_count % Self::JITTER_SAMPLE_COUNT;
        self.offset = Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5);
        self.frame_count = self.frame_count.wrapping_add(1);
    }
}

pub fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
) {
    let base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/pbr/shaders/pbr.rtx.pipeline.bin");
    let shadow_base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/pbr/shaders/shadow.rtx.pipeline.bin");
    let final_gather_base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/pbr/shaders/final_gather.rtx.pipeline.bin");

    let tonemap_pipeline: Handle<ComputePipeline> =
        asset_server.load("bazel://dust/crates/pbr/shaders/tonemap.comp.pipeline.bin");

    let res = PbrRenderState {
        pipeline: pipeline_manager.add_pipeline(),
        shadow_pipeline: pipeline_manager.add_pipeline(),
        final_gather_pipeline: pipeline_manager.add_pipeline(),
        tonemap_pipeline,
        sbt: None,
        shadow_sbt: None,
        final_gather_sbt: None,
    };
    pipeline_manager.add_library_for_pipeline(&res.pipeline, base_library);
    pipeline_manager.add_library_for_pipeline(&res.shadow_pipeline, shadow_base_library);
    pipeline_manager
        .add_library_for_pipeline(&res.final_gather_pipeline, final_gather_base_library);

    commands.insert_resource(res);
}

pub fn create_sbt(mut state: ResMut<PbrRenderState>, pipelines: Res<Assets<RayTracingPipeline>>) {
    if let Some(pipeline) = pipelines.get(&state.pipeline) {
        state.sbt = Some(pipeline.create_sbt(state.sbt.take()));
    } else {
        state.sbt = None;
    }
    if let Some(pipeline) = pipelines.get(&state.shadow_pipeline) {
        state.shadow_sbt = Some(pipeline.create_sbt(state.shadow_sbt.take()));
    } else {
        state.shadow_sbt = None;
    }
    if let Some(pipeline) = pipelines.get(&state.final_gather_pipeline) {
        state.final_gather_sbt = Some(pipeline.create_sbt(state.final_gather_sbt.take()));
    } else {
        state.final_gather_sbt = None;
    }
}

/// Holds the active DLSS-RR feature handle and the resolutions it was
/// configured for. Recreated whenever either the render (input) or output
/// (display) extent changes — i.e. on a swapchain resize or a quality-mode
/// switch.
#[derive(Resource, Default)]
pub struct DlssState {
    pub feature: Option<dust_denoiser::dlss::NgxFeature>,
    /// Internal render resolution (`InWidth`/`InHeight`) the feature was built for.
    pub configured_render_extent: UVec2,
    /// Display/output resolution (`InTargetWidth`/`InTargetHeight`) it was built for.
    pub configured_output_extent: UVec2,
}

fn ensure_dlss_feature(
    mut ctx: SubmissionState,
    mut ngx: ResMut<dust_denoiser::dlss::NgxContext>,
    state: Option<ResMut<DlssState>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    quality_settings: Res<RenderQualitySettings>,
    mut commands: Commands,
) {
    ngx.check_dlss_rr_available().unwrap();
    let Some(hdr) = hdr_target else { return };
    if hdr.display_extent.x == 0
        || hdr.display_extent.y == 0
        || hdr.render_extent.x == 0
        || hdr.render_extent.y == 0
    {
        return;
    }

    let needs_create = match state.as_deref() {
        Some(s) => {
            s.configured_render_extent != hdr.render_extent
                || s.configured_output_extent != hdr.display_extent
                || s.feature.is_none()
        }
        None => true,
    };
    if !needs_create {
        return;
    }

    // Drop any previous feature before recording the new one. NGX records
    // initialization commands into the active command buffer, so this must
    // happen during a SubmissionState::record callback.
    let mut state = state;
    if let Some(s) = state.as_mut() {
        s.feature = None;
    }

    let create_params = dust_denoiser::dlss::sys::NVSDK_NGX_DLSSD_Create_Params {
        InDenoiseMode: dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Denoise_Mode::DLUnified, // DL based unified upscaler
        InRoughnessMode: dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Roughness_Mode::Packed, // Read roughness from normals.w
        InUseHWDepth: dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Depth_Type::HW,
        InWidth: hdr.render_extent.x,
        InHeight: hdr.render_extent.y,
        InTargetWidth: hdr.display_extent.x,
        InTargetHeight: hdr.display_extent.y,
        InPerfQualityValue: quality_settings.quality.to_ngx(),
        // MVLowRes is required by DLSS-RR (the SwinDenoiser refuses to
        // initialize without it): motion vectors are sampled at the input /
        // render resolution rather than the output resolution.
        // DepthInverted matches the reverse-infinite-Z encoding written by the
        // RT closest-hit shader (depth = near / linearViewZ, so 0 = far,
        // 1 = near plane).
        InFeatureCreateFlags: dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Feature_Flags::IsHDR
            | dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Feature_Flags::MVLowRes
            | dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Feature_Flags::DepthInverted,
        InEnableOutputSubrects: false,
    };

    let mut result = None;
    ctx.record(|encoder| {
        encoder.emit_barriers();
        let cmd_buffer = encoder.buffer().vk_handle();
        result = Some(ngx.create_dlssd_feature(cmd_buffer, &create_params));
    });

    match result {
        Some(Ok(feature)) => {
            tracing::info!(
                target: "ngx",
                render = ?hdr.render_extent,
                output = ?hdr.display_extent,
                "Created DLSS-RR feature"
            );
            if let Some(mut s) = state {
                s.feature = Some(feature);
                s.configured_render_extent = hdr.render_extent;
                s.configured_output_extent = hdr.display_extent;
            } else {
                commands.insert_resource(DlssState {
                    feature: Some(feature),
                    configured_render_extent: hdr.render_extent,
                    configured_output_extent: hdr.display_extent,
                });
            }
        }
        Some(Err(e)) => {
            tracing::error!(
                target: "ngx",
                extent = ?hdr.display_extent,
                "DLSS-RR feature creation failed: {e}"
            );
        }
        None => {}
    }
}

/// Records a DLSS-RR evaluate dispatch for the current frame.
///
/// Wraps the existing G-buffers as `NVSDK_NGX_Resource_VK` descriptors and
/// drives the denoiser into [`DlssState::output`]. Required inputs that the
/// engine does not yet produce (motion vectors, specular albedo, roughness,
/// hit distances, jitter) are left as their NGX-default null/zero values —
/// NGX will return `FAIL_MissingInput` until those are wired up.
pub(crate) fn dlss_evaluate(
    mut ctx: SubmissionState,
    ngx: Option<ResMut<dust_denoiser::dlss::NgxContext>>,
    state: Option<ResMut<DlssState>>,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut jitter: ResMut<JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let (Some(mut ngx), Some(mut state), Some(mut hdr_target)) = (ngx, state, hdr_target) else {
        return;
    };
    if state.feature.is_none() {
        return;
    }
    let extent = hdr_target.display_extent;
    let render_extent = hdr_target.render_extent;
    if extent.x == 0 || extent.y == 0 || render_extent.x == 0 || render_extent.y == 0 {
        return;
    }

    ctx.record(move |encoder| {
        let DlssState { feature, .. } = &mut *state;
        let feature = feature.as_ref().unwrap();
        let hdr = &mut *hdr_target;

        let render_target_views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);

        let read_access = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        let write_access = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
        };

        encoder.use_image_resource(
            &render_target_views.hdr_output,
            &mut hdr.state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            &render_target_views.albedo,
            &mut hdr.albedo_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            &render_target_views.normal,
            &mut hdr.normal_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            &render_target_views.depth,
            &mut hdr.depth_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            &render_target_views.motion_vectors,
            &mut hdr.motion_vectors_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            &render_target_views.hdr_denoised_output,
            &mut hdr.hdr_denoised_target_state,
            write_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        encoder.emit_barriers();

        encoder.use_image_resource(
            &render_target_views.specular_albedo,
            &mut hdr.specular_albedo_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let subres = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let mut color = ngx_image_resource(
            render_target_views.hdr_output.full_view().vk_handle(),
            render_target_views.hdr_output.vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            render_extent.x,
            render_extent.y,
            false,
        );
        let mut output_resource = ngx_image_resource(
            render_target_views.hdr_denoised_output.full_view().vk_handle(),
            render_target_views.hdr_denoised_output.vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            extent.x,
            extent.y,
            true,
        );
        let mut depth = ngx_image_resource(
            render_target_views.depth.full_view().vk_handle(),
            render_target_views.depth.vk_handle(),
            subres,
            vk::Format::R32_SFLOAT,
            render_extent.x,
            render_extent.y,
            false,
        );
        let mut normals = ngx_image_resource(
            render_target_views.normal.full_view().vk_handle(),
            render_target_views.normal.vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            render_extent.x,
            render_extent.y,
            false,
        );
        let mut diffuse_albedo = ngx_image_resource(
            render_target_views.albedo.srgb_view().vk_handle(),
            render_target_views.albedo.vk_handle(),
            subres,
            vk::Format::R8G8B8A8_SRGB,
            render_extent.x,
            render_extent.y,
            false,
        );
        let mut motion_vectors = ngx_image_resource(
            render_target_views.motion_vectors.full_view().vk_handle(),
            render_target_views.motion_vectors.vk_handle(),
            subres,
            vk::Format::R16G16_SFLOAT,
            render_extent.x,
            render_extent.y,
            false,
        );
        let mut specular_albedo = ngx_image_resource(
            render_target_views.specular_albedo.full_view().vk_handle(),
            render_target_views.specular_albedo.vk_handle(),
            subres,
            vk::Format::R8G8B8A8_UNORM,
            render_extent.x,
            render_extent.y,
            false,
        );

        let mut eval_params = dust_denoiser::dlss::sys::NVSDK_NGX_VK_DLSSD_Eval_Params::zeroed();
        eval_params.pInColor = &mut color;
        eval_params.pInOutput = &mut output_resource;
        eval_params.pInDepth = &mut depth;
        eval_params.pInNormals = &mut normals;
        eval_params.pInDiffuseAlbedo = &mut diffuse_albedo;
        eval_params.pInSpecularAlbedo = &mut specular_albedo;
        eval_params.pInMotionVectors = &mut motion_vectors;
        // The valid render subrect is the internal render resolution; DLSS-RR
        // upscales it into the (display-resolution) output target.
        eval_params.InRenderSubrectDimensions = dust_denoiser::dlss::sys::NVSDK_NGX_Dimensions {
            Width: render_extent.x,
            Height: render_extent.y,
        };
        eval_params.InMVScaleX = 1.0;
        eval_params.InMVScaleY = 1.0;
        // Sub-pixel jitter applied to this frame's primary ray, in pixels.
        // Must match the value baked into the camera uniform consumed by the
        // RT pass (see `advance_jitter`).
        eval_params.InJitterOffsetX = jitter.offset.x;
        eval_params.InJitterOffsetY = jitter.offset.y;
        jitter.advance_jitter();
        // No history yet — flag every frame as a teleport so DLSS-RR
        // discards its temporal accumulation.
        eval_params.InReset = 0;

        encoder.timing_scope(profiler.as_deref_mut(), "DLSS eval", |encoder| {
            let cmd_buffer = encoder.buffer().vk_handle();
            if let Err(e) = ngx.evaluate_dlssd(cmd_buffer, feature, &mut eval_params) {
                tracing::error!(target: "ngx", "DLSS-RR evaluate failed: {e}");
            }
        });
    });
}

fn ngx_image_resource(
    image_view: vk::ImageView,
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
    read_write: bool,
) -> dust_denoiser::dlss::sys::NVSDK_NGX_Resource_VK {
    dust_denoiser::dlss::sys::NVSDK_NGX_Resource_VK {
        Resource: dust_denoiser::dlss::sys::NVSDK_NGX_Resource_VK_Resource {
            ImageViewInfo: dust_denoiser::dlss::sys::NVSDK_NGX_ImageViewInfo_VK {
                ImageView: image_view,
                Image: image,
                SubresourceRange: subresource_range,
                Format: format,
                Width: width,
                Height: height,
            },
        },
        Type: dust_denoiser::dlss::sys::NVSDK_NGX_Resource_VK_Type::ImageView,
        ReadWrite: read_write,
    }
}

fn ensure_hdr_target(
    mut commands: Commands,
    hdr_target: Option<Res<HdrRenderTarget>>,
    allocator: Res<Allocator>,
    swapchain_images: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut perf_panel: Option<ResMut<PerformancePanel>>,
    quality_settings: Res<RenderQualitySettings>,
    mut ngx: Option<ResMut<dust_denoiser::dlss::NgxContext>>,
) {
    let Ok(swapchain_image) = swapchain_images.single() else {
        return;
    };
    let Some(current) = swapchain_image.current_image() else {
        return;
    };
    let extent = UVec2::new(current.extent().x, current.extent().y);

    // Reallocate when the display resolution changes or the user picks a new
    // quality mode. Both alter the internal render resolution; otherwise the
    // (relatively expensive) optimal-settings query is skipped entirely.
    let needs_realloc = match hdr_target.as_ref() {
        Some(hdr) => hdr.display_extent != extent || quality_settings.is_changed(),
        None => true,
    };
    if !needs_realloc {
        return;
    }

    // Resolve the internal render resolution for the active quality mode. Falls
    // back to native (no upscaling) when DLSS is unavailable (non-NVIDIA GPU)
    // or the query fails — DLAA also returns render == display here.
    let quality = quality_settings.quality;
    let render_extent = match ngx.as_deref_mut() {
        Some(ngx) => match ngx.get_optimal_settings(quality.to_ngx(), extent.x, extent.y) {
            Ok(s) => UVec2::new(s.render_extent[0], s.render_extent[1]),
            Err(e) => {
                tracing::warn!(
                    target: "ngx",
                    "DLSS optimal-settings query failed ({e}); rendering at native resolution"
                );
                extent
            }
        },
        None => extent,
    };

    if let Some(panel) = perf_panel.as_deref_mut() {
        panel.report_display_resolutions(extent);
        panel.report_render_resolutions(render_extent);
    }

    // Shared image parameters; only `extent` (and per-image format/usage/flags)
    // differ between the display-resolution and render-resolution targets.
    let base_create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    // Display-resolution targets: DLSS output + the egui overlay + tonemap.
    let display_create_info = vk::ImageCreateInfo {
        extent: vk::Extent3D {
            width: extent.x,
            height: extent.y,
            depth: 1,
        },
        ..base_create_info
    };
    // Render-resolution targets: every RT-pass output / DLSS input G-buffer.
    let render_create_info = vk::ImageCreateInfo {
        extent: vk::Extent3D {
            width: render_extent.x,
            height: render_extent.y,
            depth: 1,
        },
        ..base_create_info
    };

    let hdr_output = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"HDR Render Target")
    .create_full_view()
    .unwrap()
    .with_name(c"HDR Render Target View");

    let hdr_denoised_output = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..display_create_info
        },
    )
    .unwrap()
    .with_name(c"HDR Denoised Render Target")
    .create_full_view()
    .unwrap()
    .with_name(c"HDR Denoised Render Target View");

    let sdr_target = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            ..display_create_info
        }
        .push(
            &mut vk::ImageFormatListCreateInfo::default()
                .view_formats(&[vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM]),
        ),
    )
    .unwrap()
    .with_name(c"SDR Render Target")
    .create_srgb_view(
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::ImageUsageFlags::SAMPLED,
    )
    .unwrap();

    let albedo = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
            format: vk::Format::R8G8B8A8_UNORM,
            ..render_create_info
        }
        .push(
            &mut vk::ImageFormatListCreateInfo::default().view_formats(&[
                vk::Format::R8G8B8A8_SRGB,
                vk::Format::R8G8B8A8_UNORM,
                vk::Format::R8G8B8A8_UINT,
            ]),
        ),
    )
    .unwrap()
    .with_name(c"G-Buffer Albedo Image")
    .create_srgb_view(vk::ImageUsageFlags::STORAGE, vk::ImageUsageFlags::SAMPLED)
    .unwrap()
    .create_uint_view(vk::ImageUsageFlags::STORAGE)
    .unwrap();

    let normal = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"G-Buffer Normal Image")
    .create_full_view()
    .unwrap()
    .with_name(c"G-Buffer Normal Image View");

    let depth = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R32_SFLOAT,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"G-Buffer Depth")
    .create_full_view()
    .unwrap()
    .with_name(c"G-Buffer Depth View");

    let motion_vectors = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16_SFLOAT,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"G-Buffer Motion Vectors")
    .create_full_view()
    .unwrap()
    .with_name(c"G-Buffer Motion Vectors View");

    let specular_albedo = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"DLSS-RR Specular Albedo Stand-in")
    .create_full_view()
    .unwrap()
    .with_name(c"DLSS-RR Specular Albedo Stand-in View");

    let view = GPUMutex::new(HdrRenderTargetViews {
        hdr_output,
        sdr_target,
        albedo,
        normal,
        depth,
        motion_vectors,
        specular_albedo,
        hdr_denoised_output,
    });

    commands.insert_resource(HdrRenderTarget {
        view,
        state: Default::default(),
        albedo_state: Default::default(),
        normal_state: Default::default(),
        depth_state: Default::default(),
        sdr_target_state: Default::default(),
        hdr_target_state: Default::default(),
        motion_vectors_state: Default::default(),
        specular_albedo_state: Default::default(),
        display_extent: extent,
        render_extent,
        hdr_denoised_target_state: Default::default(),
    });
}

fn shadow_pass(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS<PbrInstanceData>>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    jitter: Res<JitterState>,
    sharc_debug: Res<sharc::SharcDebugState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok((camera, transform)) = swapchain_images.single() else {
        return;
    };
    let Some(pipeline) = pipelines
        .get(&state.shadow_pipeline)
        .map(|x| x.deref().clone())
    else {
        return;
    };
    let Some(shadow_sbt) = state.shadow_sbt.as_mut() else {
        return;
    };
    let Some(per_instance_mutex) = tlas.tlas_per_instance_data.as_ref() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    shadow_sbt.push_raygen(0, 0, |_| {});
    shadow_sbt.push_miss(0, 1, |_| {});
    // Shadow pass doesn't reproject, but its ray-gen reconstructs world
    // position from the depth G-buffer via `primaryRayDirWorldSpace` — that
    // depth was written by the jittered primary ray, so we must reuse the
    // same jitter or the reconstructed shading point drifts off the surface.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, shadow_sbt.layout(), |slice| {
                shadow_sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // HDR: read-write (miss shader reads current value and adds sun contribution)
        encoder.use_image_resource(
            &render_target_views.hdr_output,
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
        // Albedo: sampled read through sRGB view (written by primary pass)
        encoder.use_image_resource(
            &render_target_views.albedo,
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
        // Normal: read (written by primary pass)
        encoder.use_image_resource(
            &render_target_views.normal,
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
        // Specular albedo: read (written by primary pass)
        encoder.use_image_resource(
            &render_target_views.specular_albedo,
            &mut hdr.specular_albedo_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Depth: read (written by primary pass)
        encoder.use_image_resource(
            &render_target_views.depth,
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
        // Lock sky atmosphere LUTs for reading in the miss shader
        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
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
                .output_texture(render_target_views.hdr_output.full_view())
                .gbuffer_albedo_linear(render_target_views.albedo.linear_view())
                .gbuffer_albedo_srgb(render_target_views.albedo.srgb_view())
                .gbuffer_albedo_uint(render_target_views.albedo.uint_view())
                .gbuffer_normal_texture(render_target_views.normal.full_view())
                .gbuffer_depth_texture(render_target_views.depth.full_view())
                .gbuffer_occlusion_texture(render_target_views.sdr_target.linear_view())
                .gbuffer_motion_vector_texture(render_target_views.motion_vectors.full_view())
                .gbuffer_specular_albedo(render_target_views.specular_albedo.full_view())
                .sky_atmosphere_params(atmo_buffer)
                .sky_transmittance_lut(transmittance_view.full_view())
                .sky_sky_view_lut(sky_view.full_view())
                .sky_linear_sampler(&atmosphere_luts.sampler)
                .per_instance_data(per_instance_buf)
                .as_slice(),
        );
        encoder.timing_scope(profiler.as_deref_mut(), "shadow ray", |encoder| {
            encoder.trace_rays(
                shadow_sbt,
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

pub(crate) fn final_gather_pass(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS<PbrInstanceData>>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut frame_index: Local<u32>,
    jitter: Res<JitterState>,
    sharc_debug: Res<sharc::SharcDebugState>,
    sharc_config: Res<sharc::SharcConfig>,
    sharc_frame_state: Res<sharc::SharcFrameState>,
    mut sharc_resources: Option<ResMut<sharc::SharcResources>>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok((camera, transform)) = swapchain_images.single() else {
        return;
    };
    let Some(pipeline) = pipelines
        .get(&state.final_gather_pipeline)
        .map(|x| x.deref().clone())
    else {
        return;
    };
    let Some(gather_sbt) = state.final_gather_sbt.as_mut() else {
        return;
    };
    let Some(per_instance_mutex) = tlas.tlas_per_instance_data.as_ref() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Some(sharc_resources) = sharc_resources.as_deref_mut() else {
        return;
    };
    gather_sbt.push_raygen(0, 0, |_| {});
    gather_sbt.push_miss(0, 1, |_| {});
    // Final-gather doesn't reproject, but it reconstructs world position from
    // the depth G-buffer via `primaryRayDirWorldSpace` — that depth was
    // written by the jittered primary ray, so we must reuse the same jitter.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let mut sharc_constants =
        sharc::build_sharc_constants(&sharc_config, &sharc_frame_state, transform.translation());
    // Carry the debug mode so the gather CHS/miss suppress their HDR writes
    // while a debug view owns the output. Final-gather still runs in that case
    // to keep the cache seeded (its CHS keeps pushing candidates on misses).
    sharc_constants.debug_mode = sharc_debug.mode.as_u32();
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    let push = sharc::FinalGatherPush::new(*frame_index);
    *frame_index = frame_index.wrapping_add(1);
    // Ping-pong pool indices for this frame.
    let (sharc_pool_read_idx, sharc_pool_write_idx) =
        sharc::SharcResources::pool_indices(sharc_frame_state.frame_index);
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, gather_sbt.layout(), |slice| {
                gather_sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // HDR: read-write (additive indirect contribution)
        encoder.use_image_resource(
            &render_target_views.hdr_output,
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
        // Albedo: sampled read through sRGB view
        encoder.use_image_resource(
            &render_target_views.albedo,
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
        // Normal: read
        encoder.use_image_resource(
            &render_target_views.normal,
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
        // Specular albedo: read F0 (specular lobe) + write — the final-gather
        // ray-gen demodulates F0 to EnvBRDF in place afterward for DLSS-RR.
        encoder.use_image_resource(
            &render_target_views.specular_albedo,
            &mut hdr.specular_albedo_state,
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
        // Depth: read
        encoder.use_image_resource(
            &render_target_views.depth,
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
        // Sky atmosphere LUTs
        let transmittance_view = encoder.lock(
            &atmosphere_luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &atmosphere_luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
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
        // SHARC working storage. Bindings 14..17 are declared in
        // pbr.playout.ron but only used when the final-gather shader actually
        // calls into the SHARC cache; this plumbing makes them available so
        // the shader-side query can land later. Access flags are RT-stage
        // storage reads — Update/Resolve declare R/W on these buffers, so the
        // Bevy tracker inserts the necessary barriers between Resolve and the
        // final-gather query.
        let sharc_hash_buf = encoder.lock(
            &sharc_resources.hash_entries,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_accum_buf = encoder.lock(
            &sharc_resources.accumulation,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_resolved_buf = encoder.lock(
            &sharc_resources.resolved,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sharc_constants_buf =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&sharc_constants));

        // Pool buffer locks. final_gather's CHS pushes new candidates on cache
        // miss (write side); next frame's Update raygen dispatches from the
        // same pool (read side). Bind both physical buffers so the Slang
        // `SharcParams` layout is satisfied regardless of which is currently
        // the read side.
        let pool_read_candidates_buf = encoder.lock(
            &sharc_resources.pool[sharc_pool_read_idx].candidates,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_read_keys_buf = encoder.lock(
            &sharc_resources.pool[sharc_pool_read_idx].keys,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_write_candidates_buf = encoder.lock(
            &sharc_resources.pool[sharc_pool_write_idx].candidates,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let pool_write_keys_buf = encoder.lock(
            &sharc_resources.pool[sharc_pool_write_idx].keys,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        let sharc_rt_read = Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        let sharc_rt_write = Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        };
        encoder.use_buffer_resource(
            sharc_hash_buf,
            &mut sharc_resources.hash_entries_state,
            sharc_rt_read,
        );
        encoder.use_buffer_resource(
            sharc_accum_buf,
            &mut sharc_resources.accumulation_state,
            sharc_rt_read,
        );
        encoder.use_buffer_resource(
            sharc_resolved_buf,
            &mut sharc_resources.resolved_state,
            sharc_rt_read,
        );
        // Split pool[0]/pool[1] mutably without violating borrow rules.
        let [pool_a, pool_b] = &mut sharc_resources.pool;
        let (pool_read, pool_write) = if sharc_pool_read_idx == 0 {
            (pool_a, pool_b)
        } else {
            (pool_b, pool_a)
        };
        encoder.use_buffer_resource(
            pool_read_candidates_buf,
            &mut pool_read.candidates_state,
            sharc_rt_read,
        );
        encoder.use_buffer_resource(pool_read_keys_buf, &mut pool_read.keys_state, sharc_rt_read);
        encoder.use_buffer_resource(
            pool_write_candidates_buf,
            &mut pool_write.candidates_state,
            sharc_rt_write,
        );
        encoder.use_buffer_resource(
            pool_write_keys_buf,
            &mut pool_write.keys_state,
            sharc_rt_write,
        );

        encoder.memory_barrier(
            Access::COPY_WRITE,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            },
        );

        encoder.emit_barriers();

        let mut params = PbrPipelineParams::new();
        params
            .scene_bvh(tlas)
            .uniforms(uniform)
            .output_texture(render_target_views.hdr_output.full_view())
            .gbuffer_albedo_linear(render_target_views.albedo.linear_view())
            .gbuffer_albedo_srgb(render_target_views.albedo.srgb_view())
            .gbuffer_normal_texture(render_target_views.normal.full_view())
            .gbuffer_depth_texture(render_target_views.depth.full_view())
            .gbuffer_occlusion_texture(render_target_views.sdr_target.linear_view())
            .gbuffer_motion_vector_texture(render_target_views.motion_vectors.full_view())
            .gbuffer_specular_albedo(render_target_views.specular_albedo.full_view())
            .sky_atmosphere_params(atmo_buffer)
            .sky_transmittance_lut(transmittance_view.full_view())
            .sky_sky_view_lut(sky_view.full_view())
            .sky_linear_sampler(&atmosphere_luts.sampler)
            .per_instance_data(per_instance_buf)
            // SHARC working storage + candidate pool via the generated helper, so
            // their binding numbers track the shader reflection automatically.
            .sharc_g_sharc_hash_entries(sharc_hash_buf)
            .sharc_g_sharc_accumulation(sharc_accum_buf)
            .sharc_g_sharc_resolved(sharc_resolved_buf)
            .sharc_g_sharc_constants(sharc_constants_buf)
            .sharc_g_sharc_candidates_read(pool_read_candidates_buf)
            .sharc_g_sharc_keys_read(pool_read_keys_buf)
            .sharc_g_sharc_candidates_write(pool_write_candidates_buf)
            .sharc_g_sharc_keys_write(pool_write_keys_buf);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            params.as_slice(),
        );

        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&push),
        );

        encoder.timing_scope(profiler.as_deref_mut(), "final gather ray", |encoder| {
            encoder.trace_rays(
                gather_sbt,
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

#[derive(Default, SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
pub struct OccludingRenderPass;

/// For egui, you are always responsible for setting up the render pass. This
/// is so that egui can piggy-back on an already existing render pass and doesn't
/// have to start a new one - important for mobile performance.
fn start_occluding_render_pass(
    mut ctx: SubmissionState,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
) {
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    ctx.record(|encoder| {
        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        encoder.use_image_resource(
            &render_target_views.sdr_target,
            &mut hdr.sdr_target_state,
            Access::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            true,
        );
        encoder.emit_barriers();
        encoder
            .begin_rendering()
            .color_attachment(0, |mut builder| {
                builder
                    .clear(Vec4::new(0.0, 0.0, 0.0, 0.0))
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(render_target_views.sdr_target.linear_view());
                // Use linear view for egui. egui does all the interpolation in srgb space.
            })
            .render_area(
                IVec2::ZERO,
                UVec2::new(hdr.display_extent.x, hdr.display_extent.y),
            )
            .begin();
    });
}
