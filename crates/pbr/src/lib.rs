pub mod camera;
pub mod hzb;
pub mod sharc;
pub mod sky;
pub mod tonemap;
pub mod upscaler;

include!(concat!(
    env!("BAZEL_BIN"),
    "/crates/pbr/shaders/pbr_module_layout.rs"
));

use std::ops::Deref;


use tonemap::{autoexposure_pass, tonemap_pass};

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
use pumicite::{
    Allocator, HasDevice, Sampler,
    ash::vk::{self, TaggedStructure},
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, SrgbImageView, SwizzledImageView},
    rtx::ShaderBindingTable,
    sync::GPUMutex,
    tracking::{Access, ResourceState},
    utils::glam_to_vk_transform,
};
use pumicite::{device::DeviceBuilder, image::UintImageView};
use pumicite_egui::{EguiPrimaryContextPass, EguiRenderSet};

use dust_gfxdebug::{GpuProfiler, GpuTimerCommands, PerformancePanel};

use bevy_pumicite::prelude::ComputePipeline;

use crate::{
    camera::Camera,
    sky::{AtmosphereLUTs, SkyAtmosphereLUTRenderSet},
    upscaler::{
        Upscaler, UpscalerSessionState, ensure_upscaler_session, setup_upscaler,
        upscaler_evaluate,
    },
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
        app.add_systems(Startup, setup_upscaler.after(CreateDevice));
        app.add_systems(Startup, hzb::setup_hzb.after(CreateDevice));
        app.init_resource::<JitterState>();
        app.init_resource::<RenderQualitySettings>();
        app.init_resource::<UpscalerSessionState>();
        app.add_systems(
            PostUpdate,
            (
                create_sbt.before(PbrRenderSet),
                ensure_hdr_target.in_set(SwapchainSet).before(render),
                ensure_upscaler_session
                    .in_set(DefaultRenderSet)
                    .after(ensure_hdr_target)
                    .before(render),
                advance_jitter
                    .in_set(DefaultRenderSet)
                    .after(ensure_upscaler_session)
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
                hzb::ensure_hzb_resources
                    .in_set(DefaultRenderSet)
                    .after(ensure_hdr_target),
                hzb::hzb_build_pass
                    .in_set(DefaultRenderSet)
                    .after(render)
                    .after(hzb::ensure_hzb_resources),
                shadow_pass
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(render)
                    .after(hzb::hzb_build_pass)
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
                autoexposure_pass
                    .in_set(DefaultRenderSet)
                    .after(final_gather_pass),
                upscaler_evaluate
                    .in_set(DefaultRenderSet)
                    .after(ensure_upscaler_session)
                    .after(final_gather_pass)
                    .after(autoexposure_pass),
                tonemap_pass
                    .in_set(DefaultRenderSet)
                    .after(render)
                    .after(shadow_pass)
                    .after(final_gather_pass)
                    .after(upscaler_evaluate),
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
    mut sharc: sharc::SharcSystemParams,
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
    let Some(sharc_resources) = sharc.resources.as_deref_mut() else {
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
        sharc::build_sharc_constants(&sharc.config, &sharc.frame_state, transform.translation());
    sharc_constants.debug_mode = sharc.debug.mode.as_u32();
    sharc_constants.debug_brightness = sharc.debug.brightness;
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();

        let pipeline = encoder.retain(pipeline).deref_inner();
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

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
            render_target_views.hdr_output(),
            &mut hdr.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.albedo(),
            &mut hdr.albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Normal G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.normal(),
            &mut hdr.normal_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Specular-albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.specular_albedo(),
            &mut hdr.specular_albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Depth G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.depth(),
            &mut hdr.depth_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Motion vectors G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.motion_vectors(),
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
            render_target_views.sdr_target(),
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
            .output_texture(render_target_views.hdr_output().full_view())
            .gbuffer_albedo_linear(
                render_target_views
                    .albedo()
                    .deref_inner()
                    .deref_inner()
                    .full_view(),
            )
            .gbuffer_albedo_srgb(render_target_views.albedo().deref_inner().srgb_view())
            .gbuffer_albedo_uint(render_target_views.albedo().uint_view())
            .gbuffer_normal_texture(render_target_views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(render_target_views.depth().full_view())
            .gbuffer_occlusion_texture(render_target_views.sdr_target().deref_inner().full_view())
            .gbuffer_motion_vector_texture(render_target_views.motion_vectors().full_view())
            .gbuffer_specular_albedo(render_target_views.specular_albedo().full_view())
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
    // R8G8B8A8_UNORM. Stores sRGB UI elements. Full view = linear, sRGB view = decoded.
    pub sdr_target: SrgbImageView<FullImageView<Image>>,
    /// R8G8B8A8_UNORM image with linear (full), sRGB, and UINT views of albedo.
    pub albedo: UintImageView<SrgbImageView<FullImageView<Image>>>,
    /// R16G16B16A16_SFLOAT. World-space normal in `.xyz`, roughness in `.w`. The
    /// full view feeds the normal G-buffer; the alpha-broadcast swizzle view
    /// feeds the roughness G-buffer.
    pub normal: SwizzledImageView<FullImageView<Image>>,
    /// R32_SFLOAT, render resolution. Reverse-infinite-Z hardware depth
    /// (`near / linearViewZ`: 1 = near, 0 = sky; UI pixels store 1). Written
    /// by the primary pass this frame.
    pub depth: FullImageView<Image>,
    /// Last frame's `depth`, for the screen trace's history depth test.
    /// `ensure_hdr_target` swaps the two images once per frame.
    pub depth_prev: FullImageView<Image>,
    /// R16G16_SFLOAT. Screen-space motion vectors in pixels
    /// (currentPixel - prevPixel). Written by the primary RT pass.
    pub motion_vectors: FullImageView<Image>,
    /// R8G8B8A8_UNORM. Specular-albedo guide for the denoiser. Written to 0
    /// every frame by the primary pass (closest-hit, miss, and UI paths):
    /// the lighting evaluates no specular lobe today, so a zero guide is the
    /// physically correct demodulation input.
    pub specular_albedo: FullImageView<Image>,

    /// R16G16B16A16_SFLOAT, display resolution. Denoised (and potentially
    /// upscaled) raw radiance. Within a frame the screen trace reads it as
    /// last frame's lit image (the radiance source for on-screen gather
    /// hits) before the upscaler overwrites it, so one image serves as both
    /// output and history; the upscaler keeps its own temporal state.
    pub hdr_denoised_output: FullImageView<Image>,

    /// R16_SFLOAT, 1×1. Auto-exposure scale: multiplying the HDR color by this
    /// value maps the scene's metered average to 18% mid-gray. Written by
    /// `autoexposure_pass`; read by the upscaler dispatch (MetalFX
    /// `exposureTexture` / DLSS-RR `pInExposureTexture`) and the tonemap pass,
    /// so all three agree on one exposure.
    pub exposure: FullImageView<Image>,
}

pumicite::project_fields! {
    /// Per-image projections of a locked [`HdrRenderTargetViews`] token.
    pub trait HdrRenderTargetViewsExt for impl[] HdrRenderTargetViews {
        hdr_output: FullImageView<Image>,
        sdr_target: SrgbImageView<FullImageView<Image>>,
        albedo: UintImageView<SrgbImageView<FullImageView<Image>>>,
        normal: SwizzledImageView<FullImageView<Image>>,
        depth: FullImageView<Image>,
        depth_prev: FullImageView<Image>,
        motion_vectors: FullImageView<Image>,
        specular_albedo: FullImageView<Image>,
        hdr_denoised_output: FullImageView<Image>,
        exposure: FullImageView<Image>,
    }
}


#[derive(Resource)]
pub struct HdrRenderTarget {
    pub view: GPUMutex<HdrRenderTargetViews>,
    pub state: ResourceState,
    pub albedo_state: ResourceState,
    pub normal_state: ResourceState,
    pub depth_state: ResourceState,
    pub depth_prev_state: ResourceState,
    pub sdr_target_state: ResourceState,
    pub hdr_target_state: ResourceState,
    pub hdr_denoised_target_state: ResourceState,
    /// Bilinear, clamp-to-edge. Samples the history color.
    pub linear_clamp_sampler: Sampler,
    /// Nearest, clamp-to-edge. Samples the HZB and the history depth.
    pub point_clamp_sampler: Sampler,
    pub motion_vectors_state: ResourceState,
    pub specular_albedo_state: ResourceState,
    pub exposure_state: ResourceState,
    /// False until `autoexposure_pass` has written the exposure image once
    /// after (re)creation; while false the shader ignores the (undefined)
    /// previous value and adopts the metered target directly.
    pub exposure_initialized: bool,
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
    /// Output-over-input render-scale for this mode, as an exact ratio. The
    /// internal render resolution is the display resolution divided by this
    /// factor; the super-resolution engine upscales back to display resolution.
    ///
    /// The ratios match NVIDIA's published DLSS quality modes, so DLSS
    /// resolves the same internal quality it would from an optimal-settings
    /// query.
    pub fn scaling_factor(self) -> ScalingFactor {
        match self {
            Self::Dlaa => ScalingFactor {
                numerator: 1,
                denominator: 1,
            },
            Self::UltraQuality => ScalingFactor {
                numerator: 13,
                denominator: 10,
            },
            Self::Quality => ScalingFactor {
                numerator: 3,
                denominator: 2,
            },
            Self::Balanced => ScalingFactor {
                numerator: 12,
                denominator: 7,
            },
            Self::Performance => ScalingFactor {
                numerator: 2,
                denominator: 1,
            },
            Self::UltraPerformance => ScalingFactor {
                numerator: 3,
                denominator: 1,
            },
        }
    }
}

/// Output-over-input ratio of an upscale, `numerator / denominator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalingFactor {
    pub numerator: u32,
    pub denominator: u32,
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
    pub autoexposure_pipeline: Handle<ComputePipeline>,
    pub sbt: Option<ShaderBindingTable>,
    pub shadow_sbt: Option<ShaderBindingTable>,
    pub final_gather_sbt: Option<ShaderBindingTable>,
}

#[derive(Default)]
struct FrameCounter(u32);

/// Sub-pixel jitter for the current frame, in texels. Refreshed once per frame
/// by [`advance_jitter`] and read by `render`, `shadow_pass`,
/// `final_gather_pass`, and `upscaler_evaluate`. All passes that sample the
/// jittered camera must use the same value, and it must match the
/// `texel_jitter` handed to the upscaler dispatch.
#[derive(Resource, Default)]
pub(crate) struct JitterState {
    pub(crate) frame_count: u32,
    pub(crate) offset: Vec2,
}

/// Advances the temporal jitter for the frame by indexing the active session's
/// recommended jitter pattern (see
/// [`SuperResolutionSession::recommended_jitter_pattern`], surfaced via
/// [`UpscalerSessionState::jitter_pattern`]). Falls back to no jitter when no
/// session is active. Runs before every pass that reads [`JitterState`].
fn advance_jitter(mut jitter: ResMut<JitterState>, state: Res<UpscalerSessionState>) {
    if state.jitter_pattern.is_empty() {
        jitter.offset = Vec2::ZERO;
        return;
    }
    let index = jitter.frame_count as usize % state.jitter_pattern.len();
    jitter.offset = state.jitter_pattern[index];
    jitter.frame_count = jitter.frame_count.wrapping_add(1);
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
    let autoexposure_pipeline: Handle<ComputePipeline> =
        asset_server.load("bazel://dust/crates/pbr/shaders/autoexposure.comp.pipeline.bin");

    let res = PbrRenderState {
        pipeline: pipeline_manager.add_pipeline(),
        shadow_pipeline: pipeline_manager.add_pipeline(),
        final_gather_pipeline: pipeline_manager.add_pipeline(),
        tonemap_pipeline,
        autoexposure_pipeline,
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

fn ensure_hdr_target(
    mut commands: Commands,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
    allocator: Res<Allocator>,
    swapchain_images: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut perf_panel: Option<ResMut<PerformancePanel>>,
    quality_settings: Res<RenderQualitySettings>,
    upscaler: Option<Res<Upscaler>>,
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
        let mut hdr = hdr_target.unwrap();
        let hdr = &mut *hdr;
        // Swap the states
        std::mem::swap(&mut hdr.depth_state, &mut hdr.depth_prev_state);
        // SAFETY: only image handles move between fields; nothing is freed.
        let views = unsafe { hdr.view.get_mut_unchecked() };
        std::mem::swap(&mut views.depth, &mut views.depth_prev);
        return;
    }

    // Resolve the internal render resolution for the active quality mode. Falls
    // back to native (no upscaling) when no upscaler is available (e.g. a
    // non-NVIDIA GPU); DLAA also yields render == display (1:1 factor).
    let quality = quality_settings.quality;
    let render_extent = match upscaler.as_ref() {
        Some(_) => {
            let factor = quality.scaling_factor();
            // Round-to-nearest of value * denominator / numerator, clamped to
            // >= 1 — the vendor's optimal render size for this quality mode.
            let scale = |value: u32| -> u32 {
                (((value as u64 * factor.denominator as u64) + factor.numerator as u64 / 2)
                    / factor.numerator as u64)
                    .max(1) as u32
            };
            UVec2::new(scale(extent.x), scale(extent.y))
        }
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

    // The blit upscaler copies `hdr_output` into `hdr_denoised_output`.
    let hdr_output = Image::new_device(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: render_create_info.usage | vk::ImageUsageFlags::TRANSFER_SRC,
            ..render_create_info
        },
    )
    .unwrap()
    .with_name(c"HDR Render Target")
    .create_full_view()
    .unwrap()
    .with_name(c"HDR Render Target View");

    let hdr_denoised_output = Image::new_device(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: display_create_info.usage | vk::ImageUsageFlags::TRANSFER_DST,
            ..display_create_info
        },
    )
    .unwrap()
    .with_name(c"HDR Denoised Render Target")
    .create_full_view()
    .unwrap()
    .with_name(c"HDR Denoised Render Target View");

    let sdr_target = Image::new_device(
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
    .create_full_view()
    .unwrap()
    .create_srgb_view(vk::ImageUsageFlags::SAMPLED)
    .unwrap();

    let albedo = Image::new_device(
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
    .create_full_view()
    .unwrap()
    .create_srgb_view(vk::ImageUsageFlags::SAMPLED)
    .unwrap()
    .create_uint_view(vk::ImageUsageFlags::STORAGE)
    .unwrap();

    // The normal image doubles as the roughness input for the denoiser: roughness
    // is packed in `normal.w`. Alongside the full view (world-space normal) we
    // build an alpha-broadcast swizzle view (`{r,g,b,a} <- A`) that presents that
    // packed roughness on every channel, sampled as the roughness G-buffer.
    let normal = Image::new_device(
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
    .with_name(c"G-Buffer Normal Image View")
    .create_swizzled_view(
        vk::ComponentMapping {
            r: vk::ComponentSwizzle::A,
            g: vk::ComponentSwizzle::A,
            b: vk::ComponentSwizzle::A,
            a: vk::ComponentSwizzle::A,
        },
        vk::ImageUsageFlags::SAMPLED,
    )
    .unwrap();

    let [depth, depth_prev] = [c"G-Buffer Depth 0", c"G-Buffer Depth 1"].map(|name| {
        Image::new_device(
            allocator.clone(),
            &vk::ImageCreateInfo {
                format: vk::Format::R32_SFLOAT,
                ..render_create_info
            },
        )
        .unwrap()
        .with_name(name)
        .create_full_view()
        .unwrap()
        .with_name(name)
    });

    let motion_vectors = Image::new_device(
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

    let specular_albedo = Image::new_device(
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

    // 1×1 exposure scale shared by the upscaler and the tonemap pass. Sized
    // independently of both resolutions; recreated with the rest of the target
    // only for simplicity.
    let exposure = Image::new_device(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16_SFLOAT,
            extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            ..base_create_info
        },
    )
    .unwrap()
    .with_name(c"Auto-Exposure Scale")
    .create_full_view()
    .unwrap()
    .with_name(c"Auto-Exposure Scale View");

    let clamp_sampler = |filter: vk::Filter, mipmap_mode: vk::SamplerMipmapMode| {
        Sampler::new(
            allocator.device().clone(),
            &vk::SamplerCreateInfo {
                mag_filter: filter,
                min_filter: filter,
                mipmap_mode,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                // Full mip range: the HZB is sampled per explicit level.
                max_lod: vk::LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let linear_clamp_sampler = clamp_sampler(vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR);
    let point_clamp_sampler = clamp_sampler(vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST);

    let view = GPUMutex::new(HdrRenderTargetViews {
        hdr_output,
        sdr_target,
        albedo,
        normal,
        depth,
        depth_prev,
        motion_vectors,
        specular_albedo,
        hdr_denoised_output,
        exposure,
    });

    commands.insert_resource(HdrRenderTarget {
        view,
        state: Default::default(),
        albedo_state: Default::default(),
        normal_state: Default::default(),
        depth_state: Default::default(),
        depth_prev_state: Default::default(),
        sdr_target_state: Default::default(),
        hdr_target_state: Default::default(),
        motion_vectors_state: Default::default(),
        specular_albedo_state: Default::default(),
        exposure_state: Default::default(),
        exposure_initialized: false,
        display_extent: extent,
        render_extent,
        hdr_denoised_target_state: Default::default(),
        linear_clamp_sampler,
        point_clamp_sampler,
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
    mut frame_index: Local<u32>,
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
    // Shadow pass doesn't reproject, but its ray-gen reconstructs world
    // position from the depth G-buffer via `primaryRayDirWorldSpace` — that
    // depth was written by the jittered primary ray, so we must reuse the
    // same jitter or the reconstructed shading point drifts off the surface.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    // Per-frame seed for the sun-disk shadow-ray jitter (ShadowPush in
    // shadow.slang). A dedicated counter, like final_gather's, so sampling
    // doesn't couple to upscaler/jitter state.
    let shadow_frame_index: u32 = *frame_index;
    *frame_index = frame_index.wrapping_add(1);
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, shadow_sbt.layout(), |slice| {
                shadow_sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();

        let pipeline = encoder.retain(pipeline).deref_inner();
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // HDR: read-write (miss shader reads current value and adds sun contribution)
        encoder.use_image_resource(
            render_target_views.hdr_output(),
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
            render_target_views.albedo(),
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
            render_target_views.normal(),
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
            render_target_views.specular_albedo(),
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
            render_target_views.depth(),
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
                .output_texture(render_target_views.hdr_output().full_view())
                .gbuffer_albedo_linear(
                    render_target_views
                        .albedo()
                        .deref_inner()
                        .deref_inner()
                        .full_view(),
                )
                .gbuffer_albedo_srgb(render_target_views.albedo().deref_inner().srgb_view())
                .gbuffer_albedo_uint(render_target_views.albedo().uint_view())
                .gbuffer_normal_texture(render_target_views.normal().deref_inner().full_view())
                .gbuffer_depth_texture(render_target_views.depth().full_view())
                .gbuffer_occlusion_texture(
                    render_target_views.sdr_target().deref_inner().full_view(),
                )
                .gbuffer_motion_vector_texture(render_target_views.motion_vectors().full_view())
                .gbuffer_specular_albedo(render_target_views.specular_albedo().full_view())
                .sky_atmosphere_params(atmo_buffer)
                .sky_transmittance_lut(transmittance_view.full_view())
                .sky_sky_view_lut(sky_view.full_view())
                .sky_linear_sampler(&atmosphere_luts.sampler)
                .per_instance_data(per_instance_buf)
                .as_slice(),
        );
        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&shadow_frame_index),
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
    mut sharc: sharc::SharcSystemParams,
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
    let Some(sharc_resources) = sharc.resources.as_deref_mut() else {
        return;
    };
    gather_sbt.push_raygen(0, 0, |_| {});
    gather_sbt.push_miss(0, 1, |_| {});
    // Final-gather doesn't reproject, but it reconstructs world position from
    // the depth G-buffer via `primaryRayDirWorldSpace` — that depth was
    // written by the jittered primary ray, so we must reuse the same jitter.
    let uniform = build_camera_uniform(camera, transform, None, jitter.offset);
    let mut sharc_constants =
        sharc::build_sharc_constants(&sharc.config, &sharc.frame_state, transform.translation());
    // Carry the debug mode so the gather CHS/miss suppress their HDR writes
    // while a debug view owns the output. Final-gather still runs in that case
    // to keep the cache seeded (its CHS keeps pushing candidates on misses).
    sharc_constants.debug_mode = sharc.debug.mode.as_u32();
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    let push = sharc::FinalGatherPush::new(*frame_index);
    *frame_index = frame_index.wrapping_add(1);
    // Ping-pong pool indices for this frame.
    let (sharc_pool_read_idx, sharc_pool_write_idx) =
        sharc::SharcResources::pool_indices(sharc.frame_state.frame_index);
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, gather_sbt.layout(), |slice| {
                gather_sbt.write_buffer(slice);
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer).deref_inner();

        let pipeline = encoder.retain(pipeline).deref_inner();
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let per_instance_buf = encoder.lock(
            per_instance_mutex,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

        let render_target_views =
            encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // HDR: read-write (additive indirect contribution)
        encoder.use_image_resource(
            render_target_views.hdr_output(),
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
            render_target_views.albedo(),
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
            render_target_views.normal(),
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
        // Specular albedo: read
        encoder.use_image_resource(
            render_target_views.specular_albedo(),
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
        // Depth: read
        encoder.use_image_resource(
            render_target_views.depth(),
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
            .output_texture(render_target_views.hdr_output().full_view())
            .gbuffer_albedo_linear(
                render_target_views
                    .albedo()
                    .deref_inner()
                    .deref_inner()
                    .full_view(),
            )
            .gbuffer_albedo_srgb(render_target_views.albedo().deref_inner().srgb_view())
            .gbuffer_normal_texture(render_target_views.normal().deref_inner().full_view())
            .gbuffer_depth_texture(render_target_views.depth().full_view())
            .gbuffer_occlusion_texture(render_target_views.sdr_target().deref_inner().full_view())
            .gbuffer_motion_vector_texture(render_target_views.motion_vectors().full_view())
            .gbuffer_specular_albedo(render_target_views.specular_albedo().full_view())
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
            render_target_views.sdr_target(),
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
                    .view(render_target_views.sdr_target().deref_inner().full_view());
                // Use linear view for egui. egui does all the interpolation in srgb space.
            })
            .render_area(
                IVec2::ZERO,
                UVec2::new(hdr.display_extent.x, hdr.display_extent.y),
            )
            .begin();
    });
}
