pub mod camera;
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
    loader::TextureAsset,
    rtx::{
        RayTracingPipeline, RtxPipelineManager,
        tlas::{TLAS, TLASInstance, tlas_build_input_upload_system},
    },
    shader::RayTracingPipelineLibrary,
    staging::{BufferInitializer, UniformRingBuffer},
    swapchain::{SwapchainImage, SwapchainSet},
};
use bytemuck::{Pod, Zeroable};
use pumicite::device::DeviceBuilder;
use pumicite::{
    Allocator, HasDevice,
    ash::vk::{self, TaggedStructure},
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike, SrgbImageView},
    rtx::ShaderBindingTable,
    sync::GPUMutex,
    tracking::{Access, ResourceState},
    utils::{AsVkHandle, glam_to_vk_transform},
};
use pumicite_egui::{EguiPrimaryContextPass, EguiRenderSet};

use bevy_pumicite::prelude::ComputePipeline;

use crate::{
    camera::Camera,
    sky::{AtmosphereLUTs, SkyAtmosphereLUTRenderSet},
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    /// row-major affine transformation matrix.
    model: vk::TransformMatrixKHR,
    tan_half_fov: f32,
    far: f32,
    near: f32,
    _padding: f32,
    /// Previous-frame world-from-view matrix; used by the primary RT pass to
    /// project hit points through last frame's camera and produce screen-space
    /// motion vectors for DLSS-RR.
    prev_model: vk::TransformMatrixKHR,
    prev_tan_half_fov: f32,
    _padding2: [f32; 3],
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
fn build_camera_uniform(
    camera: &Camera,
    transform: &GlobalTransform,
    prev: Option<&mut PreviousCameraState>,
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
        _padding: 0.0,
        prev_model,
        prev_tan_half_fov,
        _padding2: [0.0; 3],
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
                    .after(create_sbt),
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
            })
            .before(CreateDevice),
        );
    }
}

#[derive(Resource)]
struct BlueNoiseTextures {
    scalar: Handle<TextureAsset>,
    unitvec2: Handle<TextureAsset>,
    unitvec3: Handle<TextureAsset>,
    unitvec3_cosine: Handle<TextureAsset>,
    vec2: Handle<TextureAsset>,
    vec3: Handle<TextureAsset>,
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
    blue_noise: Res<BlueNoiseTextures>,
    texture_assets: Res<Assets<TextureAsset>>,
    mut prev_camera: Local<PreviousCameraState>,
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
    let Some(_noise_texture) = texture_assets.get(&blue_noise.unitvec3_cosine) else {
        tracing::warn!("Frame not rendered; missing blue noise texture");
        return;
    };
    sbt.push_raygen(0, |_| {});
    sbt.push_miss(0, |_| {});
    let uniform = build_camera_uniform(camera, transform, Some(&mut prev_camera));
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        tracing::warn!("Frame not rendered; missing atmosphere uniform buffer");
        return;
    };
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                slice.copy_from_slice(sbt.buffer())
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
            render_target_views.hdr_output.image(),
            &mut hdr.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.albedo.image(),
            &mut hdr.albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Normal G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.normal.image(),
            &mut hdr.normal_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Depth G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.depth.image(),
            &mut hdr.depth_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Motion vectors G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            render_target_views.motion_vectors.image(),
            &mut hdr.motion_vectors_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // SDR target: read (occlusion check against egui content)
        encoder.use_image_resource(
            render_target_views.sdr_target.image(),
            &mut hdr.sdr_target_state,
            Access::RTX_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
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
                .per_instance_data(per_instance_buf)
                .as_slice(),
        );
        encoder.trace_rays(
            sbt,
            0,
            sbt_buffer,
            UVec3 {
                x: hdr.extent.x,
                y: hdr.extent.y,
                z: 1,
            },
        );
    });
}

pub struct HdrRenderTargetViews {
    // R16G16B16A16_SFLOAT. Stores noisy raw light.
    pub hdr_output: FullImageView<Image>,
    // R8G8B8A8_UNORM. Stores sRGB UI elements.
    pub sdr_target: SrgbImageView<Image>,
    /// R8G8B8A8_SRGB. Stores albedo.
    pub albedo: SrgbImageView<Image>,
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
    pub extent: UVec2,
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

    commands.insert_resource(PbrRenderState {
        pipeline: pipeline_manager.add_pipeline(base_library),
        shadow_pipeline: pipeline_manager.add_pipeline(shadow_base_library),
        final_gather_pipeline: pipeline_manager.add_pipeline(final_gather_base_library),
        tonemap_pipeline,
        sbt: None,
        shadow_sbt: None,
        final_gather_sbt: None,
    });

    commands.insert_resource(BlueNoiseTextures {
        scalar: asset_server.load("bazel://dust/assets/stbn/scalar.png"),
        unitvec2: asset_server.load("bazel://dust/assets/stbn/unitvec2.png"),
        unitvec3: asset_server.load("bazel://dust/assets/stbn/unitvec3.png"),
        unitvec3_cosine: asset_server.load("bazel://dust/assets/stbn/unitvec3_cosine.png"),
        vec2: asset_server.load("bazel://dust/assets/stbn/vec2.png"),
        vec3: asset_server.load("bazel://dust/assets/stbn/vec3.png"),
    });
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

/// Holds the active DLSS-RR feature handle and the render extent it was
/// configured for. Recreated whenever the HDR target's extent changes.
#[derive(Resource, Default)]
pub struct DlssState {
    pub feature: Option<dust_denoiser::dlss::NgxFeature>,
    pub configured_extent: UVec2,
}

fn ensure_dlss_feature(
    mut ctx: SubmissionState,
    mut ngx: ResMut<dust_denoiser::dlss::NgxContext>,
    state: Option<ResMut<DlssState>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    mut commands: Commands,
) {
    ngx.check_dlss_rr_available().unwrap();
    let Some(hdr) = hdr_target else { return };
    if hdr.extent.x == 0 || hdr.extent.y == 0 {
        return;
    }

    let needs_create = match state.as_deref() {
        Some(s) => s.configured_extent != hdr.extent || s.feature.is_none(),
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
        InWidth: hdr.extent.x,
        InHeight: hdr.extent.y,
        InTargetWidth: hdr.extent.x,
        InTargetHeight: hdr.extent.y, // No upscaling for now
        InPerfQualityValue: dust_denoiser::dlss::sys::NVSDK_NGX_PerfQuality_Value::Balanced,
        // MVLowRes is required by DLSS-RR (the SwinDenoiser refuses to
        // initialize without it): motion vectors are sampled at the input /
        // render resolution rather than the output resolution.
        InFeatureCreateFlags: dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Feature_Flags::IsHDR
            | dust_denoiser::dlss::sys::NVSDK_NGX_DLSS_Feature_Flags::MVLowRes,
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
                extent = ?hdr.extent,
                "Created DLSS-RR feature"
            );
            if let Some(mut s) = state {
                s.feature = Some(feature);
                s.configured_extent = hdr.extent;
            } else {
                commands.insert_resource(DlssState {
                    feature: Some(feature),
                    configured_extent: hdr.extent,
                });
            }
        }
        Some(Err(e)) => {
            tracing::error!(
                target: "ngx",
                extent = ?hdr.extent,
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
fn dlss_evaluate(
    mut ctx: SubmissionState,
    ngx: Option<ResMut<dust_denoiser::dlss::NgxContext>>,
    state: Option<ResMut<DlssState>>,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
) {
    let (Some(mut ngx), Some(mut state), Some(mut hdr_target)) = (ngx, state, hdr_target) else {
        return;
    };
    if state.feature.is_none() {
        return;
    }
    let extent = hdr_target.extent;
    if extent.x == 0 || extent.y == 0 {
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
            render_target_views.hdr_output.image(),
            &mut hdr.state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.albedo.image(),
            &mut hdr.albedo_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.normal.image(),
            &mut hdr.normal_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.depth.image(),
            &mut hdr.depth_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            render_target_views.motion_vectors.image(),
            &mut hdr.motion_vectors_state,
            read_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Specular albedo stand-in: clear to black this frame, then read.
        encoder.use_image_resource(
            render_target_views.specular_albedo.image(),
            &mut hdr.specular_albedo_state,
            Access::CLEAR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            true,
        );
        encoder.use_image_resource(
            render_target_views.hdr_denoised_output.image(),
            &mut hdr.hdr_denoised_target_state,
            write_access,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        encoder.emit_barriers();

        // Zero-fill the specular albedo stand-in.
        let device = encoder.device().clone();
        let cmd_buffer = encoder.buffer().vk_handle();
        unsafe {
            device.cmd_clear_color_image(
                cmd_buffer,
                render_target_views.specular_albedo.image().vk_handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
        }

        // Transition specular to GENERAL for the NGX read.
        encoder.use_image_resource(
            render_target_views.specular_albedo.image(),
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
            render_target_views.hdr_output.vk_handle(),
            render_target_views.hdr_output.image().vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            extent.x,
            extent.y,
            false,
        );
        let mut output_resource = ngx_image_resource(
            render_target_views.hdr_denoised_output.vk_handle(),
            render_target_views.hdr_denoised_output.image().vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            extent.x,
            extent.y,
            true,
        );
        let mut depth = ngx_image_resource(
            render_target_views.depth.vk_handle(),
            render_target_views.depth.image().vk_handle(),
            subres,
            vk::Format::R32_SFLOAT,
            extent.x,
            extent.y,
            false,
        );
        let mut normals = ngx_image_resource(
            render_target_views.normal.vk_handle(),
            render_target_views.normal.image().vk_handle(),
            subres,
            vk::Format::R16G16B16A16_SFLOAT,
            extent.x,
            extent.y,
            false,
        );
        let mut diffuse_albedo = ngx_image_resource(
            render_target_views.albedo.srgb_view().vk_handle(),
            render_target_views.albedo.image().vk_handle(),
            subres,
            vk::Format::R8G8B8A8_SRGB,
            extent.x,
            extent.y,
            false,
        );
        let mut motion_vectors = ngx_image_resource(
            render_target_views.motion_vectors.vk_handle(),
            render_target_views.motion_vectors.image().vk_handle(),
            subres,
            vk::Format::R16G16_SFLOAT,
            extent.x,
            extent.y,
            false,
        );
        let mut specular_albedo = ngx_image_resource(
            render_target_views.specular_albedo.vk_handle(),
            render_target_views.specular_albedo.image().vk_handle(),
            subres,
            vk::Format::R8G8B8A8_UNORM,
            extent.x,
            extent.y,
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
        eval_params.InRenderSubrectDimensions = dust_denoiser::dlss::sys::NVSDK_NGX_Dimensions {
            Width: extent.x,
            Height: extent.y,
        };
        eval_params.InMVScaleX = 1.0;
        eval_params.InMVScaleY = 1.0;
        // No history yet — flag every frame as a teleport so DLSS-RR
        // discards its temporal accumulation.
        eval_params.InReset = 0;

        let cmd_buffer = encoder.buffer().vk_handle();
        if let Err(e) = ngx.evaluate_dlssd(cmd_buffer, feature, &mut eval_params) {
            tracing::error!(target: "ngx", "DLSS-RR evaluate failed: {e}");
        }
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
) {
    let Ok(swapchain_image) = swapchain_images.single() else {
        return;
    };
    let Some(current) = swapchain_image.current_image() else {
        return;
    };
    let extent = UVec2::new(current.extent().x, current.extent().y);

    if let Some(hdr) = hdr_target.as_ref() {
        if hdr.extent == extent {
            return;
        }
    }

    let create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        extent: vk::Extent3D {
            width: extent.x,
            height: extent.y,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let hdr_output = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..create_info
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
            ..create_info
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
            ..create_info
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
            ..create_info
        }
        .push(
            &mut vk::ImageFormatListCreateInfo::default()
                .view_formats(&[vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM]),
        ),
    )
    .unwrap()
    .with_name(c"G-Buffer Albedo Image")
    .create_srgb_view(vk::ImageUsageFlags::STORAGE, vk::ImageUsageFlags::SAMPLED)
    .unwrap();

    let normal = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..create_info
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
            ..create_info
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
            ..create_info
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
            ..create_info
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
        extent,
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
    shadow_sbt.push_raygen(0, |_| {});
    shadow_sbt.push_miss(0, |_| {});
    // Shadow pass doesn't reproject.
    let uniform = build_camera_uniform(camera, transform, None);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, shadow_sbt.layout(), |slice| {
                slice.copy_from_slice(shadow_sbt.buffer())
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
            render_target_views.hdr_output.image(),
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
        // Normal: read (written by primary pass)
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
        // Depth: read (written by primary pass)
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
                .per_instance_data(per_instance_buf)
                .as_slice(),
        );
        encoder.trace_rays(
            shadow_sbt,
            0,
            sbt_buffer,
            UVec3 {
                x: hdr.extent.x,
                y: hdr.extent.y,
                z: 1,
            },
        );
    });
}

fn final_gather_pass(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS<PbrInstanceData>>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    mut atmosphere_luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    blue_noise: Res<BlueNoiseTextures>,
    texture_assets: Res<Assets<TextureAsset>>,
    mut frame_counter: Local<FrameCounter>,
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
    let Some(noise_texture) = texture_assets.get(&blue_noise.unitvec3_cosine) else {
        return;
    };
    gather_sbt.push_raygen(0, |_| {});
    gather_sbt.push_miss(0, |_| {});
    let frame_index = frame_counter.0;
    frame_counter.0 = frame_counter.0.wrapping_add(1);
    // Final-gather doesn't reproject.
    let uniform = build_camera_uniform(camera, transform, None);
    let Some(atmosphere_uniform_buffer) = atmosphere_luts.param_buffer.as_ref().cloned() else {
        return;
    };
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, gather_sbt.layout(), |slice| {
                slice.copy_from_slice(gather_sbt.buffer())
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
            render_target_views.hdr_output.image(),
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
        // Normal: read
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
        // Depth: read
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
                .blue_noise_cosine(noise_texture.deref())
                .per_instance_data(per_instance_buf)
                .as_slice(),
        );

        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            bytemuck::bytes_of(&frame_index),
        );

        encoder.trace_rays(
            gather_sbt,
            0,
            sbt_buffer,
            UVec3 {
                x: hdr.extent.x,
                y: hdr.extent.y,
                z: 1,
            },
        );
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
            render_target_views.sdr_target.image(),
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
            .render_area(IVec2::ZERO, UVec2::new(hdr.extent.x, hdr.extent.y))
            .begin();
    });
}
