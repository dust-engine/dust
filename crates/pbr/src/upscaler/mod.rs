//! Turns the noisy render-resolution radiance (`hdr_output`) into the
//! display-resolution image the tonemap pass reads (`hdr_denoised_output`).
//!
//! The backends ([`UpscalerBackend`]):
//! - A linear-filtered `vkCmdBlitImage`: everywhere. Upscales only, so the
//!   output keeps the renderer's noise.
//! - MetalFX temporal denoised scaler (`metalfx`): Apple GPUs. Denoises and
//!   upscales.
//! - DLSS Ray Reconstruction (`dlss`): Bazel builds on Windows/Linux, on the
//!   NVIDIA driver. Denoises and upscales.
//!
//! [`setup_upscaler`] finds which of them this device runs, once after device
//! creation ([`AvailableUpscalers`]). [`RenderQualitySettings::upscaler`]
//! picks one at any time; an unavailable pick runs the blit.

#[cfg(feature = "dlss")]
mod dlss;
#[cfg(target_vendor = "apple")]
mod metalfx;

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy_pumicite::{CreateDevice, SubmissionState};
use dust_app::ApplicationSettings;
use dust_profiler::{GpuProfiler, GpuTimerCommands};
use pumicite::Device;
use pumicite::ash::vk;
use pumicite::command::GPURefMut;
use pumicite::image::{ImageLike, ImageViewLike};
use pumicite::tracking::Access;

use crate::camera::Camera;
use crate::{HdrRenderTarget, HdrRenderTargetViewsExt, RenderQualitySettings};

/// Enables the Vulkan instance and device extensions the upscaler backends
/// need, where the device has them.
///
/// **Ordering:** add before [`bevy_pumicite::PumicitePlugin`], which creates
/// the `VkInstance` when it is added.
pub struct UpscalerSetupPlugin;

impl Plugin for UpscalerSetupPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "dlss")]
        dlss::add_vulkan_extensions(app);
        #[cfg(target_vendor = "apple")]
        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<pumicite::device::DeviceBuilder>| {
                // Without it MetalFX is unavailable; `metalfx::is_supported`
                // checks.
                device_builder
                    .enable_extension::<pumicite::ash::ext::metal_objects::Meta>()
                    .ok();
            })
            .before(CreateDevice),
        );
        #[cfg(not(any(feature = "dlss", target_vendor = "apple")))]
        let _ = app;
    }
}

/// A way to turn `hdr_output` into `hdr_denoised_output`. Every variant exists
/// on every platform; [`AvailableUpscalers`] says which this device runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscalerBackend {
    /// Linear-filtered blit at native resolution: no denoising.
    #[default]
    Blit,
    /// MetalFX temporal denoised scaler.
    MetalFx,
    /// DLSS Ray Reconstruction.
    Dlss,
}

impl UpscalerBackend {
    pub const ALL: [Self; 3] = [Self::Blit, Self::MetalFx, Self::Dlss];

    pub fn name(self) -> &'static str {
        match self {
            Self::Blit => "Blit (no denoising)",
            Self::MetalFx => "MetalFX temporal denoised scaler",
            Self::Dlss => "DLSS Ray Reconstruction",
        }
    }

    /// Whether the backend renders below display resolution, at the scale of
    /// [`RenderQualitySettings::quality`]. The others render at display
    /// resolution.
    pub fn upscales(self) -> bool {
        match self {
            Self::Blit => false,
            Self::MetalFx | Self::Dlss => true,
        }
    }
}

/// The backends this device runs, found once after device creation by
/// [`setup_upscaler`]. The blit always runs.
#[derive(Resource)]
pub struct AvailableUpscalers {
    #[cfg(feature = "dlss")]
    ngx: Option<dlss::Ngx>,
    #[cfg(target_vendor = "apple")]
    metalfx: bool,
}

impl AvailableUpscalers {
    /// Finds which backends `device` runs.
    fn new(device: &Device, settings: &ApplicationSettings) -> Self {
        let _ = (device, settings);
        Self {
            #[cfg(feature = "dlss")]
            ngx: match dlss::Ngx::new(device, settings) {
                Ok(ngx) => Some(ngx),
                Err(reason) => {
                    tracing::info!("DLSS Ray Reconstruction unavailable: {reason}");
                    None
                }
            },
            #[cfg(target_vendor = "apple")]
            metalfx: {
                let supported = metalfx::is_supported(device);
                if !supported {
                    tracing::info!("MetalFX temporal denoised scaler unsupported on this device");
                }
                supported
            },
        }
    }

    pub fn contains(&self, backend: UpscalerBackend) -> bool {
        match backend {
            UpscalerBackend::Blit => true,
            #[cfg(target_vendor = "apple")]
            UpscalerBackend::MetalFx => self.metalfx,
            #[cfg(feature = "dlss")]
            UpscalerBackend::Dlss => self.ngx.is_some(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The backend that runs when `requested` is picked: `requested` if this
    /// device runs it, otherwise the blit.
    pub fn resolve(&self, requested: UpscalerBackend) -> UpscalerBackend {
        if self.contains(requested) {
            requested
        } else {
            UpscalerBackend::Blit
        }
    }
}

pub(crate) fn setup_upscaler(
    mut commands: Commands,
    device: Res<Device>,
    settings: Res<ApplicationSettings>,
) {
    let available = AvailableUpscalers::new(&device, &settings);
    let names: Vec<&str> = UpscalerBackend::ALL
        .into_iter()
        .filter(|&backend| available.contains(backend))
        .map(UpscalerBackend::name)
        .collect();
    tracing::info!("Upscalers available: {}", names.join(", "));
    commands.insert_resource(available);
}

/// The backend's per-resolution state, created by [`ensure_upscaler_session`].
enum UpscalerSession {
    /// In an `Arc` so each frame's commands keep the NGX feature alive after a
    /// switch or resize drops the session.
    #[cfg(feature = "dlss")]
    Dlss(std::sync::Arc<dlss::RayReconstruction>),
    #[cfg(target_vendor = "apple")]
    MetalFx(metalfx::DenoisedScaler),
    Blit,
}

impl UpscalerSession {
    fn backend(&self) -> UpscalerBackend {
        match self {
            #[cfg(feature = "dlss")]
            Self::Dlss(_) => UpscalerBackend::Dlss,
            #[cfg(target_vendor = "apple")]
            Self::MetalFx(_) => UpscalerBackend::MetalFx,
            Self::Blit => UpscalerBackend::Blit,
        }
    }
}

/// Holds the active upscaler session and what it was created for. The session
/// is recreated whenever the backend, the render extent or the display extent
/// changes: on a backend or quality-mode switch, or a swapchain resize.
#[derive(Resource, Default)]
pub struct UpscalerSessionState {
    session: Option<UpscalerSession>,
    /// Per-frame sub-pixel jitter offsets in texels, within `[-0.5, 0.5)`.
    /// `advance_jitter` cycles through them. Empty (no jitter) without a
    /// session, and for the blit, which has no temporal accumulation.
    pub(crate) jitter_pattern: Vec<Vec2>,
    /// The backend the session was created for. It runs the blit instead if
    /// creating that backend's session failed.
    configured_backend: UpscalerBackend,
    configured_render_extent: UVec2,
    configured_display_extent: UVec2,
}

impl UpscalerSessionState {
    /// The backend running this frame; `None` before the first session.
    pub fn active_backend(&self) -> Option<UpscalerBackend> {
        self.session.as_ref().map(UpscalerSession::backend)
    }

    /// How many sub-pixel jitter offsets the active backend cycles through.
    pub fn jitter_phase_count(&self) -> usize {
        self.jitter_pattern.len()
    }
}

pub(crate) fn ensure_upscaler_session(
    // Only DLSS records work when a session is created.
    #[cfg(feature = "dlss")] mut ctx: SubmissionState,
    available: Option<Res<AvailableUpscalers>>,
    settings: Res<RenderQualitySettings>,
    mut state: ResMut<UpscalerSessionState>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    // Only MetalFX needs the device to create a session.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))] device: Res<Device>,
) {
    let (Some(available), Some(hdr)) = (available, hdr_target) else {
        return;
    };
    let backend = available.resolve(settings.upscaler);
    let render_extent = hdr.render_extent;
    let display_extent = hdr.display_extent;
    if render_extent.min_element() == 0 || display_extent.min_element() == 0 {
        return;
    }
    if state.session.is_some()
        && state.configured_backend == backend
        && state.configured_render_extent == render_extent
        && state.configured_display_extent == display_extent
    {
        return;
    }

    // Drop the previous session before creating its replacement.
    state.session = None;
    state.jitter_pattern.clear();

    let session: Result<UpscalerSession, String> = match backend {
        UpscalerBackend::Blit => Ok(UpscalerSession::Blit),
        #[cfg(target_vendor = "apple")]
        UpscalerBackend::MetalFx => {
            metalfx::DenoisedScaler::new(&device, render_extent, display_extent)
                .map(UpscalerSession::MetalFx)
        }
        #[cfg(feature = "dlss")]
        UpscalerBackend::Dlss => {
            let ngx = available
                .ngx
                .as_ref()
                .expect("resolved to an available backend");
            let mut created = None;
            ctx.record(|encoder, _| {
                created =
                    Some(ngx.create_ray_reconstruction(encoder, render_extent, display_extent));
            });
            created
                .unwrap()
                .map(|session| UpscalerSession::Dlss(std::sync::Arc::new(session)))
                .map_err(|e| e.to_string())
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!("{} resolved but not compiled in", backend.name()),
    };
    // A failed backend runs the blit until the backend or an extent changes,
    // rather than leaving the frame unwritten and retrying every frame.
    let session = session.unwrap_or_else(|e| {
        tracing::error!(
            render = ?render_extent,
            display = ?display_extent,
            "Creating the {} session failed, running the blit instead: {e}",
            backend.name()
        );
        UpscalerSession::Blit
    });

    // Temporal backends need the jitter to cover each display pixel with
    // several render samples. Following Apple's MetalFX guidance, the phase
    // count grows with the pixel-count ratio: 8 · (display / render)².
    let ratio = display_extent.x as f32 / render_extent.x as f32;
    let phase_count = (8.0 * ratio * ratio) as u32;
    let phase_count = match session {
        // The DLSS Ray Reconstruction guide asks for at least 32 phases.
        #[cfg(feature = "dlss")]
        UpscalerSession::Dlss(_) => phase_count.max(32),
        #[cfg(target_vendor = "apple")]
        UpscalerSession::MetalFx(_) => phase_count,
        UpscalerSession::Blit => 0,
    };
    state.jitter_pattern = (1..=phase_count)
        .map(|i| Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5))
        .collect();

    tracing::info!(
        render = ?render_extent,
        display = ?display_extent,
        jitter_phases = phase_count,
        "Created {} session",
        session.backend().name()
    );
    state.session = Some(session);
    state.configured_backend = backend;
    state.configured_render_extent = render_extent;
    state.configured_display_extent = display_extent;
}

/// The `index`-th value of the Halton low-discrepancy sequence for `base`.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0;
    let mut result = 0.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// One image a denoising backend reads or writes, as raw handles. The image is
/// single-mip, single-layer, color-aspect, and in `GENERAL` layout.
#[derive(Clone, Copy)]
// MetalFX reads only `view`.
#[cfg_attr(not(feature = "dlss"), allow(dead_code))]
pub(crate) struct UpscalerImage {
    pub image: vk::Image,
    /// The view the backend accesses the image through.
    pub view: vk::ImageView,
    /// The view's format.
    pub format: vk::Format,
    pub extent: UVec2,
}

/// Everything a denoising backend consumes for one frame. All images except
/// `output` (display resolution) and `exposure` (1×1) are at render resolution.
pub(crate) struct UpscalerInputs {
    /// Noisy radiance.
    pub color: UpscalerImage,
    /// Denoised, upscaled radiance.
    pub output: UpscalerImage,
    /// Reverse-Z depth, `near / linearViewZ`.
    pub depth: UpscalerImage,
    /// Screen-space motion in render-resolution pixels.
    pub motion_vectors: UpscalerImage,
    /// Albedo through its sRGB view.
    pub diffuse_albedo: UpscalerImage,
    pub specular_albedo: UpscalerImage,
    /// World-space normal in `.xyz`, roughness in `.w`.
    pub normal: UpscalerImage,
    /// The normal image through a view that broadcasts `.w` to every channel.
    pub roughness: UpscalerImage,
    /// Auto-exposure scale the tonemap pass applies.
    pub exposure: UpscalerImage,
    /// This frame's sub-pixel jitter, in render-resolution texels.
    pub jitter: Vec2,
    pub world_to_view: Mat4,
    pub view_to_clip: Mat4,
}

/// Describes the image behind a locked token by its raw handles.
fn upscaler_image(
    image: GPURefMut<'_, impl ImageLike>,
    view: GPURefMut<'_, impl ImageViewLike>,
) -> UpscalerImage {
    // SAFETY: only handles and metadata are read.
    let (image, view) = unsafe { (image.get_unchecked(), view.get_unchecked()) };
    UpscalerImage {
        image: image.vk_handle(),
        view: view.vk_handle(),
        format: view.format(),
        extent: image.extent().truncate(),
    }
}

/// Records the upscale of this frame's radiance into
/// [`crate::HdrRenderTargetViews::hdr_denoised_output`].
pub(crate) fn upscaler_evaluate(
    mut ctx: SubmissionState,
    state: Res<UpscalerSessionState>,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    jitter: Res<crate::JitterState>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Some(session) = state.session.as_ref() else {
        return;
    };
    let Some(mut hdr_target) = hdr_target else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let render_extent = hdr_target.render_extent;
    let display_extent = hdr_target.display_extent;
    if render_extent.min_element() == 0 || display_extent.min_element() == 0 {
        return;
    }

    // Camera matrices for the denoisers' reprojection. dust renders with a
    // matrix-free, reverse-infinite-Z camera (rays from `tan_half_fov`, depth =
    // `near / linearViewZ`), so build the equivalent matrices: clip.w =
    // linearViewZ and clip.z = near, so ndc.z = near / linearViewZ.
    let world_to_view = Mat4::from(transform.affine().inverse());
    let t = camera.tan_half_fov();
    let aspect = render_extent.x as f32 / render_extent.y as f32;
    let view_to_clip = Mat4::from_cols(
        Vec4::new(1.0 / (t * aspect), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / t, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, camera.depth.start, 0.0),
    );
    let jitter = jitter.offset;
    // Everything below records unconditionally, so `hdr_denoised_output`
    // holds a complete frame for next frame's screen trace.
    hdr_target.denoised_written = true;

    ctx.record(move |encoder, _| {
        let hdr = &mut *hdr_target;
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);

        if let UpscalerSession::Blit = session {
            encoder.use_image_resource(
                views.hdr_output(),
                &mut hdr.state,
                Access::BLIT_SRC,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                false,
            );
            encoder.use_image_resource(
                views.hdr_denoised_output(),
                &mut hdr.hdr_denoised_target_state,
                Access::BLIT_DST,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                false, // to be able to read colors from previous frames
            );
            encoder.emit_barriers();
            let corner = |extent: UVec2| vk::Offset3D {
                x: extent.x as i32,
                y: extent.y as i32,
                z: 1,
            };
            let layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            encoder.timing_scope(profiler.as_deref_mut(), "super-resolution", |encoder| {
                encoder.blit_image_with_layout(
                    views.hdr_output(),
                    vk::ImageLayout::GENERAL,
                    views.hdr_denoised_output(),
                    vk::ImageLayout::GENERAL,
                    &[vk::ImageBlit {
                        src_subresource: layers,
                        src_offsets: [vk::Offset3D::default(), corner(render_extent)],
                        dst_subresource: layers,
                        dst_offsets: [vk::Offset3D::default(), corner(display_extent)],
                    }],
                    vk::Filter::LINEAR,
                );
            });
            return;
        }

        // The denoising backends access every image from compute work.
        encoder.use_image_resource(
            views.hdr_output(),
            &mut hdr.state,
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
            views.depth(),
            &mut hdr.depth_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.motion_vectors(),
            &mut hdr.motion_vectors_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            views.specular_albedo(),
            &mut hdr.specular_albedo_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
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
        encoder.use_image_resource(
            views.hdr_denoised_output(),
            &mut hdr.hdr_denoised_target_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false, // to be able to read colors from prev frames
        );
        encoder.emit_barriers();

        let inputs = UpscalerInputs {
            color: upscaler_image(views.hdr_output(), views.hdr_output().full_view()),
            output: upscaler_image(
                views.hdr_denoised_output(),
                views.hdr_denoised_output().full_view(),
            ),
            depth: upscaler_image(views.depth(), views.depth().full_view()),
            motion_vectors: upscaler_image(
                views.motion_vectors(),
                views.motion_vectors().full_view(),
            ),
            diffuse_albedo: upscaler_image(
                views.albedo(),
                views.albedo().deref_inner().srgb_view(),
            ),
            specular_albedo: upscaler_image(
                views.specular_albedo(),
                views.specular_albedo().full_view(),
            ),
            normal: upscaler_image(views.normal(), views.normal().deref_inner().full_view()),
            roughness: upscaler_image(views.normal(), views.normal().swizzled_view()),
            exposure: upscaler_image(views.exposure(), views.exposure().full_view()),
            jitter,
            world_to_view,
            view_to_clip,
        };
        encoder.timing_scope(
            profiler.as_deref_mut(),
            "super-resolution",
            |encoder| match session {
                #[cfg(feature = "dlss")]
                UpscalerSession::Dlss(dlss) => {
                    dlss.dispatch(encoder, &inputs);
                    // The NGX feature stays alive until this command buffer
                    // completes, even if the session is replaced.
                    encoder.retain(dlss.clone());
                }
                #[cfg(target_vendor = "apple")]
                UpscalerSession::MetalFx(scaler) => scaler.dispatch(encoder, &inputs),
                UpscalerSession::Blit => unreachable!(),
            },
        );
    });
}
