use bevy::prelude::*;
use bevy_pumicite::{
    SubmissionState, prelude::ComputePipeline, staging::UniformRingBuffer,
    swapchain::SwapchainImage,
};
use bytemuck::NoUninit;
use pumicite::{
    ash::vk,
    buffer::BufferLike,
    image::ImageLike,
    tracking::Access,
    types::format::{ColorSpace, ColorSpacePrimaries, ColorSpaceTransferFunction},
    utils::AsVkHandle,
};

use dust_gfxdebug::{GpuProfiler, GpuTimerCommands};

use crate::{HdrRenderTarget, PbrRenderState, camera::Camera};

unsafe extern "C" {
    fn dust_lpm_setup(
        ctl: *mut u32,
        preset: u32,
        saturation: *const [f32; 3],
        crosstalk: *const [f32; 3],
        fs2_red: *const [f32; 2],
        fs2_green: *const [f32; 2],
        fs2_blue: *const [f32; 2],
        fs2_white: *const [f32; 2],
        fs2_scalar: f32,
        hdr10_scalar: f32,
        soft_gap: f32,
        hdr_max: f32,
        exposure: f32,
        contrast: f32,
        shoulder_contrast: f32,
        shoulder: bool,
    );
}

/// LPM preset selector. Must match `DustLpmPreset` in `third_party/ffx/lpm_setup.cc`.
///
/// Variant naming follows the FFX `LPM_CONFIG_<Output>_<Working>` convention
/// from `ffx_lpm.h` (see `FidelityFX-LPM/ffx-lpm/ffx_lpm.h` lines 590–680):
/// the first segment is the output container, the second is the scene/working
/// color space. For the sample (ColorSpace × DisplayMode) selection matrix
/// that picks between these presets, see `sample/src/DX12/LPMPS.cpp:157`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum LpmPreset {
    /// Rec.709 working → Rec.709 / sRGB / gamma 2.2 SDR output.
    /// Baseline SDR path (`LPM_CONFIG_709_709`).
    Rec709_Rec709 = 0,
    /// DCI-P3 (D65) working → Rec.709 / sRGB SDR output.
    /// SDR display fed from a wider P3 scene (`LPM_CONFIG_709_P3`).
    Rec709_P3 = 1,
    /// Rec.2020 working → Rec.709 / sRGB SDR output.
    /// SDR display fed from a Rec.2020 scene (`LPM_CONFIG_709_2020`).
    Rec709_Rec2020 = 2,
    /// Rec.709 working → FreeSync2 raw (fast 32-bpp gamma 2.2 in native
    /// display primaries). Requires `fs2_red/green/blue/white`
    /// (`LPM_CONFIG_FS2RAW_709`).
    Fs2Raw_Rec709 = 3,
    /// Rec.709 working → FreeSync2 scRGB (slow 64-bpp, Rec.709 primaries with
    /// possible negative channels). Requires `fs2_scalar` from
    /// `LpmFs2ScrgbScalar` (`LPM_CONFIG_FS2SCRGB_709`).
    Fs2Scrgb_Rec709 = 4,
    /// Rec.709 working → HDR10 raw (fast 32-bpp, 10:10:10:2 PQ in Rec.2020
    /// primaries). Requires `hdr10_scalar` from `LpmHdr10RawScalar`
    /// (`LPM_CONFIG_HDR10RAW_709`).
    Hdr10Raw_Rec709 = 5,
    /// Rec.709 working → HDR10 scRGB (slow 64-bpp, linear FP16 Rec.709 with
    /// possible negative channels). Requires `hdr10_scalar` from
    /// `LpmHdr10ScrgbScalar` (`LPM_CONFIG_HDR10SCRGB_709`).
    Hdr10Scrgb_Rec709 = 6,
    /// DCI-P3 (D65) working → FreeSync2 raw. Requires `fs2_*` primaries
    /// (`LPM_CONFIG_FS2RAW_P3`).
    Fs2Raw_P3 = 7,
    /// DCI-P3 (D65) working → FreeSync2 scRGB. Requires `fs2_*` primaries
    /// and `fs2_scalar` (`LPM_CONFIG_FS2SCRGB_P3`).
    Fs2Scrgb_P3 = 8,
    /// DCI-P3 (D65) working → HDR10 raw. Requires `hdr10_scalar`
    /// (`LPM_CONFIG_HDR10RAW_P3`).
    Hdr10Raw_P3 = 9,
    /// DCI-P3 (D65) working → HDR10 scRGB. Requires `hdr10_scalar`
    /// (`LPM_CONFIG_HDR10SCRGB_P3`).
    Hdr10Scrgb_P3 = 10,
    /// Rec.2020 working → FreeSync2 raw. Requires `fs2_*` primaries
    /// (`LPM_CONFIG_FS2RAW_2020`).
    Fs2Raw_Rec2020 = 11,
    /// Rec.2020 working → FreeSync2 scRGB. Requires `fs2_*` primaries and
    /// `fs2_scalar` (`LPM_CONFIG_FS2SCRGB_2020`).
    Fs2Scrgb_Rec2020 = 12,
    /// Rec.2020 working → HDR10 raw. Requires `hdr10_scalar`
    /// (`LPM_CONFIG_HDR10RAW_2020`).
    Hdr10Raw_Rec2020 = 13,
    /// Rec.2020 working → HDR10 scRGB. Requires `hdr10_scalar`
    /// (`LPM_CONFIG_HDR10SCRGB_2020`).
    Hdr10Scrgb_Rec2020 = 14,
}

pub mod lpm_flags {
    // Bit layout passed to the shader as `lpm_flags`. Must match the
    // `LpmFilter()` call in `tonemap.glsl`.
    pub const SHOULDER: u32 = 1 << 0;
    pub const CON: u32 = 1 << 1;
    pub const SOFT: u32 = 1 << 2;
    pub const CON2: u32 = 1 << 3;
    pub const CLIP: u32 = 1 << 4;
    pub const SCALE_ONLY: u32 = 1 << 5;
}

impl LpmPreset {
    /// CON/SOFT/CON2/CLIP/SCALEONLY flags from the matching `LPM_CONFIG_*`
    /// macro in `ffx_lpm.h`. The shader forwards these to `LpmFilter` so the
    /// runtime code path matches the baked control block. `SHOULDER` is
    /// orthogonal and is OR'd in from `LpmConfig::shoulder`.
    pub fn filter_flags(self) -> u32 {
        use lpm_flags::*;
        match self {
            // 709 output (SDR): pure tonemap, no gamut convert, no OETF.
            Self::Rec709_Rec709 => 0,
            Self::Rec709_P3 | Self::Rec709_Rec2020 => CON | SOFT,
            // FS2 raw: gamut convert + clip (applies gamma-2.2 OETF).
            Self::Fs2Raw_Rec709 => CON2 | CLIP,
            Self::Fs2Raw_P3 | Self::Fs2Raw_Rec2020 => CON | SOFT,
            // FS2 scRGB: scale-only (linear FP16 output).
            Self::Fs2Scrgb_Rec709 => SCALE_ONLY,
            Self::Fs2Scrgb_P3 | Self::Fs2Scrgb_Rec2020 => CON | SOFT | CON2,
            // HDR10 raw: gamut convert + clip (applies PQ OETF).
            Self::Hdr10Raw_Rec709 | Self::Hdr10Raw_P3 => CON2 | CLIP,
            Self::Hdr10Raw_Rec2020 => SCALE_ONLY,
            // HDR10 scRGB: scale-only (709) or gamut convert (P3/2020).
            Self::Hdr10Scrgb_Rec709 => SCALE_ONLY,
            Self::Hdr10Scrgb_P3 | Self::Hdr10Scrgb_Rec2020 => CON2,
        }
    }

    /// True when the preset leaves linear values in the output and the shader
    /// must apply the SDR sRGB OETF. Every other preset bakes its own OETF
    /// (PQ / gamma-2.2) or intentionally emits linear scRGB, so the shader
    /// must NOT post-encode.
    pub fn is_sdr_output(self) -> bool {
        matches!(
            self,
            Self::Rec709_Rec709 | Self::Rec709_P3 | Self::Rec709_Rec2020
        )
    }

    /// Pick the LPM output preset for a swapchain surface (format + color space).
    pub fn for_swapchain_colorspace(color_space: vk::ColorSpaceKHR) -> LpmPreset {
        match color_space {
            // Linear floating-point surface → scRGB-style HDR (Windows
            // `EXTENDED_SRGB_LINEAR_EXT` or AMD FS2 scRGB). LPM's SCALEONLY path
            // applies only the HDR peak scalar; the compositor does the rest.
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
            | vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => LpmPreset::Hdr10Scrgb_Rec709,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => LpmPreset::Hdr10Raw_Rec709,
            vk::ColorSpaceKHR::SRGB_NONLINEAR | vk::ColorSpaceKHR::BT709_LINEAR_EXT => {
                LpmPreset::Rec709_Rec709
            }
            _ => todo!(),
        }
    }
}

pub const LPM_CTL_WORDS: usize = 24 * 4;

pub struct LpmConfig {
    pub swapchain_color_space: vk::ColorSpaceKHR,
    pub preset: LpmPreset,
    pub saturation: [f32; 3],
    pub crosstalk: [f32; 3],
    /// FreeSync2 display primaries (xy). Only read by `Fs2*` presets.
    pub fs2_red: [f32; 2],
    pub fs2_green: [f32; 2],
    pub fs2_blue: [f32; 2],
    pub fs2_white: [f32; 2],
    /// FreeSync2 scRGB scalar. Only read by `Fs2Scrgb_*` presets.
    pub fs2_scalar: f32,
    /// HDR10 peak scalar. Only read by `Hdr10*` presets.
    pub hdr10_peak_nits: f32,
    pub paperwhite_nits: f32,
    pub soft_gap: f32,
    pub hdr_max: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub shoulder_contrast: f32,
    pub shoulder: bool,
    pub sdr_colorspace_transform: Mat3,
}

impl LpmConfig {
    pub fn new_for_colorspace(color_space: vk::ColorSpaceKHR) -> Self {
        Self {
            preset: LpmPreset::for_swapchain_colorspace(color_space),
            sdr_colorspace_transform: ColorSpacePrimaries::BT709
                .to_color_space(&ColorSpace::from(color_space).primaries),
            swapchain_color_space: color_space,

            saturation: [0.0; 3],
            crosstalk: [1.0, 0.5, 1.0 / 32.0],
            fs2_red: [0.0; 2],
            fs2_green: [0.0; 2],
            fs2_blue: [0.0; 2],
            fs2_white: [0.0; 2],
            fs2_scalar: 1.0,
            hdr10_peak_nits: 350.0,
            paperwhite_nits: 250.0,
            soft_gap: 0.0,
            hdr_max: 256.0,
            exposure: 8.0,
            contrast: 0.0,
            shoulder_contrast: 1.0,
            shoulder: false,
        }
    }
    pub fn sdr_transform_matrix(&self) -> Mat3 {
        match self.swapchain_color_space {
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => {
                self.sdr_colorspace_transform * (self.paperwhite_nits / 10000.0)
            }
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => {
                self.sdr_colorspace_transform * (self.paperwhite_nits / 80.0)
            }
            _ => Mat3::IDENTITY,
        }
    }
    pub fn ctl_words(&self) -> [u32; LPM_CTL_WORDS] {
        let hdr10_scalar = match self.preset {
            LpmPreset::Hdr10Raw_Rec709 | LpmPreset::Hdr10Raw_P3 | LpmPreset::Hdr10Raw_Rec2020 => {
                // LpmHdr10RawScalar
                self.hdr10_peak_nits * (1.0 / 10000.0)
            }
            LpmPreset::Hdr10Scrgb_Rec709
            | LpmPreset::Hdr10Scrgb_P3
            | LpmPreset::Hdr10Scrgb_Rec2020 => {
                // LpmHdr10ScrgbScalar
                self.hdr10_peak_nits * (1.0 / 80.0)
            }
            _ => 1.0,
        };
        let mut ctl = [0u32; LPM_CTL_WORDS];
        unsafe {
            dust_lpm_setup(
                ctl.as_mut_ptr(),
                self.preset as u32,
                &self.saturation,
                &self.crosstalk,
                &self.fs2_red,
                &self.fs2_green,
                &self.fs2_blue,
                &self.fs2_white,
                self.fs2_scalar,
                hdr10_scalar,
                self.soft_gap,
                self.hdr_max,
                self.exposure,
                self.contrast,
                self.shoulder_contrast,
                self.shoulder,
            )
        };
        ctl
    }
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
struct TonemapPassUniforms {
    lpm_ctl: [u32; LPM_CTL_WORDS],

    sdr_mapping_col0: [f32; 3],
    // The OETF curve to apply after tonemapping
    gamma_mode: u32,

    sdr_mapping_col1: [f32; 3],
    lpm_flags: u32,

    sdr_mapping_col2: [f32; 3],
    _pad: u32,
}

/// Per-frame parameters for the auto-exposure metering pass. Layout must match
/// the `AutoExposureCtl` uniform block in `autoexposure.glsl`.
#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
struct AutoExposureUniforms {
    dt: f32,
    ev_compensation: f32,
    adaptation_speed: f32,
    first_frame: u32,
    min_log_luminance: f32,
    max_log_luminance: f32,
    _pad: [u32; 2],
}

/// Meters the HDR scene and writes the adapted exposure scale into the 1×1
/// [`HdrRenderTarget`] exposure image. Runs after lighting is complete and
/// before `upscaler_evaluate`, so the denoise/upscale dispatch (MetalFX
/// `exposureTexture` / DLSS-RR `pInExposureTexture`) and [`tonemap_pass`] read
/// the same per-frame exposure — both engines want the value that, multiplied
/// with the input color, matches the tonemapped brightness.
pub(crate) fn autoexposure_pass(
    mut ctx: SubmissionState,
    state: Res<PbrRenderState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    cameras: Query<&Camera, With<bevy::window::PrimaryWindow>>,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
    time: Res<Time>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&state.autoexposure_pipeline) else {
        return;
    };
    let Some(mut hdr_target) = hdr_target else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();

    let first_frame = !hdr_target.exposure_initialized;
    hdr_target.exposure_initialized = true;
    let uniforms = AutoExposureUniforms {
        dt: time.delta_secs(),
        ev_compensation: camera.exposure,
        // Reaches ~95% of a brightness step in about two seconds.
        adaptation_speed: 1.5,
        first_frame: first_frame as u32,
        min_log_luminance: -12.0,
        max_log_luminance: 16.0,
        _pad: [0; 2],
    };

    ctx.record(move |encoder| {
        let ctl_buffer = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniforms));
        let hdr = &mut *hdr_target;
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);

        // HDR scene radiance: read
        encoder.use_image_resource(
            &views.hdr_output,
            &mut hdr.state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Exposure scale: read previous frame's value, write the adapted one.
        // Contents are undefined right after (re)creation, hence the discard on
        // the first frame — the shader ignores the previous value then.
        encoder.use_image_resource(
            &views.exposure,
            &mut hdr.exposure_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            first_frame,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: views.hdr_output.full_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: views.exposure.full_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                ]),
                vk::WriteDescriptorSet {
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&[vk::DescriptorBufferInfo {
                    buffer: ctl_buffer.vk_handle(),
                    offset: ctl_buffer.offset(),
                    range: ctl_buffer.size(),
                }]),
            ],
        );

        encoder.timing_scope(profiler.as_deref_mut(), "auto-exposure", |encoder| {
            encoder.dispatch(UVec3::new(1, 1, 1));
        });
    });
}

pub(crate) fn tonemap_pass(
    mut ctx: SubmissionState,
    state: Res<PbrRenderState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut swapchain_images: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok(mut swapchain_image) = swapchain_images.single_mut() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&state.tonemap_pipeline) else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let swapchain_current_image = swapchain_image.current_image().unwrap();
    // The shader pre-multiplies the HDR input by the auto-exposure scale (see
    // `autoexposure_pass`), so its mid-gray already sits at 0.18. Run LPM at
    // its neutral exposure — `midIn = hdrMax · 0.18 · 2^-exposure = 0.18` ⇔
    // `exposure = log2(hdrMax)` — so LPM adds no exposure of its own and only
    // shapes the shoulder / gamut. Exposure intent lives in the auto-exposure
    // pass (and `Camera::exposure` EV compensation), keeping the value the
    // upscaler was told consistent with what is displayed.
    let mut lpm_ctl = LpmConfig::new_for_colorspace(swapchain_current_image.color_space());
    lpm_ctl.exposure = lpm_ctl.hdr_max.log2();
    let pipeline = pipeline.clone().into_inner();
    let swapchain_colorspace =
        pumicite::types::format::ColorSpace::from(swapchain_current_image.color_space());
    let gamma_mode = swapchain_colorspace.transfer_function;
    let lpm_filter_flags = lpm_ctl.preset.filter_flags()
        | if lpm_ctl.shoulder {
            lpm_flags::SHOULDER
        } else {
            0
        };
    let sdr_transform_matrix = lpm_ctl.sdr_transform_matrix();

    ctx.record(move |encoder| {
        let lpm_ctl_buffer = uniform_ring_buffer.create_uniform(
            encoder,
            bytemuck::bytes_of(&TonemapPassUniforms {
                lpm_ctl: lpm_ctl.ctl_words(),
                sdr_mapping_col0: sdr_transform_matrix.col(0).to_array(),
                sdr_mapping_col1: sdr_transform_matrix.col(1).to_array(),
                sdr_mapping_col2: sdr_transform_matrix.col(2).to_array(),
                gamma_mode: gamma_mode as u32,
                lpm_flags: lpm_filter_flags,
                _pad: 0,
            }),
        );
        let render_target_views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let swapchain_target = encoder.lock(
            swapchain_image.current_image().as_ref().unwrap(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        // HDR intermediary: read
        encoder.use_image_resource(
            &render_target_views.hdr_denoised_output,
            &mut hdr.hdr_denoised_target_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // SDR target: read (egui / occluding content)
        encoder.use_image_resource(
            &render_target_views.sdr_target,
            &mut hdr.sdr_target_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Auto-exposure scale: read
        encoder.use_image_resource(
            &render_target_views.exposure,
            &mut hdr.exposure_state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Swapchain: write (final composited output)
        encoder.use_image_resource(
            swapchain_target,
            &mut swapchain_image.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views
                            .hdr_denoised_output
                            .full_view()
                            .vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: swapchain_target.linear_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views.sdr_target.full_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                ]),
                vk::WriteDescriptorSet {
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&[vk::DescriptorBufferInfo {
                    buffer: lpm_ctl_buffer.vk_handle(),
                    offset: lpm_ctl_buffer.offset(),
                    range: lpm_ctl_buffer.size(),
                }]),
                vk::WriteDescriptorSet {
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: render_target_views.exposure.full_view().vk_handle(),
                    image_layout: vk::ImageLayout::GENERAL,
                }]),
            ],
        );

        encoder.timing_scope(profiler.as_deref_mut(), "tonemap", |encoder| {
            encoder.dispatch(UVec3::new(
                hdr.display_extent.x.div_ceil(8),
                hdr.display_extent.y.div_ceil(8),
                1,
            ));
        });
    });
}
