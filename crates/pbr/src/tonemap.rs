use bevy::prelude::*;
use bevy_pumicite::{
    SubmissionState, prelude::ComputePipeline, staging::UniformRingBuffer,
    swapchain::SwapchainImage,
};
use bytemuck::NoUninit;
use pumicite::{
    ash::vk, buffer::BufferLike, image::ImageLike, tracking::Access, utils::AsVkHandle
};

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

pub const LPM_CTL_WORDS: usize = 24 * 4;

pub struct LpmConfig {
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
    pub hdr10_scalar: f32,
    pub soft_gap: f32,
    pub hdr_max: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub shoulder_contrast: f32,
    pub shoulder: bool,
}

impl Default for LpmConfig {
    fn default() -> Self {
        Self {
            preset: LpmPreset::Rec709_Rec709,
            saturation: [0.0; 3],
            crosstalk: [1.0, 0.5, 1.0 / 32.0],
            fs2_red: [0.0; 2],
            fs2_green: [0.0; 2],
            fs2_blue: [0.0; 2],
            fs2_white: [0.0; 2],
            fs2_scalar: 1.0,
            hdr10_scalar: 1.0,
            soft_gap: 0.0,
            hdr_max: 256.0,
            exposure: 8.0,
            contrast: 0.0,
            shoulder_contrast: 1.0,
            shoulder: false,
        }
    }
}

impl LpmConfig {
    pub fn ctl_words(&self) -> [u32; LPM_CTL_WORDS] {
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
                self.hdr10_scalar,
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
    gamma_mode: u32,
}

pub(crate) fn tonemap_pass(
    mut ctx: SubmissionState,
    state: Res<PbrRenderState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut swapchain_images: Query<(&mut SwapchainImage, &Camera), With<bevy::window::PrimaryWindow>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
) {
    let Ok((mut swapchain_image, camera)) = swapchain_images.single_mut() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&state.tonemap_pipeline) else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let swapchain_current_image = swapchain_image.current_image().unwrap();
    let lpm_ctl = LpmConfig {
        exposure: camera.exposure,
        ..Default::default()
    };
    let pipeline = pipeline.clone().into_inner();
    let gamma_mode = swapchain_current_image.color_space().transfer_function;

    ctx.record(move |encoder| {
        let lpm_ctl_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&TonemapPassUniforms {
                lpm_ctl: lpm_ctl.ctl_words(),
                gamma_mode: gamma_mode as u32,
            }));
        let render_target_views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let swapchain_target = encoder.lock(
            swapchain_image.current_image().as_ref().unwrap(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        // HDR intermediary: read
        encoder.use_image_resource(
            render_target_views.hdr_output.image(),
            &mut hdr.state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // SDR target: read (egui / occluding content)
        encoder.use_image_resource(
            render_target_views.sdr_target.image(),
            &mut hdr.sdr_target_state,
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
                        image_view: render_target_views.hdr_output.vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: swapchain_target.linear_view().vk_handle(),
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: render_target_views.sdr_target.linear_view().vk_handle(),
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
            ],
        );

        encoder.dispatch(UVec3::new(
            hdr.extent.x.div_ceil(8),
            hdr.extent.y.div_ceil(8),
            1,
        ));
    });
}
