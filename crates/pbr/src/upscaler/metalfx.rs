//! MetalFX temporal denoised scaler (`MTL4FXTemporalDenoisedScaler`): denoises
//! and upscales the noisy radiance in one temporal pass on Apple GPUs.
//!
//! The Vulkan driver runs on Metal. `VK_EXT_metal_objects` exposes the
//! `MTLTexture` behind each image view and the Metal command buffer behind the
//! Vulkan one, so the scaler's work is encoded straight into dust's frame.

use bevy::prelude::UVec2;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTL4CompilerDescriptor, MTLDevice, MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTL4FXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use pumicite::ash::{self, vk};
use pumicite::command::CommandEncoder;
use pumicite::prelude::AsMTLDevice;
use pumicite::utils::metal::AsMTLCommandBuffer;
use pumicite::{Device, HasDevice};

use super::{UpscalerImage, UpscalerInputs};

/// Whether this GPU supports the Metal 4 temporal denoised scaler.
pub fn is_supported(device: &Device) -> bool {
    let device = device.mtl_device();
    // SAFETY: `supportsMetal4FX` only inspects the device.
    unsafe { MTLFXTemporalDenoisedScalerDescriptor::supportsMetal4FX(device) }
}

/// A denoised scaler for one pair of render and display resolutions.
#[derive(Clone)]
pub(crate) struct DenoisedScaler(Retained<ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>>);

// SAFETY: the scaler is only configured and encoded from one render system at
// a time.
unsafe impl Send for DenoisedScaler {}
unsafe impl Sync for DenoisedScaler {}

impl DenoisedScaler {
    /// Creates a scaler from `render_extent` inputs to `display_extent`. The
    /// texture formats match the images `ensure_hdr_target` allocates.
    pub fn new(
        device: &Device,
        render_extent: UVec2,
        display_extent: UVec2,
    ) -> Result<Self, String> {
        let device = device.mtl_device();
        let compiler = device
            .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
            .map_err(|e| format!("creating a Metal 4 compiler failed: {e}"))?;

        // SAFETY: every format and size is set before the scaler is created.
        let scaler = unsafe {
            let descriptor = MTLFXTemporalDenoisedScalerDescriptor::new();
            descriptor.setColorTextureFormat(MTLPixelFormat::RGBA16Float);
            descriptor.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
            descriptor.setDepthTextureFormat(MTLPixelFormat::R32Float);
            descriptor.setMotionTextureFormat(MTLPixelFormat::RG16Float);
            descriptor.setDiffuseAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm_sRGB);
            descriptor.setSpecularAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm);
            descriptor.setNormalTextureFormat(MTLPixelFormat::RGBA16Float);
            // Roughness is bound through a swizzled view of the normal image
            // (see `HdrRenderTargetViews::normal`), so it has the normal's format.
            descriptor.setRoughnessTextureFormat(MTLPixelFormat::RGBA16Float);
            descriptor.setInputWidth(render_extent.x as usize);
            descriptor.setInputHeight(render_extent.y as usize);
            descriptor.setOutputWidth(display_extent.x as usize);
            descriptor.setOutputHeight(display_extent.y as usize);
            descriptor.newTemporalDenoisedScalerWithDevice_compiler(&device, &compiler)
        }
        .ok_or("MetalFX refused the scaler descriptor")?;

        // Motion vectors are already in render-resolution pixels.
        unsafe {
            scaler.setMotionVectorScaleX(1.0);
            scaler.setMotionVectorScaleY(1.0);
        }
        Ok(Self(scaler))
    }

    /// Encodes the denoise + upscale of `inputs.color` into `inputs.output`.
    pub(crate) fn dispatch(&self, encoder: &mut CommandEncoder, inputs: &UpscalerInputs) {
        let device = encoder.device();
        let scaler = &*self.0;
        // SAFETY: the textures are exported from views the caller keeps locked
        // to this command buffer, and the scaler is fully configured before it
        // is encoded.
        unsafe {
            scaler.setColorTexture(Some(texture(device, &inputs.color)));
            scaler.setOutputTexture(Some(texture(device, &inputs.output)));
            scaler.setDepthTexture(Some(texture(device, &inputs.depth)));
            scaler.setMotionTexture(Some(texture(device, &inputs.motion_vectors)));
            scaler.setDiffuseAlbedoTexture(Some(texture(device, &inputs.diffuse_albedo)));
            scaler.setSpecularAlbedoTexture(Some(texture(device, &inputs.specular_albedo)));
            scaler.setNormalTexture(Some(texture(device, &inputs.normal)));
            scaler.setRoughnessTexture(Some(texture(device, &inputs.roughness)));
            // The color is raw radiance; the 1×1 exposure image holds the scale
            // the tonemap pass applies to it.
            scaler.setPreExposure(1.0);
            scaler.setExposureTexture(Some(texture(device, &inputs.exposure)));
            scaler.setJitterOffsetX(inputs.jitter.x);
            scaler.setJitterOffsetY(inputs.jitter.y);
            set_matrices(
                scaler,
                inputs.world_to_view.to_cols_array_2d(),
                inputs.view_to_clip.to_cols_array_2d(),
            );
            scaler.setShouldResetHistory(false);
            // Reverse-Z depth: 1 = near, 0 = far.
            scaler.setDepthReversed(true);
            scaler.encodeToCommandBuffer(encoder.buffer().mtl_command_buffer());
        }
        // Metal 4 command buffers do not retain what they use: keep the
        // scaler's internal resources alive until this command buffer
        // completes, even if `ensure_upscaler_session` replaces the session.
        encoder.retain(self.clone());
    }
}

/// The `MTLTexture` behind `image`'s view.
///
/// # Safety
///
/// The texture is owned by the Vulkan driver and stays valid only while the
/// image view does.
unsafe fn texture<'a>(
    device: &'a pumicite::Device,
    image: &UpscalerImage,
) -> &'a ProtocolObject<dyn MTLTexture> {
    let mut texture_info = vk::ExportMetalTextureInfoEXT::default()
        .image_view(image.view)
        .plane(vk::ImageAspectFlags::COLOR);
    let mut info = vk::ExportMetalObjectsInfoEXT::default();
    info.p_next = (&mut texture_info as *mut vk::ExportMetalTextureInfoEXT).cast();
    unsafe {
        device
            .extension::<ash::ext::metal_objects::Meta>()
            .export_metal_objects(&mut info);
        &*texture_info
            .mtl_texture
            .cast::<ProtocolObject<dyn MTLTexture>>()
    }
}

/// Sets the scaler's world-to-view and view-to-clip matrices.
///
/// objc2-metal-fx does not bind these setters because they take a
/// `simd_float4x4`, which objc2 cannot describe, so they are sent directly.
/// A `simd_float4x4` is four `simd_float4` columns passed in SIMD registers;
/// `float32x4_t` columns give the same calling convention on aarch64.
#[cfg(target_arch = "aarch64")]
unsafe fn set_matrices(
    scaler: &ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>,
    world_to_view: [[f32; 4]; 4],
    view_to_clip: [[f32; 4]; 4],
) {
    use core::arch::aarch64::float32x4_t;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SimdFloat4x4 {
        columns: [float32x4_t; 4],
    }
    unsafe impl objc2::encode::Encode for SimdFloat4x4 {
        const ENCODING: objc2::encode::Encoding = objc2::encode::Encoding::Struct(
            "?",
            &[objc2::encode::Encoding::Array(
                4,
                &objc2::encode::Encoding::None,
            )],
        );
    }

    fn to_simd(matrix: [[f32; 4]; 4]) -> SimdFloat4x4 {
        SimdFloat4x4 {
            // SAFETY: `[f32; 4]` and `float32x4_t` have the same size and layout.
            columns: matrix
                .map(|column| unsafe { core::mem::transmute::<[f32; 4], float32x4_t>(column) }),
        }
    }

    unsafe {
        let _: () = msg_send![scaler, setWorldToViewMatrix: to_simd(world_to_view)];
        let _: () = msg_send![scaler, setViewToClipMatrix: to_simd(view_to_clip)];
    }
}

/// Intel Macs: the `simd_float4x4` calling convention is not implemented, so
/// the scaler runs without camera matrices.
#[cfg(not(target_arch = "aarch64"))]
unsafe fn set_matrices(
    _scaler: &ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>,
    _world_to_view: [[f32; 4]; 4],
    _view_to_clip: [[f32; 4]; 4],
) {
}
