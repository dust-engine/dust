//! Screenshots: copies of a frame the game draws, for the host to save.
//!
//! [`Screenshots::take`] asks for one of the next frame. Once the tonemap pass
//! has drawn it, [`screenshot_pass`] copies it into a buffer the host can read,
//! and the future `take` returned finishes once the GPU has.

use std::future::Future;

use bevy::prelude::*;
use bevy_pumicite::{SubmissionState, prelude::ComputePipeline, staging::UniformRingBuffer};
use dust_profiler::GpuProfiler;
use dust_ui::Swapchain;
// `bevy_pumicite`'s swapchain image has these methods without a trait.
#[cfg(any(target_vendor = "apple", windows))]
use dust_ui::SwapchainImageMetadata as _;
use futures_channel::oneshot;
use pumicite::{
    Allocator,
    ash::vk,
    buffer::{Buffer, BufferLike},
    command::{CommandEncoder, GPURefMut},
    image::{Image, ImageExt, ImageLike},
    sync::GPUMutex,
    tracking::{Access, ResourceState},
};

use crate::{HdrRenderTarget, HdrRenderTargetViewsExt, PbrRenderState, tonemap::record_tonemap};

/// Which image of the frame a screenshot copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenshotKind {
    /// The frame tone-mapped for an SDR display as 8-bit sRGB, saved as PNG.
    PresentedSDR,
    /// The frame tone-mapped for the current display, saved as KTX2
    Presented,
    /// RGBA16F raw HDR image, saved as KTX2
    Scene,
}

/// A frame's image, read back from the GPU.
pub struct Screenshot {
    pub extent: UVec2,
    pub format: vk::Format,
    /// How the texels' values map to colors.
    pub color_space: vk::ColorSpaceKHR,
    texels: Box<Buffer>,
}

impl Screenshot {
    /// The texels, row after row from the top, without padding.
    pub fn texels(&self) -> &[u8] {
        self.texels
            .as_slice()
            .expect("the copy's buffer is host-visible")
    }
}

/// A screenshot the GPU is copying.
struct Copying {
    texels: GPUMutex<Buffer>,
    extent: UVec2,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
}

/// The screenshots asked for, which the next frame drawn copies.
#[derive(Resource, Default)]
pub struct Screenshots {
    requests: Vec<(ScreenshotKind, oneshot::Sender<Result<Copying, String>>)>,
}

impl Screenshots {
    /// Asks for a screenshot of the next frame drawn. The future finishes with
    /// it once the GPU has copied it, or with why it can't.
    pub fn take_screenshot(
        &mut self,
        kind: ScreenshotKind,
    ) -> impl Future<Output = Result<Screenshot, String>> + Send + use<> {
        let (sender, receiver) = oneshot::channel();
        self.requests.push((kind, sender));
        async move {
            let copying = receiver
                .await
                .map_err(|_| "the game exited before drawing another frame".to_owned())??;
            let texels = copying
                .texels
                .unwrap_block_async()
                .await
                .map_err(|error| format!("the GPU failed: {error}"))?;
            Ok(Screenshot {
                extent: copying.extent,
                format: copying.format,
                color_space: copying.color_space,
                texels,
            })
        }
    }
}

/// Copies the frame for the screenshots asked for, after the tonemap pass has
/// drawn it. For an SDR screenshot, the tonemap pass first draws the frame
/// again, into an 8-bit image of its own.
#[allow(clippy::too_many_arguments)]
pub(crate) fn screenshot_pass(
    mut ctx: SubmissionState,
    mut screenshots: ResMut<Screenshots>,
    state: Res<PbrRenderState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    allocator: Res<Allocator>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut swapchain: If<ResMut<Swapchain>>,
    hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    if screenshots.requests.is_empty() {
        return;
    }
    // Without them, the tonemap pass drew nothing either: the screenshots wait
    // for a frame it draws.
    let (Some(mut hdr_target), Some(pipeline)) =
        (hdr_target, compute_pipelines.get(&state.tonemap_pipeline))
    else {
        return;
    };
    let hdr = &mut *hdr_target;
    let pipeline = pipeline.clone().into_inner();
    let extent = hdr.display_extent;
    let presented_color_space = swapchain.current_image().unwrap().color_space();

    for (kind, sender) in std::mem::take(&mut screenshots.requests) {
        // Nothing waits for it anymore.
        if sender.is_canceled() {
            continue;
        }
        let (format, texel_size, color_space) = match kind {
            ScreenshotKind::PresentedSDR => (
                vk::Format::R8G8B8A8_UNORM,
                4,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            // Both platforms' swapchain images have this format.
            ScreenshotKind::Presented => (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                4,
                presented_color_space,
            ),
            ScreenshotKind::Scene => (
                vk::Format::R16G16B16A16_SFLOAT,
                8,
                vk::ColorSpaceKHR::BT709_LINEAR_EXT,
            ),
        };
        // The buffer the host reads, and for an SDR screenshot, the image the
        // tonemap pass draws into again.
        let size = u64::from(extent.x) * u64::from(extent.y) * texel_size;
        let allocated = Buffer::new_dynamic(
            allocator.clone(),
            size,
            texel_size,
            vk::BufferUsageFlags::TRANSFER_DST,
        )
        .and_then(|texels| {
            let image = match kind {
                ScreenshotKind::PresentedSDR => Some(
                    Image::new_device(
                        allocator.clone(),
                        &vk::ImageCreateInfo {
                            image_type: vk::ImageType::TYPE_2D,
                            format,
                            extent: vk::Extent3D {
                                width: extent.x,
                                height: extent.y,
                                depth: 1,
                            },
                            mip_levels: 1,
                            array_layers: 1,
                            samples: vk::SampleCountFlags::TYPE_1,
                            tiling: vk::ImageTiling::OPTIMAL,
                            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            ..Default::default()
                        },
                    )?
                    .create_full_view()?,
                ),
                ScreenshotKind::Presented | ScreenshotKind::Scene => None,
            };
            Ok((GPUMutex::new(texels), image))
        });
        let (texels, sdr_image) = match allocated {
            Ok(allocated) => allocated,
            Err(error) => {
                sender
                    .send(Err(format!("can't allocate a copy of the frame: {error}")))
                    .ok();
                continue;
            }
        };

        ctx.record(|encoder, _| {
            let destination = encoder.lock(&texels, vk::PipelineStageFlags2::COPY);
            match kind {
                ScreenshotKind::PresentedSDR => {
                    let image = encoder.retain(sdr_image.unwrap());
                    let mut image_state = ResourceState::default();
                    record_tonemap(
                        encoder,
                        pipeline.clone(),
                        hdr,
                        vk::ColorSpaceKHR::SRGB_NONLINEAR,
                        image.full_view(),
                        &mut image_state,
                        &mut uniform_ring_buffer,
                        profiler.as_deref_mut(),
                        "screenshot tonemap",
                    );
                    record_copy(encoder, image, &mut image_state, destination, extent);
                }
                ScreenshotKind::Presented => {
                    let image = encoder.lock(
                        swapchain.current_image().unwrap(),
                        vk::PipelineStageFlags2::COPY,
                    );
                    record_copy(encoder, image, swapchain.state_mut(), destination, extent);
                }
                ScreenshotKind::Scene => {
                    let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COPY);
                    record_copy(
                        encoder,
                        views.hdr_denoised_output(),
                        &mut hdr.hdr_denoised_target_state,
                        destination,
                        extent,
                    );
                }
            }
        });
        sender
            .send(Ok(Copying {
                texels,
                extent,
                format,
                color_space,
            }))
            .ok();
    }
}

/// Records copying all of `image`, of `extent`, whose access `state` tracks,
/// into `texels` for the host to read.
fn record_copy<'a, I: ImageLike>(
    encoder: &mut CommandEncoder<'a>,
    image: GPURefMut<'a, I>,
    state: &mut ResourceState,
    texels: GPURefMut<'a, Buffer>,
    extent: UVec2,
) {
    encoder.use_image_resource(
        image,
        state,
        Access::COPY_READ,
        vk::ImageLayout::GENERAL,
        0..1,
        0..1,
        false,
    );
    encoder.emit_barriers();
    encoder.copy_image_to_buffer_with_layout(
        image,
        vk::ImageLayout::GENERAL,
        texels,
        &[vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            ..Default::default()
        }],
    );
    // Waiting for the command buffer doesn't make the copy visible to the
    // host by itself; that takes a memory dependency on host reads.
    encoder.memory_barrier(Access::COPY_WRITE, Access::HOST_READ);
    encoder.emit_barriers();
}
