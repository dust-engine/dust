use std::fmt::{Debug, Display};

use crate::{AsyncQueues, QueuesRouter, SlicedImageArray};
use bevy_asset::{AssetLoader, AsyncReadExt};
use bevy_ecs::world::{FromWorld, World};
use bevy_utils::BoxedFuture;
use rhyolite::ensure_image_layout;
use rhyolite::utils::format::{Format, FormatType, Permutation};
use rhyolite::{
    ash::vk,
    copy_buffer_to_image,
    future::{GPUCommandFutureExt, RenderImage, RenderRes},
    macros::commands,
    ImageRequest, QueueRef,
};

pub struct PngLoader {
    allocator: crate::Allocator,
    queues: AsyncQueues,
    transfer_queue: QueueRef,
}
impl FromWorld for PngLoader {
    fn from_world(world: &mut World) -> Self {
        let allocator = world.resource::<crate::Allocator>().clone();
        let queues = world.resource::<AsyncQueues>().clone();
        let transfer_queue = world
            .resource::<QueuesRouter>()
            .of_type(rhyolite::QueueType::Transfer);
        Self {
            allocator,
            queues,
            transfer_queue,
        }
    }
}

use serde::{Deserialize, Serialize};
use thiserror::Error;
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PngSettings {
    pub format_type: FormatType,
    pub format_permutation: Option<Permutation>,
}
impl Default for PngSettings {
    fn default() -> Self {
        Self {
            format_type: FormatType::UNorm,
            format_permutation: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum PngLoadingError {
    /// Possible reasons:
    /// - The png file operates in index mode.
    /// - The png file has a bit depth of 1 or 2.
    /// - Grayscale or Grayscale alpha image with a bit depth of 4.
    #[error("Png color type wrong")]
    UnsupportedPngColorTypeError,
    #[error("unsupported format: {0:?}")]
    UnsupportedFormatError(Format),
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("vulkan error: {0}")]
    VkError(#[from] vk::Result),
}


impl AssetLoader for PngLoader {
    // TODO: make different loaders for png img arrays and single images
    type Asset = SlicedImageArray;
    type Settings = PngSettings;
    type Error = PngLoadingError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy_asset::io::Reader,
        settings: &'a Self::Settings,
        _load_context: &'a mut bevy_asset::LoadContext,
    ) -> BoxedFuture<'a, Result<SlicedImageArray, PngLoadingError>> {
        Box::pin(async move {
            let mut img_data = Vec::new();
            reader.read_to_end(&mut img_data).await?;
            let decoder = png::Decoder::new(img_data.as_slice());
            let mut reader = decoder.read_info().unwrap();
            let color_type = reader.output_color_type();

            let num_frames = if reader.info().is_animated() {
                reader.info().animation_control().unwrap().num_frames
            } else {
                1
            };
            let src_sample_size = {
                // number of bits
                let size: u32 = match color_type.1 {
                    png::BitDepth::One => 1,
                    png::BitDepth::Two => 2,
                    png::BitDepth::Four => 4,
                    png::BitDepth::Eight => 8,
                    png::BitDepth::Sixteen => 16,
                } * match color_type.0 {
                    png::ColorType::Grayscale => 1,
                    png::ColorType::GrayscaleAlpha => 2,
                    png::ColorType::Rgb => 3,
                    png::ColorType::Rgba => 4,
                    png::ColorType::Indexed => {
                        return Err(PngLoadingError::UnsupportedPngColorTypeError)
                    }
                };
                size / 8
            };
            let dst_sample_size = {
                // number of bits
                let size: u32 = match color_type.1 {
                    png::BitDepth::One => 1,
                    png::BitDepth::Two => 2,
                    png::BitDepth::Four => 4,
                    png::BitDepth::Eight => 8,
                    png::BitDepth::Sixteen => 16,
                } * match color_type.0 {
                    png::ColorType::Grayscale => 1,
                    png::ColorType::GrayscaleAlpha => 2,
                    png::ColorType::Rgb => 4,
                    png::ColorType::Rgba => 4,
                    png::ColorType::Indexed => {
                        return Err(PngLoadingError::UnsupportedPngColorTypeError)
                    }
                };
                size / 8
            };
            let dst_frame_size = dst_sample_size * reader.info().width * reader.info().height;
            let dst_buf = self
                .allocator
                .create_staging_buffer((dst_frame_size * num_frames) as u64)?;
            let dst_buf_slice = dst_buf.contents_mut().unwrap();

            // Allocate the output buffer.
            let mut buf = vec![0; reader.output_buffer_size()];

            for i in 0..num_frames {
                // Read the next frame. An APNG might contain multiple frames.
                buf.fill(0);
                let info = reader.next_frame(&mut buf).unwrap();
                // Grab the bytes of the image.
                let bytes = &buf[..info.buffer_size()];
                let dst_slice = &mut dst_buf_slice
                    [(dst_frame_size * i) as usize..(dst_frame_size * i + dst_frame_size) as usize];
                if dst_slice.len() != bytes.len() {
                    // For Rgb => Rgba copy.
                    let num_samples = reader.info().width * reader.info().height;
                    for j in 0..num_samples {
                        let dst_sample = &mut dst_slice
                            [(dst_sample_size * j) as usize..(dst_sample_size * (j + 1)) as usize];
                        let src_sample = &bytes
                            [(src_sample_size * j) as usize..(src_sample_size * (j + 1)) as usize];
                        dst_sample[0..src_sample.len()].copy_from_slice(src_sample);
                        dst_sample[src_sample.len()..].fill(0);
                    }
                } else {
                    dst_slice.copy_from_slice(bytes);
                }
            }

            #[rustfmt::skip]
            let (r, g, b, a, permutation) = match color_type {
                (png::ColorType::Grayscale, png::BitDepth::Eight) => (8, 0, 0, 0, Permutation::R),
                (png::ColorType::Grayscale, png::BitDepth::Sixteen) => (16, 0, 0, 0, Permutation::R),
                (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => (8, 8, 0, 0, Permutation::RG),
                (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen) => (16, 16, 0, 0, Permutation::RG),
                (png::ColorType::Rgb | png::ColorType::Rgba, png::BitDepth::Four) => (4, 4, 4, 4, Permutation::RGBA),
                (png::ColorType::Rgb | png::ColorType::Rgba, png::BitDepth::Eight) => (8, 8, 8, 8, Permutation::RGBA),
                (png::ColorType::Rgb | png::ColorType::Rgba, png::BitDepth::Sixteen) => (16, 16, 16, 16, Permutation::RGBA),
                (_, png::BitDepth::One | png::BitDepth::Two)
                | (png::ColorType::Indexed, _)
                | (
                    png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha,
                    png::BitDepth::Four,
                ) => {
                    return Err(PngLoadingError::UnsupportedPngColorTypeError)
                }
            };

            let format = Format {
                r,
                g,
                b,
                a,
                ty: settings.format_type,
                permutation: settings.format_permutation.unwrap_or(permutation),
            };

            let image = self.allocator.create_device_image_uninit(&ImageRequest {
                image_type: vk::ImageType::TYPE_2D,
                format: format
                    .try_into()
                    .map_err(|format| PngLoadingError::UnsupportedFormatError(format))?,
                extent: vk::Extent3D {
                    width: reader.info().width,
                    height: reader.info().height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: num_frames,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                sharing_mode: rhyolite::SharingMode::Exclusive,
                initial_layout: vk::ImageLayout::UNDEFINED,
            })?;

            let img = commands! {
                let buf = RenderRes::new(dst_buf);
                let mut img = RenderImage::new(image, vk::ImageLayout::UNDEFINED);
                copy_buffer_to_image(&buf, &mut img, vk::ImageLayout::TRANSFER_DST_OPTIMAL).await;
                ensure_image_layout(&mut img, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).await;
                retain!(buf);
                img
            }
            .schedule_on_queue(self.transfer_queue);

            let img = self
                .queues
                .submit(img, &mut Default::default())
                .await
                .into_inner();
            Ok(crate::SlicedImageArray::new(img)?)
        })
    }

    fn extensions(&self) -> &[&str] {
        &["png"]
    }
}
