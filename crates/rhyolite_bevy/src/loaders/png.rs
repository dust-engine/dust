use std::fmt::{Debug, Display};

use crate::{AsyncQueues, QueuesRouter};
use bevy_asset::{AssetLoader, LoadedAsset};
use bevy_ecs::world::{FromWorld, World};
use rhyolite::ensure_image_layout;
use rhyolite::{
    ash::vk,
    copy_buffer_to_image,
    future::{GPUCommandFutureExt, RenderImage, RenderRes},
    macros::commands,
    ImageLike, ImageRequest, QueueRef,
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

#[derive()]
pub struct Image<T: ImageLike>(T);

#[derive(Debug)]
struct UnsupportedPngColorTypeError;
impl Display for UnsupportedPngColorTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}
impl std::error::Error for UnsupportedPngColorTypeError {}

impl AssetLoader for PngLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), bevy_asset::Error>> {
        Box::pin(async move {
            let decoder = png::Decoder::new(bytes);
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
                    png::ColorType::Rgb => 3,
                    png::ColorType::Rgba => 4,
                    _ => return Err(bevy_asset::Error::new(UnsupportedPngColorTypeError)),
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
                    png::ColorType::Rgb => 4,
                    png::ColorType::Rgba => 4,
                    _ => return Err(bevy_asset::Error::new(UnsupportedPngColorTypeError)),
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

            let image = self.allocator.create_device_image_uninit(&ImageRequest {
                image_type: vk::ImageType::TYPE_2D,
                format: match color_type {
                    (png::ColorType::Grayscale, png::BitDepth::Eight) => vk::Format::R8_UNORM,
                    (png::ColorType::Grayscale, png::BitDepth::Sixteen) => vk::Format::R16_UNORM,
                    (png::ColorType::Rgb, png::BitDepth::Four) => vk::Format::R4G4B4A4_UNORM_PACK16,
                    (png::ColorType::Rgb, png::BitDepth::Eight) => vk::Format::R8G8B8A8_UNORM,
                    (png::ColorType::Rgb, png::BitDepth::Sixteen) => vk::Format::R16G16B16_UNORM,
                    (png::ColorType::Rgba, png::BitDepth::Four) => {
                        vk::Format::R4G4B4A4_UNORM_PACK16
                    }
                    (png::ColorType::Rgba, png::BitDepth::Eight) => vk::Format::R8G8B8A8_UNORM,
                    (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
                        vk::Format::R16G16B16A16_UNORM
                    }
                    _ => return Err(bevy_asset::Error::new(UnsupportedPngColorTypeError)),
                },
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
            if img.subresource_range().layer_count == 1 {
                load_context.set_default_asset(LoadedAsset::new(crate::Image::new(img)));
            } else {
                load_context
                    .set_default_asset(LoadedAsset::new(crate::SlicedImageArray::new(img)?));
            }

            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["png"]
    }
}
