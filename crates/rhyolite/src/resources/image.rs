use ash::{prelude::VkResult, vk};
use std::{ops::DerefMut, sync::Arc};

use crate::{
    future::{GPUCommandFuture, RenderData, RenderImage},
    Allocator, Device, HasDevice, SharingMode,
};

pub trait ImageLike: HasDevice {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
    fn extent(&self) -> vk::Extent3D;
    fn offset(&self) -> vk::Offset3D {
        Default::default()
    }
    fn format(&self) -> vk::Format;
}

pub trait ImageExt {
    fn crop(self, extent: vk::Extent3D, offset: vk::Offset3D) -> ImageSubregion<Self>
    where
        Self: ImageLike + Sized,
    {
        let sub_offset = self.offset();
        let sub_extent = self.extent();

        let offset = vk::Offset3D {
            x: sub_offset.x + offset.x,
            y: sub_offset.y + offset.y,
            z: sub_offset.z + offset.z,
        };
        assert!(
            extent.width <= sub_extent.width
                && extent.height <= sub_extent.height
                && extent.depth <= sub_extent.depth
        );
        ImageSubregion {
            inner: self,
            extent,
            offset,
        }
    }
    fn as_2d_view(self) -> VkResult<super::image_view::ImageView<Self>>
    where
        Self: ImageLike + Sized,
    {
        super::image_view::ImageView::new(self, vk::ImageViewType::TYPE_2D)
    }
}
impl<T> ImageExt for T where T: ImageLike {}

pub struct ImageSubregion<T: ImageLike> {
    inner: T,
    extent: vk::Extent3D,
    offset: vk::Offset3D,
}
impl<T: ImageLike> RenderData for ImageSubregion<T> {}
impl<T: ImageLike> ImageSubregion<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}
impl<T: ImageLike> HasDevice for ImageSubregion<T> {
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
impl<T: ImageLike> ImageLike for ImageSubregion<T> {
    fn raw_image(&self) -> vk::Image {
        self.inner.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.inner.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    fn offset(&self) -> vk::Offset3D {
        self.offset
    }

    fn format(&self) -> vk::Format {
        self.inner.format()
    }
}

pub struct ResidentImage {
    allocator: Allocator,
    image: vk::Image,
    allocation: vma::Allocation,
    extent: vk::Extent3D,
    format: vk::Format,
    level_count: u32,
    layer_count: u32,
}
impl RenderData for ResidentImage {}

impl HasDevice for ResidentImage {
    fn device(&self) -> &Arc<Device> {
        self.allocator.device()
    }
}

impl ImageLike for ResidentImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        let aspect_mask = match self.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::COLOR,
        };
        vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: self.level_count,
            base_array_layer: 0,
            layer_count: self.layer_count,
        }
    }

    fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for ResidentImage {
    fn drop(&mut self) {
        tracing::debug!(image = ?self.image, "drop image");
        unsafe {
            self.allocator
                .inner()
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}

#[derive(Clone)]
pub struct ImageRequest<'a> {
    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub sharing_mode: SharingMode<'a>,
    pub initial_layout: vk::ImageLayout,
}
impl<'a> Default for ImageRequest<'a> {
    fn default() -> Self {
        Self {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D::default(),
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::empty(),
            sharing_mode: SharingMode::Exclusive,
            initial_layout: vk::ImageLayout::UNDEFINED,
        }
    }
}

impl Allocator {
    pub fn create_device_image_uninit(
        &self,
        image_request: &ImageRequest,
    ) -> VkResult<ResidentImage> {
        use vma::Alloc;
        let mut build_info = vk::ImageCreateInfo {
            flags: vk::ImageCreateFlags::empty(),
            image_type: image_request.image_type,
            format: image_request.format,
            extent: image_request.extent,
            mip_levels: image_request.mip_levels,
            array_layers: image_request.array_layers,
            samples: image_request.samples,
            tiling: image_request.tiling,
            usage: image_request.usage,
            initial_layout: image_request.initial_layout,
            ..Default::default()
        };
        match image_request.sharing_mode {
            SharingMode::Concurrent {
                queue_family_indices,
            } => {
                build_info.sharing_mode = vk::SharingMode::CONCURRENT;
                build_info.p_queue_family_indices = queue_family_indices.as_ptr();
                build_info.queue_family_index_count = queue_family_indices.len() as u32;
            }
            _ => (),
        };
        let create_info = vma::AllocationCreateInfo {
            flags: vma::AllocationCreateFlags::empty(),
            usage: vma::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) = unsafe { self.inner().create_image(&build_info, &create_info) }?;
        Ok(ResidentImage {
            allocator: self.clone(),
            image,
            allocation,
            extent: image_request.extent,
            format: image_request.format,
            layer_count: image_request.array_layers,
            level_count: image_request.mip_levels,
        })
    }
}

pub fn clear_image<Img: ImageLike + RenderData, ImgRef: DerefMut<Target = RenderImage<Img>>>(
    image: ImgRef,
    value: vk::ClearColorValue,
) -> ClearImageFuture<Img, ImgRef> {
    ClearImageFuture { image, value }
}

pub struct ClearImageFuture<
    Img: ImageLike + RenderData,
    ImgRef: DerefMut<Target = RenderImage<Img>>,
> {
    image: ImgRef,
    value: vk::ClearColorValue,
}
impl<Img: ImageLike + RenderData, ImgRef: DerefMut<Target = RenderImage<Img>>> GPUCommandFuture
    for ClearImageFuture<Img, ImgRef>
{
    type Output = ();

    type RetainedState = ();

    type RecycledState = ();

    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut crate::future::CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        ctx.record(|ctx, command_buffer| unsafe {
            let range = self.image.inner().subresource_range();
            ctx.device().cmd_clear_color_image(
                command_buffer,
                self.image.inner().raw_image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &self.value,
                &[range],
            );
        });
        std::task::Poll::Ready(Default::default())
    }

    fn context(self: std::pin::Pin<&mut Self>, ctx: &mut crate::future::StageContext) {
        let image = unsafe { &mut self.get_unchecked_mut().image };
        ctx.write_image(
            image,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
    }
}
