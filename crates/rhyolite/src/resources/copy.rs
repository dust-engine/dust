use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    task::Poll,
};

use ash::vk;
use pin_project::pin_project;

use crate::{
    future::{
        CommandBufferRecordContext, GPUCommandFuture, RenderData, RenderImage, RenderRes,
        StageContext,
    },
    BufferLike, HasDevice, ImageLike,
};

#[pin_project]
pub struct CopyBufferToImageFuture<
    S: BufferLike + RenderData,
    T: ImageLike + RenderData,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderImage<T>>,
> {
    pub src: SRef,
    pub dst: TRef,

    pub buffer_row_length: u32,
    pub buffer_image_height: u32,

    /// The initial x, y, z offsets in texels of the sub-region of the source or destination image data.
    pub image_offset: vk::Offset3D,

    /// The size in texels of the image to copy in width, height and depth.
    pub image_extent: vk::Extent3D,

    pub target_layout: vk::ImageLayout,
}
impl<
        S: BufferLike + RenderData,
        T: ImageLike + RenderData,
        SRef: Deref<Target = RenderRes<S>>,
        TRef: DerefMut<Target = RenderImage<T>>,
    > GPUCommandFuture for CopyBufferToImageFuture<S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        let src = this.src.deref().inner();
        let dst = this.dst.deref_mut().inner_mut();

        let dst_subresource_range = dst.subresource_range();
        let region = vk::BufferImageCopy {
            buffer_offset: src.offset(),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_subresource_range.aspect_mask,
                mip_level: dst_subresource_range.base_mip_level,
                base_array_layer: dst_subresource_range.base_array_layer,
                layer_count: dst_subresource_range.layer_count,
            },
            buffer_image_height: *this.buffer_image_height,
            buffer_row_length: *this.buffer_row_length,
            image_extent: *this.image_extent,
            image_offset: *this.image_offset,
        };
        ctx.record(|ctx, command_buffer| unsafe {
            ctx.device().cmd_copy_buffer_to_image(
                command_buffer,
                src.raw_buffer(),
                dst.raw_image(),
                *this.target_layout,
                &[region],
            );
        });
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        ctx.read(
            this.src,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );

        ctx.write_image(
            this.dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            *this.target_layout,
        );
    }
}

/// Copy data for a tightly packed image from a buffer to an image object, covering the entire extent of the image.
pub fn copy_buffer_to_image<
    S: BufferLike + RenderData,
    T: ImageLike + RenderData,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderImage<T>>,
>(
    src: SRef,
    dst: TRef,
    target_layout: vk::ImageLayout,
) -> CopyBufferToImageFuture<S, T, SRef, TRef> {
    let dst_subresource_range = dst.inner().subresource_range();
    assert_eq!(dst_subresource_range.level_count, 1);
    assert_ne!(dst_subresource_range.layer_count, 0);
    assert_ne!(
        dst_subresource_range.layer_count,
        vk::REMAINING_ARRAY_LAYERS
    );

    let image_extent = dst.inner().extent();
    CopyBufferToImageFuture {
        src,
        dst,
        image_extent,
        image_offset: vk::Offset3D::default(),

        //  If either of these values is zero, that aspect of the buffer memory is considered to be tightly packed according to the imageExtent.
        buffer_image_height: 0,
        buffer_row_length: 0,
        target_layout,
    }
}

#[pin_project]
pub struct EnsureImageLayoutFuture<
    T: ImageLike + RenderData,
    TRef: DerefMut<Target = RenderImage<T>>,
> {
    pub dst: TRef,
    pub target_layout: vk::ImageLayout,
}
impl<T: ImageLike + RenderData, TRef: DerefMut<Target = RenderImage<T>>> GPUCommandFuture
    for EnsureImageLayoutFuture<T, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        // Guard against all future reads.
        ctx.read_image(
            this.dst,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
            *this.target_layout,
        );
    }
}

pub fn ensure_image_layout<T: ImageLike + RenderData, TRef: DerefMut<Target = RenderImage<T>>>(
    dst: TRef,
    target_layout: vk::ImageLayout,
) -> EnsureImageLayoutFuture<T, TRef> {
    EnsureImageLayoutFuture { dst, target_layout }
}
