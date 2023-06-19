use std::ops::Index;

use ash::{prelude::VkResult, vk};

use crate::{HasDevice, ImageLike, Sampler};

pub trait ImageViewLike: ImageLike {
    fn raw_image_view(&self) -> vk::ImageView;
}

pub trait ImageViewExt: ImageViewLike {
    fn as_descriptor(&self, image_layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            image_layout,
            image_view: self.raw_image_view(),
            sampler: vk::Sampler::null(),
        }
    }
    fn as_descriptor_with_sampler(
        &self,
        image_layout: vk::ImageLayout,
        sampler: &Sampler,
    ) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            image_layout,
            image_view: self.raw_image_view(),
            sampler: unsafe { sampler.raw() },
        }
    }
}
impl<T> ImageViewExt for T where T: ImageViewLike {}

pub struct ImageView<T: ImageLike> {
    image: T,
    view: vk::ImageView,
}

impl<T: ImageLike> Drop for ImageView<T> {
    fn drop(&mut self) {
        unsafe { self.image.device().destroy_image_view(self.view, None) }
    }
}
impl<T: ImageLike> HasDevice for ImageView<T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.image.device()
    }
}
impl<T: ImageLike> ImageLike for ImageView<T> {
    fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.image.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.image.extent()
    }

    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn offset(&self) -> vk::Offset3D {
        self.image.offset()
    }
}
impl<T: ImageLike> ImageViewLike for ImageView<T> {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}

impl<T: ImageLike> ImageView<T> {
    pub fn new(image: T, ty: vk::ImageViewType) -> VkResult<Self> {
        let view = unsafe {
            image.device().create_image_view(
                &vk::ImageViewCreateInfo {
                    image: image.raw_image(),
                    view_type: ty,
                    format: image.format(),
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    },
                    subresource_range: image.subresource_range(),
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(Self { view, image })
    }
}

pub struct ImageArraySlicedViews<T: ImageLike> {
    image: T,
    views: Vec<vk::ImageView>,
}
impl<T: ImageLike> Drop for ImageArraySlicedViews<T> {
    fn drop(&mut self) {
        for view in self.views.iter() {
            unsafe { self.image.device().destroy_image_view(*view, None) }
        }
    }
}
impl<T: ImageLike> ImageArraySlicedViews<T> {
    pub fn new(image: T, ty: vk::ImageViewType) -> VkResult<Self> {
        assert_ne!(
            image.subresource_range().layer_count,
            vk::REMAINING_ARRAY_LAYERS
        );
        assert_ne!(image.subresource_range().layer_count, 0);

        let mut views: Vec<vk::ImageView> =
            Vec::with_capacity(image.subresource_range().layer_count as usize);
        for i in 0..image.subresource_range().layer_count {
            let view = unsafe {
                image.device().create_image_view(
                    &vk::ImageViewCreateInfo {
                        image: image.raw_image(),
                        view_type: ty,
                        format: image.format(),
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            base_array_layer: i,
                            layer_count: 1,
                            ..image.subresource_range()
                        },
                        ..Default::default()
                    },
                    None,
                )
            };
            match view {
                Ok(view) => views.push(view),
                Err(err) => unsafe {
                    for view in views {
                        image.device().destroy_image_view(view, None);
                    }
                    return Err(err);
                },
            }
        }
        Ok(Self { views, image })
    }
}
impl<T: ImageLike> HasDevice for ImageArraySlicedViews<T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.image.device()
    }
}
impl<T: ImageLike> ImageLike for ImageArraySlicedViews<T> {
    fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.image.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.image.extent()
    }

    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn offset(&self) -> vk::Offset3D {
        self.image.offset()
    }
}
pub struct ImageArraySliceView<'a, T: ImageLike> {
    image: &'a ImageArraySlicedViews<T>,
    view: vk::ImageView,
    index: u32,
}
impl<'a, T: ImageLike> HasDevice for ImageArraySliceView<'a, T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.image.device()
    }
}
impl<'a, T: ImageLike> ImageLike for ImageArraySliceView<'a, T> {
    fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            base_array_layer: self.index,
            layer_count: 1,
            ..self.image.subresource_range()
        }
    }

    fn extent(&self) -> vk::Extent3D {
        self.image.extent()
    }

    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn offset(&self) -> vk::Offset3D {
        self.image.offset()
    }
}
impl<'a, T: ImageLike> ImageViewLike for ImageArraySliceView<'a, T> {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}
pub struct ImageArraySliceViewMut<'a, T: ImageLike> {
    image: &'a mut ImageArraySlicedViews<T>,
    view: vk::ImageView,
    index: u32,
}

impl<'a, T: ImageLike> HasDevice for ImageArraySliceViewMut<'a, T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.image.device()
    }
}
impl<'a, T: ImageLike> ImageLike for ImageArraySliceViewMut<'a, T> {
    fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            base_array_layer: self.index,
            layer_count: 1,
            ..self.image.subresource_range()
        }
    }

    fn extent(&self) -> vk::Extent3D {
        self.image.extent()
    }

    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn offset(&self) -> vk::Offset3D {
        self.image.offset()
    }
}
impl<'a, T: ImageLike> ImageViewLike for ImageArraySliceViewMut<'a, T> {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}
impl<T: ImageLike> ImageArraySlicedViews<T> {
    pub fn slice(&self, index: usize) -> ImageArraySliceView<T> {
        // TODO: convert to Index and IndexMut trait implementations, pending Rust GAT
        ImageArraySliceView {
            image: self,
            index: index as u32,
            view: self.views[index],
        }
    }
    pub fn slice_mut(&mut self, index: usize) -> ImageArraySliceViewMut<T> {
        // TODO: convert to Index and IndexMut trait implementations, pending Rust GAT
        ImageArraySliceViewMut {
            view: self.views[index],
            index: index as u32,
            image: self,
        }
    }
}
