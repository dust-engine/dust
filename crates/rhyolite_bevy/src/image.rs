use bevy_reflect::{TypePath, TypeUuid};
use rhyolite::HasDevice;
use rhyolite::{
    ash::{prelude::VkResult, vk},
    ImageArraySlicedViews, ImageLike, ResidentImage,
};

#[derive(TypeUuid, TypePath)]
#[uuid = "55b9fd4f-bd81-4a49-94ea-21d50236f6b0"]
pub struct Image(ResidentImage);
impl Image {
    pub fn new(image: ResidentImage) -> Self {
        Self(image)
    }
}
impl HasDevice for Image {
    fn device(&self) -> &std::sync::Arc<rhyolite::Device> {
        self.0.device()
    }
}

impl ImageLike for Image {
    fn raw_image(&self) -> rhyolite::ash::vk::Image {
        self.0.raw_image()
    }
    fn subresource_range(&self) -> rhyolite::ash::vk::ImageSubresourceRange {
        self.0.subresource_range()
    }
    fn offset(&self) -> rhyolite::ash::vk::Offset3D {
        self.0.offset()
    }
    fn format(&self) -> rhyolite::ash::vk::Format {
        self.0.format()
    }
    fn extent(&self) -> rhyolite::ash::vk::Extent3D {
        self.0.extent()
    }
}

#[derive(TypeUuid, TypePath)]
#[uuid = "55b9fd4f-bd81-4a49-94ea-21d50236f6b1"]
pub struct SlicedImageArray(ImageArraySlicedViews<ResidentImage>);
impl SlicedImageArray {
    pub fn new(image: ResidentImage) -> VkResult<Self> {
        let image = ImageArraySlicedViews::new(image, vk::ImageViewType::TYPE_2D)?;
        Ok(Self(image))
    }
    pub fn slice(&self, index: usize) -> rhyolite::ImageArraySliceView<ResidentImage> {
        self.0.slice(index)
    }
    pub fn slice_mut(&mut self, index: usize) -> rhyolite::ImageArraySliceViewMut<ResidentImage> {
        self.0.slice_mut(index)
    }
}
impl HasDevice for SlicedImageArray {
    fn device(&self) -> &std::sync::Arc<rhyolite::Device> {
        self.0.device()
    }
}
impl ImageLike for SlicedImageArray {
    fn raw_image(&self) -> vk::Image {
        self.0.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.0.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.0.extent()
    }

    fn format(&self) -> vk::Format {
        self.0.format()
    }

    fn offset(&self) -> vk::Offset3D {
        self.0.offset()
    }
}
