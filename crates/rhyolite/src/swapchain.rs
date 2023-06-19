use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use pin_project::pin_project;

use std::sync::Arc;
use std::{ops::Deref, pin::Pin};

use crate::future::{Access, RenderData, RenderImage, StageContextImage};
use crate::utils::format::ColorSpace;
use crate::{
    Device, HasDevice, ImageLike, ImageViewLike, PhysicalDevice, QueueFuture, QueueFuturePoll,
    QueueMask, QueueRef, QueueSubmissionContextExport, QueueSubmissionContextSemaphoreWait,
    QueueSubmissionType, SharingMode, Surface,
};

pub struct SwapchainLoader {
    loader: khr::Swapchain,
    device: Arc<Device>,
}

impl Deref for SwapchainLoader {
    type Target = khr::Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

impl SwapchainLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::Swapchain::new(device.instance(), &device);
        Self { loader, device }
    }
}

pub struct SwapchainInner {
    device: Arc<Device>,
    swapchain: vk::SwapchainKHR,
    images: Vec<(vk::Image, vk::ImageView)>,
    format: vk::Format,
    generation: u64,

    surface: Arc<Surface>,
    color_space: ColorSpace,
    extent: vk::Extent2D,
    layer_count: u32,
}

pub struct Swapchain {
    inner: Arc<SwapchainInner>,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.device
                .swapchain_loader()
                .destroy_swapchain(self.swapchain, None);
            for (_, view) in self.images.drain(..) {
                self.device.destroy_image_view(view, None);
            }
        }
    }
}

pub struct SwapchainCreateInfo<'a> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub image_sharing_mode: SharingMode<'a>,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
}

pub fn color_space_area(color_space: vk::ColorSpaceKHR) -> f32 {
    match color_space {
        vk::ColorSpaceKHR::SRGB_NONLINEAR => 0.112,
        vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => 0.151,
        vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => 0.152,
        vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => 0.152,
        vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => 0.5,
        vk::ColorSpaceKHR::BT709_LINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::BT2020_LINEAR_EXT => 0.212,
        vk::ColorSpaceKHR::HDR10_ST2084_EXT => 0.212,
        vk::ColorSpaceKHR::HDR10_HLG_EXT => 0.212,
        vk::ColorSpaceKHR::DOLBYVISION_EXT => 0.212,
        vk::ColorSpaceKHR::PASS_THROUGH_EXT => 0.0,
        vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD => 1.0,
        _ => 0.0,
    }
}

impl<'a> SwapchainCreateInfo<'a> {
    pub fn pick(
        surface: &Surface,
        pdevice: &PhysicalDevice,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<Self> {
        let formats = surface
            .pick_format(pdevice, usage)?
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
        Ok(Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: formats.format,
            image_color_space: formats.color_space,
            image_extent: Default::default(),
            image_array_layers: 1,
            image_usage: usage,
            image_sharing_mode: SharingMode::Exclusive,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: true,
        })
    }
}

impl HasDevice for Swapchain {
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

/// Unsafe APIs for Swapchain
impl Swapchain {
    pub fn surface(&self) -> &Arc<Surface> {
        &self.inner.surface
    }
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub fn create(
        device: Arc<Device>,
        surface: Arc<Surface>,
        info: SwapchainCreateInfo,
    ) -> VkResult<Self> {
        tracing::info!(format = ?info.image_format, color_space = ?info.image_color_space, usage = ?info.image_usage, "Creating swapchain");
        unsafe {
            let mut create_info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: surface.surface,
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                ..Default::default()
            };
            match &info.image_sharing_mode {
                SharingMode::Exclusive => (),
                SharingMode::Concurrent {
                    queue_family_indices,
                } => {
                    create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
                    create_info.p_queue_family_indices = queue_family_indices.as_ptr();
                }
            }
            let swapchain = device
                .swapchain_loader()
                .create_swapchain(&create_info, None)?;
            let images = get_swapchain_images(&device, swapchain, info.image_format)?;
            let inner = SwapchainInner {
                device,
                surface,
                swapchain,
                images,
                generation: 0,
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
    }

    pub fn recreate(&mut self, info: SwapchainCreateInfo) -> VkResult<()> {
        unsafe {
            let mut create_info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: self.inner.surface.surface,
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                old_swapchain: self.inner.swapchain,
                ..Default::default()
            };
            match &info.image_sharing_mode {
                SharingMode::Exclusive => (),
                SharingMode::Concurrent {
                    queue_family_indices,
                } => {
                    create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
                    create_info.p_queue_family_indices = queue_family_indices.as_ptr();
                }
            }
            let swapchain = self
                .inner
                .device
                .swapchain_loader()
                .create_swapchain(&create_info, None)?;

            let images = get_swapchain_images(self.device(), swapchain, info.image_format)?;
            let inner = SwapchainInner {
                device: self.inner.device.clone(),
                surface: self.inner.surface.clone(),
                swapchain,
                images,
                generation: self.inner.generation.wrapping_add(1),
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            self.inner = Arc::new(inner);
        }
        Ok(())
    }

    pub fn acquire_next_image(&mut self, semaphore: vk::Semaphore) -> AcquireFuture {
        let (image_indice, suboptimal) = unsafe {
            self.inner.device.swapchain_loader().acquire_next_image(
                self.inner.swapchain,
                !0,
                semaphore,
                vk::Fence::null(),
            )
        }
        .unwrap();
        let (image, view) = self.inner.images[image_indice as usize];
        let swapchain_image = SwapchainImage {
            device: self.device().clone(),
            swapchain: self.inner.swapchain,
            format: self.inner.format,
            image,
            view,
            indice: image_indice,
            suboptimal,
            generation: self.inner.generation,
            extent: self.inner.extent,
            layer_count: self.inner.layer_count,
            presented: false,
            color_space: self.inner.color_space.clone(),
        };
        AcquireFuture {
            image: Some(swapchain_image),
            semaphore,
        }
    }
}

pub struct SwapchainImage {
    device: Arc<Device>,
    swapchain: vk::SwapchainKHR,
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    indice: u32,
    suboptimal: bool,
    generation: u64,
    extent: vk::Extent2D,
    layer_count: u32,
    color_space: ColorSpace,
    presented: bool,
}
impl Drop for SwapchainImage {
    fn drop(&mut self) {
        if !self.presented && !std::thread::panicking() {
            panic!("SwapchainImage must be returned to the OS by calling Present!")
        }
    }
}
impl SwapchainImage {
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }
}
impl HasDevice for SwapchainImage {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl RenderData for SwapchainImage {}
impl ImageLike for SwapchainImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: self.layer_count,
        }
    }
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent.width,
            height: self.extent.height,
            depth: 1,
        }
    }

    fn format(&self) -> vk::Format {
        self.format
    }
}
impl ImageViewLike for SwapchainImage {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}

#[pin_project]
pub struct PresentFuture {
    queue: QueueRef,
    prev_queue: QueueMask,
    swapchain: Vec<RenderImage<SwapchainImage>>,
}

impl RenderImage<SwapchainImage> {
    pub fn present(self) -> PresentFuture {
        PresentFuture {
            queue: QueueRef::null(),
            prev_queue: QueueMask::empty(),
            swapchain: vec![self],
        }
    }
}

impl QueueFuture for PresentFuture {
    type Output = ();

    type RecycledState = ();

    type RetainedState = Vec<RenderImage<SwapchainImage>>;

    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut crate::SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        prev_queue: crate::QueueMask,
    ) {
        let this = self.project();
        *this.prev_queue = prev_queue;

        if this.queue.is_null() {
            let mut iter = prev_queue.iter();
            if let Some(inherited_queue) = iter.next() {
                *this.queue = inherited_queue;
                assert!(
                    iter.next().is_none(),
                    "Cannot use derived queue when the future depends on more than one queues"
                );
            } else {
                // Default to the first queue, if the queue does not have predecessor.
                *this.queue = QueueRef(0);
            }
        }
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();

        if !this.prev_queue.is_empty() {
            *this.prev_queue = QueueMask::empty();
            return QueueFuturePoll::Semaphore(Vec::new());
        }
        for swapchain in this.swapchain.iter() {
            let tracking = swapchain.res.tracking_info.borrow_mut();
            assert!(
                !tracking.queue_index.is_null(),
                "The swapchain image was never written to!"
            );

            // If we consider the queue present operation as a read, then we only need to syncronize with previous writes.
            ctx.queues[tracking.queue_index.0 as usize]
                .signals
                .insert((tracking.current_stage_access.write_stages, true));

            let export = QueueSubmissionContextExport::Image {
                image: StageContextImage {
                    image: swapchain.res.inner.image,
                    subresource_range: swapchain.res.inner.subresource_range(),
                    extent: swapchain.res.inner.extent(),
                },
                barrier: vk::MemoryBarrier2 {
                    src_stage_mask: tracking.current_stage_access.write_stages,
                    src_access_mask: tracking.current_stage_access.write_access,

                    // We set dst_stage_mask to be write_stages so that we can establish dependency
                    // between the layout transition and the semaphore signal operation.
                    dst_stage_mask: tracking.current_stage_access.write_stages,

                    // No need for memory dependency - handled automatically by the semaphore signal operation.
                    dst_access_mask: vk::AccessFlags2::empty(),
                    ..Default::default()
                },
                dst_queue_family: ctx.queues[this.queue.0 as usize].queue_family_index,
                src_layout: swapchain.layout.get(),
                dst_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            };
            ctx.queues[tracking.queue_index.0 as usize]
                .exports
                .push(export);

            ctx.queues[this.queue.0 as usize].waits.push(
                QueueSubmissionContextSemaphoreWait::WaitForSignal {
                    dst_stages: vk::PipelineStageFlags2::empty(),
                    queue: tracking.queue_index,
                    src_stages: tracking.current_stage_access.write_stages,
                },
            );
        }
        assert!(matches!(
            ctx.submission[this.queue.0 as usize],
            QueueSubmissionType::Unknown
        ));
        ctx.submission[this.queue.0 as usize] = QueueSubmissionType::Present(
            this.swapchain
                .iter()
                .map(|a| (a.res.inner.swapchain, a.res.inner.indice))
                .collect(),
        );
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output: (),
        }
    }

    fn dispose(mut self) -> Self::RetainedState {
        for i in self.swapchain.iter_mut() {
            i.inner_mut().presented = true;
        }
        self.swapchain
    }
}

#[pin_project]

pub struct AcquireFuture {
    image: Option<SwapchainImage>,
    semaphore: vk::Semaphore,
}
impl QueueFuture for AcquireFuture {
    type Output = RenderImage<SwapchainImage>;

    type RecycledState = ();

    type RetainedState = ();

    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut crate::SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        _prev_queue: QueueMask,
    ) {
    }

    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut crate::SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        let output = RenderImage::new(this.image.take().unwrap(), vk::ImageLayout::UNDEFINED);
        {
            let mut tracking = output.res.tracking_info.borrow_mut();
            tracking.untracked_semaphore = Some(*this.semaphore);
            tracking.current_stage_access = Access {
                read_stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                write_stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                ..Default::default()
            };
        }
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output,
        }
    }

    fn dispose(self) -> Self::RetainedState {}
}

unsafe fn get_swapchain_images(
    device: &Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
) -> VkResult<Vec<(vk::Image, vk::ImageView)>> {
    let images = device.swapchain_loader().get_swapchain_images(swapchain)?;
    let mut image_views: Vec<(vk::Image, vk::ImageView)> = Vec::with_capacity(images.len());
    for image in images.into_iter() {
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            },
            None,
        );
        match view {
            Ok(view) => image_views.push((image, view)),
            Err(err) => {
                // Destroy existing
                for (_image, view) in image_views.into_iter() {
                    device.destroy_image_view(view, None);
                }
                return Err(err);
            }
        }
    }
    Ok(image_views)
}
