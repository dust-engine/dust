//! The game's [`Swapchain`](crate::Swapchain) on macOS, made from the display link's drawables instead of
//! a Vulkan swapchain.
//!
//! Each frame, [`acquire`] imports the drawable's texture into Vulkan as a `VkImage`
//! (`VK_EXT_metal_objects`) and makes it the swapchain's current image; [`present`]
//! presents the drawable once the frame's work is committed. kosmickrisp runs the
//! Vulkan queue on a Metal 4 command queue and commits to it inside `vkQueueSubmit`,
//! so both talk to that queue directly: `waitForDrawable` before the frame's work,
//! `signalDrawable` after it, as Metal 4 requires of anything drawing into a
//! drawable.

use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;
use bevy_pumicite::{
    CreateDevice,
    pumicite::{
        Device,
        ash::{ext, vk, vk::TaggedStructure as _},
        device::DeviceBuilder,
        image::ImageLike,
        project_host_metadata,
        sync::GPUMutex,
        tracking::ResourceState,
        utils::AsVkHandle,
        utils::metal::{AsMTLCommandQueue, AsMTLDevice},
    },
    queue::{Queue, RenderQueue},
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTL4CommandQueue, MTLDrawable, MTLResource, MTLTexture};
use objc2_quartz_core::CAMetalDrawable;

use crate::SwapchainSet;

/// Inserts the [`MetalDrawable`], and acquires and presents its images around
/// [`SwapchainSet`].
pub(super) struct CAMetalDrawablePlugin;

impl Plugin for CAMetalDrawablePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MetalDrawable {
            drawable: None,
            image: None,
            state: ResourceState::default(),
        })
        .add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                device_builder
                    .enable_extension::<ext::metal_objects::Meta>()
                    .unwrap();
            })
            .before(CreateDevice),
        )
        .add_systems(
            PostUpdate,
            (acquire.before(SwapchainSet), present.after(SwapchainSet)),
        );
    }
}

/// The swapchain's state on macOS: the frame's drawable, and its texture imported
/// into Vulkan.
#[derive(Resource)]
pub struct MetalDrawable {
    /// The drawable of the frame being updated, from
    /// `BevyDisplayLinkDelegate::update` until [`present`].
    pub(crate) drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    /// The drawable's texture imported as a Vulkan image, from [`acquire`] until
    /// [`present`]. Metal makes no promise to hand out the same textures again, so
    /// each frame imports its own; dropping one still locked by the frame's work
    /// defers destroying it until that work is done.
    image: Option<GPUMutex<DrawableImage>>,
    /// The current image's resource state.
    state: ResourceState,
}

// SAFETY: only `BevyDisplayLinkDelegate::update`, `acquire` and `present` touch
// `drawable`, all on the Bevy thread (`acquire` and `present` take
// `NonSendMarker`). The rest is `Send` and `Sync`.
unsafe impl Send for MetalDrawable {}
unsafe impl Sync for MetalDrawable {}

impl MetalDrawable {
    /// The image the current frame renders into, from its acquisition until it's
    /// presented: in [`SwapchainSet`].
    pub fn current_image(&self) -> Option<&GPUMutex<DrawableImage>> {
        self.image.as_ref()
    }

    /// The current image's resource state, for the barriers of the work using it.
    pub fn state_mut(&mut self) -> &mut ResourceState {
        &mut self.state
    }
}

/// Imports the drawable's texture as the swapchain's current image, and has the
/// queue wait until the display is done with it.
fn acquire(
    _bevy_thread: NonSendMarker,
    device: Res<Device>,
    queue: Queue<RenderQueue>,
    metal_drawable: ResMut<MetalDrawable>,
) {
    let metal_drawable = metal_drawable.into_inner();
    let Some(drawable) = &metal_drawable.drawable else {
        return;
    };
    metal_drawable.image = Some(GPUMutex::new(DrawableImage::new(
        device.clone(),
        &drawable.texture(),
    )));
    metal_drawable.state = ResourceState::default();
    metal_drawable.state.write.stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    // Work committed from here on, the frame's included, waits for the drawable.
    queue
        .mtl_command_queue()
        .waitForDrawable(ProtocolObject::from_ref(&**drawable));
}

/// Presents the drawable once the frame's work, committed in [`SwapchainSet`], is
/// done.
fn present(
    _bevy_thread: NonSendMarker,
    queue: Queue<RenderQueue>,
    mut metal_drawable: ResMut<MetalDrawable>,
) {
    metal_drawable.image = None;
    let Some(drawable) = metal_drawable.drawable.take() else {
        return;
    };
    queue
        .mtl_command_queue()
        .signalDrawable(ProtocolObject::from_ref(&*drawable));
    drawable.present();
}

/// A drawable's texture imported as a `VkImage`, with a view of all of it: the
/// image a frame renders into.
pub struct DrawableImage {
    device: Device,
    image: vk::Image,
    view: DrawableImageView,
    extent: UVec2,
}

impl DrawableImage {
    fn new(device: Device, texture: &ProtocolObject<dyn MTLTexture>) -> Self {
        debug_assert!(
            std::ptr::eq(&*texture.device(), device.mtl_device()),
            "the layer's drawables come from another GPU than kosmickrisp's"
        );
        let extent = UVec2::new(texture.width() as u32, texture.height() as u32);
        let mut import = vk::ImportMetalTextureInfoEXT::default()
            .plane(vk::ImageAspectFlags::PLANE_0)
            .mtl_texture(std::ptr::from_ref(texture).cast_mut().cast());
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::A2B10G10R10_UNORM_PACK32)
            .extent(vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push(&mut import);
        // The texture is the image's storage: kosmickrisp binds no memory to an
        // imported image, so none is allocated for it.
        let image = unsafe { device.create_image(&info, None) }.unwrap();
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::A2B10G10R10_UNORM_PACK32)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .unwrap();
        DrawableImage {
            device,
            image,
            view: DrawableImageView(view),
            extent,
        }
    }

    /// A view of the whole image, in its own format.
    pub fn linear_view(&self) -> &DrawableImageView {
        &self.view
    }

    /// The color space the layer shows the image in.
    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        vk::ColorSpaceKHR::HDR10_ST2084_EXT
    }
}

project_host_metadata! {
    /// [`DrawableImage`]'s accessors on a `GPUMutex` holding it, and on the tokens
    /// locking it.
    pub trait DrawableImageMetadata for impl[] DrawableImage {
        /// A view of the whole image, in its own format.
        fn linear_view(&self) -> &DrawableImageView;
        /// The color space the layer shows the image in.
        fn color_space(&self) -> vk::ColorSpaceKHR;
    }
}

impl AsVkHandle for DrawableImage {
    type Handle = vk::Image;

    fn vk_handle(&self) -> Self::Handle {
        self.image
    }
}

// SAFETY: one `DrawableImage` names each imported image, which it creates and
// destroys. It isn't `Clone`.
unsafe impl ImageLike for DrawableImage {
    fn aspects(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::COLOR
    }

    fn array_layer_count(&self) -> u32 {
        1
    }

    fn mip_level_count(&self) -> u32 {
        1
    }

    fn extent(&self) -> UVec3 {
        self.extent.extend(1)
    }

    fn format(&self) -> vk::Format {
        vk::Format::A2B10G10R10_UNORM_PACK32
    }

    fn ty(&self) -> vk::ImageType {
        vk::ImageType::TYPE_2D
    }
}

impl Drop for DrawableImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view.0, None);
            self.device.destroy_image(self.image, None);
        }
    }
}

/// The view [`DrawableImage::linear_view`] returns.
pub struct DrawableImageView(vk::ImageView);

impl AsVkHandle for DrawableImageView {
    type Handle = vk::ImageView;

    fn vk_handle(&self) -> Self::Handle {
        self.0
    }
}
