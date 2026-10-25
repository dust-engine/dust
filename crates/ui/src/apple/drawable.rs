//! The game's [`Swapchain`](crate::Swapchain) on macOS, made from the display link's drawables instead of
//! a Vulkan swapchain.
//!
//! Each frame, [`acquire`] imports the drawable's texture into Vulkan as a `VkImage`
//! (`VK_EXT_metal_objects`) and makes it the swapchain's current image; [`present`]
//! presents the drawable once the work writing into it is committed. kosmickrisp
//! runs the Vulkan queue on a Metal 4 command queue and commits to it inside
//! `vkQueueSubmit`, so both talk to that queue directly, as Metal 4 requires of
//! anything drawing into a drawable: `waitForDrawable` between `DefaultRenderSet`'s
//! submission and [`SwapchainSet`]'s, so that only the latter waits for the
//! drawable, and `signalDrawable` after [`SwapchainSet`]'s.

use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;
use bevy_pumicite::{
    CreateDevice, DefaultRenderSet, PumiciteApp, SubmissionSetConfig,
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
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

/// Where systems using the [`MetalDrawable`]'s current image run: in a submission
/// of their own on the render queue, after `DefaultRenderSet`'s, between the
/// image's acquisition and its presentation. It's skipped on frames without an
/// image.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapchainSet;

/// The run condition for [`SwapchainSet`].
fn swapchain_image_acquired(swapchain: Res<MetalDrawable>) -> bool {
    swapchain.current_image().is_some()
}

/// Inserts the [`MetalDrawable`], makes [`SwapchainSet`] a submission after
/// `DefaultRenderSet`'s, and acquires and presents the drawables' images around it.
pub(super) struct CAMetalLayerPlugin {
    pub(super) layer: Retained<CAMetalLayer>,
}
// SAFETY: the runner makes it on the Bevy thread and adds it to the app there,
// which builds it and, when dropped, drops it there too.
unsafe impl Send for CAMetalLayerPlugin {}
unsafe impl Sync for CAMetalLayerPlugin {}

impl Plugin for CAMetalLayerPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MetalDrawable {
            layer: self.layer.clone(),
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
        .add_systems(Startup, add_layer_to_residency_set.after(CreateDevice))
        .add_submission_set::<RenderQueue>(
            SwapchainSet,
            SubmissionSetConfig {
                debug_color: Vec4::new(1.0, 0.0, 1.0, 1.0),
            },
        )
        .configure_sets(
            PostUpdate,
            (
                SwapchainSet.run_if(swapchain_image_acquired),

                // TODO: This is also probably bad. Why DefaultRenderSet has to
                // run before SwapchainSet but not DefaultComputeSet / DefaultTransferSet?
                // Feels like AI slop to me.
                DefaultRenderSet.before(SwapchainSet),
            ),
        )
        .add_systems(
            PostUpdate,
            (
                // TODO: This is really temporary. Idea is that waitForDrawable will
                // run as late as possible. But on Metal all presentation should
                // probably happen on a separate MTL4CommandQueue to make this more
                // maintainable.
                acquire.after(DefaultRenderSet).before(SwapchainSet),
                present.after(SwapchainSet),
            ),
        );
    }
}

/// The swapchain's state on macOS: the frame's drawable, and its texture imported
/// into Vulkan.
#[derive(Resource)]
pub struct MetalDrawable {
    layer: Retained<CAMetalLayer>,
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

// SAFETY: only `BevyDisplayLinkDelegate::update`, `acquire` and `present` set and
// take `drawable`, and only `add_layer_to_residency_set` touches `layer`, all on
// the Bevy thread (the systems take `NonSendMarker`). The app is dropped there
// too, and with it both. `extent` reads the drawable's texture size from any
// thread. The rest is `Send` and `Sync`.
unsafe impl Send for MetalDrawable {}
unsafe impl Sync for MetalDrawable {}

impl MetalDrawable {
    /// The size of the current frame's image, or zero outside of a frame. Unlike
    /// the image, it's known for the whole frame, e.g. to size the images that
    /// `DefaultRenderSet` renders.
    pub fn extent(&self) -> UVec2 {
        self.drawable.as_ref().map_or(UVec2::ZERO, |drawable| {
            let texture = drawable.texture();
            UVec2::new(texture.width() as u32, texture.height() as u32)
        })
    }

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

fn add_layer_to_residency_set(
    _bevy_thread: NonSendMarker,
    metal_drawable: Res<MetalDrawable>,
    queue: Queue<RenderQueue>,
) {
    queue
        .mtl_command_queue()
        .addResidencySet(&metal_drawable.layer.residencySet());
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
    // Work committed from here on, `SwapchainSet`'s included, waits for the
    // drawable. `DefaultRenderSet`'s is already committed.
    queue
        .mtl_command_queue()
        .waitForDrawable(ProtocolObject::from_ref(&**drawable));
}

/// Presents the drawable once the work writing into it, committed in
/// [`SwapchainSet`]'s submission, is done.
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
