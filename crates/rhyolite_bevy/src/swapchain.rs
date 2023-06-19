use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};
use rhyolite::{
    ash::vk,
    utils::format::{ColorSpace, ColorSpaceType, FormatType},
    AcquireFuture, HasDevice, PhysicalDevice, Surface,
};

use crate::{queue::Queues, Device, Frame, SharingMode};

#[derive(Component)]
pub struct Swapchain(rhyolite::Swapchain);

impl Swapchain {
    pub fn acquire_next_image(&mut self, current_frame: &mut Frame) -> AcquireFuture {
        self.0
            .acquire_next_image(current_frame.shared_semaphore_pool.get_binary_semaphore())
    }
    fn get_create_info<'a>(
        surface: &'_ rhyolite::Surface,
        pdevice: &'_ PhysicalDevice,
        window: &'_ Window,
        config: &'a SwapchainConfigExt,
    ) -> rhyolite::SwapchainCreateInfo<'a> {
        let surface_capabilities = surface.get_capabilities(pdevice).unwrap();
        let supported_present_modes = surface.get_present_modes(pdevice).unwrap();
        let image_format = config.image_format.unwrap_or_else(|| {
            if config.hdr {
                get_surface_preferred_format(
                    surface,
                    pdevice,
                    config.required_feature_flags,
                    config.srgb_format,
                )
            } else {
                if config.srgb_format {
                    vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_SRGB,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    }
                } else {
                    vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_UNORM,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    }
                }
            }
        });
        rhyolite::SwapchainCreateInfo {
            flags: config.flags,
            min_image_count: config.min_image_count,
            image_format: image_format.format,
            image_color_space: image_format.color_space,
            image_extent: vk::Extent2D {
                width: window.resolution.physical_width(),
                height: window.resolution.physical_height(),
            },
            image_array_layers: config.image_array_layers,
            image_usage: config.image_usage,
            image_sharing_mode: (&config.sharing_mode).into(),
            pre_transform: config.pre_transform,
            composite_alpha: match window.composite_alpha_mode {
                bevy_window::CompositeAlphaMode::Auto => {
                    if surface_capabilities
                        .supported_composite_alpha
                        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    {
                        vk::CompositeAlphaFlagsKHR::OPAQUE
                    } else {
                        vk::CompositeAlphaFlagsKHR::INHERIT
                    }
                }
                bevy_window::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
                bevy_window::CompositeAlphaMode::PreMultiplied => {
                    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
                }
                bevy_window::CompositeAlphaMode::PostMultiplied => {
                    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
                }
                bevy_window::CompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
            },
            present_mode: match window.present_mode {
                bevy_window::PresentMode::AutoVsync => {
                    if supported_present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                        vk::PresentModeKHR::FIFO_RELAXED
                    } else {
                        vk::PresentModeKHR::FIFO
                    }
                }
                bevy_window::PresentMode::AutoNoVsync => {
                    if supported_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                        vk::PresentModeKHR::IMMEDIATE
                    } else if supported_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                        vk::PresentModeKHR::MAILBOX
                    } else {
                        vk::PresentModeKHR::FIFO
                    }
                }
                bevy_window::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
                bevy_window::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
                bevy_window::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            },
            clipped: config.clipped,
        }
    }
    pub fn create(
        device: Arc<rhyolite::Device>,
        surface: Arc<rhyolite::Surface>,
        window: &Window,
        config: &SwapchainConfigExt,
    ) -> Self {
        let create_info = Self::get_create_info(&surface, device.physical_device(), window, config);
        Swapchain(rhyolite::Swapchain::create(device, surface, create_info).unwrap())
    }
    pub fn recreate(&mut self, window: &Window, config: &SwapchainConfigExt) {
        let create_info = Self::get_create_info(
            self.0.surface(),
            self.0.device().physical_device(),
            window,
            config,
        );
        self.0.recreate(create_info).unwrap()
    }
}

/// Runs in RenderSystems::SetUp
pub(super) fn extract_windows(
    mut commands: Commands,
    device: Res<Device>,
    mut queues: ResMut<Queues>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    mut window_resized_events: EventReader<bevy_window::WindowResized>,

    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS s
    _marker: NonSend<NonSendResource>,
    mut query: Query<(
        &Window,
        &RawHandleWrapper,
        Option<&SwapchainConfigExt>,
        Option<&mut Swapchain>,
    )>,
) {
    queues.next_frame();
    for resize_event in window_resized_events.iter() {
        let (window, _, config, swapchain) = query.get_mut(resize_event.window).unwrap();
        if let Some(mut swapchain) = swapchain {
            let default_config = SwapchainConfigExt::default();
            let swapchain_config = config.unwrap_or(&default_config);
            swapchain.recreate(window, swapchain_config);
        }
    }
    for create_event in window_created_events.iter() {
        let (window, raw_handle, config, swapchain) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(swapchain.is_none());
        let new_surface = Arc::new(
            rhyolite::Surface::create(device.instance().clone(), &raw_handle, &raw_handle).unwrap(),
        );

        let default_config = SwapchainConfigExt::default();
        let swapchain_config = config.unwrap_or(&default_config);
        let new_swapchain = Swapchain::create(
            device.inner().clone(),
            new_surface,
            window,
            swapchain_config,
        );
        commands.entity(create_event.window).insert(new_swapchain);
    }
}

#[derive(Component)]
pub struct SwapchainConfigExt {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    /// If set to None, the implementation will select the best available color space.
    pub image_format: Option<vk::SurfaceFormatKHR>,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub required_feature_flags: vk::FormatFeatureFlags,
    pub sharing_mode: SharingMode,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub clipped: bool,

    /// If set to true and the `image_format` property weren't set,
    /// the implementation will select the best available HDR color space.
    /// On Windows 11, it is recommended to turn this off when the application was started in Windowed mode
    /// and the system HDR toggle was turned off. Otherwise, the screen may flash when the application is started.
    pub hdr: bool,

    /// If set to true, SDR swapchains will be created with a sRGB format.
    /// If set to false, SDR swapchains will be created with a UNORM format.
    ///
    /// Set this to false if the data will be written to the swapchain image as a storage image,
    /// and the tonemapper will apply gamma correction manually. This is the default.
    ///
    /// Set this to true if the swapchain will be directly used as a render target. In this case,
    /// the sRGB gamma correction will be applied automatically.
    pub srgb_format: bool,
}

impl Default for SwapchainConfigExt {
    fn default() -> Self {
        Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: None,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            sharing_mode: SharingMode::Exclusive,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            clipped: false,
            required_feature_flags: vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            hdr: true,
            srgb_format: false,
        }
    }
}

pub fn get_surface_preferred_format(
    surface: &Surface,
    physical_device: &PhysicalDevice,
    required_feature_flags: vk::FormatFeatureFlags,
    use_srgb_format: bool,
) -> vk::SurfaceFormatKHR {
    let supported_formats = physical_device.get_surface_formats(surface).unwrap();

    supported_formats
        .iter()
        .filter(|&surface_format| {
            let format_properties = unsafe {
                physical_device
                    .instance()
                    .get_physical_device_format_properties(
                        physical_device.raw(),
                        surface_format.format,
                    )
            };
            format_properties
                .optimal_tiling_features
                .contains(required_feature_flags)
                | format_properties
                    .linear_tiling_features
                    .contains(required_feature_flags)
        })
        .max_by_key(|&surface_format| {
            // Select color spaces based on the following criteria:
            // Prefer larger color spaces. For extended srgb, consider it the same as Rec2020 but after all other Rec2020 color spaces.
            // Prefer formats with larger color depth.
            // If small swapchain format, prefer non-linear. Otherwise, prefer linear.
            let format: rhyolite::utils::format::Format = surface_format.format.into();
            let format_priority = format.r + format.g + format.b;

            let color_space: ColorSpace = surface_format.color_space.into();
            let color_space_priority = match color_space.ty {
                ColorSpaceType::ExtendedSrgb => {
                    ColorSpaceType::HDR10_ST2084.primaries().area_size()
                }
                _ => color_space.primaries().area_size(),
            } * 4096.0;
            let color_space_priority = color_space_priority as u32;
            let linearity_priority: u8 = if format_priority < 30 {
                // < 8 bit color. Prefer non-linear color space
                if color_space.linear {
                    0
                } else {
                    1
                }
            } else {
                // >= 8 bit color, for example 10 bit color and above. Prefer linear color space
                if color_space.linear {
                    1
                } else {
                    0
                }
            };

            let srgb_priority = if (format.ty == FormatType::sRGB) ^ use_srgb_format {
                0_u8
            } else {
                1_u8
            };
            (
                color_space_priority,
                format_priority,
                linearity_priority,
                srgb_priority,
            )
        })
        .cloned()
        .unwrap_or(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        })
}

#[derive(Default)]
pub(super) struct NonSendResource;
