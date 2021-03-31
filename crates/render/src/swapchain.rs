use crate::device_info::Quirks;
use ash::version::DeviceV1_0;
use ash::vk;

struct Frame {
    swapchain_image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}
struct SwapchainImage {
    view: vk::ImageView,
    fence: vk::Fence, // This fence was borrowed from the last rendered frame.
    // The reason we need a separate command buffer for each swapchain image
    // is that cmd_begin_render_pass contains a reference to the framebuffer
    // which is unique to each swapchain image.
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
}
pub trait RenderPassProvider {
    unsafe fn record_command_buffer(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        config: &SwapchainConfig,
    );
    unsafe fn get_render_pass(&self) -> vk::RenderPass;
}
impl Frame {
    unsafe fn new(device: &ash::Device) -> Self {
        Self {
            swapchain_image_available_semaphore: device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap(),
            render_finished_semaphore: device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap(),
            fence: device
                .create_fence(
                    &vk::FenceCreateInfo::builder()
                        .flags(vk::FenceCreateFlags::SIGNALED)
                        .build(),
                    None,
                )
                .unwrap(),
        }
    }
}
pub struct Swapchain {
    device: ash::Device,
    current_frame: usize,
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    frames_in_flight: Vec<Frame>,          // number of frames in flight
    swapchain_images: Vec<SwapchainImage>, // number of images in swapchain
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pub config: SwapchainConfig,
}
pub struct SwapchainConfig {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub flip_y_requires_shift: bool,
}
impl Swapchain {
    pub unsafe fn get_config(
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: ash::extensions::khr::Surface,
        quirks: &Quirks,
    ) -> SwapchainConfig {
        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .unwrap();
        let _supported_formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .unwrap();
        let _supported_present_mode = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .unwrap();

        let format = vk::Format::R8G8B8A8_SRGB;
        let extent = caps.current_extent;
        SwapchainConfig {
            format,
            extent,
            flip_y_requires_shift: quirks.flip_y_requires_shift,
        }
    }
    pub unsafe fn new<T: RenderPassProvider>(
        instance: &ash::Instance,
        device: ash::Device,
        surface: vk::SurfaceKHR,
        config: SwapchainConfig,
        graphics_queue_family_index: u32,
        graphics_queue: vk::Queue,
        render_pass_provider: &T,
    ) -> Self {
        let num_frames_in_flight = 3;
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, &device);
        let swapchain = swapchain_loader
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(surface)
                    .min_image_count(3)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_format(config.format)
                    .image_extent(config.extent)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true)
                    .image_array_layers(1)
                    .build(),
                None,
            )
            .unwrap();
        let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        // First, create the command pool
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::empty())
                    .build(),
                None,
            )
            .unwrap();
        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .command_buffer_count(images.len() as u32)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build(),
            )
            .unwrap();
        let swapchain_images = images
            .into_iter()
            .zip(command_buffers.into_iter())
            .map(|(image, command_buffer)| {
                let view = device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(config.format)
                            .components(vk::ComponentMapping {
                                r: vk::ComponentSwizzle::R,
                                g: vk::ComponentSwizzle::G,
                                b: vk::ComponentSwizzle::B,
                                a: vk::ComponentSwizzle::A,
                            })
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image(image),
                        None,
                    )
                    .unwrap();
                // Create the framebuffer
                let framebuffer = device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .height(config.extent.height)
                            .width(config.extent.width)
                            .layers(1)
                            .attachments(&[view])
                            .render_pass(render_pass_provider.get_render_pass())
                            .flags(vk::FramebufferCreateFlags::empty())
                            .build(),
                        None,
                    )
                    .unwrap();
                // record the command buffer
                device
                    .begin_command_buffer(
                        command_buffer,
                        &vk::CommandBufferBeginInfo::builder()
                            .flags(vk::CommandBufferUsageFlags::empty())
                            .build(),
                    )
                    .unwrap();
                render_pass_provider.record_command_buffer(
                    &device,
                    command_buffer,
                    framebuffer,
                    &config,
                );
                device.end_command_buffer(command_buffer).unwrap();

                SwapchainImage {
                    view,
                    fence: vk::Fence::null(),
                    command_buffer,
                    framebuffer,
                }
            })
            .collect();
        let mut frames_in_flight = Vec::with_capacity(num_frames_in_flight);
        for _i in 0..num_frames_in_flight {
            frames_in_flight.push(Frame::new(&device));
        }
        Self {
            command_pool,
            device,
            loader: swapchain_loader,
            swapchain,
            current_frame: 0,
            frames_in_flight,
            swapchain_images,
            graphics_queue,
            config,
        }
    }

    pub unsafe fn recreate(
        &mut self,
        new_config: &SwapchainConfig,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
    ) {
        // reclaim resources
        for swapchain_image in self.swapchain_images.iter() {
            self.device
                .destroy_framebuffer(swapchain_image.framebuffer, None);
            self.device.destroy_image_view(swapchain_image.view, None);
        }
        self.loader.destroy_swapchain(self.swapchain, None);
        self.device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty());
    }

    pub unsafe fn render_frame(&mut self) {
        let frame_in_flight = &self.frames_in_flight[self.current_frame];
        self.device
            .wait_for_fences(&[frame_in_flight.fence], true, u64::MAX)
            .unwrap();
        let (image_index, suboptimal) = self
            .loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                frame_in_flight.swapchain_image_available_semaphore,
                vk::Fence::null(),
            )
            .unwrap();
        if suboptimal {
            tracing::warn!("Suboptimal image acquired.");
        }
        let swapchain_image = &mut self.swapchain_images[image_index as usize];
        {
            if swapchain_image.fence != vk::Fence::null() {
                self.device
                    .wait_for_fences(&[swapchain_image.fence], true, u64::MAX)
                    .unwrap();
            }
            swapchain_image.fence = frame_in_flight.fence;
        }
        self.device.reset_fences(&[frame_in_flight.fence]).unwrap();
        self.device
            .queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::builder()
                    .wait_semaphores(&[frame_in_flight.swapchain_image_available_semaphore])
                    .signal_semaphores(&[frame_in_flight.render_finished_semaphore])
                    .command_buffers(&[swapchain_image.command_buffer])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .build()],
                frame_in_flight.fence,
            )
            .unwrap();

        self.loader
            .queue_present(
                self.graphics_queue,
                &vk::PresentInfoKHR::builder()
                    .wait_semaphores(&[frame_in_flight.render_finished_semaphore])
                    .swapchains(&[self.swapchain])
                    .image_indices(&[image_index]),
            )
            .unwrap();

        self.current_frame = self.current_frame + 1;
        if self.current_frame >= 3 {
            self.current_frame = 0;
        }
    }
}
