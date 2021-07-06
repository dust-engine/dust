use crate::raytracer::RayTracer;
use crate::renderer::RenderContext;
use crate::utils::div_round_up;
use ash::vk;
use std::mem::MaybeUninit;
use std::sync::Arc;
use vk_mem as vma;

struct Frame {
    swapchain_image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}
pub struct SwapchainImage {
    pub desc_set: vk::DescriptorSet,
    pub view: vk::ImageView,
    pub image: vk::Image,
    fence: vk::Fence, // This fence was borrowed from the last rendered frame.
    // The reason we need a separate command buffer for each swapchain image
    // is that cmd_begin_render_pass contains a reference to the framebuffer
    // which is unique to each swapchain image.
    pub command_buffer: vk::CommandBuffer,
}
pub(crate) struct DepthImage {
    image: vk::Image,
    pub(crate) view: vk::ImageView,
    allocation: vma::Allocation,
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

unsafe fn create_depth_image(
    device: &ash::Device,
    allocator: &vma::Allocator,
    config: &SwapchainConfig,
) -> DepthImage {
    let (image, allocation, _allocation_info) = allocator
        .create_image(
            &vk::ImageCreateInfo::builder()
                .flags(vk::ImageCreateFlags::empty())
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .extent(vk::Extent3D {
                    width: config.extent.width,
                    height: config.extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .build(),
            &vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                flags: vma::AllocationCreateFlags::NONE,
                ..Default::default()
            },
        )
        .unwrap();
    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image),
            None,
        )
        .unwrap();
    DepthImage {
        image,
        view,
        allocation,
    }
}

unsafe fn create_beam_image(
    device: &ash::Device,
    allocator: &vma::Allocator,
    config: &SwapchainConfig,
) -> DepthImage {
    let (image, allocation, _allocation_info) = allocator
        .create_image(
            &vk::ImageCreateInfo::builder()
                .flags(vk::ImageCreateFlags::empty())
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(vk::Extent3D {
                    width: div_round_up(config.extent.width, 8),
                    height: div_round_up(config.extent.height, 8),
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::INPUT_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .build(),
            &vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                flags: vma::AllocationCreateFlags::NONE,
                ..Default::default()
            },
        )
        .unwrap();
    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image),
            None,
        )
        .unwrap();
    DepthImage {
        image,
        view,
        allocation,
    }
}

unsafe fn create_swapchain(
    loader: &ash::extensions::khr::Swapchain,
    surface: vk::SurfaceKHR,
    config: &SwapchainConfig,
) -> vk::SwapchainKHR {
    loader
        .create_swapchain(
            &vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(3)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_format(config.format)
                .image_extent(config.extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::IMMEDIATE)
                .clipped(true)
                .image_array_layers(1)
                .build(),
            None,
        )
        .unwrap()
}

unsafe fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    config: &SwapchainConfig,
) -> vk::ImageView {
    device
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
        .unwrap()
}
unsafe fn update_image_desc_sets(
    device: &ash::Device,
    swapchain_images: &[SwapchainImage],
    beam_image: &DepthImage,
) {
    let image_infos = swapchain_images
        .iter()
        .map(|swapchain_image| vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: swapchain_image.view,
            image_layout: vk::ImageLayout::GENERAL,
        })
        .collect::<Vec<vk::DescriptorImageInfo>>();
    let beam_info = [vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: beam_image.view,
        image_layout: vk::ImageLayout::GENERAL,
    }];
    let descriptor_writes = swapchain_images
        .iter()
        .enumerate()
        .flat_map(|(i, swapchain_image)| {
            std::iter::once(
                vk::WriteDescriptorSet::builder()
                    .dst_set(swapchain_image.desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_infos[i..=i])
                    .build(),
            )
            .chain(std::iter::once(
                vk::WriteDescriptorSet::builder()
                    .dst_set(swapchain_image.desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&beam_info)
                    .build(),
            ))
        })
        .collect::<Vec<vk::WriteDescriptorSet>>();
    device.update_descriptor_sets(descriptor_writes.as_slice(), &[]);
}

const NUM_FRAMES_IN_FLIGHT: usize = 3;
pub struct Swapchain {
    context: Arc<RenderContext>,
    allocator: Arc<vma::Allocator>,
    surface: vk::SurfaceKHR,
    current_frame: usize,
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    frames_in_flight: [Frame; NUM_FRAMES_IN_FLIGHT], // number of frames in flight
    swapchain_images: Vec<SwapchainImage>,           // number of images in swapchain
    depth_image: DepthImage,
    beam_image: DepthImage,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pub config: SwapchainConfig,
    pub images_desc_set_layout: vk::DescriptorSetLayout,
    pub images_desc_set_pool: vk::DescriptorPool,
}
pub struct SwapchainConfig {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}
impl Swapchain {
    pub unsafe fn get_config(
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> SwapchainConfig {
        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .unwrap();
        let supported_formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .unwrap();
        let _supported_present_mode = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .unwrap();

        let format = supported_formats[0].format;
        let extent = caps.current_extent;
        SwapchainConfig { format, extent }
    }
    pub unsafe fn new(
        context: Arc<RenderContext>,
        allocator: Arc<vma::Allocator>,
        surface: vk::SurfaceKHR,
        config: SwapchainConfig,
        graphics_queue_family_index: u32,
        graphics_queue: vk::Queue,
    ) -> Self {
        let instance = &context.instance;
        let device = &context.device;
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = create_swapchain(&swapchain_loader, surface, &config);
        let depth_image = create_depth_image(device, &allocator, &config);
        let beam_image = create_beam_image(device, &allocator, &config);
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
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(images.len() as u32)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: images.len() as u32 * 2,
                    }])
                    .build(),
                None,
            )
            .unwrap();
        let desc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .build(), // Primary Output Image
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .build(), // Beam Image
                ]),
                None,
            )
            .unwrap();
        let desc_sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(desc_pool)
                    .set_layouts(vec![desc_set_layout; images.len()].as_slice())
                    .build(),
            )
            .unwrap();
        let swapchain_images = images
            .into_iter()
            .zip(command_buffers.into_iter())
            .zip(desc_sets.into_iter())
            .map(|((image, command_buffer), desc_set)| {
                let view = create_image_view(&device, image, &config);
                SwapchainImage {
                    desc_set,
                    image,
                    view,
                    fence: vk::Fence::null(),
                    command_buffer,
                }
            })
            .collect::<Vec<_>>();

        update_image_desc_sets(device, &swapchain_images, &beam_image);
        let mut frames_in_flight: [MaybeUninit<Frame>; NUM_FRAMES_IN_FLIGHT] =
            MaybeUninit::uninit().assume_init();
        for i in 0..NUM_FRAMES_IN_FLIGHT {
            frames_in_flight[i].write(Frame::new(&device));
        }
        let frames_in_flight: [Frame; NUM_FRAMES_IN_FLIGHT] = std::mem::transmute(frames_in_flight);
        Self {
            command_pool,
            context,
            loader: swapchain_loader,
            swapchain,
            current_frame: 0,
            frames_in_flight,
            swapchain_images,
            depth_image,
            beam_image,
            graphics_queue,
            config,
            surface,
            images_desc_set_layout: desc_set_layout,
            images_desc_set_pool: desc_pool,
            allocator,
        }
    }

    pub unsafe fn recreate(&mut self, allocator: &vma::Allocator, config: SwapchainConfig) {
        // reclaim resources
        self.context
            .device
            .destroy_image_view(self.depth_image.view, None);
        allocator.destroy_image(self.depth_image.image, &self.depth_image.allocation);
        for swapchain_image in self.swapchain_images.iter() {
            self.context
                .device
                .destroy_image_view(swapchain_image.view, None);
        }
        self.loader.destroy_swapchain(self.swapchain, None);
        self.context
            .device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();

        // create new
        self.swapchain = create_swapchain(&self.loader, self.surface, &config);
        let images = self.loader.get_swapchain_images(self.swapchain).unwrap();
        for (swapchain_image, image) in self.swapchain_images.iter_mut().zip(images.into_iter()) {
            swapchain_image.view = create_image_view(&self.context.device, image, &config);
            swapchain_image.image = image;
        }
        update_image_desc_sets(
            &self.context.device,
            &self.swapchain_images,
            &self.beam_image,
        );
        self.depth_image = create_depth_image(&self.context.device, allocator, &config);
        self.config = config;
    }

    pub unsafe fn render_frame(&mut self) {
        let frame_in_flight = &self.frames_in_flight[self.current_frame];
        self.context
            .device
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
                self.context
                    .device
                    .wait_for_fences(&[swapchain_image.fence], true, u64::MAX)
                    .unwrap();
            }
            swapchain_image.fence = frame_in_flight.fence;
        }
        self.context
            .device
            .reset_fences(&[frame_in_flight.fence])
            .unwrap();
        self.context
            .device
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
        if self.current_frame >= self.frames_in_flight.len() {
            self.current_frame = 0;
        }
    }
    pub unsafe fn bind_render_pass(&mut self, render_pass_provider: &mut RayTracer) {
        for swapchain_image in self.swapchain_images.iter_mut() {
            self.context
                .device
                .begin_command_buffer(
                    swapchain_image.command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::empty())
                        .build(),
                )
                .unwrap();
            render_pass_provider.record_command_buffer(
                &self.context.device,
                swapchain_image,
                &self.config,
            );
            self.context
                .device
                .end_command_buffer(swapchain_image.command_buffer)
                .unwrap();
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(self.images_desc_set_layout, None);
            self.context
                .device
                .destroy_descriptor_pool(self.images_desc_set_pool, None);
            for swapchain_image in self.swapchain_images.iter() {
                self.context
                    .device
                    .destroy_image_view(swapchain_image.view, None);
            }
            self.context
                .device
                .destroy_image_view(self.depth_image.view, None);
            self.allocator
                .destroy_image(self.depth_image.image, &self.depth_image.allocation);
            self.context
                .device
                .destroy_image_view(self.beam_image.view, None);
            self.allocator
                .destroy_image(self.beam_image.image, &self.beam_image.allocation);
            // allocator destroy image
            for frame in self.frames_in_flight.iter() {
                self.context.device.destroy_fence(frame.fence, None);
                self.context
                    .device
                    .destroy_semaphore(frame.swapchain_image_available_semaphore, None);
                self.context
                    .device
                    .destroy_semaphore(frame.render_finished_semaphore, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
