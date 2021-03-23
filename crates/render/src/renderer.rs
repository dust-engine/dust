use crate::back;
use crate::hal;
use crate::hal::window::Extent2D;
use crate::raytracer::Raytracer;
use hal::prelude::*;
use std::borrow::Borrow;
use std::mem::ManuallyDrop;
use std::sync::Arc;

pub struct RenderState {
    pub extent: Extent2D,
    pub surface: ManuallyDrop<<back::Backend as hal::Backend>::Surface>,
    pub device: Arc<<back::Backend as hal::Backend>::Device>,
    pub graphics_queue_group: hal::queue::QueueGroup<back::Backend>,
    pub transfer_binding_queue_group: hal::queue::QueueGroup<back::Backend>,
    pub surface_format: hal::format::Format,
}

pub struct Renderer {
    pub instance: ManuallyDrop<back::Instance>,
    pub adapter: hal::adapter::Adapter<back::Backend>,
    // arc device, queue, window resized event reader, window created event reader, initliazed
    pub state: RenderState,
    raytracer: ManuallyDrop<Raytracer>,
    pub device_properties: hal::PhysicalDeviceProperties,
    pub memory_properties: hal::adapter::MemoryProperties,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.state.device.wait_idle().unwrap();

            let raytracer = ManuallyDrop::take(&mut self.raytracer);
            raytracer.destroy();

            let surface = ManuallyDrop::take(&mut self.state.surface);
            self.instance.destroy_surface(surface);
        }
    }
}

impl Renderer {
    pub fn new(window: &impl raw_window_handle::HasRawWindowHandle) -> Self {
        let instance = back::Instance::create("dust engine", 0).unwrap();
        let surface = unsafe { instance.create_surface(window).unwrap() };
        let adapter = instance.enumerate_adapters().pop().unwrap();

        let device_properties = adapter.physical_device.properties();
        let memory_properties = adapter.physical_device.memory_properties();

        let span = tracing::info_span!("renderer new");
        let _enter = span.enter();

        let supported_surface_formats = surface.supported_formats(&adapter.physical_device);
        tracing::info!("supported surface formats: {:?}", supported_surface_formats);
        let surface_format =
            supported_surface_formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
                formats
                    .iter()
                    .find(|format| {
                        format.base_format().1 == hal::format::ChannelType::Srgb
                            && format.base_format().0 == hal::format::SurfaceType::R8_G8_B8_A8
                    })
                    .copied()
                    .unwrap_or(formats[0])
            });
        tracing::info!("selected surface formats: {:?}", surface_format);

        tracing::info!(
            "self.adapter.queue_families: \n{:?}\n",
            adapter.queue_families
        );
        let graphics_queue_family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        let transfer_binding_queue_family = adapter
            .queue_families
            .iter()
            .find(|family| {
                family.id() != graphics_queue_family.id()
                    && family.supports_sparse_binding()
                    && family.queue_type().supports_transfer()
                    && !family.queue_type().supports_graphics()
                    && !family.queue_type().supports_compute()
            })
            .expect("Can not find a queue family that supports sparse binding");

        let physical_device = &adapter.physical_device;
        let mut gpu = unsafe {
            physical_device.open(
                &[
                    (graphics_queue_family, &[1.0]),
                    (transfer_binding_queue_family, &[0.5]),
                ],
                hal::Features::SPARSE_BINDING
                    | hal::Features::SPARSE_RESIDENCY_BUFFER
                    | hal::Features::NDC_Y_UP,
            )
        }
        .unwrap();
        let device = gpu.device;
        assert_eq!(gpu.queue_groups.len(), 2);
        tracing::info!("queues returned:\n{:?}\n", gpu.queue_groups);
        let transfer_binding_queue_group = gpu.queue_groups.pop().unwrap();
        let graphics_queue_group = gpu.queue_groups.pop().unwrap();

        let mut state = RenderState {
            extent: Extent2D {
                width: 1920,
                height: 1080,
            },
            surface: ManuallyDrop::new(surface),
            device: Arc::new(device),
            graphics_queue_group,
            transfer_binding_queue_group,
            surface_format,
        };

        let surface_capabilities = state.surface.capabilities(&adapter.physical_device);
        let framebuffer_attachment = Self::rebuild_swapchain(&mut state, &surface_capabilities);
        let raytracer = Raytracer::new(&mut state, &memory_properties, framebuffer_attachment);

        Renderer {
            instance: ManuallyDrop::new(instance),
            adapter,
            state,
            raytracer: ManuallyDrop::new(raytracer),
            device_properties,
            memory_properties,
        }
    }
    pub fn on_resize(&mut self) {
        let surface_capabilities = self
            .state
            .surface
            .capabilities(&self.adapter.physical_device);
        let framebuffer_attachment =
            Self::rebuild_swapchain(&mut self.state, &surface_capabilities);
        self.raytracer
            .rebuild_framebuffer(self.state.extent, framebuffer_attachment);
    }
    fn rebuild_swapchain(
        state: &mut RenderState,
        surface_capabilities: &hal::window::SurfaceCapabilities,
    ) -> hal::image::FramebufferAttachment {
        let config = hal::window::SwapchainConfig::from_caps(
            surface_capabilities,
            state.surface_format,
            Extent2D {
                width: 1920,
                height: 1080,
            },
        );
        tracing::info!(
            "Swapchain rebuilt with size {}x{}",
            config.extent.width,
            config.extent.height
        );
        state.extent = config.extent;
        let framebuffer = config.framebuffer_attachment();
        unsafe {
            state
                .surface
                .configure_swapchain(&state.device, config)
                .unwrap();
        }
        framebuffer
    }
    pub fn update(&mut self, state: &crate::State) {
        unsafe {
            let (surface_image, suboptimal) = self.state.surface.acquire_image(!0).unwrap();
            if suboptimal.is_some() {
                tracing::warn!("Suboptimal swapchain image acquired");
            }

            let raytracer_submission_semaphore = self.raytracer.update(
                surface_image.borrow(),
                &mut self.state.graphics_queue_group.queues[0],
                state,
            );

            let suboptimal = self.state.graphics_queue_group.queues[0]
                .present(
                    &mut self.state.surface,
                    surface_image,
                    Some(raytracer_submission_semaphore),
                )
                .unwrap();
            if suboptimal.is_some() {
                tracing::warn!("Suboptimal surface presented");
            }
        }
    }

    pub fn create_block_allocator(
        &mut self,
    ) -> Result<Box<svo::alloc::ArenaBlockAllocator>, hal::buffer::CreationError> {
        use block_alloc::{DiscreteBlockAllocator, IntegratedBlockAllocator};
        use hal::adapter::DeviceType;
        const SIZE: usize = svo::alloc::CHUNK_SIZE;
        match self.adapter.info.device_type {
            DeviceType::DiscreteGpu | DeviceType::VirtualGpu | DeviceType::Other => {
                let allocator: DiscreteBlockAllocator<back::Backend, SIZE> =
                    DiscreteBlockAllocator::new(
                        self.state.device.clone(),
                        self.state
                            .transfer_binding_queue_group
                            .queues
                            .pop()
                            .unwrap(),
                        self.state.transfer_binding_queue_group.family,
                        &self.memory_properties,
                    )?;
                return Ok(Box::new(allocator));
            }
            DeviceType::IntegratedGpu | DeviceType::Cpu => {
                let allocator: IntegratedBlockAllocator<back::Backend, SIZE> =
                    IntegratedBlockAllocator::new(
                        self.state.device.clone(),
                        self.state
                            .transfer_binding_queue_group
                            .queues
                            .pop()
                            .unwrap(),
                        &self.memory_properties,
                    )?;

                return Ok(Box::new(allocator));
            }
        }
    }
}
