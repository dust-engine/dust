use crate::back;
use crate::hal;
use crate::hal::window::Extent2D;
use crate::raytracer::Raytracer;
use hal::prelude::*;
use std::borrow::{Borrow, Cow};
use std::mem::ManuallyDrop;

pub struct RenderState {
    pub extent: Extent2D,
    pub surface: <back::Backend as hal::Backend>::Surface,
    pub device: <back::Backend as hal::Backend>::Device,
    pub graphics_queue_group: hal::queue::QueueGroup<back::Backend>,
    pub transfer_binding_queue_group: hal::queue::QueueGroup<back::Backend>,
    pub surface_format: hal::format::Format,

    pub shared_staging_memory: Option<<back::Backend as hal::Backend>::Memory>,
}

pub struct Renderer {
    pub instance: ManuallyDrop<back::Instance>,
    pub adapter: hal::adapter::Adapter<back::Backend>,
    // arc device, queue, window resized event reader, window created event reader, initliazed
    pub state: Option<RenderState>,
    raytracer: Option<Raytracer>,
    pub device_properties: hal::PhysicalDeviceProperties,
    pub memory_properties: hal::adapter::MemoryProperties,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if let Some(state) = self.state.take() {
                state.device.wait_idle().unwrap();
                self.instance.destroy_surface(state.surface);
            }
        }
    }
}

pub struct Config {
    pub name: Cow<'static, str>,
    pub version: u32,
}
impl Renderer {
    pub fn new(options: Config) -> Self {
        let instance = back::Instance::create(options.name.as_ref(), options.version).unwrap();
        let adapter = instance.enumerate_adapters().pop().unwrap();

        let device_properties = adapter.physical_device.properties();
        let memory_properties = adapter.physical_device.memory_properties();
        Renderer {
            instance: ManuallyDrop::new(instance),
            adapter,
            state: None,
            raytracer: None,
            device_properties,
            memory_properties,
        }
    }
    pub fn set_surface(&mut self, surface: <back::Backend as hal::Backend>::Surface) {
        let span = tracing::info_span!("set_surface");
        let _enter = span.enter();

        let supported_surface_formats = surface.supported_formats(&self.adapter.physical_device);
        tracing::info!("supported surface formats: {:?}", supported_surface_formats);
        let surface_format =
            supported_surface_formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
                formats
                    .iter()
                    .find(|format| {
                        format.base_format().1 == hal::format::ChannelType::Srgb
                            && format.base_format().0 == hal::format::SurfaceType::R8_G8_B8_A8
                    })
                    .map(|format| *format)
                    .unwrap_or(formats[0])
            });
        tracing::info!("selected surface formats: {:?}", surface_format);

        tracing::info!(
            "self.adapter.queue_families: \n{:?}\n",
            self.adapter.queue_families
        );
        let graphics_queue_family = self.adapter.queue_families.iter().find(|family| {
            surface.supports_queue_family(family) && family.queue_type().supports_graphics()
        });
        if graphics_queue_family.is_none() {
            return;
        }
        let graphics_queue_family = graphics_queue_family.unwrap();
        let transfer_binding_queue_family = self
            .adapter
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

        let physical_device = &self.adapter.physical_device;
        let mut gpu = unsafe {
            physical_device.open(
                &[
                    (graphics_queue_family, &[1.0]),
                    (transfer_binding_queue_family, &[0.5]),
                ],
                hal::Features::SPARSE_BINDING | hal::Features::SPARSE_RESIDENCY_IMAGE_2D,
            )
        }
        .unwrap();
        let device = gpu.device;
        assert_eq!(gpu.queue_groups.len(), 2);
        tracing::info!("queues returned:\n{:?}\n", gpu.queue_groups);
        let transfer_binding_queue_group = gpu.queue_groups.pop().unwrap();
        let graphics_queue_group = gpu.queue_groups.pop().unwrap();

        // Allocate some memory
        let shared_staging_memory = unsafe {
            let staging_type = self
                .memory_properties
                .memory_types
                .iter()
                .position(|memory_type| {
                    memory_type.properties.contains(
                        hal::memory::Properties::CPU_VISIBLE | hal::memory::Properties::COHERENT,
                    )
                })
                .unwrap()
                .into();

            device
                .allocate_memory(
                    staging_type,
                    128, //TODO
                )
                .unwrap()
        };

        let state = RenderState {
            extent: Extent2D {
                width: 1920,
                height: 1080,
            },
            surface,
            device,
            graphics_queue_group,
            transfer_binding_queue_group,
            surface_format,
            shared_staging_memory: Some(shared_staging_memory),
        };
        self.state = Some(state);
        let framebuffer_attachment = self.rebuild_swapchain();
        let raytracer = Raytracer::new(self, framebuffer_attachment);
        self.raytracer = Some(raytracer);
    }
    pub fn on_resize(&mut self) {
        let framebuffer_attachment = self.rebuild_swapchain();
        let state = self.state.as_ref().unwrap();
        self.raytracer
            .as_mut()
            .unwrap()
            .rebuild_framebuffer(state, framebuffer_attachment);
    }
    fn rebuild_swapchain(&mut self) -> hal::image::FramebufferAttachment {
        let mut state = self.state.as_mut().unwrap();

        let surface_capabilities = state.surface.capabilities(&self.adapter.physical_device);
        let config = hal::window::SwapchainConfig::from_caps(
            &surface_capabilities,
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
    pub fn update(&mut self) {
        if self.state.is_none() {
            return;
        }
        let state = self.state.as_mut().unwrap();
        if self.raytracer.is_none() {
            return;
        }
        let raytracer = self.raytracer.as_mut().unwrap();
        unsafe {
            let (surface_image, suboptimal) = state.surface.acquire_image(!0).unwrap();
            if suboptimal.is_some() {
                tracing::warn!("Suboptimal swapchain image acquired");
            }

            let raytracer_submission_semaphore = raytracer.update(state, surface_image.borrow());

            let suboptimal = state.graphics_queue_group.queues[0]
                .present(
                    &mut state.surface,
                    surface_image,
                    Some(raytracer_submission_semaphore),
                )
                .unwrap();
            if suboptimal.is_some() {
                tracing::warn!("Suboptimal surface presented");
            }
        }
    }
}
