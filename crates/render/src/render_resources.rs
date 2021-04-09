use crate::material::Material;
use crate::material_repo::TextureRepo;
use crate::swapchain::Swapchain;
use crate::Renderer;
use ash::vk;
use dust_core::svo::alloc::BlockAllocator;
use dust_core::svo::alloc::BLOCK_SIZE;
use vk_mem as vma;
use std::sync::Arc;
use crate::renderer::RenderContext;

pub struct RenderResources {
    pub swapchain: Swapchain,
    pub allocator: vma::Allocator,
    pub block_allocator_buffer: vk::Buffer,
    pub texture_repo: TextureRepo,
}

impl RenderResources {
    pub unsafe fn new(renderer: &Renderer) -> (Self, Box<dyn BlockAllocator>) {
        let allocator = vk_mem::Allocator::new(&vma::AllocatorCreateInfo {
            physical_device: renderer.physical_device,
            device: &renderer.context.device,
            instance: &renderer.context.instance,
            flags: Default::default(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        })
            .unwrap();
        let swapchain_config = Swapchain::get_config(
            renderer.physical_device,
            renderer.context.surface,
            &renderer.context.surface_loader,
            &renderer.quirks,
        );
        let swapchain = Swapchain::new(
            renderer.context.clone(),
            &allocator,
            renderer.context.surface,
            swapchain_config,
            renderer.graphics_queue_family,
            renderer.graphics_queue,
        );
        let mut texture_repo = TextureRepo::new();
        texture_repo.materials.push(Material {
            name: "Stone".into(),
            scale: 1.0,
            diffuse: image::io::Reader::open("./assets/stone.png")
                .unwrap()
                .decode()
                .unwrap(),
            normal: None,
        });

        let block_allocator_buffer: vk::Buffer;
        let device_type = renderer.info.physical_device_properties.device_type;
        let block_allocator: Box<dyn BlockAllocator> = match device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                let allocator = crate::block_alloc::DiscreteBlockAllocator::new(
                    renderer.context.clone(),
                    renderer.transfer_binding_queue,
                    renderer.transfer_binding_queue_family,
                    renderer.graphics_queue_family,
                    BLOCK_SIZE,
                    renderer
                        .info
                        .physical_device_properties
                        .limits
                        .max_storage_buffer_range as u64,
                    &renderer.info,
                );
                block_allocator_buffer = allocator.device_buffer;
                Box::new(allocator)
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                let allocator = crate::block_alloc::IntegratedBlockAllocator::new(
                    renderer.context.clone(),
                    renderer.transfer_binding_queue,
                    renderer.transfer_binding_queue_family,
                    renderer.graphics_queue_family,
                    BLOCK_SIZE,
                    renderer
                        .info
                        .physical_device_properties
                        .limits
                        .max_storage_buffer_range as u64,
                    &renderer.info,
                );
                block_allocator_buffer = allocator.buffer;
                Box::new(allocator)
            }
            _ => panic!("Unsupported GPU"),
        };
        let render_resources = RenderResources {
            swapchain,
            allocator,
            block_allocator_buffer,
            texture_repo,
        };
        (render_resources, block_allocator)
    }
}
