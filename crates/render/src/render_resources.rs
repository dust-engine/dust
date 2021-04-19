use crate::material::{Material, ColoredMaterialDeviceLayout, ColoredMaterial};
use crate::material_repo::{TextureRepo, TextureRepoUploadState};

use crate::swapchain::Swapchain;
use crate::Renderer;
use ash::vk;
use dust_core::svo::alloc::BlockAllocator;
use dust_core::svo::alloc::BLOCK_SIZE;
use std::sync::Arc;
use vk_mem as vma;
use ash::version::DeviceV1_0;
use glam::Vec3;

pub struct RenderResources {
    pub swapchain: Swapchain,
    pub allocator: vma::Allocator,
    pub block_allocator_buffer: vk::Buffer,
    pub texture_repo: TextureRepoUploadState,
    pub block_allocator: Arc<dyn BlockAllocator>,
}

impl RenderResources {
    pub unsafe fn new(renderer: &Renderer) -> Self {
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
            scale: 10.0,
            diffuse: image::io::Reader::open("./assets/stone.png")
                .unwrap()
                .decode()
                .unwrap(),
        });
        texture_repo.materials.push(Material {
            name: "Dirt".into(),
            scale: 10.0,
            diffuse: image::io::Reader::open("./assets/dirt.png")
                .unwrap()
                .decode()
                .unwrap(),
        });
        texture_repo.colored_materials.push(ColoredMaterial {
            name: "Grass".into(),
            scale: 10.0,
            diffuse: image::io::Reader::open("./assets/grass_block_top.png")
                .unwrap()
                .decode()
                .unwrap(),
            color_palette: [Vec3::ZERO; 128]
        });
        texture_repo.colored_materials[0].color_palette[0] = Vec3::new(0.1, 1.0, 0.1);
        let upload = unsafe {
            let command_pool = renderer.context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(renderer.graphics_queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None
            )
                .unwrap();
            let mut command_buffer = vk::CommandBuffer::null();
            renderer.context.device
                .fp_v1_0()
                .allocate_command_buffers(
                renderer.context.device.handle(),
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build(),
                    &mut command_buffer
            );
            renderer.context.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build()
                )
                .unwrap();
            let upload = texture_repo.upload(
                &renderer.context.device,
                &allocator,
                command_buffer,
                renderer.graphics_queue_family,
            );
            let fence = renderer.context.device.create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            renderer.context.device.end_command_buffer(command_buffer).unwrap();
            renderer.context.device.queue_submit(
                renderer.graphics_queue,
                &[
                    vk::SubmitInfo::builder()
                        .command_buffers(&[command_buffer])
                        .build(),
                ],
                fence
            );
            renderer.context.device.wait_for_fences(&[fence], true, u64::MAX);
            renderer.context.device.destroy_fence(fence, None);
            renderer.context.device.destroy_command_pool(command_pool, None);
            upload
        };

        let block_allocator_buffer: vk::Buffer;
        let device_type = renderer.info.physical_device_properties.device_type;
        let block_allocator: Arc<dyn BlockAllocator> = match device_type {
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
                Arc::new(allocator)
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
                Arc::new(allocator)
            }
            _ => panic!("Unsupported GPU"),
        };
        RenderResources {
            swapchain,
            allocator,
            block_allocator_buffer,
            texture_repo: upload,
            block_allocator,
        }
    }
}
