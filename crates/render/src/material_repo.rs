use crate::material::{ColoredMaterial, Material};
use ash::version::DeviceV1_0;
use ash::vk;
use image::GenericImageView;
use vk_mem as vma;

pub struct TextureRepo {
    pub materials: Vec<Material>,
    pub colored_materials: Vec<ColoredMaterial>,
}

pub struct TextureRepoUploadState {
    pub image: vk::Image,
    pub image_allocation: vma::Allocation,
    pub image_allocation_info: vma::AllocationInfo,
    pub staging_buffer: vk::Buffer,
    pub staging_buffer_allocation: vma::Allocation,
    pub staging_buffer_allocation_info: vma::AllocationInfo,
}
impl TextureRepo {
    pub fn new() -> Self {
        TextureRepo {
            materials: Vec::new(),
            colored_materials: Vec::new(),
        }
    }
    pub fn upload(
        self,
        device: &ash::Device,
        allocator: &vma::Allocator,
        command_buffer: vk::CommandBuffer,
        graphics_queue_family: u32,
        transfer_queue_family: u32,
    ) -> TextureRepoUploadState {
        let (image, image_allocation, image_allocation_info) = allocator
            .create_image(
                &vk::ImageCreateInfo::builder()
                    .flags(vk::ImageCreateFlags::empty())
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .extent(vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(self.materials.len() as u32)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .build(),
                &vma::AllocationCreateInfo {
                    usage: vma::MemoryUsage::GpuOnly,
                    flags: vma::AllocationCreateFlags::DEDICATED_MEMORY,
                    ..Default::default()
                },
            )
            .unwrap();
        let (staging_buffer, staging_buffer_allocation, staging_buffer_allocation_info) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(image_allocation_info.get_size() as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .build(),
                &vma::AllocationCreateInfo {
                    usage: vma::MemoryUsage::CpuOnly,
                    flags: vma::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
            .unwrap();
        let staging_ptr = staging_buffer_allocation_info.get_mapped_data();

        let indices = {
            // Copy data into the buffer
            let mut current_offset: usize = 0;
            let mut indices: Vec<usize> = Vec::with_capacity(self.materials.len());
            for (_i, material) in self.materials.iter().enumerate() {
                let rgba8img = material.diffuse.as_rgba8().unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        rgba8img.as_ptr(),
                        staging_ptr.add(current_offset),
                        rgba8img.len(),
                    );
                }
                indices.push(current_offset);
                current_offset += rgba8img.len();
            }
            indices
        };

        let buffer_copies: Vec<_> = self
            .materials
            .iter()
            .zip(indices.iter())
            .enumerate()
            .map(|(i, (material, &indice))| {
                vk::BufferImageCopy {
                    buffer_offset: indice as u64,
                    // If either of these values is zero, that aspect of the buffer memory is
                    // considered to be tightly packed according to the imageExtent.
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                }
            })
            .collect();
        unsafe {
            let image_memory_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(transfer_queue_family)
                .dst_queue_family_index(transfer_queue_family)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                })
                .build();
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier]
            );
            let mut image_memory_barrier2 = image_memory_barrier.clone();
            image_memory_barrier2.dst_queue_family_index = graphics_queue_family;
            image_memory_barrier2.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier2.dst_access_mask = vk::AccessFlags::SHADER_READ;
            image_memory_barrier2.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_memory_barrier2.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_copies,
            );
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier2]
            );
        }
        TextureRepoUploadState {
            image,
            image_allocation,
            image_allocation_info,
            staging_buffer,
            staging_buffer_allocation,
            staging_buffer_allocation_info,
        }
    }
}
