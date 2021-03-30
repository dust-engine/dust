use ash::vk as vk;

use crate::camera_projection::CameraProjection;
use glam::{Mat4, TransformRT};
use std::sync::Arc;
use ash::version::DeviceV1_0;

pub struct SharedStagingBuffer {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
}

pub struct SharedBuffer {
    device: ash::Device,
    memory: vk::DeviceMemory,
    staging: Option<SharedStagingBuffer>,
    pub buffer: vk::Buffer,
}

impl SharedBuffer {
    pub unsafe fn new(
        device: ash::Device,
        queue: vk::Queue,
        queue_family: u32,
        vertex_buffer: &[(f32, f32, f32); 8],
        index_buffer: &[u16; 14],
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let span = tracing::info_span!("shared_buffer_new");
        let _enter = span.enter();

        let mut needs_staging = true;
        for i in 0..memory_properties.memory_type_count {
            let ty: &vk::MemoryType = &memory_properties.memory_types[i as usize];
            if ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE) {
                // We've found a device local and host visible memory type. Don't need staging.
                needs_staging = false;
                break;
            }
        }
        debug_assert_eq!(
            std::mem::size_of_val(vertex_buffer) + std::mem::size_of_val(index_buffer),
            124
        );
        let mut buffer = device.create_buffer(
            &vk::BufferCreateInfo::builder()
                .flags(vk::BufferCreateFlags::empty())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size(256)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER |
                    vk::BufferUsageFlags::INDEX_BUFFER |
                    vk::BufferUsageFlags::VERTEX_BUFFER |
                    if needs_staging { vk::BufferUsageFlags::TRANSFER_DST } else { vk::BufferUsageFlags::empty() }
                ),
            None
        ).unwrap();
        let buffer_requirments = device.get_buffer_memory_requirements(buffer);
        let mut buffer_memory_type: u32 = u32::MAX;
        for i in 0..memory_properties.memory_type_count {
            let ty: &vk::MemoryType = &memory_properties.memory_types[i as usize];
            if buffer_requirments.memory_type_bits & (1 << i) == 0 {
                continue;
            }
            if needs_staging {
                if ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) &&
                    !ty.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    buffer_memory_type = i;
                    break;
                }
            } else {
                if ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) &&
                    ty.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    buffer_memory_type = i;
                    break;
                }
            }
        }
        assert_ne!(buffer_memory_type, u32::MAX);
        let memory =
            device.allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(buffer_memory_type)
                    .allocation_size(buffer_requirments.size),
                None
            ).unwrap();
        device.bind_buffer_memory(
            buffer,
            memory,
            0
        ).unwrap();
        let mut shared_buffer = Self {
            memory,
            staging: None,
            buffer,
            device,
        };
        unsafe {
            // Write data into buffer
            let mem_to_write = if needs_staging {
                let mut staging_buffer = shared_buffer
                    .device
                    .create_buffer(
                        &vk::BufferCreateInfo::builder()
                            .size(128)
                            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                            .flags(vk::BufferCreateFlags::empty())
                            .sharing_mode(vk::SharingMode::EXCLUSIVE),
                        None
                    ).unwrap();
                let staging_buffer_requirements = shared_buffer
                    .device
                    .get_buffer_memory_requirements(staging_buffer);
                let mut memory_type = u32::MAX;
                for i in 0..memory_properties.memory_type_count {
                    let ty: &vk::MemoryType = &memory_properties.memory_types[i as usize];
                    if staging_buffer_requirements.memory_type_bits & (1 << i) == 0 {
                        continue;
                    }
                    if ty.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                        && !ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                        memory_type = i;
                    }
                }
                assert_ne!(memory_type, u32::MAX);
                let staging_mem = shared_buffer
                    .device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::builder()
                            .allocation_size(staging_buffer_requirements.size)
                            .memory_type_index(memory_type)
                            .build(),
                        None,
                    )
                    .unwrap();
                shared_buffer
                    .device
                    .bind_buffer_memory(
                    staging_buffer,
                    staging_mem,
                    0
                ).unwrap();
                shared_buffer.staging = Some(SharedStagingBuffer {
                    memory: staging_mem,
                    buffer: staging_buffer
                });
                shared_buffer.staging.as_mut().unwrap().memory
            } else {
                // Directly map and write
                shared_buffer.memory
            };
            let ptr = shared_buffer
                .device
                .map_memory(
                mem_to_write,
                0,
                128,
                vk::MemoryMapFlags::empty()
            ).unwrap() as *mut u8;
            std::ptr::copy_nonoverlapping(
                vertex_buffer.as_ptr() as *const u8,
                ptr as *mut u8,
                std::mem::size_of_val(vertex_buffer),
            );
            std::ptr::copy_nonoverlapping(
                index_buffer.as_ptr() as *const u8,
                ptr.add(std::mem::size_of_val(vertex_buffer)),
                std::mem::size_of_val(index_buffer),
            );
            shared_buffer
                .device
                .flush_mapped_memory_ranges(
                &[
                    vk::MappedMemoryRange::builder()
                        .memory(mem_to_write)
                        .size(128)
                        .offset(0)
                        .build()
                ]
            ).unwrap();
            shared_buffer
                .device
                .unmap_memory(mem_to_write);
            // copy buffer to buffer
            if let Some(ref staging) = shared_buffer.staging {
                let pool = shared_buffer
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::builder()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(queue_family)
                            .build(),
                        None
                    ).unwrap();
                let cmd_buf = shared_buffer
                    .device
                    .allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1)
                        .build()
                    )
                    .unwrap()
                    .pop()
                    .unwrap();
                shared_buffer
                    .device.begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                ).unwrap();
                shared_buffer
                    .device.cmd_copy_buffer(
                    cmd_buf,
                    staging.buffer,
                    shared_buffer.buffer,
                    &[
                        vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: std::mem::size_of_val(vertex_buffer) as u64 + std::mem::size_of_val(index_buffer) as u64
                        }
                    ]
                );
                shared_buffer
                    .device.end_command_buffer(
                    cmd_buf
                ).unwrap();
                let fence = shared_buffer
                    .device.create_fence(
                    &vk::FenceCreateInfo::default(),
                    None
                ).unwrap();
                shared_buffer
                    .device.queue_submit(
                    queue,
                    &[
                        vk::SubmitInfo::builder()
                            .command_buffers(&[cmd_buf])
                            .build()
                    ],
                    fence,
                ).unwrap();
                shared_buffer
                    .device.wait_for_fences(
                    &[fence],
                    true,
                    u64::MAX
                ).unwrap();
                shared_buffer
                    .device.destroy_command_pool(pool, None);
                shared_buffer
                    .device.destroy_fence(fence, None);
            }
        }
        shared_buffer
    }

    pub fn update_camera(
        &mut self,
        camera_projection: &CameraProjection,
        transform: &TransformRT,
        aspect_ratio: f32,
    ) {
        let transform = Mat4::from_rotation_translation(transform.rotation, transform.translation);
        let view_proj = camera_projection.get_projection_matrix(aspect_ratio) * transform.inverse();
        let transform_cols_arr = transform.to_cols_array();
        let view_proj_cols_arr = view_proj.to_cols_array();
        let (mem_to_write, offset, size) = if let Some(staging_buffer) = self.staging.as_mut() {
            // needs staging
            (
                staging_buffer.memory,
                0_u64,
                128_u64
            )
        } else {
            // direct write
            (
                self.memory,
                128_u64,
                128_u64
            )
        };
        unsafe {
            let ptr = self.device
                .map_memory(
                    mem_to_write,
                    offset,
                    size,
                    vk::MemoryMapFlags::empty()
                )
                .unwrap() as *mut u8;
            std::ptr::copy_nonoverlapping(
                view_proj_cols_arr.as_ptr() as *const u8,
                ptr,
                std::mem::size_of_val(&view_proj_cols_arr),
            );
            std::ptr::copy_nonoverlapping(
                transform_cols_arr.as_ptr() as *const u8,
                ptr.add(std::mem::size_of_val(&view_proj_cols_arr)),
                std::mem::size_of_val(&transform_cols_arr),
            );
            self.device
                .flush_mapped_memory_ranges(&[
                    vk::MappedMemoryRange::builder()
                        .size(size)
                        .offset(offset)
                        .memory(mem_to_write)
                        .build()
                ])
                .unwrap();
            self.device.unmap_memory(mem_to_write);
        }
    }

    pub unsafe fn record_cmd_buffer_copy_buffer(
        &self,
        cmd_buffer: vk::CommandBuffer,
    ) {
        if let Some(staging_buffer) = self.staging.as_ref() {
            self.device
                .cmd_copy_buffer(
                    cmd_buffer,
                    staging_buffer.buffer,
                    self.buffer,
                    &[
                        vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 128,
                            size: 128
                        }
                    ]
                );
        }
    }
}
