use ash::vk;

use crate::camera_projection::CameraProjection;
use ash::version::DeviceV1_0;
use glam::{Mat4, TransformRT, Vec3};

struct StagingState {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    queue: vk::Queue,
    queue_family: u32,
}

pub struct SharedBuffer {
    device: ash::Device,
    memory: vk::DeviceMemory,
    staging: Option<StagingState>,
    pub buffer: vk::Buffer,
}

impl SharedBuffer {
    pub unsafe fn new(
        device: ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue: vk::Queue,
        queue_family: u32,
    ) -> Self {
        let span = tracing::info_span!("shared_buffer_new");
        let _enter = span.enter();

        let needs_staging = !memory_properties.memory_types
            [0..memory_properties.memory_type_count as usize]
            .iter()
            .any(|ty| {
                ty.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            });
        tracing::info!("SharedBuffer using staging: {:?}", needs_staging);
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .flags(vk::BufferCreateFlags::empty())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size(256)
                    .usage(
                        vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::VERTEX_BUFFER
                            | if needs_staging {
                                vk::BufferUsageFlags::TRANSFER_DST
                            } else {
                                vk::BufferUsageFlags::empty()
                            },
                    ),
                None,
            )
            .unwrap();
        let buffer_requirements = device.get_buffer_memory_requirements(buffer);
        let buffer_memory_type: u32 = memory_properties.memory_types
            [0..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .position(|(i, ty)| {
                if buffer_requirements.memory_type_bits & (1 << i) == 0 {
                    return false;
                }
                if needs_staging {
                    if ty
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        && !ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    {
                        return true;
                    }
                } else {
                    if ty
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        && ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    {
                        return true;
                    }
                }
                return false;
            })
            .unwrap() as u32;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(buffer_memory_type)
                    .allocation_size(buffer_requirements.size),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();
        let mut shared_buffer = Self {
            memory,
            staging: None,
            buffer,
            device,
        };
        if needs_staging {
            let staging_buffer = shared_buffer
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(128)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .flags(vk::BufferCreateFlags::empty())
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let staging_buffer_requirements = shared_buffer
                .device
                .get_buffer_memory_requirements(staging_buffer);
            let memory_type = memory_properties.memory_types
                [0..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .position(|(i, ty)| {
                    if staging_buffer_requirements.memory_type_bits & (1 << i) == 0 {
                        return false;
                    }
                    if ty
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                        && !ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    {
                        return true;
                    }
                    return false;
                })
                .unwrap() as u32;
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
                .bind_buffer_memory(staging_buffer, staging_mem, 0)
                .unwrap();
            shared_buffer.staging = Some(StagingState {
                memory: staging_mem,
                buffer: staging_buffer,
                queue,
                queue_family,
            });
        }
        shared_buffer
    }

    pub unsafe fn write_vertex_index(
        &self,
        vertex_buffer: &[(f32, f32, f32); 8],
        index_buffer: &[u16; 14],
    ) {
        // Write data into buffer
        let mem_to_write = if let Some(staging_buffer) = self.staging.as_ref() {
            staging_buffer.memory
        } else {
            // Directly map and write
            self.memory
        };
        let ptr = self
            .device
            .map_memory(mem_to_write, 0, 128, vk::MemoryMapFlags::empty())
            .unwrap() as *mut u8;
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
        self.device
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                .memory(mem_to_write)
                .size(128)
                .offset(0)
                .build()])
            .unwrap();
        self.device.unmap_memory(mem_to_write);
        // copy buffer to buffer
        if let Some(ref staging) = self.staging {
            let pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                        .queue_family_index(staging.queue_family)
                        .build(),
                    None,
                )
                .unwrap();
            let cmd_buf = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1)
                        .build(),
                )
                .unwrap()
                .pop()
                .unwrap();
            self.device
                .begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )
                .unwrap();
            self.device.cmd_copy_buffer(
                cmd_buf,
                staging.buffer,
                self.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: std::mem::size_of_val(vertex_buffer) as u64
                        + std::mem::size_of_val(index_buffer) as u64,
                }],
            );
            self.device.end_command_buffer(cmd_buf).unwrap();
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            self.device
                .queue_submit(
                    staging.queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[cmd_buf])
                        .build()],
                    fence,
                )
                .unwrap();
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            self.device.destroy_command_pool(pool, None);
            self.device.destroy_fence(fence, None);
        }
    }

    pub fn update_camera(
        &self,
        camera_projection: &CameraProjection,
        transform: &TransformRT,
        aspect_ratio: f32,
    ) {
        let rotation = Mat4::from_rotation_translation(transform.rotation, Vec3::ZERO);
        let transform = Mat4::from_rotation_translation(transform.rotation, transform.translation);
        let view_proj = camera_projection.get_projection_matrix(aspect_ratio) * rotation.inverse();
        let transform_cols_arr = transform.to_cols_array();
        let view_proj_cols_arr = view_proj.to_cols_array();
        let (mem_to_write, offset, size) = if let Some(staging_buffer) = self.staging.as_ref() {
            // needs staging
            (staging_buffer.memory, 0_u64, 128_u64)
        } else {
            // direct write
            (self.memory, 128_u64, 128_u64)
        };
        unsafe {
            let ptr = self
                .device
                .map_memory(mem_to_write, offset, size, vk::MemoryMapFlags::empty())
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
                .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                    .size(size)
                    .offset(offset)
                    .memory(mem_to_write)
                    .build()])
                .unwrap();
            self.device.unmap_memory(mem_to_write);
        }
    }

    pub unsafe fn record_cmd_buffer_copy_buffer(&self, cmd_buffer: vk::CommandBuffer) {
        if let Some(staging_buffer) = self.staging.as_ref() {
            self.device.cmd_copy_buffer(
                cmd_buffer,
                staging_buffer.buffer,
                self.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 128,
                    size: 128,
                }],
            );
        }
    }
}
