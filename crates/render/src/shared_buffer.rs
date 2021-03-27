use crate::{back, hal};
use hal::prelude::*;

use crate::camera_projection::CameraProjection;
use glam::{Mat4, TransformRT};
use std::sync::Arc;

pub struct SharedStagingBuffer {
    memory: <back::Backend as hal::Backend>::Memory,
    buffer: <back::Backend as hal::Backend>::Buffer,
}

pub struct SharedBuffer {
    device: Arc<<back::Backend as hal::Backend>::Device>,
    mem: <back::Backend as hal::Backend>::Memory,
    staging: Option<SharedStagingBuffer>,
    pub buffer: <back::Backend as hal::Backend>::Buffer,
}

impl SharedBuffer {
    pub fn new(
        device: Arc<<back::Backend as hal::Backend>::Device>,
        transfer_queue: &mut hal::queue::QueueGroup<back::Backend>,
        vertex_buffer: &[(f32, f32, f32); 8],
        index_buffer: &[u16; 14],
        memory_properties: &hal::adapter::MemoryProperties,
    ) -> Result<Self, hal::buffer::CreationError> {
        let span = tracing::info_span!("shared_buffer_new");
        let _enter = span.enter();
        let needs_staging = memory_properties
            .memory_types
            .iter()
            .find(|ty| {
                ty.properties.contains(
                    hal::memory::Properties::DEVICE_LOCAL | hal::memory::Properties::CPU_VISIBLE,
                )
            })
            .is_none();
        tracing::info!("Needs staging: {:?}", needs_staging);
        debug_assert_eq!(
            std::mem::size_of_val(vertex_buffer) + std::mem::size_of_val(index_buffer),
            124
        );
        let mut buffer = unsafe {
            device.create_buffer(
                256,
                hal::buffer::Usage::INDEX
                    | hal::buffer::Usage::VERTEX
                    | hal::buffer::Usage::UNIFORM
                    | if needs_staging {
                        hal::buffer::Usage::TRANSFER_DST
                    } else {
                        hal::buffer::Usage::empty()
                    },
                hal::memory::SparseFlags::empty(),
            )
        }?;
        let buffer_requirements = unsafe { device.get_buffer_requirements(&buffer) };
        let memory_type: hal::MemoryTypeId = memory_properties
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                (if needs_staging {
                    ty.properties
                        .contains(hal::memory::Properties::DEVICE_LOCAL)
                        && !ty.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                } else {
                    ty.properties
                        .contains(hal::memory::Properties::DEVICE_LOCAL)
                        && ty.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                }) && (buffer_requirements.type_mask & (1 << id) != 0)
            })
            .unwrap()
            .into();
        let memory = unsafe {
            device
                .allocate_memory(memory_type, buffer_requirements.size)
                .unwrap()
        };
        unsafe {
            // Bind buffer with memory
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
        }
        let mut shared_buffer = Self {
            device,
            mem: memory,
            staging: None,
            buffer,
        };
        let device = &shared_buffer.device;
        unsafe {
            // Write data into buffer
            let mem_to_write = if needs_staging {
                let mut staging_buffer = device
                    .create_buffer(
                        128,
                        hal::buffer::Usage::TRANSFER_SRC,
                        hal::memory::SparseFlags::empty(),
                    )
                    .unwrap();
                let staging_buffer_requirements = device.get_buffer_requirements(&staging_buffer);

                let memory_type: hal::MemoryTypeId = memory_properties
                    .memory_types
                    .iter()
                    .enumerate()
                    .position(|(id, ty)| {
                        ty.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                            && !ty
                                .properties
                                .contains(hal::memory::Properties::DEVICE_LOCAL)
                            && staging_buffer_requirements.type_mask & (1 << id) != 0
                    })
                    .unwrap()
                    .into();
                let staging_mem = device
                    .allocate_memory(memory_type, staging_buffer_requirements.size)
                    .unwrap();
                device
                    .bind_buffer_memory(&staging_mem, 0, &mut staging_buffer)
                    .unwrap();
                shared_buffer.staging = Some(SharedStagingBuffer {
                    memory: staging_mem,
                    buffer: staging_buffer,
                });
                &mut shared_buffer.staging.as_mut().unwrap().memory
            } else {
                // Directly map and write
                &mut shared_buffer.mem
            };
            let segment = hal::memory::Segment {
                offset: 0,
                size: Some(
                    (std::mem::size_of_val(vertex_buffer) + std::mem::size_of_val(index_buffer))
                        as u64,
                ),
            };
            let ptr = device.map_memory(mem_to_write, segment.clone()).unwrap();
            std::ptr::copy_nonoverlapping(
                vertex_buffer.as_ptr() as *const u8,
                ptr,
                std::mem::size_of_val(vertex_buffer),
            );
            std::ptr::copy_nonoverlapping(
                index_buffer.as_ptr() as *const u8,
                ptr.add(std::mem::size_of_val(vertex_buffer)),
                std::mem::size_of_val(index_buffer),
            );
            device
                .flush_mapped_memory_ranges(std::iter::once((&*mem_to_write, segment.clone())))
                .unwrap();
            device.unmap_memory(mem_to_write);
            // copy buffer to buffer
            if let Some(ref staging) = shared_buffer.staging {
                // TODO: Maybe a better way to initialize this?
                let mut pool = device
                    .create_command_pool(
                        transfer_queue.family,
                        hal::pool::CommandPoolCreateFlags::TRANSIENT,
                    )
                    .unwrap();
                let mut cmd_buf = pool.allocate_one(hal::command::Level::Primary);
                cmd_buf.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
                cmd_buf.copy_buffer(
                    &staging.buffer,
                    &shared_buffer.buffer,
                    std::iter::once(hal::command::BufferCopy {
                        src: segment.offset,
                        dst: segment.offset,
                        size: segment.size.unwrap(),
                    }),
                );
                cmd_buf.finish();
                let mut fence = device.create_fence(false).unwrap();
                transfer_queue.queues[0].submit(
                    std::iter::once(&cmd_buf),
                    std::iter::empty(),
                    std::iter::empty(),
                    Some(&mut fence),
                );
                device.wait_for_fence(&mut fence, !0).unwrap();
                device.destroy_command_pool(pool);
                device.destroy_fence(fence);
            }
        }
        Ok(shared_buffer)
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
        let (mem_to_write, segment) = if let Some(staging_buffer) = self.staging.as_mut() {
            // needs staging
            (
                &mut staging_buffer.memory,
                hal::memory::Segment {
                    offset: 0,
                    size: Some(128),
                },
            )
        } else {
            // direct write
            (
                &mut self.mem,
                hal::memory::Segment {
                    offset: 128,
                    size: Some(128),
                },
            )
        };
        unsafe {
            let ptr = self
                .device
                .map_memory(mem_to_write, segment.clone())
                .unwrap();
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
                .flush_mapped_memory_ranges(std::iter::once((&*mem_to_write, segment)))
                .unwrap();
            self.device.unmap_memory(mem_to_write);
        }
    }

    pub unsafe fn record_cmd_buffer(
        &self,
        cmd_buffer: &mut <back::Backend as hal::Backend>::CommandBuffer,
    ) {
        if let Some(staging_buffer) = self.staging.as_ref() {
            cmd_buffer.copy_buffer(
                &staging_buffer.buffer,
                &self.buffer,
                std::iter::once(hal::command::BufferCopy {
                    src: 0,
                    dst: 128,
                    size: 128,
                }),
            )
        }
    }

    pub unsafe fn destroy(self, device: &<back::Backend as hal::Backend>::Device) {
        device.destroy_buffer(self.buffer);
        if let Some(staging_buffer) = self.staging {
            device.destroy_buffer(staging_buffer.buffer);
            device.free_memory(staging_buffer.memory);
        }
        device.free_memory(self.mem);
    }
}
