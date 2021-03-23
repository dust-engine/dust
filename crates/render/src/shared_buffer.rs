use crate::back;
use crate::hal;
use hal::prelude::*;

use crate::camera_projection::CameraProjection;
use glam::{Mat4, TransformRT};

pub struct SharedBuffer {
    mem: <back::Backend as hal::Backend>::Memory,
    staging: Option<(
        <back::Backend as hal::Backend>::Memory,
        <back::Backend as hal::Backend>::Buffer,
    )>,
    pub buffer: <back::Backend as hal::Backend>::Buffer,
}

impl SharedBuffer {
    pub fn new(
        device: &<back::Backend as hal::Backend>::Device,
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
                ty.properties.contains(if needs_staging {
                    hal::memory::Properties::DEVICE_LOCAL
                } else {
                    hal::memory::Properties::DEVICE_LOCAL | hal::memory::Properties::CPU_VISIBLE
                }) && buffer_requirements.type_mask & (1 << id) != 0
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
            mem: memory,
            staging: None,
            buffer,
        };
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
                shared_buffer.staging = Some((staging_mem, staging_buffer));
                &mut shared_buffer.staging.as_mut().unwrap().0
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
                .flush_mapped_memory_ranges(std::iter::once((&*mem_to_write, segment)))
                .unwrap();
            device.unmap_memory(mem_to_write);
            // TODO: copy buffer to buffer
        }
        Ok(shared_buffer)
    }

    pub fn update_camera(
        &mut self,
        device: &<back::Backend as hal::Backend>::Device,
        camera_projection: &CameraProjection,
        transform: &TransformRT,
    ) {
        let transform = Mat4::from_rotation_translation(transform.rotation, transform.translation);
        let view_proj = camera_projection.get_projection_matrix() * transform.inverse();
        let transform_cols_arr = transform.to_cols_array();
        let view_proj_cols_arr = view_proj.to_cols_array();
        let mem_to_write = if let Some((staging_buffer, staging_mem)) = self.staging.as_ref() {
            // needs staging
            todo!()
        } else {
            // direct write
            &mut self.mem
        };
        unsafe {
            let segment = hal::memory::Segment {
                offset: 128,
                size: Some(128),
            };
            let ptr = device.map_memory(mem_to_write, segment.clone()).unwrap();
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
            device
                .flush_mapped_memory_ranges(std::iter::once((&*mem_to_write, segment)))
                .unwrap();
            device.unmap_memory(mem_to_write);
        }
    }

    pub unsafe fn destroy(self, device: &<back::Backend as hal::Backend>::Device) {
        device.destroy_buffer(self.buffer);
        if let Some((staging_mem, staging_buf)) = self.staging {
            device.destroy_buffer(staging_buf);
            device.free_memory(staging_mem);
        }
        device.free_memory(self.mem);
    }
}
