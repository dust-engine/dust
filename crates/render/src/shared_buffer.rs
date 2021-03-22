use crate::back;
use crate::hal;
use hal::prelude::*;
use std::alloc::Layout;
use tracing;

pub struct SharedBuffer {
    mem: <back::Backend as hal::Backend>::Memory,
    staging: Option<(
        <back::Backend as hal::Backend>::Memory,
        <back::Backend as hal::Backend>::Buffer
    )>,
    pub vertex_index_buffer: <back::Backend as hal::Backend>::Buffer,
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
                    hal::memory::Properties::DEVICE_LOCAL
                        | hal::memory::Properties::CPU_VISIBLE
                        | hal::memory::Properties::COHERENT,
                )
            })
            .is_none();
        tracing::info!("Needs staging: {:?}", needs_staging);
        debug_assert_eq!(
            std::mem::size_of_val(vertex_buffer) + std::mem::size_of_val(index_buffer),
            124
        );
        let mut vertex_index_buffer = unsafe {
            device.create_buffer(
                124,
                if needs_staging {
                    hal::buffer::Usage::INDEX
                        | hal::buffer::Usage::VERTEX
                        | hal::buffer::Usage::TRANSFER_DST
                } else {
                    hal::buffer::Usage::INDEX | hal::buffer::Usage::VERTEX
                },
                hal::memory::SparseFlags::empty(),
            )
        }?;
        let vertex_index_requirements =
            unsafe { device.get_buffer_requirements(&vertex_index_buffer) };
        let mut uniform_buffer = unsafe {
            device.create_buffer(
                (std::mem::size_of::<glam::Mat4>() * 2) as u64,
                if needs_staging {
                    hal::buffer::Usage::UNIFORM | hal::buffer::Usage::TRANSFER_DST
                } else {
                    hal::buffer::Usage::UNIFORM
                },
                hal::memory::SparseFlags::empty(),
            )
        }?;
        let uniform_buffer_requirements =
            unsafe { device.get_buffer_requirements(&uniform_buffer) };
        let memory_type: hal::MemoryTypeId = memory_properties
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                ty.properties.contains(
                    if needs_staging {
                        hal::memory::Properties::DEVICE_LOCAL
                    } else {
                        hal::memory::Properties::DEVICE_LOCAL
                            | hal::memory::Properties::CPU_VISIBLE
                            | hal::memory::Properties::COHERENT
                    },
                ) && vertex_index_requirements.type_mask & (1 << id) != 0
                    && uniform_buffer_requirements.type_mask & (1 << id) != 0
            })
            .unwrap()
            .into();
        let mut memory = unsafe {
            device.allocate_memory(
                memory_type,
                256,
            ).unwrap()
        };
        unsafe {
            // Bind buffer with memory
            device.bind_buffer_memory(
                &memory,
                0,
                &mut vertex_index_buffer
            ).unwrap();
            device.bind_buffer_memory(
                &memory,
                128,
                &mut uniform_buffer
            ).unwrap();
        }
        let mut shared_buffer = Self {
            mem: memory,
            staging: None,
            vertex_index_buffer,
        };
        unsafe {
            // Write data into buffer
            let mem_to_write = if needs_staging {
                let mut staging_buffer = device.create_buffer(
                    128,
                    hal::buffer::Usage::TRANSFER_SRC,
                    hal::memory::SparseFlags::empty(),
                ).unwrap();
                let staging_buffer_requirements = device.get_buffer_requirements(&staging_buffer);

                let memory_type: hal::MemoryTypeId = memory_properties
                    .memory_types
                    .iter()
                    .enumerate()
                    .position(|(id, ty)| {
                        ty.properties.contains(
                            hal::memory::Properties::CPU_VISIBLE | hal::memory::Properties::COHERENT,
                        ) && staging_buffer_requirements.type_mask & (1 << id) != 0
                    })
                    .unwrap()
                    .into();
                let staging_mem = device.allocate_memory(
                    memory_type,
                    staging_buffer_requirements.size,
                ).unwrap();
                device.bind_buffer_memory(
                    &staging_mem,
                    0,
                    &mut staging_buffer
                ).unwrap();
                shared_buffer.staging = Some((staging_mem, staging_buffer));
                &mut shared_buffer.staging.as_mut().unwrap().0
            } else {
                // Directly map and write
                &mut shared_buffer.mem
            };
            let ptr = device.map_memory(
                mem_to_write,
                hal::memory::Segment{
                    offset: (std::mem::size_of_val(vertex_buffer) + std::mem::size_of_val(index_buffer)) as u64,
                    size: None
                }
            ).unwrap();
            std::ptr::copy_nonoverlapping(
                vertex_buffer.as_ptr() as *const u8,
                ptr,
                std::mem::size_of_val(vertex_buffer),
            );
            std::ptr::copy_nonoverlapping(
                index_buffer.as_ptr() as *const u8,
                ptr.offset(std::mem::size_of_val(vertex_buffer) as isize),
                std::mem::size_of_val(index_buffer),
            );
            device.unmap_memory(
                mem_to_write,
            );
            // TODO: copy buffer to buffer
        }
        Ok(shared_buffer)
    }
}
