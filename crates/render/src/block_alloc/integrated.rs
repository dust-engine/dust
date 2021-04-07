use crate::device_info::DeviceInfo;
use ash::version::DeviceV1_0;
use ash::vk;

use crate::renderer::RenderContext;
use dust_core::svo::alloc::{AllocError, BlockAllocator};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

pub struct IntegratedBlockAllocator {
    context: Arc<RenderContext>,
    bind_transfer_queue: vk::Queue,
    memtype: u32,
    pub buffer: vk::Buffer,

    current_offset: u64,
    free_offsets: Vec<u64>,
    allocations: HashMap<*mut u8, vk::DeviceMemory>,
    block_size: u64,
}
unsafe impl Send for IntegratedBlockAllocator {}
unsafe impl Sync for IntegratedBlockAllocator {}

impl IntegratedBlockAllocator {
    pub unsafe fn new(
        context: Arc<RenderContext>,
        bind_transfer_queue: vk::Queue,
        bind_transfer_queue_family: u32,
        graphics_queue_family: u32,
        block_size: u64,
        max_storage_buffer_size: u64,
        device_info: &DeviceInfo,
    ) -> Self {
        let queue_family_indices = [graphics_queue_family, bind_transfer_queue_family];
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(max_storage_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

        if graphics_queue_family == bind_transfer_queue_family {
            buffer_create_info = buffer_create_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        } else {
            buffer_create_info = buffer_create_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        }

        let device_buffer = context
            .device
            .create_buffer(&buffer_create_info.build(), None)
            .unwrap();
        let requirements = context.device.get_buffer_memory_requirements(device_buffer);
        let memtype = select_integrated_memtype(&device_info.memory_properties, &requirements);
        Self {
            context,
            bind_transfer_queue,
            memtype,
            buffer: device_buffer,
            current_offset: 0,
            free_offsets: Vec::new(),
            allocations: HashMap::new(),
            block_size,
        }
    }
}

impl BlockAllocator for IntegratedBlockAllocator {
    unsafe fn allocate_block(&mut self) -> Result<*mut u8, AllocError> {
        let resource_offset = self.free_offsets.pop().unwrap_or_else(|| {
            let val = self.current_offset;
            self.current_offset += 1;
            val
        });
        let mem = self
            .context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(self.block_size)
                    .memory_type_index(self.memtype)
                    .build(),
                None,
            )
            .unwrap();
        let ptr = self
            .context
            .device
            .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(super::utils::map_err)? as *mut u8;
        self.context
            .device
            .queue_bind_sparse(
                self.bind_transfer_queue,
                &[vk::BindSparseInfo::builder()
                    .buffer_binds(&[vk::SparseBufferMemoryBindInfo::builder()
                        .buffer(self.buffer)
                        .binds(&[vk::SparseMemoryBind {
                            resource_offset: resource_offset * self.block_size as u64,
                            size: self.block_size,
                            memory: mem,
                            memory_offset: 0,
                            flags: vk::SparseMemoryBindFlags::empty(),
                        }])
                        .build()])
                    .build()],
                vk::Fence::null(),
            )
            .map_err(super::utils::map_err)?;
        self.allocations.insert(ptr, mem);
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, block: *mut u8) {
        let memory = self.allocations.remove(&block).unwrap();
        self.context.device.unmap_memory(memory);
        self.context.device.free_memory(memory, None);
    }

    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (*mut u8, Range<u32>)>) {
        // TODO: only do this for non-coherent memory
        let allocations = &self.allocations;
        self.context
            .device
            .flush_mapped_memory_ranges(
                &ranges
                    .map(|(ptr, range)| {
                        let memory = allocations[&ptr];
                        vk::MappedMemoryRange::builder()
                            .memory(memory)
                            .offset(range.start as u64)
                            .size((range.end - range.start) as u64)
                            .build()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    fn can_flush(&self) -> bool {
        true
    }
}

fn select_integrated_memtype(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: &vk::MemoryRequirements,
) -> u32 {
    memory_properties.memory_types[0..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            requirements.memory_type_bits & (1 << id) != 0
                && memory_type.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                )
        })
        .or_else(|| {
            memory_properties.memory_types[0..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .position(|(id, memory_type)| {
                    requirements.memory_type_bits & (1 << id) != 0
                        && memory_type.property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                })
        })
        .unwrap() as u32
}

impl Drop for IntegratedBlockAllocator {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();
            self.context.device.destroy_buffer(self.buffer, None);
            for i in self.allocations.values() {
                self.context.device.free_memory(*i, None);
            }
        }
    }
}
