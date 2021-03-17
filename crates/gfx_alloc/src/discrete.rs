use crate::{AllocError, BlockAllocator, MAX_BUFFER_SIZE};
use gfx_hal as hal;
use gfx_hal::prelude::*;
use std::ops::Range;
use std::ptr::NonNull;
use std::collections::HashMap;

/// The voxel repository resides on both system RAM and VRAM
///
/// This provides support for discrete graphics cards, such as
/// NVIDIA, AMD where an explicit copy command needs to be issued
/// to initiate transfer between system RAM and VRAM
pub struct DiscreteBlock<B: hal::Backend, const SIZE: usize> {
    system_mem: B::Memory,
    device_mem: B::Memory,
    ptr: NonNull<[u8; SIZE]>,
    offset: usize,
}

pub struct DiscreteBlockAllocator<'a, B: hal::Backend, const SIZE: usize> {
    device: &'a B::Device,
    bind_queue: &'a mut B::Queue,
    device_buf: B::Buffer,
    device_memtype: hal::MemoryTypeId,
    system_buf: B::Buffer,
    system_memtype: hal::MemoryTypeId,

    copy_regions: Vec<hal::command::BufferCopy>,
    current_offset: usize,
    free_offsets: Vec<usize>,

    command_pool: B::CommandPool,
    command_buffer: B::CommandBuffer,

    allocations: HashMap<NonNull<[u8; SIZE]>, DiscreteBlock<B, SIZE>>
}

impl<'a, B: hal::Backend, const SIZE: usize> DiscreteBlockAllocator<'a, B, SIZE> {
    pub fn new(
        device: &'a B::Device,
        bind_queue: &'a mut B::Queue,
        transfer_queue_family: hal::queue::QueueFamilyId,
        memory_properties: &hal::adapter::MemoryProperties,
    ) -> Result<Self, hal::buffer::CreationError> {
        unsafe {
            let device_buf = device.create_buffer(
                MAX_BUFFER_SIZE,
                hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
                hal::memory::SparseFlags::SPARSE_BINDING
                    | hal::memory::SparseFlags::SPARSE_RESIDENCY,
            )?;
            let system_buf = device.create_buffer(
                MAX_BUFFER_SIZE,
                hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_SRC,
                hal::memory::SparseFlags::SPARSE_BINDING
                    | hal::memory::SparseFlags::SPARSE_RESIDENCY,
            )?;
            let device_buf_requirements = device.get_buffer_requirements(&device_buf);
            let system_buf_requirements = device.get_buffer_requirements(&system_buf);
            let (system_memtype, device_memtype) = select_discrete_memtype(
                memory_properties,
                &system_buf_requirements,
                &device_buf_requirements,
            );

            let mut command_pool = device
                .create_command_pool(
                    transfer_queue_family,
                    hal::pool::CommandPoolCreateFlags::TRANSIENT,
                )
                .unwrap();
            let command_buffer = command_pool.allocate_one(hal::command::Level::Primary);
            Ok(Self {
                device,
                bind_queue,
                device_buf,
                device_memtype,
                system_buf,
                system_memtype,
                copy_regions: Vec::new(),
                current_offset: 0,
                free_offsets: Vec::new(),
                command_pool,
                command_buffer,
                allocations: HashMap::new()
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<SIZE>
    for DiscreteBlockAllocator<'_, B, SIZE> {

    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError> {
        let resource_offset = self.free_offsets.pop().unwrap_or_else(|| {
            let val = self.current_offset;
            self.current_offset += 1;
            val
        });
        let mut system_mem = self
            .device
            .allocate_memory(self.system_memtype, SIZE as u64)
            .map_err(crate::utils::map_alloc_err)?;
        let device_mem = self
            .device
            .allocate_memory(self.device_memtype, SIZE as u64)
            .map_err(crate::utils::map_alloc_err)?;
        let ptr = self
            .device
            .map_memory(&mut system_mem, hal::memory::Segment::ALL)
            .map_err(crate::utils::map_map_err)?;
        self.bind_queue.bind_sparse(
            std::iter::empty::<&B::Semaphore>(),
            std::iter::empty::<&B::Semaphore>(),
            std::iter::once((
                &mut self.device_buf,
                std::iter::once(&hal::memory::SparseBind {
                    resource_offset: (resource_offset * SIZE) as u64,
                    size: SIZE as u64,
                    memory: Some((&device_mem, 0)),
                }),
            ))
            .chain(std::iter::once((
                &mut self.system_buf,
                std::iter::once(&hal::memory::SparseBind {
                    resource_offset: (resource_offset * SIZE) as u64,
                    size: SIZE as u64,
                    memory: Some((&system_mem, 0)),
                }),
            ))),
            std::iter::empty(),
            std::iter::empty::<(
                &mut B::Image,
                std::iter::Empty<&hal::memory::SparseImageBind<&B::Memory>>,
            )>(),
            self.device,
            None,
        );

        let ptr = NonNull::new_unchecked(ptr as *mut [u8; SIZE]);
        let block = DiscreteBlock {
            system_mem: system_mem,
            device_mem: device_mem,
            ptr,
            offset: resource_offset,
        };
        self.allocations.insert(ptr, block);
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, mut block: NonNull<[u8; SIZE]>) {
        let mut block = self.allocations.remove(&block).unwrap();
        self.device.unmap_memory(&mut block.system_mem);
        self.device.free_memory(block.system_mem);
        self.device.free_memory(block.device_mem);
        if self.current_offset == block.offset {
            self.current_offset -= 1;
        } else {
            self.free_offsets.push(block.offset);
        }
    }

    unsafe fn updated_block(&mut self, _block: NonNull<[u8; SIZE]>, _block_range: Range<u64>) {
        self.copy_regions.push(hal::command::BufferCopy {
            src: 0,
            dst: 0,
            size: 0,
        });
    }

    unsafe fn flush(&mut self) {
        self.command_buffer.reset(false);
        // todo: wait for semaphores
        self.command_buffer
            .begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        self.command_buffer.copy_buffer(
            &self.system_buf,
            &self.device_buf,
            self.copy_regions.drain(..),
        );
        self.command_buffer.finish();
        self.bind_queue.submit(
            std::iter::once(&self.command_buffer),
            std::iter::empty(),
            std::iter::empty(),
            None,
        );
    }
}

/// Returns SystemMemId, DeviceMemId
fn select_discrete_memtype(
    memory_properties: &hal::adapter::MemoryProperties,
    system_buf_requirements: &hal::memory::Requirements,
    device_buf_requirements: &hal::memory::Requirements,
) -> (hal::MemoryTypeId, hal::MemoryTypeId) {
    let system_buf_mem_type: hal::MemoryTypeId = memory_properties
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            system_buf_requirements.type_mask & (1 << id) != 0
                && memory_type.properties.contains(
                    hal::memory::Properties::CPU_VISIBLE
                        | hal::memory::Properties::COHERENT
                        | hal::memory::Properties::CPU_CACHED,
                )
        })
        .unwrap()
        .into();

    // Search for the largest DEVICE_LOCAL heap
    let (device_heap_index, _device_heap) = memory_properties
        .memory_heaps
        .iter()
        .filter(|heap| heap.flags.contains(hal::memory::HeapFlags::DEVICE_LOCAL))
        .enumerate()
        .max_by_key(|(_, heap)| heap.size)
        .unwrap();

    let device_buf_mem_type: hal::MemoryTypeId = memory_properties
        .memory_types
        .iter()
        .filter(|ty| ty.heap_index == device_heap_index)
        .enumerate()
        .position(|(id, memory_type)| {
            device_buf_requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();

    (system_buf_mem_type, device_buf_mem_type)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::discrete::DiscreteBlockAllocator;
    use crate::BlockAllocator;
    use gfx_backend_vulkan as back;
    use gfx_hal as hal;

    pub(crate) fn get_block_allocator<const SIZE: usize>(
        gpu: &mut hal::adapter::Gpu<back::Backend>,
        memory_properties: hal::adapter::MemoryProperties,
    ) -> DiscreteBlockAllocator<back::Backend, SIZE> {
        let queue_group = gpu.queue_groups.first_mut().unwrap();
        let device = &gpu.device;
        DiscreteBlockAllocator::new(
            device,
            &mut queue_group.queues[0],
            queue_group.family,
            &memory_properties,
        )
        .unwrap()
    }

    #[test]
    fn test_discrete() {
        let (_instance, mut gpu, memory_properties) = crate::tests::get_gpu();
        let mut allocator: DiscreteBlockAllocator<back::Backend, 16777216> =
            get_block_allocator(&mut gpu, memory_properties);

        unsafe {
            let _block1 = allocator.allocate_block().unwrap();
            let block2 = allocator.allocate_block().unwrap();
            let block3 = allocator.allocate_block().unwrap();
            allocator.deallocate_block(block2);

            let block4 = allocator.allocate_block().unwrap();
            assert_eq!(block4.offset, 1);

            allocator.deallocate_block(block3);
            let block5 = allocator.allocate_block().unwrap();
            assert_eq!(block5.offset, 2);
        };
    }
}