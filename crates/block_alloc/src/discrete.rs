use super::MAX_BUFFER_SIZE;
use gfx_hal as hal;
use gfx_hal::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use svo::alloc::{AllocError, BlockAllocator};

/// The voxel repository resides on both system RAM and VRAM
///
/// This provides support for discrete graphics cards, such as
/// NVIDIA, AMD where an explicit copy command needs to be issued
/// to initiate transfer between system RAM and VRAM
pub struct DiscreteBlock<B: hal::Backend, const SIZE: usize> {
    system_mem: B::Memory,
    device_mem: B::Memory,
    offset: u64,
}

pub struct DiscreteBlockAllocator<B: hal::Backend, const SIZE: usize> {
    device: Arc<B::Device>,
    bind_queue: B::Queue,
    pub device_buf: B::Buffer,
    device_memtype: hal::MemoryTypeId,
    system_buf: B::Buffer,
    system_memtype: hal::MemoryTypeId,

    copy_regions: Vec<hal::command::BufferCopy>,
    current_offset: u64,
    free_offsets: Vec<u64>,

    command_pool: B::CommandPool,
    command_buffer: B::CommandBuffer,

    allocations: HashMap<NonNull<[u8; SIZE]>, DiscreteBlock<B, SIZE>>,
}

impl<B: hal::Backend, const SIZE: usize> DiscreteBlockAllocator<B, SIZE> {
    pub fn new(
        device: Arc<B::Device>,
        bind_queue: B::Queue,
        transfer_queue_family: hal::queue::QueueFamilyId,
        memory_properties: &hal::adapter::MemoryProperties,
    ) -> Result<Self, hal::buffer::CreationError> {
        unsafe {
            let mut device_buf = device.create_buffer(
                MAX_BUFFER_SIZE,
                hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
                hal::memory::SparseFlags::SPARSE_BINDING
                    | hal::memory::SparseFlags::SPARSE_RESIDENCY,
            )?;
            device.set_buffer_name(&mut device_buf, "DiscreteBlockAllocatorDeviceBuffer");
            let mut system_buf = device.create_buffer(
                MAX_BUFFER_SIZE,
                hal::buffer::Usage::TRANSFER_SRC,
                hal::memory::SparseFlags::SPARSE_BINDING
                    | hal::memory::SparseFlags::SPARSE_RESIDENCY,
            )?;
            device.set_buffer_name(&mut system_buf, "DiscreteBlockAllocatorSystemBuffer");
            let device_buf_requirements = device.get_buffer_requirements(&device_buf);
            let system_buf_requirements = device.get_buffer_requirements(&system_buf);
            let (system_memtype, device_memtype) = select_discrete_memtype(
                memory_properties,
                &system_buf_requirements,
                &device_buf_requirements,
            );
            println!(
                "sys_memtype: {:?} \ndevice_memtype: {:?}",
                memory_properties.memory_types[system_memtype.0],
                memory_properties.memory_types[device_memtype.0]
            );

            let mut command_pool = device
                .create_command_pool(
                    transfer_queue_family,
                    hal::pool::CommandPoolCreateFlags::TRANSIENT,
                )
                .unwrap();
            let mut command_buffer = command_pool.allocate_one(hal::command::Level::Primary);
            device.set_command_buffer_name(
                &mut command_buffer,
                "DiscreteBlockAllocatorCommandBuffer",
            );
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
                allocations: HashMap::new(),
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<SIZE> for DiscreteBlockAllocator<B, SIZE> {
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
                    resource_offset: resource_offset * SIZE as u64,
                    size: SIZE as u64,
                    memory: Some((&device_mem, 0)),
                }),
            ))
            .chain(std::iter::once((
                &mut self.system_buf,
                std::iter::once(&hal::memory::SparseBind {
                    resource_offset: resource_offset * SIZE as u64,
                    size: SIZE as u64,
                    memory: Some((&system_mem, 0)),
                }),
            ))),
            std::iter::empty(),
            std::iter::empty::<(
                &mut B::Image,
                std::iter::Empty<&hal::memory::SparseImageBind<&B::Memory>>,
            )>(),
            &self.device,
            None,
        );

        let ptr = NonNull::new_unchecked(ptr as *mut [u8; SIZE]);
        let block = DiscreteBlock {
            system_mem,
            device_mem,
            offset: resource_offset,
        };
        self.allocations.insert(ptr, block);
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>) {
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

    unsafe fn flush(
        &mut self,
        ranges: &mut dyn Iterator<Item = (NonNull<[u8; SIZE]>, Range<u32>)>,
    ) {
        self.command_buffer.reset(false);
        // todo: wait for semaphores
        self.command_buffer
            .begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        let allocations = &self.allocations;
        self.command_buffer.copy_buffer(
            &self.system_buf,
            &self.device_buf,
            ranges.map(|(block, range)| {
                let location = allocations[&block].offset * SIZE as u64 + range.start as u64;
                hal::command::BufferCopy {
                    src: location,
                    dst: location,
                    size: (range.end - range.start) as u64,
                }
            }),
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
