use crate::{AllocError, BlockAllocator, MAX_BUFFER_SIZE};
use gfx_hal as hal;
use gfx_hal::prelude::*;
use std::ptr::NonNull;

/// The voxel repository resides on both system RAM and VRAM
///
/// This provides support for discrete graphics cards, such as
/// NVIDIA, AMD where an explicit copy command needs to be issued
/// to initiate transfer between system RAM and VRAM
pub struct DiscreteBlock<B: hal::Backend, const SIZE: usize> {
    system_mem: B::Memory,
    device_mem: B::Memory,
    ptr: NonNull<[u8; SIZE]>,
}

pub struct DiscreteBlockAllocator<'a, B: hal::Backend> {
    device: &'a B::Device,
    device_buf: B::Buffer,
    device_memtype: hal::MemoryTypeId,
    system_buf: B::Buffer,
    system_memtype: hal::MemoryTypeId,
}

impl<'a, B: hal::Backend> DiscreteBlockAllocator<'a, B> {
    pub fn new(
        device: &'a B::Device,
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
            Ok(Self {
                device,
                device_buf,
                device_memtype,
                system_buf,
                system_memtype,
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<B, SIZE> for DiscreteBlockAllocator<'_, B> {
    type Block = DiscreteBlock<B, SIZE>;

    unsafe fn allocate_block(&self) -> Result<Self::Block, AllocError> {
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
        Ok(Self::Block {
            system_mem: system_mem,
            device_mem: device_mem,
            ptr: NonNull::new_unchecked(ptr as *mut [u8; SIZE]),
        })
    }

    unsafe fn deallocate_block(&self, mut block: Self::Block) {
        self.device.unmap_memory(&mut block.system_mem);
        self.device.free_memory(block.system_mem);
        self.device.free_memory(block.device_mem);
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
    let (device_heap_index, device_heap) = memory_properties
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
