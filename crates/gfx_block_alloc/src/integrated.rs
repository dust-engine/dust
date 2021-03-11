use crate::{AllocError, BlockAllocator, MAX_BUFFER_SIZE};
use gfx_hal as hal;
use gfx_hal::prelude::*;
use std::ops::Range;
use std::ptr::NonNull;

/// The voxel repository resides on both system RAM and VRAM
///
/// This provides support for discrete graphics cards, such as
/// NVIDIA, AMD where an explicit copy command needs to be issued
/// to initiate transfer between system RAM and VRAM
pub struct IntegratedBlock<B: hal::Backend, const SIZE: usize> {
    mem: B::Memory,
    ptr: NonNull<[u8; SIZE]>,
}

pub struct IntegratedBlockAllocator<'a, B: hal::Backend> {
    device: &'a B::Device,
    buf: B::Buffer,
    memtype: hal::MemoryTypeId,
}

impl<'a, B: hal::Backend> IntegratedBlockAllocator<'a, B> {
    pub fn new(
        device: &'a B::Device,
        memory_properties: &hal::adapter::MemoryProperties,
    ) -> Result<Self, hal::buffer::CreationError> {
        unsafe {
            let buf = device.create_buffer(
                MAX_BUFFER_SIZE,
                hal::buffer::Usage::STORAGE,
                hal::memory::SparseFlags::SPARSE_BINDING
                    | hal::memory::SparseFlags::SPARSE_RESIDENCY,
            )?;
            let requirements = device.get_buffer_requirements(&buf);
            let memtype = select_integrated_memtype(memory_properties, &requirements);
            Ok(Self {
                device,
                buf,
                memtype,
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<SIZE> for IntegratedBlockAllocator<'_, B> {
    type Block = IntegratedBlock<B, SIZE>;

    unsafe fn allocate_block(&mut self) -> Result<Self::Block, AllocError> {
        let mut mem = self
            .device
            .allocate_memory(self.memtype, SIZE as u64)
            .map_err(crate::utils::map_alloc_err)?;
        let ptr = self
            .device
            .map_memory(&mut mem, hal::memory::Segment::ALL)
            .map_err(crate::utils::map_map_err)?;
        Ok(Self::Block {
            mem,
            ptr: NonNull::new_unchecked(ptr as *mut [u8; SIZE]),
        })
    }

    unsafe fn deallocate_block(&mut self, mut block: Self::Block) {
        self.device.unmap_memory(&mut block.mem);
        self.device.free_memory(block.mem);
    }

    unsafe fn updated_block(&mut self, block: &Self::Block, block_range: Range<u64>) {
        // Do exactly nothing. Nothing needs to be done to sync data to the GPU.
        unimplemented!()
    }

    unsafe fn flush(&mut self) {
        unimplemented!()
    }
}

/// Returns SystemMemId, DeviceMemId
fn select_integrated_memtype(
    memory_properties: &hal::adapter::MemoryProperties,
    requirements: &hal::memory::Requirements,
) -> hal::MemoryTypeId {
    // Search for the largest DEVICE_LOCAL heap
    let (device_heap_index, device_heap) = memory_properties
        .memory_heaps
        .iter()
        .filter(|heap| heap.flags.contains(hal::memory::HeapFlags::DEVICE_LOCAL))
        .enumerate()
        .max_by_key(|(_, heap)| heap.size)
        .unwrap();

    let mut mem_properties = memory_properties
        .memory_types
        .iter()
        .filter(|ty| ty.heap_index == device_heap_index)
        .enumerate();

    mem_properties
        .position(|(id, memory_type)| {
            requirements.type_mask & (1 << id) != 0
                && memory_type.properties.contains(
                    hal::memory::Properties::DEVICE_LOCAL
                        | hal::memory::Properties::CPU_VISIBLE
                        | hal::memory::Properties::COHERENT
                        | hal::memory::Properties::CPU_CACHED,
                )
        })
        .or_else(|| {
            mem_properties.position(|(id, memory_type)| {
                requirements.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(
                        hal::memory::Properties::DEVICE_LOCAL
                            | hal::memory::Properties::CPU_VISIBLE
                            | hal::memory::Properties::COHERENT,
                    )
            })
        })
        .unwrap()
        .into()
}
