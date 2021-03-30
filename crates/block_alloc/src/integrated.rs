use super::MAX_BUFFER_SIZE;
use gfx_hal as hal;
use gfx_hal::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use svo::alloc::{AllocError, BlockAllocator};

pub struct IntegratedBlockAllocator<B: hal::Backend, const SIZE: usize> {
    device: Arc<B::Device>,
    bind_queue: B::Queue,
    pub buf: B::Buffer,
    memtype: hal::MemoryTypeId,
    current_offset: u64,
    free_offsets: Vec<u64>,

    allocations: HashMap<NonNull<[u8; SIZE]>, B::Memory>,
}

impl<B: hal::Backend, const SIZE: usize> IntegratedBlockAllocator<B, SIZE> {
    pub fn new(
        device: Arc<B::Device>,
        bind_queue: B::Queue,
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
                bind_queue,
                buf,
                memtype,
                free_offsets: Vec::new(),
                current_offset: 0,
                allocations: HashMap::new(),
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<SIZE>
    for IntegratedBlockAllocator<B, SIZE>
{
    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError> {
        let resource_offset = self.free_offsets.pop().unwrap_or_else(|| {
            let val = self.current_offset;
            self.current_offset += 1;
            val
        });
        println!("Allocated {}", SIZE);
        let mut mem = self
            .device
            .allocate_memory(self.memtype, SIZE as u64)
            .map_err(crate::utils::map_alloc_err)?;
        let ptr = self
            .device
            .map_memory(&mut mem, hal::memory::Segment::ALL)
            .map_err(crate::utils::map_map_err)?;

        self.bind_queue.bind_sparse(
            std::iter::empty::<&B::Semaphore>(),
            std::iter::empty::<&B::Semaphore>(),
            std::iter::once((
                &mut self.buf,
                std::iter::once(&hal::memory::SparseBind {
                    resource_offset: resource_offset * SIZE as u64,
                    size: SIZE as u64,
                    memory: Some((&mem, 0)),
                }),
            )),
            std::iter::empty(),
            std::iter::empty::<(
                &mut B::Image,
                std::iter::Empty<&hal::memory::SparseImageBind<&B::Memory>>,
            )>(),
            &self.device,
            None,
        );
        let ptr = NonNull::new_unchecked(ptr as *mut [u8; SIZE]);
        self.allocations.insert(ptr, mem);
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>) {
        let mut memory = self.allocations.remove(&block).unwrap();
        self.device.unmap_memory(&mut memory);
        self.device.free_memory(memory);
    }

    unsafe fn flush(
        &mut self,
        ranges: &mut dyn Iterator<Item = (NonNull<[u8; SIZE]>, Range<u32>)>,
    ) {
        println!("Flushed {:?}", ranges.size_hint());
        let allocations = &self.allocations;
        self.device
            .flush_mapped_memory_ranges(ranges.map(|(ptr, range)| {
                let memory = &allocations[&ptr];
                (
                    memory,
                    hal::memory::Segment {
                        offset: range.start as u64,
                        size: Some((range.end - range.start) as u64),
                    },
                )
            }));
    }
}

/// Returns SystemMemId, DeviceMemId
fn select_integrated_memtype(
    memory_properties: &hal::adapter::MemoryProperties,
    requirements: &hal::memory::Requirements,
) -> hal::MemoryTypeId {
    memory_properties
        .memory_types
        .iter()
        .enumerate()
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
            memory_properties
                .memory_types
                .iter()
                .enumerate()
                .position(|(id, memory_type)| {
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
