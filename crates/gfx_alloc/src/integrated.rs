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
pub struct IntegratedBlock<B: hal::Backend, const SIZE: usize> {
    mem: B::Memory,
    ptr: NonNull<[u8; SIZE]>,
}

pub struct IntegratedBlockAllocator<'a, B: hal::Backend, const SIZE: usize> {
    device: &'a B::Device,
    bind_queue: &'a mut B::Queue,
    buf: B::Buffer,
    memtype: hal::MemoryTypeId,
    current_offset: usize,
    free_offsets: Vec<usize>,

    allocations: HashMap<NonNull<[u8; SIZE]>, IntegratedBlock<B, SIZE>>
}

impl<'a, B: hal::Backend, const SIZE: usize> IntegratedBlockAllocator<'a, B, SIZE> {
    pub fn new(
        device: &'a B::Device,
        bind_queue: &'a mut B::Queue,
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
                allocations: HashMap::new()
            })
        }
    }
}

impl<B: hal::Backend, const SIZE: usize> BlockAllocator<SIZE>
    for IntegratedBlockAllocator<'_, B, SIZE>
{

    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError> {
        let resource_offset = self.free_offsets.pop().unwrap_or_else(|| {
            let val = self.current_offset;
            self.current_offset += 1;
            val
        });
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
                    resource_offset: (resource_offset * SIZE) as u64,
                    size: SIZE as u64,
                    memory: Some((&mem, 0)),
                }),
            )),
            std::iter::empty(),
            std::iter::empty::<(
                &mut B::Image,
                std::iter::Empty<&hal::memory::SparseImageBind<&B::Memory>>,
            )>(),
            self.device,
            None,
        );
        let ptr = NonNull::new_unchecked(ptr as *mut [u8; SIZE]);
        self.allocations.insert(ptr, IntegratedBlock {
            mem,
            ptr,
        });
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>) {
        let mut block = self.allocations.remove(&block).unwrap();
        self.device.unmap_memory(&mut block.mem);
        self.device.free_memory(block.mem);
    }

    unsafe fn updated_block(&mut self, _block: NonNull<[u8; SIZE]>, _block_range: Range<u64>) {
        // Do exactly nothing. Nothing needs to be done to sync data to the GPU.
    }

    unsafe fn flush(&mut self) {}
}

/// Returns SystemMemId, DeviceMemId
fn select_integrated_memtype(
    memory_properties: &hal::adapter::MemoryProperties,
    requirements: &hal::memory::Requirements,
) -> hal::MemoryTypeId {
    // Search for the largest DEVICE_LOCAL heap
    let (device_heap_index, _device_heap) = memory_properties
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

#[cfg(test)]
mod tests {
    use super::IntegratedBlockAllocator;
    use crate::BlockAllocator;
    use gfx_backend_vulkan as back;
    use gfx_hal as hal;
    use gfx_hal::prelude::*;

    //#[test]
    fn test_integrated() {
        let instance = back::Instance::create("gfx_test", 1).expect("Unable to create an instance");
        let adapters = instance.enumerate_adapters();
        let adapter = {
            for adapter in &instance.enumerate_adapters() {
                println!("{:?}", adapter);
            }
            adapters
                .iter()
                .find(|adapter| adapter.info.device_type == hal::adapter::DeviceType::DiscreteGpu)
        }
        .expect("Unable to find a discrete GPU");

        let physical_device = &adapter.physical_device;
        let memory_properties = physical_device.memory_properties();
        let family = adapter
            .queue_families
            .iter()
            .find(|family| family.queue_type() == hal::queue::QueueType::Transfer)
            .expect("Can't find transfer queue family!");
        let mut gpu = unsafe {
            physical_device.open(
                &[(family, &[1.0])],
                hal::Features::SPARSE_BINDING | hal::Features::SPARSE_RESIDENCY_IMAGE_2D,
            )
        }
        .expect("Unable to open the physical device!");
        let mut queue_group = gpu.queue_groups.pop().unwrap();
        let device = gpu.device;
        let mut allocator: IntegratedBlockAllocator<back::Backend, 16777216> =
            IntegratedBlockAllocator::new(&device, &mut queue_group.queues[0], &memory_properties)
                .unwrap();

        unsafe {
            let _block1 = allocator.allocate_block().unwrap();
            let block2 = allocator.allocate_block().unwrap();
            let block3 = allocator.allocate_block().unwrap();
            allocator.deallocate_block(block2);

            let _block4 = allocator.allocate_block().unwrap();

            allocator.deallocate_block(block3);
            let _block5 = allocator.allocate_block().unwrap();
        };
    }
}
