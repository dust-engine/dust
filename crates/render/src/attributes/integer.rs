use std::{marker::PhantomData, sync::Arc};

use dustash::resources::alloc::{BufferRequest, MemoryAllocScenario, MemBuffer};
use num_integer::Integer;
use vk_mem::AllocationCreateFlags;

use super::AttributeWriter;
pub struct IntegerWriter<T: Integer> {
    staging_buffer: MemBuffer,
    i: usize,
    _marker: PhantomData<T>
}

impl<T: Integer> AttributeWriter<T> for IntegerWriter<T> {
    fn new(allocator: &std::sync::Arc<dustash::resources::alloc::Allocator>, n: usize) -> Self {
        let layout = std::alloc::Layout::new::<T>().repeat(n).unwrap().0.pad_to_align();
        let staging_buffer = allocator.allocate_buffer(&BufferRequest {
            size: layout.size() as u64,
            alignment: layout.align() as u64,
            usage: ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            scenario: MemoryAllocScenario::StagingBuffer,
            allocation_flags: AllocationCreateFlags::MAPPED,
            ..Default::default()
        }).unwrap();
        Self { staging_buffer, _marker: PhantomData, i: 0 }
    }

    fn write_item(&mut self, item: T) {
        let ptr = self.staging_buffer.ptr as *mut T;
        unsafe {
            let p = ptr.add(self.i);
            *p = item;
        }
        self.i += 1;
    }

    type Resource = Arc<MemBuffer>;

    fn into_resource(self) -> Self::Resource {
        Arc::new(self.staging_buffer)
    }
}






