use rhyolite::{
    ash::{
        prelude::VkResult,
        vk::{self, BufferCreateInfo},
    },
    Allocator, Device, ManagedBuffer,
};

/// Suballocating ManagedBuffer.
/// This is designed specifically for allocating attributes.
pub struct AttributeAllocator {
    buffer: ManagedBuffer,
    freelists: Box<[Vec<u32>]>,
    alignment: u32,
    max_allocation: u32,
    head: u32,
    wasted_bytes: u32,
}
unsafe impl Send for AttributeAllocator {}
unsafe impl Sync for AttributeAllocator {}

impl AttributeAllocator {
    fn freelist_for_size(&mut self, size: u32) -> &mut Vec<u32> {
        let freelist_index = (size - 1) / self.alignment;
        &mut self.freelists[freelist_index as usize]
    }
    pub fn new_with_capacity(
        allocator: Allocator,
        capacity: u64,
        alignment: u32,
        max_allocation: u32,
    ) -> VkResult<Self> {
        let buffer = ManagedBuffer::new(
            allocator,
            capacity,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let num_freelists = max_allocation.div_ceil(alignment);
        Ok(Self {
            buffer,
            alignment,
            max_allocation,
            freelists: vec![Vec::new(); num_freelists as usize].into_boxed_slice(),
            head: 0,
            wasted_bytes: 0,
        })
    }
    pub fn allocate(&mut self, size: u32) -> u32 {
        assert!(size <= self.max_allocation);
        let increment = size.next_multiple_of(self.alignment);
        self.wasted_bytes += increment - size;
        if let Some(indice) = self.freelist_for_size(size).pop() {
            return indice;
        }
        if self.head + increment >= self.buffer.len() as u32 {
            // overflow. panic
            panic!()
        }
        let old_head = self.head;
        self.head += increment;
        return old_head;
    }
    pub fn realloc(&mut self, ptr: u32, old_size: u32, new_size: u32) -> u32 {
        let old_increment = old_size.next_multiple_of(self.alignment);
        let new_increment = new_size.next_multiple_of(self.alignment);
        if old_increment == new_increment {
            return ptr;
        }
        self.free(ptr, old_size);
        self.allocate(new_size)
    }
    pub fn free(&mut self, ptr: u32, size: u32) {
        assert!(size <= self.max_allocation);
        self.freelist_for_size(size).push(ptr);
        self.wasted_bytes -= size.next_multiple_of(self.alignment) - size;
    }

    pub fn buffer(&self) -> &ManagedBuffer {
        &self.buffer
    }
    pub fn buffer_mut(&mut self) -> &mut ManagedBuffer {
        &mut self.buffer
    }
}
