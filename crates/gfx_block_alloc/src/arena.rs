use crate::AllocatorBlock;
use crate::BlockAllocator;
use std::ptr::NonNull;
use std::mem::size_of;
use std::mem::ManuallyDrop;

const CHUNK_DEGREE: usize = 24;
const CHUNK_SIZE: usize = 1 << CHUNK_DEGREE; // 16MB per block

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Handle(u32);
impl Handle {
    pub const fn none() -> Self {
        Handle(std::u32::MAX)
    }
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == std::u32::MAX
    }
    pub(crate) fn offset(&self, n: u32) -> Self {
        Handle(self.0 + n)
    }
    pub fn get_slot_num(&self) -> u32 {
        let mask = CHUNK_SIZE as u32 - 1;
        self.0 & mask
    }
    pub fn get_chunk_num(&self) -> u32 {
        self.0 >> CHUNK_DEGREE
    }
    pub fn from_index(chunk_index: u32, block_index: u32) -> Handle {
        Handle(chunk_index << CHUNK_DEGREE | block_index)
    }
}



type ArenaAllocatorChunk<T> = [ArenaSlot<T>; CHUNK_SIZE / size_of::<T>()];

#[repr(C)]
struct FreeSlot {
    block_size: u8, // This value is 0 for free blocks
    freemask: u8,              // 0 means no children, 1 means has children
    _reserved: u16,
    next: Handle, // 32 bits
}


union ArenaSlot<T: ArenaAllocated> {
    occupied: ManuallyDrop<T>,
    free: FreeSlot
}

pub unsafe trait ArenaAllocated: Sized {

}

pub struct ArenaAllocator<BA: BlockAllocator<CHUNK_SIZE>, T: ArenaAllocated>
    where [T; CHUNK_SIZE / size_of::<T>()]: Sized {
    block_allocator: BA,
    chunks: Vec<NonNull<ArenaAllocatorChunk<T>>>,
    freelist_heads: [Handle; 8],
    newspace_top: Handle, // new space to be allocated
    pub(crate) size: u32,         // number of allocated slots
    pub(crate) num_blocks: u32,   // number of allocated blocks
    pub(crate) capacity: u32,     // number of available slots
}

impl<BA: BlockAllocator<CHUNK_SIZE>, T: ArenaAllocated> ArenaAllocator<BA, T>
    where [T; CHUNK_SIZE / size_of::<T>()]: Sized {
    const NUM_SLOTS_IN_CHUNK: usize = CHUNK_SIZE / size_of::<T>();
    pub fn new(block_allocator: BA) -> Self {
        debug_assert_eq!(CHUNK_SIZE % size_of::<T>(), 0);
        debug_assert!(size_of::<T>() >= size_of::<FreeSlot>(), "Improper implementation of ArenaAllocated");
        Self {
            block_allocator,
            chunks: vec![],
            freelist_heads: [
                Handle::none(),
                Handle::none(),
                Handle::none(),
                Handle::none(),
                Handle::none(),
                Handle::none(),
                Handle::none(),
                Handle::none(),
            ],
            // Space pointed by this is guaranteed to have free space > 8
            newspace_top: Handle::none(),
            size: 0,
            num_blocks: 0,
            capacity: 0,
        }
    }
    pub fn alloc(&mut self, len: u32) -> Handle{
        assert!(0 < len && len <= 8, "Only supports block size between 1-8!");
        self.size += len;
        self.num_blocks += 1;

        // Retrieve the head of the freelist
        let sized_head = self.freelist_heads[len as usize - 1];
        let handle: Handle = if sized_head.is_none() {
            // If the head is none, it means we need to allocate some new slots
            if self.newspace_top.is_none() {
                println!("Allocating new");
                // We've run out of newspace.
                // Allocate a new memory chunk from the underlying block allocator.
                let chunk_index = self.chunks.len() as u32;
                let chunk = unsafe {
                    self.block_allocator.allocate_block().unwrap().ptr()
                };
                self.chunks.push(chunk.cast());
                self.capacity += Self::NUM_SLOTS_IN_CHUNK as u32;
                self.newspace_top = Handle::from_index(chunk_index, len);
                println!("Allocating new done");
                Handle::from_index(chunk_index, 0)
            } else {
                // There's still space remains to be allocated in the current chunk.
                let handle = self.newspace_top;
                let mask: u32 = CHUNK_SIZE as u32 - 1;
                let slot_index = handle.get_slot_num();
                let chunk_index = handle.get_chunk_num();
                let remaining_space = Self::NUM_SLOTS_IN_CHUNK as u32 - slot_index - len;

                let new_handle = Handle::from_index(chunk_index, slot_index + len);
                if remaining_space > 8 {
                    self.newspace_top = new_handle;
                } else {
                    println!("Starinasdfasdf");
                    if remaining_space > 0 {
                        println!("Adding some remaining space");
                        // freelist push
                        self.get_slot_mut(new_handle).free.next = self.freelist_heads[remaining_space as usize - 1];
                        self.freelist_heads[remaining_space as usize - 1] = new_handle;
                    }
                    self.newspace_top = Handle::none();
                }
                handle
            }
        } else {
            println!("Reuse");
            // There's previously used blocks stored in the freelist. Use them first.
            self.freelist_heads[len as usize - 1] = unsafe {
                self.get_slot(sized_head).free.next
            };
            sized_head
        };

        for i in 0..len {
            unsafe {
                let slot_handle = handle.offset(i);
                let mut slot = self.get_slot_mut(slot_handle);
                slot.free.block_size = len as u8;
                slot.free.freemask = 0;
            }
        }
        handle
    }

    fn get_slot(&self, handle: Handle) -> &ArenaSlot<T> {
        let slot_index = handle.get_slot_num();
        let chunk_index = handle.get_chunk_num();
        unsafe {
            let slice = self.chunks[chunk_index as usize].as_ref();
            &slice[slot_index as usize]
        }
    }
    fn get_slot_mut(&mut self, handle: Handle) -> &mut ArenaSlot<T> {
        let slot_index = handle.get_slot_num();
        let chunk_index = handle.get_chunk_num();
        unsafe {
            let slice = self.chunks[chunk_index as usize].as_mut();
            &mut slice[slot_index as usize]
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use crate::DiscreteBlockAllocator;
    use gfx_backend_vulkan as back;
    use gfx_hal as hal;
    use hal::prelude::*;

    unsafe impl ArenaAllocated for u128 {}

    #[test]
    fn test_alloc() {
        {
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
            let allocator: DiscreteBlockAllocator<back::Backend, 16777216> = DiscreteBlockAllocator::new(
                &device,
                &mut queue_group.queues[0],
                queue_group.family,
                &memory_properties,
            )
                .unwrap();
            type Data = u128;
            let mut arena: ArenaAllocator<_, Data> = ArenaAllocator::new(allocator);
            let num_slots_in_chunk = CHUNK_SIZE / size_of::<Data>();
            for i in 0..(num_slots_in_chunk as u32 - 8) {
                let handle = arena.alloc(1);
                assert_eq!(handle.get_slot_num(), i);
                assert_eq!(handle.get_chunk_num(), 0);
            }
            assert_eq!(arena.capacity, num_slots_in_chunk as u32);
            for i in 0..10 {
                let handle = arena.alloc(1);
                assert_eq!(handle.get_slot_num(), i);
                assert_eq!(handle.get_chunk_num(), 1);
            }
            assert_eq!(arena.capacity, num_slots_in_chunk as u32 * 2);
            assert_eq!(
                arena.freelist_heads[7],
                Handle(num_slots_in_chunk as u32 - 8)
            );
            let handle = arena.alloc(5);
            assert_eq!(handle.get_slot_num(), 10);
            assert_eq!(handle.get_chunk_num(), 1);
            let handle = arena.alloc(8);
            assert_eq!(handle.get_slot_num(), num_slots_in_chunk as u32 - 8);
            assert_eq!(handle.get_chunk_num(), 0);
        }
    }
}