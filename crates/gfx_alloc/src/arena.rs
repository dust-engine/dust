use crate::BlockAllocator;
use std::mem::{size_of, ManuallyDrop};
use std::ptr::NonNull;
use std::ops::{Index, IndexMut};

pub const CHUNK_DEGREE: usize = 24;
pub const CHUNK_SIZE: usize = 1 << CHUNK_DEGREE; // 16MB per block

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
    pub fn offset(&self, n: u32) -> Self {
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
    freemask: u8,   // 0 means no children, 1 means has children
    _reserved: u16,
    next: Handle, // 32 bits
}

union ArenaSlot<T: ArenaAllocated> {
    occupied: ManuallyDrop<T>,
    free: FreeSlot,
}

pub unsafe trait ArenaAllocated: Sized {}

pub struct ArenaAllocator<T: ArenaAllocated>
where
    [T; CHUNK_SIZE / size_of::<T>()]: Sized,
{
    block_allocator: Box<dyn BlockAllocator<CHUNK_SIZE>>,
    chunks: Vec<NonNull<ArenaAllocatorChunk<T>>>,
    freelist_heads: [Handle; 8],
    newspace_top: Handle,       // new space to be allocated
    pub(crate) size: u32,       // number of allocated slots
    pub(crate) num_blocks: u32, // number of allocated blocks
    pub(crate) capacity: u32,   // number of available slots
}

impl<T: ArenaAllocated> ArenaAllocator<T>
where
    [T; CHUNK_SIZE / size_of::<T>()]: Sized,
{
    const NUM_SLOTS_IN_CHUNK: usize = CHUNK_SIZE / size_of::<T>();
    pub fn new(block_allocator: Box<dyn BlockAllocator<CHUNK_SIZE>>) -> Self {
        debug_assert_eq!(CHUNK_SIZE % size_of::<T>(), 0);
        debug_assert!(
            size_of::<T>() >= size_of::<FreeSlot>(),
            "Improper implementation of ArenaAllocated"
        );
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
    pub fn alloc(&mut self, len: u32) -> Handle {
        assert!(0 < len && len <= 8, "Only supports block size between 1-8!");
        self.size += len;
        self.num_blocks += 1;

        // Retrieve the head of the freelist
        let sized_head = self.freelist_heads[len as usize - 1];
        let handle: Handle = if sized_head.is_none() {
            // If the head is none, it means we need to allocate some new slots
            if self.newspace_top.is_none() {
                // We've run out of newspace.
                // Allocate a new memory chunk from the underlying block allocator.
                let chunk_index = self.chunks.len() as u32;
                let chunk = unsafe { self.block_allocator.allocate_block().unwrap() };
                self.chunks.push(chunk.cast());
                self.capacity += Self::NUM_SLOTS_IN_CHUNK as u32;
                self.newspace_top = Handle::from_index(chunk_index, len);
                Handle::from_index(chunk_index, 0)
            } else {
                // There's still space remains to be allocated in the current chunk.
                let handle = self.newspace_top;
                let slot_index = handle.get_slot_num();
                let chunk_index = handle.get_chunk_num();
                let remaining_space = Self::NUM_SLOTS_IN_CHUNK as u32 - slot_index - len;

                let new_handle = Handle::from_index(chunk_index, slot_index + len);
                if remaining_space > 8 {
                    self.newspace_top = new_handle;
                } else {
                    if remaining_space > 0 {
                        self.freelist_push(remaining_space as u8, new_handle);
                    }
                    self.newspace_top = Handle::none();
                }
                handle
            }
        } else {
            // There's previously used blocks stored in the freelist. Use them first.
            self.freelist_heads[len as usize - 1] = unsafe {
                // TODO: function?
                self.get_slot(sized_head).free.next
            };
            sized_head
        };

        for i in 0..len {
            // TODO: Get rid of the loop here
            // TODO: Use occupied here.
            let slot_handle = handle.offset(i);
            let mut slot = self.get_slot_mut(slot_handle);
            slot.free.block_size = len as u8;
            slot.free.freemask = 0;
        }
        handle
    }
    pub fn free(&mut self, handle: Handle) {
        let block_size = unsafe { self.get_slot(handle).free.block_size }; // TODO: use occupied here
        assert!(block_size > 0, "Double free detected");
        for i in 0..block_size {
            let new_handle = handle.offset(i as u32);
            let slot = self.get_slot_mut(new_handle);
            unsafe {
                debug_assert_eq!(
                    slot.free.block_size, block_size,
                    "Overlapping handle detected"
                );
                slot.occupied = std::mem::zeroed();
            }
        }
        self.freelist_push(block_size, handle);
        self.size -= block_size as u32;
        self.num_blocks -= 1;
    }
    fn freelist_push(&mut self, n: u8, handle: Handle) {
        assert!(1 <= n && n <= 8);
        self.get_slot_mut(handle).free.next = self.freelist_heads[(n - 1) as usize];
        self.freelist_heads[(n - 1) as usize] = handle;
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

impl<T: ArenaAllocated> Index<Handle> for ArenaAllocator<T>
    where
        [T; CHUNK_SIZE / size_of::<T>()]: Sized,
{
    type Output = T;

    fn index(&self, index: Handle) -> &Self::Output {
        // TODO: check that the chunk was allocated
        unsafe { &self.get_slot(index).occupied }
    }
}

impl<T: ArenaAllocated> IndexMut<Handle> for ArenaAllocator<T>
    where
        [T; CHUNK_SIZE / size_of::<T>()]: Sized,
{
    fn index_mut(&mut self, index: Handle) -> &mut Self::Output {
        // TODO: check that the chunk was allocated
        unsafe { &mut self.get_slot_mut(index).occupied }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::size_of;

    unsafe impl ArenaAllocated for u128 {}

    #[test]
    fn test_alloc() {
        let (_instance, mut gpu, memory_properties) = crate::tests::get_gpu();
        let allocator = crate::discrete::tests::get_block_allocator(&mut gpu, memory_properties);
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

    #[test]
    #[should_panic(expected = "Double free detected")]
    fn test_doublefree() {
        let (_instance, mut gpu, memory_properties) = crate::tests::get_gpu();
        let allocator = crate::discrete::tests::get_block_allocator(&mut gpu, memory_properties);
        type Data = u128;
        let mut arena: ArenaAllocator<_, Data> = ArenaAllocator::new(allocator);
        let handle = arena.alloc(3);
        arena.free(handle);
        arena.free(handle);
    }

    #[test]
    #[should_panic(expected = "Overlapping handle detected")]
    fn test_invalid_overlapping_handle() {
        let (_instance, mut gpu, memory_properties) = crate::tests::get_gpu();
        let allocator = crate::discrete::tests::get_block_allocator(&mut gpu, memory_properties);
        type Data = u128;
        let mut arena: ArenaAllocator<_, Data> = ArenaAllocator::new(allocator);
        let handle = arena.alloc(8);
        arena.free(handle.offset(4));
    }
}
