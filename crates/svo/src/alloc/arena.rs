use super::BlockAllocator;
use crate::alloc::changeset::ChangeSet;

use std::mem::{size_of, ManuallyDrop};

use crate::alloc::BlockAllocation;
use std::ptr::NonNull;
use std::sync::Arc;

pub const BLOCK_MASK_DEGREE: u32 = 20;
pub const NUM_SLOTS_IN_BLOCK: u32 = 1 << BLOCK_MASK_DEGREE;
pub const BLOCK_SIZE: u64 = NUM_SLOTS_IN_BLOCK as u64 * 24;
pub const BLOCK_MASK: u32 = NUM_SLOTS_IN_BLOCK - 1;

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Handle(u32);
impl Handle {
    #[inline]
    pub const fn none() -> Self {
        Handle(u32::MAX)
    }
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == u32::MAX
    }
    #[inline]
    pub fn offset(&self, n: u32) -> Self {
        Handle(self.0 + n)
    }
    #[inline]
    pub fn get_slot_num(&self) -> u32 {
        self.0 & BLOCK_MASK
    }
    #[inline]
    pub fn get_chunk_num(&self) -> u32 {
        self.0 >> BLOCK_MASK_DEGREE
    }
    #[inline]
    pub fn from_index(chunk_index: u32, block_index: u32) -> Handle {
        Handle(chunk_index << BLOCK_MASK_DEGREE | block_index)
    }
}

impl Default for Handle {
    fn default() -> Self {
        Handle::none()
    }
}

pub type ArenaBlockAllocator = dyn BlockAllocator;

#[repr(C)]
struct FreeSlot {
    next: Handle, // 32 bits
}

union ArenaSlot<T: ArenaAllocated> {
    occupied: ManuallyDrop<T>,
    free: FreeSlot,
}

pub trait ArenaAllocated: Sized + Default {}

pub struct ArenaAllocator<T: ArenaAllocated> {
    block_allocator: Arc<ArenaBlockAllocator>,
    chunks: Vec<(NonNull<ArenaSlot<T>>, BlockAllocation)>,
    freelist_heads: [Handle; 16],
    newspace_top: Handle,       // new space to be allocated
    pub(crate) size: u32,       // number of allocated slots
    pub(crate) num_blocks: u32, // number of allocated blocks
    pub(crate) capacity: u32,   // number of available slots

    pub changeset: ChangeSet,
}

// ArenaAllocator contains NunNull which makes it !Send and !Sync.
// NonNull is !Send and !Sync because the data they reference may be aliased.
// Here we guarantee that NonNull will never be aliased.
// Therefore ArenaAllocator should be Send and Sync.
unsafe impl<T: ArenaAllocated> Send for ArenaAllocator<T> {}
unsafe impl<T: ArenaAllocated> Sync for ArenaAllocator<T> {}

impl<T: ArenaAllocated> ArenaAllocator<T> {
    pub fn new(block_allocator: Arc<ArenaBlockAllocator>) -> Self {
        debug_assert!(
            size_of::<T>() >= size_of::<FreeSlot>(),
            "Improper implementation of ArenaAllocated"
        );
        Self {
            block_allocator,
            chunks: vec![],
            freelist_heads: [Handle::none(); 16],
            // Space pointed by this is guaranteed to have free space > 8
            newspace_top: Handle::none(),
            size: 0,
            num_blocks: 0,
            capacity: 0,
            changeset: ChangeSet::new(0),
        }
    }
    pub fn alloc_block(&mut self) -> Handle {
        let chunk_index = self.chunks.len() as u32;
        let (chunk, allocation) = unsafe { self.block_allocator.allocate_block().unwrap() };
        self.chunks
            .push(unsafe { (NonNull::new_unchecked(chunk as _), allocation) });
        self.capacity += NUM_SLOTS_IN_BLOCK;
        Handle::from_index(chunk_index, 0)
    }
    pub fn alloc(&mut self, len: u32) -> Handle {
        assert!(0 < len && len <= 16, "Only supports block size between 1-16!");
        self.size += len;
        self.num_blocks += 1;


        // println!("Allocating");
        // Retrieve the head of the freelist
        let sized_head = self.freelist_pop(len as u8);
        let handle: Handle = if sized_head.is_none() {
            // If the head is none, it means we need to allocate some new slots
            if self.newspace_top.is_none() {
                // We've run out of newspace.
                // Allocate a new memory chunk from the underlying block allocator.
                let alloc_head = self.alloc_block();
                self.newspace_top = Handle::from_index(alloc_head.get_chunk_num(), len);
                alloc_head
            } else {
                // There's still space remains to be allocated in the current chunk.
                let handle = self.newspace_top;
                let slot_index = handle.get_slot_num();
                let chunk_index = handle.get_chunk_num();
                let remaining_space = NUM_SLOTS_IN_BLOCK - slot_index - len;

                let new_handle = Handle::from_index(chunk_index, slot_index + len);
                if remaining_space > 16 {
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
            sized_head
        };

        // println!("Finished allocating");

        // initialize to zero
        let slot_index = handle.get_slot_num();
        let chunk_index = handle.get_chunk_num();
        unsafe {
            let base = self.chunks[chunk_index as usize]
                .0
                .as_ptr()
                .add(slot_index as usize);
            for i in 0..len {
                let i = &mut *base.add(i as usize);
                i.occupied = Default::default();
            }
        }
        handle
    }
    pub unsafe fn free(&mut self, handle: Handle, block_size: u8) {
        self.freelist_push(block_size, handle);
        self.size -= block_size as u32;
        self.num_blocks -= 1;
    }
    fn freelist_push(&mut self, n: u8, handle: Handle) {
        debug_assert!(1 <= n && n <= 16);
        let index: usize = (n - 1) as usize;
        self.get_slot_mut(handle).free.next = self.freelist_heads[index];
        self.freelist_heads[index] = handle;
    }
    fn freelist_pop(&mut self, n: u8) -> Handle {
        let index: usize = (n - 1) as usize;
        let sized_head = self.freelist_heads[index];
        if !sized_head.is_none() {
            self.freelist_heads[index] = unsafe { self.get_slot(sized_head).free.next };
        }
        sized_head
    }
    #[inline]
    fn get_slot(&self, handle: Handle) -> &ArenaSlot<T> {
        let slot_index = handle.get_slot_num();
        let chunk_index = handle.get_chunk_num();
        unsafe {
            let base = self.chunks[chunk_index as usize].0.as_ptr();
            &*base.add(slot_index as usize)
        }
    }
    #[inline]
    fn get_slot_mut(&mut self, handle: Handle) -> &mut ArenaSlot<T> {
        let slot_index = handle.get_slot_num();
        let chunk_index = handle.get_chunk_num();
        unsafe {
            let base = self.chunks[chunk_index as usize].0.as_ptr();
            &mut *base.add(slot_index as usize)
        }
    }

    // method here due to compiler bug
    #[inline]
    pub fn get(&self, index: Handle) -> &T {
        unsafe {
            let slot = self.get_slot(index);
            &slot.occupied
        }
    }
    #[inline]
    pub fn get_mut(&mut self, index: Handle) -> &mut T {
        unsafe {
            let slot = self.get_slot_mut(index);
            &mut slot.occupied
        }
    }
    pub fn changed(&mut self, index: Handle) {
        self.changeset.changed(index)
    }
    pub fn changed_block(&mut self, index: Handle, len: u32) {
        self.changeset.changed_block(index, len)
    }
    pub fn flush(&mut self) {
        if self.changeset.len() == 0 {
            return;
        }
        if !self.block_allocator.can_flush() {
            return;
        }
        let chunks = &self.chunks;
        let mut iter = self.changeset.drain().map(|(chunk_index, range)| {
            let ptr = &chunks[chunk_index].1;
            let size: u32 = size_of::<T>() as u32;
            let range = (range.start * size)..(range.end * size);
            (ptr, range)
        });
        unsafe {
            self.block_allocator.flush(&mut iter);
        }
    }
}

impl<T: ArenaAllocated> Drop for ArenaAllocator<T> {
    fn drop(&mut self) {
        for (_, allocation) in self.chunks.drain(..) {
            unsafe {
                self.block_allocator.deallocate_block(allocation);
            }
        }
    }
}
/* Disabled due to compiler bug
impl<T: ArenaAllocated> Index<Handle> for ArenaAllocator<T>
{
    type Output = T;

    fn index(&self, index: Handle) -> &Self::Output {
        unsafe {
            let slot = self.get_slot(index);
            &slot.occupied
        }
    }
}

impl<T: ArenaAllocated> IndexMut<Handle> for ArenaAllocator<T>
{
    fn index_mut(&mut self, index: Handle) -> &mut Self::Output {
        unsafe {
            let slot = self.get_slot_mut(index);
            &mut slot.occupied
        }
    }
}
*/
#[cfg(untested)]
mod tests {
    use super::*;

    use std::mem::size_of;

    impl ArenaAllocated for u128 {}

    #[test]
    fn test_alloc() {
        let block_allocator = crate::alloc::SystemBlockAllocator::new();
        let mut arena: ArenaAllocator<u128> = ArenaAllocator::new(Box::new(block_allocator));
        let num_slots_in_chunk = CHUNK_SIZE / size_of::<u128>();
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
    fn test_free() {
        let block_allocator = crate::alloc::SystemBlockAllocator::new();
        let mut arena: ArenaAllocator<u128> = ArenaAllocator::new(Box::new(block_allocator));
        let handles: Vec<Handle> = (0..8).map(|_| arena.alloc(4)).collect();
        for handle in handles.iter().rev() {
            unsafe { arena.free(*handle, 4) };
        }
        assert_eq!(arena.alloc(1), Handle(8 * 4));
        for handle in handles.iter() {
            let new_handle = arena.alloc(4);
            assert_eq!(*handle, new_handle);
        }
    }
}
