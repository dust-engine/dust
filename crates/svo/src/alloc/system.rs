use std::ops::Range;
use std::ptr::NonNull;
//use crate::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout, WriteCloneIntoRaw};
use super::AllocError;
use super::BlockAllocator;
use crate::alloc::BlockAllocation;
use std::alloc::{Allocator, Global, Layout};

pub struct SystemBlockAllocator<A: Allocator = Global> {
    allocator: A,
    block_size: u32,
}

impl SystemBlockAllocator {
    pub fn new(block_size: u32) -> SystemBlockAllocator<Global> {
        SystemBlockAllocator {
            allocator: Global,
            block_size,
        }
    }
}

impl BlockAllocator for SystemBlockAllocator {
    unsafe fn allocate_block(&self) -> Result<(*mut u8, BlockAllocation, u32), AllocError> {
        let mem = self
            .allocator
            .allocate(Layout::from_size_align_unchecked(
                self.block_size as usize,
                1,
            ))
            .map_err(|_| AllocError::OutOfHostMemory)?;
        let ptr = mem.as_mut_ptr();
        Ok((mem.as_mut_ptr(), BlockAllocation(ptr as u64), 0))
    }

    unsafe fn deallocate_block(&self, block: BlockAllocation) {
        let _layout = Layout::new::<u8>()
            .repeat(self.block_size as usize)
            .unwrap();
        self.allocator.deallocate(
            NonNull::new(block.0 as *mut u8).unwrap(),
            Layout::from_size_align_unchecked(self.block_size as usize, 1),
        );
        std::mem::forget(block);
    }

    unsafe fn flush(&self, _ranges: &mut dyn Iterator<Item = (&BlockAllocation, Range<u32>)>) {}

    fn can_flush(&self) -> bool {
        true
    }
}
